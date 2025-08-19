# app.py — MDTWRR Portal (robust CSV mapping, clean Dietz + Brinson, reporting, Excel export)

from __future__ import annotations
import io
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# -------------------------------------------------
# App & constants
# -------------------------------------------------
st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.title("MDTWRR Performance Portal")
st.caption("Modified Dietz returns + Brinson attribution. No fluff.")

MIN_DATE = date(1900, 1, 1)
MAX_DATE = date(2100, 12, 31)
INFLOW, OUTFLOW = "INFLOW", "OUTFLOW"
ALLOWED_TYPES = [INFLOW, OUTFLOW]

# -------------------------------------------------
# Utilities
# -------------------------------------------------
def pct(x: float, dp: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{x*100:.{dp}f}%"

def to_float(x):
    if x is None: return np.nan
    try: return float(str(x).replace(",", "").strip())
    except Exception: return np.nan

def clamp01(x: float) -> float: return max(0.0, min(1.0, x))

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw))  # last attempt (may throw)

def modified_dietz(bv: float, ev: float, flows: pd.DataFrame, t0: datetime, t1: datetime, eps=1e-12) -> float:
    """
    R = (EV - BV - sum(CF)) / (BV + sum(w_i * CF_i)), w_i = (t1 - t_i)/(t1 - t0)
    flows: ['when','amount'] signed (+ INFLOW, - OUTFLOW). Returns decimal (0.12 = 12%).
    """
    if not np.isfinite(bv) or not np.isfinite(ev):
        raise ValueError("BV/EV must be finite")
    if not (isinstance(t0, datetime) and isinstance(t1, datetime) and t1 > t0):
        raise ValueError("Bad dates")
    T = (t1 - t0).total_seconds()
    if T <= 0: raise ValueError("Zero/negative period")

    f = pd.DataFrame(columns=["when","amount"]) if flows is None else flows.copy()
    if not f.empty:
        f = f[(f["when"] > t0) & (f["when"] <= t1)]
        f = f[np.isfinite(f["amount"])]

    sum_cf = 0.0 if f.empty else f["amount"].sum()
    denom = bv
    if not f.empty:
        f["w"] = f["when"].apply(lambda w: clamp01((t1 - w).total_seconds() / T))
        denom += (f["w"] * f["amount"]).sum()
    if abs(denom) < eps:  # avoid explosions
        raise ValueError("Unstable denominator")

    numer = ev - bv - sum_cf
    return numer / denom

def brinson_bhb(rows: list[dict]) -> pd.DataFrame:
    """
    rows: dicts with name, wP, wB, rP, rB  (weights & returns as decimals)
    Effects: Allocation=(wP-wB)*rB, Selection=wB*(rP-rB), Interaction=(wP-wB)*(rP-rB)
    """
    out = []
    for r in rows:
        alloc = (r["wP"] - r["wB"]) * r["rB"]
        select = r["wB"] * (r["rP"] - r["rB"])
        timing = (r["wP"] - r["wB"]) * (r["rP"] - r["rB"])
        active = alloc + select + timing
        out.append({
            "Sector": r["name"],
            "Portfolio Weight": r["wP"],
            "Benchmark Weight": r["wB"],
            "Portfolio Return": r["rP"],
            "Benchmark Return": r["rB"],
            "Asset Allocation": alloc,
            "Stock Selection": select,
            "Market Timing": timing,
            "Total Active": active
        })
    df = pd.DataFrame(out)
    if not df.empty:
        total = {
            "Sector": "Total",
            "Portfolio Weight": df["Portfolio Weight"].sum(),
            "Benchmark Weight": df["Benchmark Weight"].sum(),
            "Portfolio Return": (df["Portfolio Weight"] * df["Portfolio Return"]).sum(),
            "Benchmark Return": (df["Benchmark Weight"] * df["Benchmark Return"]).sum(),
            "Asset Allocation": df["Asset Allocation"].sum(),
            "Stock Selection": df["Stock Selection"].sum(),
            "Market Timing": df["Market Timing"].sum(),
            "Total Active": df["Total Active"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([total])], ignore_index=True)
    return df

def canon(s: str) -> str:
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())

# -------------------------------------------------
# Session state
# -------------------------------------------------
if "assets" not in st.session_state:
    st.session_state.assets = pd.DataFrame(
        columns=["Asset Class","Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]
    )
if "flows" not in st.session_state:
    st.session_state.flows = pd.DataFrame(
        columns=["Transaction Date","Transaction Type","Transaction Details","Amount","Asset Class"]
    )
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2024, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

# -------------------------------------------------
# UI Tabs
# -------------------------------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Global Settings", "Cashflow Upload", "Results", "Reporting"])

# =================================================
# 1) Global Settings
# =================================================
with tab1:
    st.subheader("Period")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.start_date = st.date_input(
            "Start Date", st.session_state.start_date, min_value=MIN_DATE, max_value=MAX_DATE
        )
    with c2:
        st.session_state.end_date = st.date_input(
            "End Date", st.session_state.end_date, min_value=MIN_DATE, max_value=MAX_DATE
        )

    st.markdown("### Asset Classes")
    st.caption("CSV headers (flexible): Asset Class, Beginning MV, Ending MV, Benchmark Weight %, Benchmark Return %")

    # --- Upload (auto-map + manual mapping fallback)
    up = st.file_uploader("Upload Assets CSV", type=["csv"], key="assets_csv")
    if up is not None:
        try:
            df = read_csv_robust(up)
            st.caption(f"Detected columns: {list(df.columns)}")

            rename = {}
            for c in df.columns:
                cn = canon(c)
                if cn in {"assetclass","asset","class","sector","strategy","bucket"}:
                    rename[c] = "Asset Class"
                elif cn in {"beginningmv","beginningmarketvalue","openingmv","openingmarketvalue","bv","bmv","startmv","startvalue"}:
                    rename[c] = "Beginning MV"
                elif cn in {"endingmv","endingmarketvalue","closingmv","closingmarketvalue","ev","emv","endmv","endvalue"}:
                    rename[c] = "Ending MV"
                elif cn in {"benchmarkweight","benchmarkweightpct","bmweight","targetweight","weightpct","weightpercent","weight","bw"}:
                    rename[c] = "Benchmark Weight %"
                elif cn in {"benchmarkreturn","benchmarkreturnpct","bmreturn","indexreturn","returnpct","retpct","br","benchmarkret"}:
                    rename[c] = "Benchmark Return %"
            df = df.rename(columns=rename)

            required = ["Asset Class","Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]
            missing = [c for c in required if c not in df.columns]

            if missing:
                st.warning(f"Missing columns after auto-detect: {missing}. Map them below.")
                options = ["<none>"] + list(df.columns)
                chosen = {}
                col1, col2 = st.columns(2)
                with col1:
                    sel_ac = st.selectbox("Map to **Asset Class**", options, key="map_ac")
                    sel_bv = st.selectbox("Map to **Beginning MV**", options, key="map_bv")
                    sel_ev = st.selectbox("Map to **Ending MV**", options, key="map_ev")
                with col2:
                    sel_bw = st.selectbox("Map to **Benchmark Weight %**", options, key="map_bw")
                    sel_br = st.selectbox("Map to **Benchmark Return %**", options, key="map_br")
                if st.button("Apply Mapping"):
                    chosen = {
                        sel_ac: "Asset Class",
                        sel_bv: "Beginning MV",
                        sel_ev: "Ending MV",
                        sel_bw: "Benchmark Weight %",
                        sel_br: "Benchmark Return %"
                    }
                    if "<none>" in chosen.keys():
                        chosen.pop("<none>", None)
                    df2 = df.rename(columns=chosen)
                    missing2 = [c for c in required if c not in df2.columns]
                    if missing2:
                        st.error(f"Still missing: {missing2}. Fix mapping or CSV headers.")
                    else:
                        for col in ["Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]:
                            df2[col] = df2[col].apply(to_float)
                        st.session_state.assets = df2[required].copy()
                        st.success("Assets loaded.")
            else:
                for col in ["Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]:
                    df[col] = df[col].apply(to_float)
                st.session_state.assets = df[required].copy()
                st.success("Assets loaded.")

        except Exception as e:
            st.error(f"Could not read assets CSV: {e}")

    # Editor
    assets_editor = st.data_editor(
        st.session_state.assets,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset Class": st.column_config.TextColumn(required=True),
            "Beginning MV": st.column_config.NumberColumn(min_value=0.0),
            "Ending MV": st.column_config.NumberColumn(min_value=0.0),
            "Benchmark Weight %": st.column_config.NumberColumn(help="Must sum to 100%"),
            "Benchmark Return %": st.column_config.NumberColumn(help="e.g., 4.5 for +4.5%"),
        },
        key="assets_editor"
    )
    st.session_state.assets = assets_editor

    bw_sum = pd.to_numeric(st.session_state.assets.get("Benchmark Weight %", pd.Series(dtype=float)),
                           errors="coerce").fillna(0).sum()
    st.info(f"Benchmark weights sum: **{bw_sum:.2f}%**" + (" ✅" if abs(bw_sum-100)<1e-6 else " ⚠️ should total 100%"))

# =================================================
# 2) Cashflow Upload
# =================================================
with tab2:
    st.subheader("Upload Cashflows (CSV)")
    st.caption("Headers (flexible): Transaction Date, Transaction Type (INFLOW/OUTFLOW), Transaction Details, Amount, Asset Class")

    # Template
    st.download_button(
        "Download Cashflow Template",
        pd.DataFrame({
            "Transaction Date": [(date.today()-timedelta(days=30)).isoformat(), (date.today()-timedelta(days=10)).isoformat()],
            "Transaction Type": [INFLOW, OUTFLOW],
            "Transaction Details": ["Initial funding", "Fees"],
            "Amount": [100000, 1500],
            "Asset Class": ["Equity","Equity"],
        }).to_csv(index=False).encode("utf-8"),
        "cashflows_template.csv","text/csv"
    )

    cf_file = st.file_uploader("Choose CSV", type=["csv"])
    if cf_file is not None:
        try:
            df = read_csv_robust(cf_file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            ren = {}
            for c in df.columns:
                cn = canon(c)
                if cn in {"transactiondate","date"}: ren[c] = "Transaction Date"
                elif cn in {"transactiontype","type"}: ren[c] = "Transaction Type"
                elif cn in {"transactiondetails","details","description"}: ren[c] = "Transaction Details"
                elif cn in {"amount","amt","value"}: ren[c] = "Amount"
                elif cn in {"assetclass","asset","class","sector"}: ren[c] = "Asset Class"
            df = df.rename(columns=ren)

            need = ["Transaction Date","Transaction Type","Transaction Details","Amount","Asset Class"]
            missing = [c for c in need if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.tz_localize(None)
                df["Transaction Type"] = df["Transaction Type"].astype(str).str.upper().str.strip()
                df["Transaction Details"] = df["Transaction Details"].astype(str).fillna("")
                df["Amount"] = df["Amount"].apply(to_float)
                df["Asset Class"] = df["Asset Class"].astype(str).str.strip()

                t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
                t1 = datetime.combine(st.session_state.end_date, datetime.max.time())
                asset_set = set(st.session_state.assets["Asset Class"].dropna().astype(str).str.strip())

                errs = []
                for i, r in df.iterrows():
                    if pd.isna(r["Transaction Date"]): errs.append(f"Row {i+1}: bad date")
                    elif not (t0 < r["Transaction Date"] <= t1): errs.append(f"Row {i+1}: date outside Start/End")
                    if r["Transaction Type"] not in ALLOWED_TYPES: errs.append(f"Row {i+1}: type must be {ALLOWED_TYPES}")
                    if not np.isfinite(r["Amount"]) or r["Amount"] <= 0: errs.append(f"Row {i+1}: Amount must be > 0")
                    if r["Asset Class"] not in asset_set: errs.append(f"Row {i+1}: Asset Class not in Global Settings")

                if errs:
                    st.error("Issues found:\n- " + "\n- ".join(errs))
                else:
                    st.session_state.flows = df.copy()
                    st.success("Cashflows loaded.")

    st.markdown("#### Loaded Cashflows")
    st.dataframe(st.session_state.flows, use_container_width=True)

# =================================================
# 3) Results
# =================================================
with tab3:
    st.subheader("Performance & Attribution")

    assets = st.session_state.assets.copy()
    if assets.empty:
        st.warning("Add or upload assets in Global Settings."); st.stop()
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("End Date must be after Start Date."); st.stop()
    if abs(pd.to_numeric(assets.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() - 100) > 1e-6:
        st.error("Benchmark weights must sum to 100%."); st.stop()

    assets["Beginning MV"] = assets["Beginning MV"].apply(to_float).fillna(0.0)
    assets["Ending MV"] = assets["Ending MV"].apply(to_float).fillna(0.0)
    assets["Benchmark Weight %"] = assets["Benchmark Weight %"].apply(to_float).fillna(0.0)
    assets["Benchmark Return %"] = assets["Benchmark Return %"].apply(to_float).fillna(0.0)

    total_bv = assets["Beginning MV"].sum()

    flows = st.session_state.flows.copy()
    if not flows.empty:
        flows["when"] = pd.to_datetime(flows["Transaction Date"], errors="coerce").dt.tz_localize(None)
        flows["amount"] = np.where(flows["Transaction Type"].str.upper()==INFLOW, flows["Amount"], -flows["Amount"])
        flows["asset"] = flows["Asset Class"].astype(str).str.strip()
    else:
        flows = pd.DataFrame(columns=["when","amount","asset"])

    t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
    t1 = datetime.combine(st.session_state.end_date, datetime.max.time())

    issues, rows = [], []
    for _, r in assets.iterrows():
        name = str(r["Asset Class"]).strip()
        bv, ev = float(r["Beginning MV"]), float(r["Ending MV"])
        wB, rB = float(r["Benchmark Weight %"])/100.0, float(r["Benchmark Return %"])/100.0
        if name == "" or bv < 0 or ev < 0: continue

        fA = flows[flows["asset"] == name][["when","amount"]].copy() if not flows.empty else pd.DataFrame(columns=["when","amount"])
        try:
            if bv == 0 and (fA.empty or abs(fA["amount"].sum()) < 1e-12):
                rP = 0.0
            else:
                rP = modified_dietz(bv, ev, fA, t0, t1)
        except Exception as e:
            issues.append(f"{name}: Dietz error → {e}")
            rP = 0.0

        wP = 0.0 if total_bv <= 0 else (bv / total_bv)
        rows.append({"name": name, "wP": wP, "wB": wB, "rP": rP, "rB": rB, "bv": bv, "ev": ev})

    ev_port = sum(x["ev"] for x in rows)
    flows_port = flows[["when","amount"]].copy()
    try:
        r_port = modified_dietz(max(total_bv, 0.0), ev_port, flows_port, t0, t1)
    except Exception as e:
        st.error(f"Portfolio Dietz error → {e}"); st.stop()

    r_bench = sum(x["wB"] * x["rB"] for x in rows)
    excess = r_port - r_bench

    attrib = brinson_bhb(rows)

    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Return (Dietz)", pct(r_port))
    c2.metric("Benchmark Return", pct(r_bench))
    c3.metric("Excess Return", pct(excess))

    show = attrib.copy()
    for col in ["Portfolio Weight","Benchmark Weight","Portfolio Return","Benchmark Return",
                "Asset Allocation","Stock Selection","Market Timing","Total Active"]:
        show[col] = show[col].apply(pct)
    st.dataframe(show, use_container_width=True)

    if issues: st.warning("Data issues:\n- " + "\n- ".join(issues))
    if not attrib.empty:
        drift = float(attrib.loc[attrib["Sector"]=="Total","Total Active"].values[0]) - excess
        st.caption(f"Attribution total-active vs. excess drift: {pct(drift)}")

    st.session_state.results_payload = {
        "period": (st.session_state.start_date, st.session_state.end_date),
        "metrics": {"Portfolio": r_port, "Benchmark": r_bench, "Excess": excess},
        "attrib": attrib,
        "assets": st.session_state.assets.copy(),
        "flows": st.session_state.flows.copy(),
    }

# =================================================
# 4) Reporting
# =================================================
with tab4:
    st.subheader("Reporting & Export")
    rep = st.session_state.get("results_payload", {})
    if not rep:
        st.info("Compute results first in the Results tab.")
    else:
        m = rep["metrics"]
        st.write({
            "Start": str(rep["period"][0]),
            "End": str(rep["period"][1]),
            "Portfolio Return (Dietz)": pct(m["Portfolio"]),
            "Benchmark Return": pct(m["Benchmark"]),
            "Excess Return": pct(m["Excess"]),
        })

        dfc = rep["attrib"].copy()
        if not dfc.empty and "Total" in dfc["Sector"].values:
            dfc = dfc[dfc["Sector"] != "Total"]
        if not dfc.empty:
            st.write("Top/Bottom Contributors (Total Active)")
            st.bar_chart(dfc.set_index("Sector")["Total Active"])

        # Export to Excel
        def build_excel_bytes(data: dict) -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                period = data.get("period", (None, None))
                metr = data.get("metrics", {})
                summary = pd.DataFrame({
                    "Metric": ["Start Date","End Date","Portfolio Return (Dietz)","Benchmark Return","Excess Return"],
                    "Value": [
                        str(period[0]) if period[0] else "",
                        str(period[1]) if period[1] else "",
                        pct(metr.get("Portfolio")),
                        pct(metr.get("Benchmark")),
                        pct(metr.get("Excess")),
                    ],
                })
                summary.to_excel(xw, sheet_name="Summary", index=False)
                data.get("attrib", pd.DataFrame()).to_excel(xw, sheet_name="Attribution", index=False)
                data.get("assets", pd.DataFrame()).to_excel(xw, sheet_name="Assets", index=False)
                data.get("flows", pd.DataFrame()).to_excel(xw, sheet_name="Cashflows", index=False)
            return buf.getvalue()

        xls = build_excel_bytes(rep)
        st.download_button("Download Excel Report", xls, "MDTWRR_Report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
