# app.py — MDTWRR Portal (clean Dietz + Brinson attribution)
# Run: streamlit run app.py
# Requirements: streamlit, pandas, numpy, xlsxwriter

from __future__ import annotations
import io
from datetime import date, datetime, timedelta
import numpy as np
import pandas as pd
import streamlit as st

# ---------- App setup ----------
st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.title("MDTWRR Performance Portal")
st.caption("Modified Dietz returns + Brinson attribution. No fluff.")

MIN_DATE = date(1900, 1, 1)
MAX_DATE = date(2100, 12, 31)
INFLOW, OUTFLOW = "INFLOW", "OUTFLOW"
ALLOWED_TYPES = [INFLOW, OUTFLOW]

# ---------- Helpers ----------
def pct(x: float, dp: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and (np.isnan(x))) else f"{x*100:.{dp}f}%"

def to_float(x):
    if x is None:
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))

def read_csv_robust(uploaded_file) -> pd.DataFrame:
    raw = uploaded_file.getvalue()
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    # last attempt plain read (will throw)
    return pd.read_csv(io.BytesIO(raw))

def modified_dietz(bv: float, ev: float, flows: pd.DataFrame, t0: datetime, t1: datetime, eps=1e-12) -> float:
    """
    R = (EV - BV - sum(CF)) / (BV + sum(w_i * CF_i)), with w_i = (t1 - t_i)/(t1 - t0)
    flows: columns ['when','amount'] with signed amounts (+in, -out)
    Return is decimal (e.g., 0.1234 = 12.34%)
    """
    if not np.isfinite(bv) or not np.isfinite(ev):
        raise ValueError("BV/EV must be finite")
    if not (isinstance(t0, datetime) and isinstance(t1, datetime) and t1 > t0):
        raise ValueError("Bad period dates")
    T = (t1 - t0).total_seconds()
    if T <= 0:
        raise ValueError("Zero/negative period")

    f = pd.DataFrame(columns=["when", "amount"]) if flows is None else flows.copy()
    if not f.empty:
        f = f[(f["when"] > t0) & (f["when"] <= t1)]
        f = f[np.isfinite(f["amount"])]

    sum_cf = 0.0 if f.empty else f["amount"].sum()

    denom = bv
    if not f.empty:
        f["w"] = f["when"].apply(lambda w: clamp01((t1 - w).total_seconds() / T))
        denom += (f["w"] * f["amount"]).sum()

    if abs(denom) < eps:
        # If absolutely no base to earn on, returns are undefined; treat as 0 to avoid explosions.
        raise ValueError("Unstable denominator")

    numer = ev - bv - sum_cf
    return numer / denom

def brinson_rows(rows: list[dict]) -> pd.DataFrame:
    """
    BHB with interaction = Market Timing.
    rows elements need: name, wP, wB, rP, rB
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

# ---------- Session Defaults ----------
if "assets" not in st.session_state:
    st.session_state.assets = pd.DataFrame(
        columns=["Asset Class", "Beginning MV", "Ending MV", "Benchmark Weight %", "Benchmark Return %"]
    )
if "flows" not in st.session_state:
    st.session_state.flows = pd.DataFrame(
        columns=["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
    )
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2024, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

# ---------- UI Tabs ----------
tab1, tab2, tab3, tab4 = st.tabs(["Global Settings", "Cashflow Upload", "Results", "Reporting"])

# =========================
# Global Settings
# =========================
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
    st.caption("Enter manually or upload CSV with headers: Asset Class, Beginning MV, Ending MV, Benchmark Weight %, Benchmark Return %")

    # Upload
    up = st.file_uploader("Upload Assets CSV", type=["csv"], key="assets_csv")
    if up is not None:
        try:
            df = read_csv_robust(up)
            rename = {}
            for c in df.columns:
                cl = c.strip().lower()
                if cl in ("asset class", "assetclass", "class"): rename[c] = "Asset Class"
                elif cl in ("beginning mv", "beginningmv", "bv", "opening mv", "start mv"): rename[c] = "Beginning MV"
                elif cl in ("ending mv", "endingmv", "ev", "closing mv", "end mv"): rename[c] = "Ending MV"
                elif cl in ("benchmark weight %", "benchmark weight", "bm weight", "weight %", "weight"): rename[c] = "Benchmark Weight %"
                elif cl in ("benchmark return %", "benchmark return", "bm return", "index return", "return %"): rename[c] = "Benchmark Return %"
            df = df.rename(columns=rename)
            req = ["Asset Class", "Beginning MV", "Ending MV", "Benchmark Weight %", "Benchmark Return %"]
            miss = [c for c in req if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                for col in ["Beginning MV", "Ending MV", "Benchmark Weight %", "Benchmark Return %"]:
                    df[col] = df[col].apply(to_float)
                st.session_state.assets = df[req].copy()
                st.success("Assets loaded.")
        except Exception as e:
            st.error(f"Could not read assets CSV: {e}")

    # Editor
    assets = st.data_editor(
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
    st.session_state.assets = assets

    bw_sum = pd.to_numeric(assets.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    st.info(f"Benchmark weights sum: **{bw_sum:.2f}%**" + (" ✅" if abs(bw_sum-100) < 1e-6 else " ⚠️ should total 100%"))

# =========================
# Cashflow Upload
# =========================
with tab2:
    st.subheader("Upload Cashflows (CSV)")
    st.caption("Headers: Transaction Date, Transaction Type, Transaction Details, Amount, Asset Class")

    cf_tpl = st.download_button(
        "Download Cashflow Template",
        pd.DataFrame({
            "Transaction Date": [(date.today()-timedelta(days=20)).isoformat()],
            "Transaction Type": [INFLOW],
            "Transaction Details": ["Initial contribution"],
            "Amount": [100000.0],
            "Asset Class": ["Equity"]
        }).to_csv(index=False).encode("utf-8"),
        "cashflows_template.csv", "text/csv"
    )

    file = st.file_uploader("Choose CSV", type=["csv"])
    if file is not None:
        try:
            df = read_csv_robust(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            ren = {}
            for c in df.columns:
                cl = c.strip().lower()
                if cl in ("transaction date", "date"): ren[c] = "Transaction Date"
                elif cl in ("transaction type", "type"): ren[c] = "Transaction Type"
                elif cl in ("transaction details", "details", "description"): ren[c] = "Transaction Details"
                elif cl in ("amount",): ren[c] = "Amount"
                elif cl in ("asset class", "assetclass", "class"): ren[c] = "Asset Class"
            df = df.rename(columns=ren)
            need = ["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.tz_localize(None)
                df["Transaction Type"] = df["Transaction Type"].astype(str).str.upper().str.strip()
                df["Transaction Details"] = df["Transaction Details"].astype(str).fillna("")
                df["Amount"] = df["Amount"].apply(to_float)
                df["Asset Class"] = df["Asset Class"].astype(str).str.strip()

                t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
                t1 = datetime.combine(st.session_state.end_date, datetime.max.time())
                assets_set = set(st.session_state.assets["Asset Class"].dropna().astype(str).str.strip().tolist())

                errs = []
                for i, r in df.iterrows():
                    if pd.isna(r["Transaction Date"]):
                        errs.append(f"Row {i+1}: bad date"); continue
                    if not (t0 < r["Transaction Date"] <= t1):
                        errs.append(f"Row {i+1}: date outside Start/End")
                    if r["Transaction Type"] not in ALLOWED_TYPES:
                        errs.append(f"Row {i+1}: type must be {ALLOWED_TYPES}")
                    if not np.isfinite(r["Amount"]) or r["Amount"] <= 0:
                        errs.append(f"Row {i+1}: Amount must be > 0")
                    if r["Asset Class"] not in assets_set:
                        errs.append(f"Row {i+1}: Asset Class not in Global Settings")

                if errs:
                    st.error("Issues found:\n- " + "\n- ".join(errs))
                else:
                    st.session_state.flows = df.copy()
                    st.success("Cashflows loaded.")

    st.markdown("#### Loaded Cashflows")
    st.dataframe(st.session_state.flows, use_container_width=True)

# =========================
# Results
# =========================
with tab3:
    st.subheader("Performance & Attribution")

    # validations
    assets = st.session_state.assets.copy()
    if assets.empty:
        st.warning("Add/upload assets in Global Settings."); st.stop()
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("End Date must be after Start Date."); st.stop()
    if abs(pd.to_numeric(assets.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() - 100) > 1e-6:
        st.error("Benchmark Weights must sum to 100%."); st.stop()

    # clean numbers
    assets["Beginning MV"] = assets["Beginning MV"].apply(to_float).fillna(0.0)
    assets["Ending MV"] = assets["Ending MV"].apply(to_float).fillna(0.0)
    assets["Benchmark Weight %"] = assets["Benchmark Weight %"].apply(to_float).fillna(0.0)
    assets["Benchmark Return %"] = assets["Benchmark Return %"].apply(to_float).fillna(0.0)

    total_bv = assets["Beginning MV"].sum()

    # flows to signed
    flows = st.session_state.flows.copy()
    if not flows.empty:
        flows["when"] = pd.to_datetime(flows["Transaction Date"], errors="coerce").dt.tz_localize(None)
        flows["amount"] = np.where(flows["Transaction Type"].str.upper() == INFLOW, flows["Amount"], -flows["Amount"])
        flows["asset"] = flows["Asset Class"].astype(str).str.strip()
    else:
        flows = pd.DataFrame(columns=["when", "amount", "asset"])

    t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
    t1 = datetime.combine(st.session_state.end_date, datetime.max.time())

    # Per-asset returns
    issues = []
    rows = []
    for _, r in assets.iterrows():
        name = str(r["Asset Class"]).strip()
        bv, ev = float(r["Beginning MV"]), float(r["Ending MV"])
        wB, rB = float(r["Benchmark Weight %"]) / 100.0, float(r["Benchmark Return %"]) / 100.0
        if name == "" or bv < 0 or ev < 0:  # skip malformed
            continue

        fA = flows[flows["asset"] == name][["when", "amount"]].copy() if not flows.empty else pd.DataFrame(columns=["when", "amount"])
        try:
            # If BV is zero but there are flows, Dietz can still work. If no flows and BV=0, return set to 0.
            if bv == 0 and (fA.empty or fA["amount"].sum() == 0):
                rP = 0.0
            else:
                rP = modified_dietz(bv, ev, fA, t0, t1)
        except Exception as e:
            issues.append(f"{name}: Dietz error → {e}")
            rP = 0.0

        wP = 0.0 if total_bv <= 0 else (bv / total_bv)
        rows.append({"name": name, "wP": wP, "wB": wB, "rP": rP, "rB": rB, "bv": bv, "ev": ev})

    # Portfolio Dietz (aggregate)
    ev_port = sum(x["ev"] for x in rows)
    flows_port = flows[["when", "amount"]].copy()
    try:
        r_port = modified_dietz(max(total_bv, 0.0), ev_port, flows_port, t0, t1)
    except Exception as e:
        st.error(f"Portfolio Dietz error → {e}"); st.stop()

    # Benchmark & Excess
    r_bench = sum(x["wB"] * x["rB"] for x in rows)
    excess = r_port - r_bench

    # Attribution
    attrib = brinson_rows(rows)

    # -------- display --------
    c1, c2, c3 = st.columns(3)
    c1.metric("Portfolio Return (Dietz)", pct(r_port))
    c2.metric("Benchmark Return", pct(r_bench))
    c3.metric("Excess Return", pct(excess))

    # Table (pretty)
    show = attrib.copy()
    for col in ["Portfolio Weight","Benchmark Weight","Portfolio Return","Benchmark Return",
                "Asset Allocation","Stock Selection","Market Timing","Total Active"]:
        show[col] = show[col].apply(pct)
    st.dataframe(show, use_container_width=True)

    if issues:
        st.warning("Data issues:\n- " + "\n- ".join(issues))
    # reconciliation hint
    if not attrib.empty:
        drift = float(attrib.loc[attrib["Sector"] == "Total", "Total Active"].values[0]) - excess
        st.caption(f"Attribution total-active vs. excess drift: {pct(drift)} (small rounding/weighting differences are normal)")

    # Save for reporting/export
    st.session_state.results_payload = {
        "period": (st.session_state.start_date, st.session_state.end_date),
        "metrics": {"Portfolio": r_port, "Benchmark": r_bench, "Excess": excess},
        "attrib": attrib,
        "assets": st.session_state.assets.copy(),
        "flows": st.session_state.flows.copy(),
    }

# =========================
# Reporting
# =========================
with tab4:
    st.subheader("Reporting")
    payload = st.session_state.get("results_payload", {})
    if not payload:
        st.info("Compute results first.")
    else:
        m = payload["metrics"]
        st.write({
            "Start": str(payload["period"][0]),
            "End": str(payload["period"][1]),
            "Portfolio Return (Dietz)": pct(m["Portfolio"]),
            "Benchmark Return": pct(m["Benchmark"]),
            "Excess Return": pct(m["Excess"])
        })

        # Simple contribution chart
        dfc = payload["attrib"].copy()
        if not dfc.empty and "Total" in dfc["Sector"].values:
            dfc = dfc[dfc["Sector"] != "Total"]
        if not dfc.empty:
            chart_df = dfc[["Sector", "Total Active"]].set_index("Sector")
            st.bar_chart(chart_df)

        # ---- Export (Excel) ----
        def build_excel_bytes(rep: dict) -> bytes:
            buf = io.BytesIO()
            with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
                period = rep.get("period", (None, None))
                metr = rep.get("metrics", {})
                summary = pd.DataFrame({
                    "Metric": ["Start Date", "End Date", "Portfolio Return (Dietz)", "Benchmark Return", "Excess Return"],
                    "Value": [
                        str(period[0]) if period[0] else "",
                        str(period[1]) if period[1] else "",
                        pct(metr.get("Portfolio")),
                        pct(metr.get("Benchmark")),
                        pct(metr.get("Excess")),
                    ],
                })
                summary.to_excel(xw, sheet_name="Summary", index=False)
                rep.get("attrib", pd.DataFrame()).to_excel(xw, sheet_name="Attribution", index=False)
                rep.get("assets", pd.DataFrame()).to_excel(xw, sheet_name="Assets", index=False)
                rep.get("flows", pd.DataFrame()).to_excel(xw, sheet_name="Cashflows", index=False)
            return buf.getvalue()

        xls = build_excel_bytes(payload)
        st.download_button("Download Excel Report", xls, "MDTWRR_Report.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
