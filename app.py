# app.py
# MDTWRR Performance Portal (Streamlit) — robust CSV + tolerant Dietz
# Run: pip install streamlit pandas numpy
# Start: streamlit run app.py

from datetime import datetime, date, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.title("MDTWRR Performance Portal")
st.caption("Inputs → Cashflows → Performance & Attribution. No fluff.")

# -----------------------------
# Constants & helpers
# -----------------------------
INFLOW = "INFLOW"
OUTFLOW = "OUTFLOW"
ALLOWED_TYPES = [INFLOW, OUTFLOW]

def to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def clamp01(x): return max(0.0, min(1.0, x))

def read_csv_robust(uploaded_file):
    """Read a CSV with tolerant encoding & delimiter sniffing."""
    raw = uploaded_file.getvalue()  # bytes
    encodings = ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]
    last_err = None
    for enc in encodings:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception as e:
            last_err = e
            continue
    raise last_err or ValueError("Unknown CSV encoding")

def modified_dietz(bv, ev, flows_df, t0, t1, eps=1e-12):
    """Modified Dietz: t0 < when <= t1, amount signed (+in, -out)."""
    if not (isinstance(t0, datetime) and isinstance(t1, datetime) and t1 > t0):
        raise ValueError("Bad dates for Dietz calculation")
    if not np.isfinite(bv) or not np.isfinite(ev):
        raise ValueError("BV/EV must be finite")

    T = (t1 - t0).total_seconds()
    if T <= 0: raise ValueError("Zero/negative period length")

    if flows_df is None or flows_df.empty:
        denom = bv
        numer = ev - bv
        if abs(denom) < eps:
            raise ValueError("Unstable denominator (no flows)")
        return numer / denom

    f = flows_df[(flows_df["when"] > t0) & (flows_df["when"] <= t1)].copy()
    sum_cf = f["amount"].sum() if not f.empty else 0.0

    denom = bv
    if not f.empty:
        f["w"] = f["when"].apply(lambda w: clamp01((t1 - w).total_seconds() / T))
        denom += (f["w"] * f["amount"]).sum()

    if abs(denom) < eps: raise ValueError("Unstable denominator; split period or check flows")
    numer = ev - bv - sum_cf
    return numer / denom

def brinson_attribution(rows):
    """BHB with interaction as Market Timing."""
    recs = []
    for x in rows:
        alloc = (x["wP"] - x["wB"]) * x["rB"]
        select = x["wB"] * (x["rP"] - x["rB"])
        timing = (x["wP"] - x["wB"]) * (x["rP"] - x["rB"])
        active = alloc + select + timing
        recs.append({
            "Sector": x["name"],
            "Portfolio Weight": x["wP"],
            "Benchmark Weight": x["wB"],
            "Portfolio Return": x["rP"],
            "Benchmark Return": x["rB"],
            "Asset Allocation": alloc,
            "Stock Selection": select,
            "Market Timing": timing,
            "Total Active": active,
        })
    df = pd.DataFrame(recs)
    if not df.empty:
        totals = {
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
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df

def pct(x, dp=2): return "-" if pd.isna(x) else f"{x*100:.{dp}f}%"

# -----------------------------
# Session defaults
# -----------------------------
if "assets" not in st.session_state:
    st.session_state.assets = pd.DataFrame(
        columns=["Asset Class", "Beginning MV", "Ending MV", "Benchmark Weight %", "Benchmark Return %"]
    )
if "flows" not in st.session_state:
    st.session_state.flows = pd.DataFrame(
        columns=["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
    )
if "start_date" not in st.session_state:
    st.session_state.start_date = date.today().replace(day=1)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()

# -----------------------------
# Tabs
# -----------------------------
tab_settings, tab_cashflows, tab_results = st.tabs(["1) Global Settings", "2) Cashflow Upload", "3) Results"])

# =============================
# 1) GLOBAL SETTINGS (CSV-enabled)
# =============================
with tab_settings:
    st.subheader("Period")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.start_date = st.date_input("Start Date", value=st.session_state.start_date, key="start_date_input")
    with c2:
        st.session_state.end_date = st.date_input("End Date", value=st.session_state.end_date, key="end_date_input")

    st.markdown("#### Load Period via CSV (optional)")
    st.caption("One row with headers **Start Date, End Date**.")
    period_sample = pd.DataFrame([{
        "Start Date": (date.today().replace(day=1)).isoformat(),
        "End Date": date.today().isoformat(),
    }])
    st.download_button("Download Period CSV Template", period_sample.to_csv(index=False).encode("utf-8"),
                       "period_template.csv", "text/csv", key="dl_period_tpl")
    period_file = st.file_uploader("Upload Period CSV (optional)", type=["csv"], key="period_csv")
    if period_file is not None:
        try:
            pdf = read_csv_robust(period_file)
            rename_map = {}
            for c in pdf.columns:
                k = c.strip().lower()
                if k in ("start date", "start_date", "start"):
                    rename_map[c] = "Start Date"
                if k in ("end date", "end_date", "end"):
                    rename_map[c] = "End Date"
            pdf = pdf.rename(columns=rename_map)
            if {"Start Date","End Date"}.issubset(set(pdf.columns)) and len(pdf) >= 1:
                s = pd.to_datetime(pdf.loc[0,"Start Date"], errors="coerce")
                e = pd.to_datetime(pdf.loc[0,"End Date"], errors="coerce")
                if pd.notna(s): st.session_state.start_date = s.date()
                if pd.notna(e): st.session_state.end_date = e.date()
                st.success("Period loaded from CSV")
            else:
                st.error("Period CSV must have headers: Start Date, End Date (one row)")
        except Exception as ex:
            st.error(f"Could not read Period CSV: {ex}")

    st.markdown("#### Asset Classes")
    st.caption("Enter manually **or** load from CSV. Required headers below.")

    # Flexible header matching utilities
    def _canon(s: str) -> str:
        return "".join(ch for ch in str(s).lower().strip() if ch.isalnum())
    def _match(col: str, targets: list[str]) -> bool:
        c = _canon(col); return any(t in c for t in targets)

    assets_tpl = pd.DataFrame({
        "Asset Class": ["Equity","Fixed Income","Real Estate"],
        "Beginning MV": [1_000_000, 500_000, 250_000],
        "Ending MV": [1_115_000, 520_000, 260_000],
        "Benchmark Weight %": [60, 30, 10],
        "Benchmark Return %": [2.5, 1.0, 1.2],
    })
    st.download_button("Download Assets CSV Template", assets_tpl.to_csv(index=False).encode("utf-8"),
                       "assets_template.csv", "text/csv", key="dl_assets_tpl")

    assets_file = st.file_uploader("Upload Assets CSV", type=["csv"], key="assets_csv")
    if assets_file is not None:
        try:
            df = read_csv_robust(assets_file)
            st.caption(f"Detected columns: {list(df.columns)}")
            rename = {}
            for c in df.columns:
                if _match(c, ["assetclass","asset","class"]): rename[c] = "Asset Class"
                elif _match(c, ["beginningmv","beginmv","openingmv","bv","startmv"]): rename[c] = "Beginning MV"
                elif _match(c, ["endingmv","endmv","closingmv","ev","finishmv"]): rename[c] = "Ending MV"
                elif _match(c, ["benchmarkweight","benchweight","bmweight","bw","weightpct","weightpercent","weight%"]): rename[c] = "Benchmark Weight %"
                elif _match(c, ["benchmarkreturn","benchreturn","bmreturn","indexreturn","br","retpct","returnpercent","return%"]): rename[c] = "Benchmark Return %"
            df = df.rename(columns=rename)

            required = ["Asset Class","Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing columns: {missing}")
            else:
                for col in ["Beginning MV","Ending MV","Benchmark Weight %","Benchmark Return %"]:
                    df[col] = df[col].apply(to_float)
                st.session_state.assets = df[required].copy()
                st.success("Assets loaded from CSV")
        except Exception as e:
            st.error(f"Could not read Assets CSV: {e}")

    # Manual editor (CSV pre-fills it)
    assets_df = st.data_editor(
        st.session_state.assets,
        num_rows="dynamic",
        use_container_width=True,
        key="assets_editor",
        column_config={
            "Asset Class": st.column_config.TextColumn(required=True),
            "Beginning MV": st.column_config.NumberColumn(min_value=0.0),
            "Ending MV": st.column_config.NumberColumn(min_value=0.0),
            "Benchmark Weight %": st.column_config.NumberColumn(help="Must sum to 100%"),
            "Benchmark Return %": st.column_config.NumberColumn(help="e.g., 2.5 for +2.5%"),
        },
    )
    st.session_state.assets = assets_df

    bw_sum = pd.to_numeric(assets_df.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    st.info(f"Benchmark weights sum: **{bw_sum:.2f}%**" + (" ✅" if abs(bw_sum - 100) < 1e-6 else " ⚠️ should be 100%"))

# =============================
# 2) CASHFLOWS
# =============================
with tab_cashflows:
    st.subheader("Upload Cashflows (CSV)")
    st.caption("Required headers: Transaction Date, Transaction Type, Transaction Details, Amount, Asset Class")

    sample = pd.DataFrame({
        "Transaction Date": [(date.today()-timedelta(days=20)).isoformat(), (date.today()-timedelta(days=10)).isoformat()],
        "Transaction Type": [INFLOW, OUTFLOW],
        "Transaction Details": ["Initial contribution", "Fees"],
        "Amount": [100000.00, 2000.00],
        "Asset Class": ["Equity", "Equity"],
    })
    st.download_button("Download Cashflows CSV Template", sample.to_csv(index=False).encode("utf-8"),
                       "cashflows_template.csv", "text/csv")

    file = st.file_uploader("Choose CSV file", type=["csv"])
    if file is not None:
        try:
            df = read_csv_robust(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            rename_map = {}
            for col in df.columns:
                c = col.strip().lower()
                if c in ("transaction date", "date"): rename_map[col] = "Transaction Date"
                elif c in ("transaction type", "type"): rename_map[col] = "Transaction Type"
                elif c in ("transaction details", "details", "description"): rename_map[col] = "Transaction Details"
                elif c in ("amount",): rename_map[col] = "Amount"
                elif c in ("asset class", "assetclass", "class"): rename_map[col] = "Asset Class"
            df = df.rename(columns=rename_map)

            required = ["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.tz_localize(None)
                df["Transaction Type"] = df["Transaction Type"].astype(str).str.upper().str.strip()
                df["Transaction Details"] = df["Transaction Details"].astype(str).fillna("")
                df["Amount"] = df["Amount"].apply(to_float)
                df["Asset Class"] = df["Asset Class"].astype(str).str.strip()

                errs = []
                t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
                t1 = datetime.combine(st.session_state.end_date, datetime.max.time())
                defined_assets = set(st.session_state.assets["Asset Class"].dropna().astype(str).str.strip().tolist())

                for i, r in df.iterrows():
                    if pd.isna(r["Transaction Date"]):
                        errs.append(f"Row {i+1}: bad date"); continue
                    if not (t0 < r["Transaction Date"] <= t1):
                        errs.append(f"Row {i+1}: date outside Start/End")
                    if r["Transaction Type"] not in ALLOWED_TYPES:
                        errs.append(f"Row {i+1}: type must be {ALLOWED_TYPES}")
                    if not np.isfinite(r["Amount"]) or r["Amount"] <= 0:
                        errs.append(f"Row {i+1}: Amount must be > 0")
                    if r["Asset Class"] not in defined_assets:
                        errs.append(f"Row {i+1}: Asset Class not in Global Settings")

                if errs:
                    st.error("Issues found:\n- " + "\n- ".join(errs))
                else:
                    st.success("Cashflows loaded")
                    st.session_state.flows = df.copy()

    st.markdown("#### Loaded Cashflows")
    st.dataframe(st.session_state.flows, use_container_width=True)

# =============================
# 3) RESULTS
# =============================
with tab_results:
    st.subheader("Performance & Attribution")

    assets = st.session_state.assets.copy()
    if assets.empty:
        st.warning("Add or upload Asset Classes in Global Settings."); st.stop()
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("End Date must be after Start Date."); st.stop()
    if abs(pd.to_numeric(assets.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() - 100) > 1e-6:
        st.error("Benchmark Weights must sum to 100%."); st.stop()

    assets["Beginning MV"] = assets["Beginning MV"].apply(to_float).fillna(0.0)
    assets["Ending MV"] = assets["Ending MV"].apply(to_float).fillna(0.0)
    assets["Benchmark Weight %"] = assets["Benchmark Weight %"].apply(to_float).fillna(0.0)
    assets["Benchmark Return %"] = assets["Benchmark Return %"].apply(to_float).fillna(0.0)

    if assets["Beginning MV"].sum() <= 0:
        st.error("Total Beginning MV must be > 0."); st.stop()

    flows = st.session_state.flows.copy()
    if not flows.empty:
        flows["when"] = pd.to_datetime(flows["Transaction Date"], errors="coerce").dt.tz_localize(None)
        flows["amount"] = np.where(flows["Transaction Type"] == INFLOW, flows["Amount"], -flows["Amount"])
        flows["asset"] = flows["Asset Class"].astype(str).str.strip()
    else:
        flows = pd.DataFrame(columns=["when", "amount", "asset"])

    t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
    t1 = datetime.combine(st.session_state.end_date, datetime.max.time())

    total_bv = assets["Beginning MV"].sum()

    # Per-class Dietz (tolerant; collect issues)
    issues = []
    class_rows = []
    for _, row in assets.iterrows():
        name = str(row["Asset Class"]).strip()
        bv = float(row["Beginning MV"])
        ev = float(row["Ending MV"])
        wB = float(row["Benchmark Weight %"]) / 100.0
        rB = float(row["Benchmark Return %"]) / 100.0
        if name == "" or bv < 0 or ev < 0:
            continue

        f_cls = flows[flows["asset"] == name][["when", "amount"]].copy() if not flows.empty else pd.DataFrame(columns=["when", "amount"])
        has_flows = (not f_cls.empty) and np.isfinite(f_cls["amount"].sum())

        if bv == 0 and not has_flows:
            if ev == 0:
                # inactive this period — zero weight/return
                class_rows.append({"name": name, "wP": 0.0, "wB": wB, "rP": 0.0, "rB": rB, "bv": bv, "ev": ev})
            else:
                issues.append(f"{name}: BV is 0 and no cashflows provided; add INFLOW(s) for purchases. Skipped in return calc.")
                class_rows.append({"name": name, "wP": 0.0, "wB": wB, "rP": 0.0, "rB": rB, "bv": bv, "ev": ev})
            continue

        try:
            rP = modified_dietz(bv, ev, f_cls, t0, t1)
        except Exception as e:
            issues.append(f"{name}: Dietz error → {e}. Row ignored in return calc.")
            class_rows.append({"name": name, "wP": 0.0, "wB": wB, "rP": 0.0, "rB": rB, "bv": bv, "ev": ev})
            continue

        wP = 0.0 if total_bv <= 0 else bv / total_bv
        class_rows.append({"name": name, "wP": wP, "wB": wB, "rP": rP, "rB": rB, "bv": bv, "ev": ev})

    # Portfolio Dietz
    ev_port = sum(c["ev"] for c in class_rows)
    flows_port = flows[["when", "amount"]].copy() if not flows.empty else pd.DataFrame(columns=["when", "amount"])
    try:
        r_port = modified_dietz(total_bv, ev_port, flows_port, t0, t1)
    except Exception as e:
        st.error(f"Portfolio Dietz error → {e}"); st.stop()

    r_bench = sum(c["wB"] * c["rB"] for c in class_rows)
    excess = r_port - r_bench

    attrib_df = brinson_attribution(class_rows)

    # Metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio Return (Dietz)", pct(r_port))
    m2.metric("Benchmark Return", pct(r_bench))
    m3.metric("Excess Return", pct(excess))

    # Table
    show = attrib_df.copy()
    pct_cols = ["Portfolio Weight","Benchmark Weight","Portfolio Return","Benchmark Return",
                "Asset Allocation","Stock Selection","Market Timing","Total Active"]
    for c in pct_cols: show[c] = show[c].apply(pct)
    st.dataframe(show, use_container_width=True)

    # Warnings
    if issues:
        st.warning("Data issues detected:\n- " + "\n- ".join(issues))

    # Download
    out_csv = attrib_df.to_csv(index=False).encode("utf-8")
    st.download_button("Download Attribution CSV", out_csv, "attribution.csv", "text/csv")
