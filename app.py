# app.py
# --------------------------------------------------------------------
# MDTWRR Investment Performance Portal (Streamlit)
# - One file, no extras. Run:  pip install streamlit pandas numpy
# - Start:  streamlit run app.py
# --------------------------------------------------------------------

import io
from datetime import datetime, date, timedelta

import numpy as np
import pandas as pd
import streamlit as st

st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.title("MDTWRR Performance Portal")
st.caption("Inputs → Cashflows → Performance & Attribution. No fluff.")

# -----------------------------
# Helpers
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

def clamp01(x):
    return max(0.0, min(1.0, x))

def modified_dietz(bv, ev, flows_df, t0, t1, eps=1e-12):
    """
    bv, ev: floats
    flows_df: DataFrame with columns ["when", "amount"] where amount is signed (+in, -out)
              and t0 < when <= t1
    t0, t1: datetime (t1 > t0)
    """
    if not (isinstance(t0, datetime) and isinstance(t1, datetime) and t1 > t0):
        raise ValueError("Bad dates for Dietz calculation")
    if not np.isfinite(bv) or not np.isfinite(ev):
        raise ValueError("BV/EV must be finite")

    T = (t1 - t0).total_seconds()
    if T <= 0:
        raise ValueError("Zero/negative period length")

    if flows_df is None or flows_df.empty:
        denom = bv
        numer = ev - bv
        if abs(denom) < eps:
            raise ValueError("Unstable denominator (no flows)")
        return numer / denom

    # Only flows within (t0, t1]
    f = flows_df[(flows_df["when"] > t0) & (flows_df["when"] <= t1)].copy()
    sum_cf = f["amount"].sum() if not f.empty else 0.0

    denom = bv
    if not f.empty:
        f["w"] = f["when"].apply(lambda w: clamp01((t1 - w).total_seconds() / T))
        denom += (f["w"] * f["amount"]).sum()

    if abs(denom) < eps:
        raise ValueError("Unstable denominator; split period or check flows")

    numer = ev - bv - sum_cf
    return numer / denom

def brinson_attribution(rows):
    """
    rows: list of dicts per asset class with:
      name, wP, wB, rP, rB
    Returns DataFrame with Allocation, Selection, Timing, Active per class + totals row.
    """
    recs = []
    for x in rows:
        alloc = (x["wP"] - x["wB"]) * x["rB"]
        select = x["wB"] * (x["rP"] - x["rB"])
        timing = (x["wP"] - x["wB"]) * (x["rP"] - x["rB"])  # interaction
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
    # Totals row
    if not df.empty:
        totals = {
            "Sector": "Total",
            "Portfolio Weight": df["Portfolio Weight"].sum(),
            "Benchmark Weight": df["Benchmark Weight"].sum(),
            "Portfolio Return": (df["Portfolio Weight"] * df["Portfolio Return"]).sum(),   # weighted by P weights
            "Benchmark Return": (df["Benchmark Weight"] * df["Benchmark Return"]).sum(),   # weighted by B weights
            "Asset Allocation": df["Asset Allocation"].sum(),
            "Stock Selection": df["Stock Selection"].sum(),
            "Market Timing": df["Market Timing"].sum(),
            "Total Active": df["Total Active"].sum(),
        }
        df = pd.concat([df, pd.DataFrame([totals])], ignore_index=True)
    return df

def pct(x, dp=2):
    if pd.isna(x):
        return "-"
    return f"{x*100:.{dp}f}%"

def money(x, dp=2):
    if pd.isna(x):
        return "-"
    return f"{x:,.{dp}f}"

# -----------------------------
# Session state defaults
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
# Tabs (pages)
# -----------------------------
tab_settings, tab_cashflows, tab_results = st.tabs(["1) Global Settings", "2) Cashflow Upload", "3) Results"])

# =============================
# 1) GLOBAL SETTINGS
# =============================
with tab_settings:
    colA, colB, colC = st.columns([1,1,1])
    with colA:
        st.subheader("Period")
        st.session_state.start_date = st.date_input("Start Date", value=st.session_state.start_date, key="start_date_input")
    with colB:
        st.write("")
        st.session_state.end_date = st.date_input("End Date", value=st.session_state.end_date, key="end_date_input")
    with colC:
        st.subheader("Transaction Types")
        st.write(", ".join(ALLOWED_TYPES))

    st.markdown("#### Asset Classes")
    st.caption("Enter BV/EV, Benchmark Weight (%) and Benchmark Return (%) per class.")
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

    # Weight sum check
    bw_sum = pd.to_numeric(assets_df.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    st.info(f"Benchmark weights sum: **{bw_sum:.2f}%**" + (" ✅" if abs(bw_sum - 100) < 1e-6 else " ⚠️ should be 100%"))

# =============================
# 2) CASHFLOWS
# =============================
with tab_cashflows:
    st.subheader("Upload Cashflows (CSV)")
    st.caption("Required headers: Transaction Date, Transaction Type, Transaction Details, Amount, Asset Class")

    # Template download
    sample = pd.DataFrame({
        "Transaction Date": [(date.today()-timedelta(days=20)).isoformat(), (date.today()-timedelta(days=10)).isoformat()],
        "Transaction Type": [INFLOW, OUTFLOW],
        "Transaction Details": ["Initial contribution", "Fees"],
        "Amount": [100000.00, 2000.00],
        "Asset Class": ["Equity", "Equity"],
    })
    st.download_button("Download CSV Template", sample.to_csv(index=False).encode("utf-8"), "cashflows_template.csv", "text/csv")

    file = st.file_uploader("Choose CSV file", type=["csv"])
    if file is not None:
        try:
            df = pd.read_csv(file)
        except Exception as e:
            st.error(f"Could not read CSV: {e}")
            df = None

        if df is not None:
            # Normalize columns (case-insensitive match)
            rename_map = {}
            for col in df.columns:
                c = col.strip().lower()
                if c in ("transaction date", "date"):
                    rename_map[col] = "Transaction Date"
                elif c in ("transaction type", "type"):
                    rename_map[col] = "Transaction Type"
                elif c in ("transaction details", "details", "description"):
                    rename_map[col] = "Transaction Details"
                elif c in ("amount",):
                    rename_map[col] = "Amount"
                elif c in ("asset class", "assetclass", "class"):
                    rename_map[col] = "Asset Class"
            df = df.rename(columns=rename_map)

            required = ["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
            missing = [c for c in required if c not in df.columns]
            if missing:
                st.error(f"Missing required columns: {missing}")
            else:
                # Clean types
                df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.tz_localize(None)
                df["Transaction Type"] = df["Transaction Type"].astype(str).str.upper().str.strip()
                df["Transaction Details"] = df["Transaction Details"].astype(str).fillna("")
                df["Amount"] = df["Amount"].apply(to_float)
                df["Asset Class"] = df["Asset Class"].astype(str).str.strip()

                # Validations
                errs = []
                t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
                t1 = datetime.combine(st.session_state.end_date, datetime.max.time())
                defined_assets = set(st.session_state.assets["Asset Class"].dropna().astype(str).str.strip().tolist())

                for i, r in df.iterrows():
                    if pd.isna(r["Transaction Date"]):
                        errs.append(f"Row {i+1}: bad date")
                        continue
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

    # Validate settings
    assets = st.session_state.assets.copy()
    if assets.empty:
        st.warning("Add at least one Asset Class in Global Settings.")
        st.stop()

    # Basic checks
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("End Date must be after Start Date.")
        st.stop()

    if abs(pd.to_numeric(assets.get("Benchmark Weight %", pd.Series(dtype=float)), errors="coerce").fillna(0).sum() - 100) > 1e-6:
        st.error("Benchmark Weights must sum to 100%.")
        st.stop()

    # Clean assets table numbers
    assets["Beginning MV"] = assets["Beginning MV"].apply(to_float).fillna(0.0)
    assets["Ending MV"] = assets["Ending MV"].apply(to_float).fillna(0.0)
    assets["Benchmark Weight %"] = assets["Benchmark Weight %"].apply(to_float).fillna(0.0)
    assets["Benchmark Return %"] = assets["Benchmark Return %"].apply(to_float).fillna(0.0)

    if assets["Beginning MV"].sum() <= 0:
        st.error("Total Beginning MV must be > 0.")
        st.stop()

    # Build flows df (signed)
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
    # Per-class Dietz
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
        try:
            rP = modified_dietz(bv, ev, f_cls, t0, t1)
        except Exception as e:
            st.error(f"{name}: Dietz error → {e}")
            st.stop()

        wP = 0.0 if total_bv <= 0 else bv / total_bv
        class_rows.append({"name": name, "wP": wP, "wB": wB, "rP": rP, "rB": rB, "bv": bv, "ev": ev})

    # Portfolio Dietz
    ev_port = sum(c["ev"] for c in class_rows)
    flows_port = flows[["when", "amount"]].copy() if not flows.empty else pd.DataFrame(columns=["when", "amount"])
    try:
        r_port = modified_dietz(total_bv, ev_port, flows_port, t0, t1)
    except Exception as e:
        st.error(f"Portfolio Dietz error → {e}")
        st.stop()

    r_bench = sum(c["wB"] * c["rB"] for c in class_rows)
    excess = r_port - r_bench

    # Attribution table
    attrib_df = brinson_attribution(class_rows)

    # Display metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Portfolio Return (Dietz)", pct(r_port))
    m2.metric("Benchmark Return", pct(r_bench))
    m3.metric("Excess Return", pct(excess))

    # Display table
    show = attrib_df.copy()
    pct_cols = [
        "Portfolio Weight","Benchmark Weight","Portfolio Return","Benchmark Return",
        "Asset Allocation","Stock Selection","Market Timing","Total Active",
    ]
    for c in pct_cols:
        show[c] = show[c].apply(pct)
    st.dataframe(show, use_container_width=True)

    # Download
    out = attrib_df.copy()
    out_csv = out.to_csv(index=False).encode("utf-8")
    st.download_button("Download Attribution CSV", out_csv, "attribution.csv", "text/csv")

    # Raw per-class detail (optional)
    with st.expander("Per-asset inputs (raw)"):
        raw = pd.DataFrame(class_rows)
        st.dataframe(raw, use_container_width=True)
