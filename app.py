import streamlit as st
st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.caption("Initializing…")

# app.py — MDTWRR (Modified Dietz) Performance Portal
# Pages: Global Settings • Cashflows • Adjusted Cashflows • Results
# Returns: one-period Modified Dietz per asset class, annualized, plus contribution

from __future__ import annotations
import io
from datetime import date, datetime
import numpy as np
import pandas as pd
import streamlit as st

# ----------------------- App setup -----------------------
st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")
st.title("MDTWRR Performance Portal")
st.caption("One-period Modified Dietz (TWRR). Clean math. No attribution.")

MIN_DATE = date(1900, 1, 1)
MAX_DATE = date(2100, 12, 31)
INFLOW, OUTFLOW = "INFLOW", "OUTFLOW"
ALLOWED_TYPES = [INFLOW, OUTFLOW]

# ----------------------- Helpers -------------------------
def pct(x: float, dp: int = 2) -> str:
    return "-" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x*100:.{dp}f}%"

def to_float(x):
    if x is None:
        return np.nan
    try:
        return float(str(x).replace(",", "").strip())
    except Exception:
        return np.nan

def canon(s: str) -> str:
    # normalize header names for auto-mapping
    return "".join(ch.lower() for ch in str(s).strip() if ch.isalnum())

def read_csv_robust(uploaded) -> pd.DataFrame:
    raw = uploaded.getvalue()
    for enc in ["utf-8", "utf-8-sig", "cp1252", "latin1", "iso-8859-1", "utf-16", "utf-16le", "utf-16be"]:
        try:
            return pd.read_csv(io.BytesIO(raw), encoding=enc, sep=None, engine="python")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(raw))  # last attempt

def mdietz(bv: float, ev: float, flows: pd.DataFrame, t0: datetime, t1: datetime, eps=1e-12) -> float:
    """
    Modified Dietz:
      R = (EV - BV - ΣCF) / (BV + Σ(w_i * CF_i)), where w_i = (t1 - t_i) / (t1 - t0)
    Notes:
      • Include flows on Start Date (t_i == t0) with weight 1.0; End Date flows weight 0.0
      • CF are signed: +INFLOW, −OUTFLOW
      • Returns decimal (0.12 = 12%)
    """
    if not np.isfinite(bv) or not np.isfinite(ev):
        raise ValueError("BV/EV must be finite")
    if not (isinstance(t0, datetime) and isinstance(t1, datetime) and t1 > t0):
        raise ValueError("Bad period")
    T = (t1 - t0).total_seconds()
    if T <= 0:
        raise ValueError("Zero period")

    f = pd.DataFrame(columns=["when", "amount"]) if flows is None else flows.copy()
    if not f.empty:
        f = f[(f["when"] >= t0) & (f["when"] <= t1)]
        f = f[np.isfinite(f["amount"])]

    sum_cf = 0.0 if f.empty else f["amount"].sum()

    denom = bv
    if not f.empty:
        w = (t1 - f["when"]).dt.total_seconds() / T
        w = w.clip(lower=0.0, upper=1.0)
        denom += (w * f["amount"]).sum()

    if abs(denom) < eps:
        raise ValueError("Unstable denominator")

    numer = ev - bv - sum_cf
    return numer / denom

def annualize(r: float, t0: datetime, t1: datetime) -> float:
    """ACT/365 annualization."""
    days = (t1 - t0).days or 1
    return (1.0 + r) ** (365.0 / days) - 1.0

# ----------------------- Session state -------------------
if "assets" not in st.session_state:
    st.session_state.assets = pd.DataFrame(
        columns=["Asset Class", "Beginning MV", "Ending MV"]
    )
if "flows" not in st.session_state:
    st.session_state.flows = pd.DataFrame(
        columns=["Transaction Date", "Transaction Type", "Transaction Details", "Amount", "Asset Class"]
    )
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2024, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = date(2024, 12, 31)

# ----------------------- Tabs ----------------------------
tab1, tab2, tab3, tab4 = st.tabs(["Global Settings", "Cashflows Upload", "Adjusted Cashflows", "Results"])

# =========================================================
# 1) Global Settings
# =========================================================
with tab1:
    st.subheader("Period & Asset Classes")

    c1, c2 = st.columns(2)
    with c1:
        st.session_state.start_date = st.date_input(
            "Start Date", st.session_state.start_date, min_value=MIN_DATE, max_value=MAX_DATE
        )
    with c2:
        st.session_state.end_date = st.date_input(
            "End Date", st.session_state.end_date, min_value=MIN_DATE, max_value=MAX_DATE
        )

    st.markdown("#### Asset Classes & Market Values")
    st.caption("Upload CSV with headers (flexible): Asset Class, Beginning MV, Ending MV")

    up = st.file_uploader("Upload Assets CSV", type=["csv"], key="assets_csv")
    if up is not None:
        try:
            df = read_csv_robust(up)
            rename = {}
            for c in df.columns:
                cn = canon(c)
                if cn in {"assetclass", "asset", "class", "sector"}:
                    rename[c] = "Asset Class"
                elif cn in {"beginningmv", "openingmv", "bv", "bmv", "startmv"}:
                    rename[c] = "Beginning MV"
                elif cn in {"endingmv", "closingmv", "ev", "emv", "endmv"}:
                    rename[c] = "Ending MV"
            df = df.rename(columns=rename)
            need = ["Asset Class", "Beginning MV", "Ending MV"]
            miss = [c for c in need if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                for col in ["Beginning MV", "Ending MV"]:
                    df[col] = df[col].apply(to_float)
                st.session_state.assets = df[need].copy()
                st.success("Assets loaded.")
        except Exception as e:
            st.error(f"Could not read assets CSV: {e}")

    st.session_state.assets = st.data_editor(
        st.session_state.assets,
        num_rows="dynamic",
        use_container_width=True,
        column_config={
            "Asset Class": st.column_config.TextColumn(required=True),
            "Beginning MV": st.column_config.NumberColumn(min_value=0.0),
            "Ending MV": st.column_config.NumberColumn(min_value=0.0),
        },
        key="assets_editor"
    )

# =========================================================
# 2) Cashflows Upload
# =========================================================
with tab2:
    st.subheader("Cashflows Upload")
    st.caption("CSV headers (flexible): Transaction Date, Transaction Type (INFLOW/OUTFLOW), Transaction Details, Amount, Asset Class")

    st.download_button(
        "Download Cashflow Template",
        pd.DataFrame({
            "Transaction Date": [str(st.session_state.start_date), str(st.session_state.end_date)],
            "Transaction Type": [INFLOW, OUTFLOW],
            "Transaction Details": ["Initial investment", "Fees"],
            "Amount": [100000, 1500],
            "Asset Class": ["Equity", "Equity"],
        }).to_csv(index=False).encode("utf-8"),
        "cashflows_template.csv", "text/csv"
    )

    f = st.file_uploader("Upload Cashflow CSV", type=["csv"])
    if f is not None:
        try:
            df = read_csv_robust(f)
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
            miss = [c for c in need if c not in df.columns]
            if miss:
                st.error(f"Missing columns: {miss}")
            else:
                df["Transaction Date"] = pd.to_datetime(df["Transaction Date"], errors="coerce").dt.tz_localize(None)
                df["Transaction Type"] = df["Transaction Type"].astype(str).str.upper().str.strip()
                df["Transaction Details"] = df["Transaction Details"].astype(str).fillna("")
                df["Amount"] = df["Amount"].apply(to_float)
                df["Asset Class"] = df["Asset Class"].astype(str).str.strip()
                st.session_state.flows = df.copy()
                st.success("Cashflows loaded.")
        except Exception as e:
            st.error(f"Could not read cashflow CSV: {e}")

    st.markdown("#### Loaded Cashflows")
    st.dataframe(st.session_state.flows, use_container_width=True)

# =========================================================
# 3) Adjusted Cashflows (weights view)
# =========================================================
with tab3:
    st.subheader("Time-Weighted Cashflows (Modified Dietz weights)")

    if st.session_state.flows.empty:
        st.info("Upload cashflows to see the weighted view.")
    else:
        t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
        t1 = datetime.combine(st.session_state.end_date, datetime.max.time())
        T = (t1 - t0).total_seconds() or 1

        flows = st.session_state.flows.copy()
        flows["when"] = pd.to_datetime(flows["Transaction Date"], errors="coerce").dt.tz_localize(None)
        flows["sign_amount"] = np.where(flows["Transaction Type"].str.upper() == INFLOW, flows["Amount"], -flows["Amount"])
        flows = flows[(flows["when"] >= t0) & (flows["when"] <= t1)].copy()

        flows["w"] = ((t1 - flows["when"]).dt.total_seconds() / T).clip(lower=0.0, upper=1.0)
        flows["weighted_amount"] = flows["w"] * flows["sign_amount"]

        pretty = flows[["Transaction Date","Transaction Type","Transaction Details","Amount","Asset Class","w","weighted_amount"]].copy()
        pretty.rename(columns={"w":"Weight (w)","weighted_amount":"w × Amount"}, inplace=True)
        st.dataframe(pretty, use_container_width=True)

# =========================================================
# 4) Results
# =========================================================
with tab4:
    st.subheader("Performance Results (Modified Dietz)")

    # Validations
    assets = st.session_state.assets.copy()
    if assets.empty:
        st.warning("Add/upload asset classes first."); st.stop()
    if st.session_state.start_date >= st.session_state.end_date:
        st.error("End Date must be after Start Date."); st.stop()

    # Clean numbers
    assets["Beginning MV"] = assets["Beginning MV"].apply(to_float).fillna(0.0)
    assets["Ending MV"] = assets["Ending MV"].apply(to_float).fillna(0.0)

    total_bv = assets["Beginning MV"].sum()
    if total_bv <= 0:
        st.error("Total Beginning MV must be > 0."); st.stop()

    # Prepare flows (signed)
    flows_all = st.session_state.flows.copy()
    if not flows_all.empty:
        flows_all["when"] = pd.to_datetime(flows_all["Transaction Date"], errors="coerce").dt.tz_localize(None)
        flows_all["amount"] = np.where(flows_all["Transaction Type"].str.upper()==INFLOW, flows_all["Amount"], -flows_all["Amount"])
        flows_all["asset"] = flows_all["Asset Class"].astype(str).str.strip()
    else:
        flows_all = pd.DataFrame(columns=["when","amount","asset"])

    t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
    t1 = datetime.combine(st.session_state.end_date, datetime.max.time())

    # Per-asset Dietz & contribution
    rows = []
    issues = []
    for _, r in assets.iterrows():
        name = str(r["Asset Class"]).strip()
        bv, ev = float(r["Beginning MV"]), float(r["Ending MV"])
        fA = flows_all[flows_all["asset"] == name][["when","amount"]].copy() if not flows_all.empty else pd.DataFrame(columns=["when","amount"])
        try:
            if bv == 0 and (fA.empty or abs(fA["amount"].sum()) < 1e-12):
                rP = 0.0
            else:
                rP = mdietz(bv, ev, fA, t0, t1)
        except Exception as e:
            issues.append(f"{name}: {e}")
            rP = 0.0

        wP = bv / total_bv  # beginning-weight contribution base
        contrib = wP * rP   # simple contribution approximation
        rows.append({"Asset Class": name, "Beginning MV": bv, "Ending MV": ev, "Weight (BV)": wP, "Return (period)": rP, "Contribution": contrib})

    df_res = pd.DataFrame(rows)
    df_res["Return (annualized)"] = df_res["Return (period)"].apply(lambda x: annualize(x, t0, t1))

    # Portfolio (aggregate Dietz)
    ev_port = df_res["Ending MV"].sum()
    flows_port = flows_all[["when","amount"]].copy()
    try:
        r_port = mdietz(total_bv, ev_port, flows_port, t0, t1)
    except Exception as e:
        st.error(f"Portfolio Dietz error → {e}"); st.stop()
    r_port_ann = annualize(r_port, t0, t1)

    # Display
    c1, c2 = st.columns(2)
    c1.metric("Portfolio Return (period)", pct(r_port))
    c2.metric("Portfolio Return (annualized)", pct(r_port_ann))

    pretty = df_res.copy()
    for c in ["Weight (BV)", "Return (period)", "Return (annualized)", "Contribution"]:
        pretty[c] = pretty[c].apply(pct)
    st.dataframe(pretty[["Asset Class","Beginning MV","Ending MV","Weight (BV)","Return (period)","Return (annualized)","Contribution"]],
                 use_container_width=True)

    if issues:
        st.warning("Data issues:\n- " + "\n- ".join(issues))
