# app.py
# MDTWRR Performance Portal (Streamlit) — robust CSV, tolerant Dietz, Reporting + Export (Excel/PDF)
# Run locally:  pip install -r requirements.txt
# Start:        streamlit run app.py

from datetime import datetime, date, timedelta
import io
import numpy as np
import pandas as pd
import streamlit as st

# PDF export
from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Table, TableStyle, Spacer
from reportlab.lib.styles import getSampleStyleSheet

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

def chain_link(returns: pd.Series) -> pd.Series:
    """Turn period returns into cumulative index (start=1.0)."""
    if returns.empty: return pd.Series(dtype=float)
    idx = (1.0 + returns.fillna(0)).cumprod()
    idx.iloc[0] = (1.0 + returns.iloc[0])
    return idx

def max_drawdown(index: pd.Series) -> pd.Series:
    if index.empty: return pd.Series(dtype=float)
    running_max = index.cummax()
    return index / running_max - 1.0

def subperiod_dietz(bv, ev, flows, t0, t1):
    return modified_dietz(bv, ev, flows, t0, t1)

def compute_timeseries_from_valuations(vals_df, flows_df, bench_ret_series=None):
    """
    vals_df columns: Date, Asset Class, Market Value
    flows_df: ['when','amount','asset'] signed
    """
    v = vals_df.copy()
    v["Date"] = pd.to_datetime(v["Date"], errors="coerce").dt.tz_localize(None)
    v = v.dropna(subset=["Date"])
    v["Asset Class"] = v["Asset Class"].astype(str).str.strip()
    v["Market Value"] = v["Market Value"].apply(to_float).fillna(0.0)
    if v.empty or v["Date"].nunique() < 2:
        return {"error": "Valuations CSV must include at least two dates."}
    dates = sorted(v["Date"].unique().tolist())
    periods = list(zip(dates[:-1], dates[1:]))

    flows = flows_df.copy() if flows_df is not None else pd.DataFrame(columns=["when","amount","asset"])
    if not flows.empty:
        flows["when"] = pd.to_datetime(flows["when"], errors="coerce").dt.tz_localize(None)

    period_rets = []
    for (t0, t1) in periods:
        bv_df = v[v["Date"] == t0].groupby("Asset Class", as_index=False)["Market Value"].sum()
        ev_df = v[v["Date"] == t1].groupby("Asset Class", as_index=False)["Market Value"].sum()
        assets = sorted(set(bv_df["Asset Class"]).union(set(ev_df["Asset Class"])))
        port_bv, port_ev = 0.0, 0.0
        flows_period = []
        for a in assets:
            bv_a = float(bv_df.loc[bv_df["Asset Class"] == a, "Market Value"].sum())
            ev_a = float(ev_df.loc[ev_df["Asset Class"] == a, "Market Value"].sum())
            f_a = pd.DataFrame(columns=["when","amount"])
            if not flows.empty:
                fa = flows[(flows["asset"] == a) & (flows["when"] > t0) & (flows["when"] <= t1)][["when","amount"]]
                f_a = fa.copy() if not fa.empty else f_a
            port_bv += bv_a
            port_ev += ev_a
            if not f_a.empty: flows_period.append(f_a)
        fp = pd.concat(flows_period, ignore_index=True) if flows_period else pd.DataFrame(columns=["when","amount"])
        try:
            rP = subperiod_dietz(port_bv, port_ev, fp, t0, t1)
        except Exception:
            rP = 0.0
        period_rets.append(rP)

    r_port = pd.Series(period_rets, index=[pd.Timestamp(t1) for (_, t1) in periods], name="Portfolio")
    cum_port = chain_link(r_port)
    result = {"periods": periods, "r_port": r_port, "cum_port": cum_port, "drawdown": max_drawdown(cum_port)}
    if bench_ret_series is not None and not bench_ret_series.empty:
        r_bench = bench_ret_series.reindex(r_port.index).fillna(0.0)
        result["r_bench"] = r_bench
        result["cum_bench"] = chain_link(r_bench)
    return result

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
if "report" not in st.session_state:
    st.session_state.report = {}  # we’ll stash exportables here

# -----------------------------
# Tabs
# -----------------------------
tab_settings, tab_cashflows, tab_results, tab_reporting = st.tabs(
    ["1) Global Settings", "2) Cashflow Upload", "3) Results", "4) Reporting"]
)

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
        flows["amount"] = np.where(flows["Transaction Type"].str.upper() == "INFLOW", flows["Amount"], -flows["Amount"])
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

    if issues: st.warning("Data issues detected:\n- " + "\n- ".join(issues))

    # Save for export (Results tab)
    st.session_state.report.update({
        "period": (st.session_state.start_date, st.session_state.end_date),
        "metrics": {"Portfolio": r_port, "Benchmark": r_bench, "Excess": excess},
        "attrib": attrib_df.copy(),
        "assets": st.session_state.assets.copy(),
        "flows": st.session_state.flows.copy(),
    })

# =============================
# 4) REPORTING
# =============================
with tab_reporting:
    st.subheader("Reporting & Analytics")

    assets = st.session_state.assets.copy()
    flows = st.session_state.flows.copy()

    total_bv = pd.to_numeric(assets.get("Beginning MV", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    total_ev = pd.to_numeric(assets.get("Ending MV", pd.Series(dtype=float)), errors="coerce").fillna(0).sum()
    n_assets = assets["Asset Class"].dropna().astype(str).str.strip().nunique()

    if not flows.empty:
        flows_disp = flows.copy()
        flows_disp["amount_signed"] = np.where(flows_disp["Transaction Type"].str.upper() == "INFLOW",
                                              flows_disp["Amount"], -flows_disp["Amount"])
        net_flows = pd.to_numeric(flows_disp["amount_signed"], errors="coerce").fillna(0).sum()
    else:
        net_flows = 0.0

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Beginning MV", f"{total_bv:,.0f}")
    c2.metric("Ending MV", f"{total_ev:,.0f}")
    c3.metric("Net Flows", f"{net_flows:,.0f}")
    c4.metric("# Asset Classes", f"{n_assets}")

    # --- B) Contribution bars
    st.markdown("#### Contribution to Active Return by Asset Class")
    contrib = pd.DataFrame()
    if not assets.empty:
        # Quick recompute to get contributions
        assets_calc = assets.copy()
        assets_calc["Beginning MV"] = assets_calc["Beginning MV"].apply(to_float).fillna(0.0)
        assets_calc["Ending MV"] = assets_calc["Ending MV"].apply(to_float).fillna(0.0)
        assets_calc["Benchmark Weight %"] = assets_calc["Benchmark Weight %"].apply(to_float).fillna(0.0)
        assets_calc["Benchmark Return %"] = assets_calc["Benchmark Return %"].apply(to_float).fillna(0.0)

        total_bv2 = assets_calc["Beginning MV"].sum()
        if not flows.empty:
            flows_calc = flows.copy()
            flows_calc["when"] = pd.to_datetime(flows_calc["Transaction Date"], errors="coerce").dt.tz_localize(None)
            flows_calc["amount"] = np.where(flows_calc["Transaction Type"].str.upper() == "INFLOW", flows_calc["Amount"], -flows_calc["Amount"])
            flows_calc["asset"] = flows_calc["Asset Class"].astype(str).str.strip()
        else:
            flows_calc = pd.DataFrame(columns=["when","amount","asset"])

        t0 = datetime.combine(st.session_state.start_date, datetime.min.time())
        t1 = datetime.combine(st.session_state.end_date, datetime.max.time())

        rows = []
        for _, r in assets_calc.iterrows():
            name = str(r["Asset Class"]).strip()
            bv = float(r["Beginning MV"]); ev = float(r["Ending MV"])
            wB = float(r["Benchmark Weight %"]) / 100.0; rB = float(r["Benchmark Return %"]) / 100.0
            if name == "" or bv < 0 or ev < 0: continue
            fA = flows_calc[flows_calc["asset"] == name][["when","amount"]] if not flows_calc.empty else pd.DataFrame(columns=["when","amount"])
            try:
                rP = modified_dietz(bv, ev, fA, t0, t1)
            except Exception:
                rP = 0.0
            wP = 0.0 if total_bv2 <= 0 else bv / total_bv2
            alloc = (wP - wB) * rB
            select = wB * (rP - rB)
            timing = (wP - wB) * (rP - rB)
            active = alloc + select + timing
            rows.append({"Asset Class": name, "Active": active, "Allocation": alloc, "Selection": select, "Timing": timing})
        contrib = pd.DataFrame(rows).sort_values("Active", ascending=False)
        st.dataframe(contrib, use_container_width=True)
        st.bar_chart(contrib.set_index("Asset Class")[["Active"]])
    else:
        st.info("Add assets in Global Settings to see contributions.")

    # --- C) Cashflow analytics
    st.markdown("#### Cashflow Analytics")
    if flows.empty:
        st.info("No cashflows loaded.")
        cf_by_type = pd.DataFrame(); cf_by_asset = pd.DataFrame(); cf_ts = pd.Series(dtype=float)
    else:
        cf = flows.copy()
        cf["Transaction Date"] = pd.to_datetime(cf["Transaction Date"], errors="coerce").dt.tz_localize(None)
        cf["signed"] = np.where(cf["Transaction Type"].str.upper() == "INFLOW", cf["Amount"], -cf["Amount"])
        cf_by_type = cf.groupby("Transaction Type", as_index=False)["Amount"].sum()
        cf_by_asset = cf.groupby("Asset Class", as_index=False)["signed"].sum().sort_values("signed", ascending=False)
        cA, cB = st.columns(2)
        with cA:
            st.write("By Type")
            st.dataframe(cf_by_type, use_container_width=True)
            st.bar_chart(cf_by_type.set_index("Transaction Type"))
        with cB:
            st.write("Net by Asset Class")
            st.dataframe(cf_by_asset, use_container_width=True)
            st.bar_chart(cf_by_asset.set_index("Asset Class"))
        st.write("Cashflow Timeline")
        cf_ts = cf.set_index("Transaction Date")["signed"].resample("W").sum().fillna(0)
        st.line_chart(cf_ts)

    # --- D) Optional: time-series TWRR (Valuations CSV)
    st.markdown("#### Optional: Upload Valuations CSV for Chain-Linked TWRR")
    st.caption("Headers: **Date, Asset Class, Market Value** (period ends).")
    if not assets.empty:
        d0 = st.session_state.start_date; d1 = st.session_state.end_date
        val_tpl = pd.DataFrame({
            "Date": [d0.isoformat(), d1.isoformat()],
            "Asset Class": [assets.iloc[0]["Asset Class"], assets.iloc[0]["Asset Class"]],
            "Market Value": [assets.iloc[0].get("Beginning MV", 0.0), assets.iloc[0].get("Ending MV", 0.0)],
        })
    else:
        val_tpl = pd.DataFrame({"Date": [date.today().isoformat(), (date.today()+timedelta(days=30)).isoformat()],
                                "Asset Class": ["Equity","Equity"], "Market Value": [1000000, 1010000]})
    st.download_button("Download Valuations CSV Template", val_tpl.to_csv(index=False).encode("utf-8"),
                       "valuations_template.csv", "text/csv")

    val_file = st.file_uploader("Upload Valuations CSV (optional)", type=["csv"], key="vals_csv")
    ts_res = {}
    if val_file is not None:
        try:
            vals_df = read_csv_robust(val_file)
            rnm = {}
            for c in vals_df.columns:
                cl = c.strip().lower()
                if cl in ("date","valuation date","period end"): rnm[c] = "Date"
                elif cl in ("asset class","assetclass","class"): rnm[c] = "Asset Class"
                elif cl in ("market value","mv","value"): rnm[c] = "Market Value"
            vals_df = vals_df.rename(columns=rnm)
            if not {"Date","Asset Class","Market Value"}.issubset(vals_df.columns):
                st.error("Valuations CSV must have headers: Date, Asset Class, Market Value")
            else:
                if st.session_state.flows.empty:
                    flows_ts = pd.DataFrame(columns=["when","amount","asset"])
                else:
                    flows_ts = st.session_state.flows.copy()
                    flows_ts["when"] = pd.to_datetime(flows_ts["Transaction Date"], errors="coerce").dt.tz_localize(None)
                    flows_ts["amount"] = np.where(flows_ts["Transaction Type"].str.upper()=="INFLOW",
                                                  flows_ts["Amount"], -st.session_state.flows["Amount"])
                    flows_ts["asset"] = flows_ts["Asset Class"].astype(str).str.strip()
                bench_const = None
                if not st.session_state.assets.empty:
                    a = st.session_state.assets.copy()
                    a["Benchmark Weight %"] = a["Benchmark Weight %"].apply(to_float).fillna(0.0)
                    a["Benchmark Return %"] = a["Benchmark Return %"].apply(to_float).fillna(0.0)
                    bench_const = (a["Benchmark Weight %"]/100.0 * a["Benchmark Return %"]/100.0).sum()
                bench_series = None
                if bench_const is not None:
                    tmp_idx = pd.to_datetime(sorted(vals_df["Date"].unique()[1:]))
                    bench_series = pd.Series([bench_const]*len(tmp_idx), index=tmp_idx, name="Benchmark")
                ts_res = compute_timeseries_from_valuations(vals_df, flows_ts, bench_series)
                if "error" in ts_res:
                    st.error(ts_res["error"])
                else:
                    cX, cY = st.columns(2)
                    with cX:
                        st.write("Cumulative Index")
                        df_plot = pd.DataFrame({"Portfolio": ts_res["cum_port"]})
                        if ts_res.get("cum_bench") is not None:
                            df_plot["Benchmark"] = ts_res["cum_bench"]
                        st.line_chart(df_plot)
                    with cY:
                        st.write("Drawdown")
                        st.area_chart(ts_res["drawdown"])
                    st.write("Per-period returns")
                    show_r = pd.DataFrame({"Portfolio": ts_res["r_port"]})
                    if ts_res.get("r_bench") is not None:
                        show_r["Benchmark"] = ts_res["r_bench"]
                    st.dataframe(show_r.applymap(lambda x: f"{x*100:.2f}%"), use_container_width=True)
        except Exception as e:
            st.error(f"Could not process Valuations CSV: {e}")

    # ========== EXPORT ==========
    st.markdown("### Export")
    # Prepare export payload
    report_payload = {
        "period": st.session_state.report.get("period"),
        "metrics": st.session_state.report.get("metrics"),
        "attrib": st.session_state.report.get("attrib", pd.DataFrame()),
        "contrib": contrib.copy(),
        "assets": st.session_state.report.get("assets", pd.DataFrame()),
        "flows": st.session_state.report.get("flows", pd.DataFrame()),
        "cf_by_type": cf_by_type.copy(),
        "cf_by_asset": cf_by_asset.copy(),
        "cf_ts": cf_ts.copy(),
        "timeseries": ts_res,  # may be {}
    }

    def build_excel_bytes(rep: dict) -> bytes:
        buf = io.BytesIO()
        with pd.ExcelWriter(buf, engine="xlsxwriter") as xw:
            # Summary
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
            rep.get("contrib", pd.DataFrame()).to_excel(xw, sheet_name="Contributions", index=False)
            rep.get("assets", pd.DataFrame()).to_excel(xw, sheet_name="Assets", index=False)
            rep.get("flows", pd.DataFrame()).to_excel(xw, sheet_name="Cashflows", index=False)
            rep.get("cf_by_type", pd.DataFrame()).to_excel(xw, sheet_name="CF By Type", index=False)
            rep.get("cf_by_asset", pd.DataFrame()).to_excel(xw, sheet_name="CF By Asset", index=False)
            # Time series (if any)
            ts = rep.get("timeseries", {})
            if ts and "r_port" in ts:
                pd.DataFrame({"Portfolio": ts["r_port"]}).to_excel(xw, sheet_name="Period Returns", index=True)
            if ts and "cum_port" in ts:
                pd.DataFrame({"Portfolio": ts["cum_port"]}).to_excel(xw, sheet_name="Cumulative", index=True)
            if ts and "drawdown" in ts:
                pd.DataFrame({"Drawdown": ts["drawdown"]}).to_excel(xw, sheet_name="Drawdown", index=True)
        return buf.getvalue()

    def build_pdf_bytes(rep: dict) -> bytes:
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4, leftMargin=36, rightMargin=36, topMargin=36, bottomMargin=36)
        styles = getSampleStyleSheet()
        els = []
        # Title
        els.append(Paragraph("MDTWRR Performance Report", styles["Title"]))
        period = rep.get("period", (None, None))
        els.append(Paragraph(f"Period: {period[0]} → {period[1]}", styles["Normal"]))
        els.append(Spacer(1, 8))

        metr = rep.get("metrics", {})
        mt = [["Metric", "Value"],
              ["Portfolio Return (Dietz)", pct(metr.get("Portfolio"))],
              ["Benchmark Return", pct(metr.get("Benchmark"))],
              ["Excess Return", pct(metr.get("Excess"))]]
        table = Table(mt, hAlign="LEFT")
        table.setStyle(TableStyle([("BACKGROUND",(0,0),(1,0), colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
                                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
        els.append(table)
        els.append(Spacer(1, 12))

        # Attribution (top 15 rows)
        attrib = rep.get("attrib", pd.DataFrame())
        if not attrib.empty:
            els.append(Paragraph("Attribution (Top Contributors)", styles["Heading2"]))
            top = attrib.copy()
            if "Total" in top["Sector"].values:
                top = top[top["Sector"] != "Total"]
            top = top.sort_values("Total Active", ascending=False).head(15)
            data = [list(["Sector","Portfolio Weight","Benchmark Weight","Portfolio Return","Benchmark Return","Asset Allocation","Stock Selection","Market Timing","Total Active"])]
            data += [[
                r["Sector"],
                f"{r['Portfolio Weight']:.4f}",
                f"{r['Benchmark Weight']:.4f}",
                f"{r['Portfolio Return']:.4f}",
                f"{r['Benchmark Return']:.4f}",
                f"{r['Asset Allocation']:.4f}",
                f"{r['Stock Selection']:.4f}",
                f"{r['Market Timing']:.4f}",
                f"{r['Total Active']:.4f}",
            ] for _, r in top.iterrows()]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8),
                                   ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1), 0.25, colors.grey),
                                   ("FONTNAME",(0,0),(-1,0),"Helvetica-Bold")]))
            els.append(t)
            els.append(Spacer(1, 12))

        # Cashflow summary
        cf_by_type = rep.get("cf_by_type", pd.DataFrame())
        if not cf_by_type.empty:
            els.append(Paragraph("Cashflows by Type", styles["Heading2"]))
            data = [["Type","Amount"]]+[[r["Transaction Type"], f"{r['Amount']:,.2f}"] for _, r in cf_by_type.iterrows()]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            els.append(t)
            els.append(Spacer(1, 8))

        cf_by_asset = rep.get("cf_by_asset", pd.DataFrame())
        if not cf_by_asset.empty:
            els.append(Paragraph("Net Cashflows by Asset Class", styles["Heading2"]))
            data = [["Asset Class","Net Flow"]]+[[r["Asset Class"], f"{r['signed']:,.2f}"] for _, r in cf_by_asset.iterrows()]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            els.append(t)
            els.append(Spacer(1, 8))

        # Timeseries (if present)
        ts = rep.get("timeseries", {})
        if ts and "r_port" in ts:
            els.append(Paragraph("Per-Period Returns (Portfolio)", styles["Heading2"]))
            data = [["Period End","Return %"]]+[[str(idx.date()), f"{val*100:.2f}%"] for idx, val in ts["r_port"].items()]
            t = Table(data, hAlign="LEFT")
            t.setStyle(TableStyle([("FONTSIZE",(0,0),(-1,-1),8),
                                   ("BACKGROUND",(0,0),(-1,0), colors.lightgrey),
                                   ("GRID",(0,0),(-1,-1), 0.25, colors.grey)]))
            els.append(t)

        doc.build(els)
        return buf.getvalue()

    # Download buttons
    xls_bytes = build_excel_bytes(report_payload)
    st.download_button("⬇️ Download Excel Report", xls_bytes, "MDTWRR_Report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    pdf_bytes = build_pdf_bytes(report_payload)
    st.download_button("⬇️ Download PDF Summary", pdf_bytes, "MDTWRR_Summary.pdf", "application/pdf")
