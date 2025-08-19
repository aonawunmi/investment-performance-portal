import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, date
import io

st.set_page_config(page_title="MDTWRR Performance Portal", layout="wide")

# ------------------------------------------------
# GLOBAL CONSTANTS
# ------------------------------------------------
MIN_DATE = date(1900, 1, 1)
MAX_DATE = date(2100, 12, 31)

# ------------------------------------------------
# SESSION STATE INIT
# ------------------------------------------------
if "start_date" not in st.session_state:
    st.session_state.start_date = date(2015, 1, 1)
if "end_date" not in st.session_state:
    st.session_state.end_date = date.today()
if "cashflows" not in st.session_state:
    st.session_state.cashflows = pd.DataFrame()
if "global_settings" not in st.session_state:
    st.session_state.global_settings = {}

# ------------------------------------------------
# NAVIGATION
# ------------------------------------------------
tabs = st.tabs(["Global Settings", "Cashflow Upload", "Results", "Reporting"])

# ------------------------------------------------
# TAB 1 – GLOBAL SETTINGS
# ------------------------------------------------
with tabs[0]:
    st.header("Period")
    c1, c2 = st.columns(2)
    with c1:
        st.session_state.start_date = st.date_input(
            "Start Date",
            value=st.session_state.start_date,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            key="start_date_input",
        )
    with c2:
        st.session_state.end_date = st.date_input(
            "End Date",
            value=st.session_state.end_date,
            min_value=MIN_DATE,
            max_value=MAX_DATE,
            key="end_date_input",
        )

    st.markdown("### Load Period via CSV (optional)")
    file = st.file_uploader("Upload Global Settings CSV", type=["csv"], key="global_settings")
    if file:
        df = pd.read_csv(file)
        st.session_state.global_settings = df.to_dict(orient="records")[0]

        # apply period from CSV if present
        if "start_date" in df.columns and pd.notna(df.loc[0, "start_date"]):
            s = pd.to_datetime(df.loc[0, "start_date"]).date()
            s = max(MIN_DATE, min(MAX_DATE, s))
            st.session_state.start_date = s
        if "end_date" in df.columns and pd.notna(df.loc[0, "end_date"]):
            e = pd.to_datetime(df.loc[0, "end_date"]).date()
            e = max(MIN_DATE, min(MAX_DATE, e))
            st.session_state.end_date = e

        st.success("Global settings loaded from CSV.")

# ------------------------------------------------
# TAB 2 – CASHFLOW UPLOAD
# ------------------------------------------------
with tabs[1]:
    st.header("Cashflow Upload")
    f = st.file_uploader("Upload Cashflow CSV", type=["csv"], key="cashflows")
    if f:
        df = pd.read_csv(f)
        st.session_state.cashflows = df
        st.dataframe(df)

# ------------------------------------------------
# TAB 3 – RESULTS
# ------------------------------------------------
with tabs[2]:
    st.header("Performance & Attribution")
    if st.session_state.cashflows.empty:
        st.warning("Please upload cashflows first.")
    else:
        df = st.session_state.cashflows.copy()

        # sanity check
        if "begin_mv" not in df.columns or df["begin_mv"].sum() <= 0:
            st.error("Total Beginning MV must be > 0.")
        else:
            df["period_return"] = (df["end_mv"] - df["begin_mv"] - df["net_cf"]) / df["begin_mv"]
            total_return = np.prod(1 + df["period_return"]) - 1
            st.metric("Total Time-Weighted Return", f"{total_return:.2%}")
            st.dataframe(df)

# ------------------------------------------------
# TAB 4 – REPORTING
# ------------------------------------------------
with tabs[3]:
    st.header("Reporting Dashboard")
    if st.session_state.cashflows.empty:
        st.info("Upload data to view reports.")
    else:
        df = st.session_state.cashflows.copy()

        # summary stats
        st.subheader("Summary Statistics")
        total_invested = df["net_cf"].sum()
        total_mv_end = df["end_mv"].iloc[-1]
        avg_return = df["period_return"].mean() if "period_return" in df else np.nan

        st.write({
            "Total Invested": total_invested,
            "Ending Market Value": total_mv_end,
            "Average Period Return": avg_return
        })

        # cumulative curve
        if "period_return" in df:
            df["cumulative"] = (1 + df["period_return"]).cumprod()
            st.line_chart(df.set_index("date")["cumulative"])

        # export
        buffer = io.BytesIO()
        df.to_excel(buffer, index=False)
        st.download_button(
            label="Download Results as Excel",
            data=buffer,
            file_name="results.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
        )
