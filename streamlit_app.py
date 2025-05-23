# streamlit_app.py
import streamlit as st
import yfinance as yf
import numpy as np
from datetime import datetime

# Hardcoded Nifty 50 tickers (you can keep this updated manually or fetch dynamically if needed)
NIFTY_50_TICKERS = {
    "RELIANCE": "RELIANCE.NS",
    "TCS": "TCS.NS",
    "INFY": "INFY.NS",
    "HDFCBANK": "HDFCBANK.NS",
    "ICICIBANK": "ICICIBANK.NS",
    "HINDUNILVR": "HINDUNILVR.NS",
    "ITC": "ITC.NS",
    "KOTAKBANK": "KOTAKBANK.NS",
    "SBIN": "SBIN.NS",
    "LT": "LT.NS",
    "AXISBANK": "AXISBANK.NS",
    "ASIANPAINT": "ASIANPAINT.NS",
    "MARUTI": "MARUTI.NS",
    "BAJFINANCE": "BAJFINANCE.NS",
    "BHARTIARTL": "BHARTIARTL.NS",
    "SUNPHARMA": "SUNPHARMA.NS",
    "ULTRACEMCO": "ULTRACEMCO.NS",
    "WIPRO": "WIPRO.NS",
    "TECHM": "TECHM.NS",
    "TITAN": "TITAN.NS"
    # Add more as needed
}

# Discount cash flow calculation
def calculate_dcf(fcff_values, discount_rate, terminal_growth_rate, terminal_year):
    present_value_fcff = 0
    for year, fcff in enumerate(fcff_values, 1):
        present_value_fcff += fcff / ((1 + discount_rate) ** year)

    terminal_value = fcff_values[-1] * (1 + terminal_growth_rate) / (discount_rate - terminal_growth_rate)
    present_value_terminal = terminal_value / ((1 + discount_rate) ** terminal_year)

    total_value = present_value_fcff + present_value_terminal
    return total_value, present_value_fcff, present_value_terminal

# App Title
st.title("💸 FCFF Valuation App (DCF Model)")

# Inputs
st.sidebar.header("Input Parameters")

# Ticker suggestion box
search_input = st.sidebar.text_input("Enter company name or ticker (min 2 letters)", value="")
filtered_tickers = {name: symbol for name, symbol in NIFTY_50_TICKERS.items() if search_input.upper() in name}

if filtered_tickers:
    selected_name = st.sidebar.selectbox("Select Nifty 50 Company", list(filtered_tickers.keys()))
    ticker = filtered_tickers[selected_name]
else:
    st.sidebar.warning("Please enter at least 2 letters to search Nifty 50 stocks.")
    ticker = None

year = st.sidebar.number_input("Starting Year", min_value=2000, max_value=datetime.now().year, value=2024)
quarter = st.sidebar.selectbox("Quarter", ["Q1", "Q2", "Q3", "Q4"])
projection_years = st.sidebar.slider("Years of FCFF Projection", min_value=3, max_value=10, value=5)
discount_rate = st.sidebar.number_input("Discount Rate (WACC, %)", min_value=0.0, max_value=100.0, value=10.0) / 100
terminal_growth_rate = st.sidebar.number_input("Terminal Growth Rate (%)", min_value=0.0, max_value=10.0, value=2.0) / 100

# FCFF Inputs
st.subheader("Projected Free Cash Flow to Firm (FCFF)")
fcff_values = []
for i in range(1, projection_years + 1):
    fcff = st.number_input(f"Year {i} FCFF ($M)", min_value=0.0, value=100.0)
    fcff_values.append(fcff)

# Calculate button
if st.button("Calculate Valuation") and ticker:
    total_value, pv_fcff, pv_terminal = calculate_dcf(
        fcff_values, discount_rate, terminal_growth_rate, projection_years
    )
    st.success(f"Estimated Enterprise Value: ${total_value:,.2f}M")
    st.metric("Present Value of FCFF", f"${pv_fcff:,.2f}M")
    st.metric("Present Value of Terminal Value", f"${pv_terminal:,.2f}M")

    st.markdown("---")
    st.subheader("🧠 Model Summary")
    st.markdown(f"- **Discount Rate (WACC)**: {discount_rate * 100:.1f}%")
    st.markdown(f"- **Terminal Growth Rate**: {terminal_growth_rate * 100:.1f}%")
    st.markdown(f"- **Projection Period**: {projection_years} years")
    st.markdown(f"- **Terminal Value Year**: Year {projection_years}")
    st.markdown(f"- **Stock Ticker**: {ticker.upper()} for {quarter} {year}")
elif st.button("Calculate Valuation") and not ticker:
    st.error("Please select a valid Nifty 50 company to continue.")
