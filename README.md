# 💸 FCFF Valuation App (DCF Model)

This Streamlit application allows you to estimate the enterprise value of a company using the Discounted Cash Flow (DCF) method based on Free Cash Flow to Firm (FCFF). Only Nifty 50 companies are supported.

## 📊 Features
- Search Nifty 50 tickers by partial input
- Input FCFF projections
- Set WACC and terminal growth rate
- Calculates present value of FCFF and terminal value
- Summary metrics and breakdown

## 🛠 Tech Stack
- Python
- Streamlit
- yFinance
- NumPy

## 🚀 Run Locally

```bash
git clone https://github.com/yourusername/fcff-valuation-app.git
cd fcff-valuation-app
pip install -r requirements.txt
streamlit run streamlit_app.py
