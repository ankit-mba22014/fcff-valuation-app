import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from datetime import datetime, timedelta
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Configure page
st.set_page_config(
    page_title="NIFTY 50 Financial Analyst",
    page_icon="📈",
    layout="wide"
)

# NIFTY 50 stocks list with NSE suffixes
NIFTY_50_STOCKS = {
    'ADANIPORTS.NS': 'Adani Ports',
    'ASIANPAINT.NS': 'Asian Paints',
    'AXISBANK.NS': 'Axis Bank',
    'BAJAJ-AUTO.NS': 'Bajaj Auto',
    'BAJFINANCE.NS': 'Bajaj Finance',
    'BAJAJFINSV.NS': 'Bajaj Finserv',
    'BPCL.NS': 'BPCL',
    'BHARTIARTL.NS': 'Bharti Airtel',
    'BRITANNIA.NS': 'Britannia',
    'CIPLA.NS': 'Cipla',
    'COALINDIA.NS': 'Coal India',
    'DIVISLAB.NS': 'Divi\'s Labs',
    'DRREDDY.NS': 'Dr. Reddy\'s',
    'EICHERMOT.NS': 'Eicher Motors',
    'GRASIM.NS': 'Grasim',
    'HCLTECH.NS': 'HCL Tech',
    'HDFCBANK.NS': 'HDFC Bank',
    'HDFCLIFE.NS': 'HDFC Life',
    'HEROMOTOCO.NS': 'Hero MotoCorp',
    'HINDALCO.NS': 'Hindalco',
    'HINDUNILVR.NS': 'Hindustan Unilever',
    'HDFC.NS': 'HDFC Ltd',
    'ICICIBANK.NS': 'ICICI Bank',
    'ITC.NS': 'ITC',
    'INDUSINDBK.NS': 'IndusInd Bank',
    'INFY.NS': 'Infosys',
    'JSWSTEEL.NS': 'JSW Steel',
    'KOTAKBANK.NS': 'Kotak Bank',
    'LT.NS': 'L&T',
    'M&M.NS': 'M&M',
    'MARUTI.NS': 'Maruti Suzuki',
    'NTPC.NS': 'NTPC',
    'NESTLEIND.NS': 'Nestle India',
    'ONGC.NS': 'ONGC',
    'POWERGRID.NS': 'Power Grid',
    'RELIANCE.NS': 'Reliance',
    'SBILIFE.NS': 'SBI Life',
    'SBIN.NS': 'SBI',
    'SUNPHARMA.NS': 'Sun Pharma',
    'TCS.NS': 'TCS',
    'TATACONSUM.NS': 'Tata Consumer',
    'TATAMOTORS.NS': 'Tata Motors',
    'TATASTEEL.NS': 'Tata Steel',
    'TECHM.NS': 'Tech Mahindra',
    'TITAN.NS': 'Titan',
    'UPL.NS': 'UPL',
    'ULTRACEMCO.NS': 'UltraTech Cement',
    'WIPRO.NS': 'Wipro',
    'APOLLOHOSP.NS': 'Apollo Hospital',
    'SHREECEM.NS': 'Shree Cement'
}

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'financial_cache' not in st.session_state:
    st.session_state.financial_cache = {}
if 'current_company_data' not in st.session_state:
    st.session_state.current_company_data = None

def create_session_with_retry():
    """Create requests session with retry strategy"""
    session = requests.Session()
    retry_strategy = Retry(
        total=2,
        backoff_factor=2,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    adapter = HTTPAdapter(max_retries=retry_strategy)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    return session

@st.cache_data(ttl=7200)  # Cache for 2 hours
def get_minimal_financial_data(ticker):
    """Fetch only essential data for FCFF calculation - minimal API calls"""
    try:
        time.sleep(2)  # Rate limiting
        
        session = create_session_with_retry()
        company = yf.Ticker(ticker, session=session)
        
        # Only fetch what we absolutely need
        data = {}
        
        # Get basic info (single API call)
        try:
            info = company.info
            data['company_name'] = info.get('longName', NIFTY_50_STOCKS.get(ticker, ticker))
            data['sector'] = info.get('sector', 'Unknown')
        except:
            data['company_name'] = NIFTY_50_STOCKS.get(ticker, ticker)
            data['sector'] = 'Unknown'
        
        # Get only cash flow statement (single API call)
        try:
            cashflow = company.cashflow
            if not cashflow.empty:
                data['cashflow'] = cashflow
                data['has_cashflow'] = True
            else:
                data['has_cashflow'] = False
        except:
            data['has_cashflow'] = False
        
        # Get only financials for ratios (single API call)
        try:
            financials = company.financials
            if not financials.empty:
                data['financials'] = financials
                data['has_financials'] = True
            else:
                data['has_financials'] = False
        except:
            data['has_financials'] = False
            
        # Get balance sheet for ratios (single API call)
        try:
            balance_sheet = company.balance_sheet
            if not balance_sheet.empty:
                data['balance_sheet'] = balance_sheet
                data['has_balance_sheet'] = True
            else:
                data['has_balance_sheet'] = False
        except:
            data['has_balance_sheet'] = False
        
        data['ticker'] = ticker
        data['fetch_time'] = datetime.now()
        
        return data
        
    except Exception as e:
        st.error(f"Error fetching {ticker}: {str(e)}")
        return None

def calculate_fcff_minimal(data):
    """Calculate FCFF using minimal available data"""
    fcff_values = []
    
    try:
        if data.get('has_cashflow') and 'Operating Cash Flow' in data['cashflow'].index:
            ocf = data['cashflow'].loc['Operating Cash Flow']
            
            # Try to get Capital Expenditure
            if 'Capital Expenditure' in data['cashflow'].index:
                capex = data['cashflow'].loc['Capital Expenditure']
                fcff = ocf + capex  # capex is usually negative
            else:
                # Estimate capex as 8% of OCF
                fcff = ocf * 0.92
                
            fcff_values = [float(x) for x in fcff.dropna() if pd.notna(x)]
            
        elif data.get('has_financials'):
            # Fallback: estimate from net income
            if 'Net Income' in data['financials'].index:
                net_income = data['financials'].loc['Net Income']
                # Rough FCFF estimate: Net Income * 1.2 (add back non-cash items)
                fcff_values = [float(x) * 1.2 for x in net_income.dropna() if pd.notna(x)]
        
        return fcff_values[-4:] if len(fcff_values) >= 4 else fcff_values
        
    except Exception as e:
        st.warning(f"FCFF calculation issue: {str(e)}")
        return []

def simple_dcf_valuation(fcff_history, growth_rate=0.08, discount_rate=0.12, terminal_growth=0.03):
    """Simplified DCF calculation"""
    if not fcff_history or len(fcff_history) == 0:
        return 0
        
    try:
        last_fcff = abs(float(fcff_history[-1]))  # Use absolute value
        
        # Project 5 years
        pv_sum = 0
        for year in range(1, 6):
            projected_fcff = last_fcff * ((1 + growth_rate) ** year)
            present_value = projected_fcff / ((1 + discount_rate) ** year)
            pv_sum += present_value
        
        # Terminal value
        terminal_fcff = last_fcff * ((1 + growth_rate) ** 5) * (1 + terminal_growth)
        terminal_value = terminal_fcff / (discount_rate - terminal_growth)
        terminal_pv = terminal_value / ((1 + discount_rate) ** 5)
        
        enterprise_value = pv_sum + terminal_pv
        return max(enterprise_value, 0)
        
    except Exception as e:
        st.error(f"Valuation error: {str(e)}")
        return 0

def calculate_financial_ratio(data, ratio_name, fy_year, quarter=None):
    """Calculate specific financial ratio for given FY and quarter"""
    try:
        # Parse FY year (e.g., "FY23" -> 2023)
        if fy_year.upper().startswith('FY'):
            year = int('20' + fy_year[2:]) if len(fy_year) == 4 else int(fy_year[2:])
        else:
            year = int(fy_year)
        
        # Get the appropriate data columns (yearly for now, quarterly later)
        financials = data.get('financials', pd.DataFrame())
        balance_sheet = data.get('balance_sheet', pd.DataFrame())
        cashflow = data.get('cashflow', pd.DataFrame())
        
        if financials.empty and balance_sheet.empty:
            return f"No financial data available for {data.get('company_name', 'company')}"
        
        # Get the closest year column
        available_years = []
        for col in financials.columns:
            if hasattr(col, 'year'):
                available_years.append(col.year)
        
        if not available_years:
            return "No yearly financial data found"
        
        # Find closest year
        closest_year = min(available_years, key=lambda x: abs(x - year))
        target_col = None
        for col in financials.columns:
            if hasattr(col, 'year') and col.year == closest_year:
                target_col = col
                break
        
        if target_col is None:
            return f"No data found for FY{year%100:02d}"
        
        # Calculate ratios based on ratio_name
        ratio_lower = ratio_name.lower()
        
        # Profitability Ratios
        if 'roe' in ratio_lower or 'return on equity' in ratio_lower:
            net_income = get_value_safe(financials, 'Net Income', target_col)
            equity = get_value_safe(balance_sheet, 'Stockholders Equity', target_col)
            if net_income and equity:
                roe = (net_income / equity) * 100
                return f"ROE for FY{year%100:02d}: {roe:.2f}%"
        
        elif 'roa' in ratio_lower or 'return on assets' in ratio_lower:
            net_income = get_value_safe(financials, 'Net Income', target_col)
            total_assets = get_value_safe(balance_sheet, 'Total Assets', target_col)
            if net_income and total_assets:
                roa = (net_income / total_assets) * 100
                return f"ROA for FY{year%100:02d}: {roa:.2f}%"
        
        elif 'net margin' in ratio_lower or 'profit margin' in ratio_lower:
            net_income = get_value_safe(financials, 'Net Income', target_col)
            revenue = get_value_safe(financials, 'Total Revenue', target_col)
            if net_income and revenue:
                margin = (net_income / revenue) * 100
                return f"Net Profit Margin for FY{year%100:02d}: {margin:.2f}%"
        
        elif 'gross margin' in ratio_lower:
            revenue = get_value_safe(financials, 'Total Revenue', target_col)
            cogs = get_value_safe(financials, 'Cost Of Revenue', target_col)
            if revenue and cogs:
                gross_margin = ((revenue - cogs) / revenue) * 100
                return f"Gross Margin for FY{year%100:02d}: {gross_margin:.2f}%"
        
        # Liquidity Ratios
        elif 'current ratio' in ratio_lower:
            current_assets = get_value_safe(balance_sheet, 'Current Assets', target_col)
            current_liab = get_value_safe(balance_sheet, 'Current Liabilities', target_col)
            if current_assets and current_liab:
                ratio = current_assets / current_liab
                return f"Current Ratio for FY{year%100:02d}: {ratio:.2f}"
        
        elif 'quick ratio' in ratio_lower or 'acid test' in ratio_lower:
            current_assets = get_value_safe(balance_sheet, 'Current Assets', target_col)
            inventory = get_value_safe(balance_sheet, 'Inventory', target_col) or 0
            current_liab = get_value_safe(balance_sheet, 'Current Liabilities', target_col)
            if current_assets and current_liab:
                quick_ratio = (current_assets - inventory) / current_liab
                return f"Quick Ratio for FY{year%100:02d}: {quick_ratio:.2f}"
        
        # Leverage Ratios
        elif 'debt to equity' in ratio_lower or 'debt equity' in ratio_lower:
            total_debt = get_value_safe(balance_sheet, 'Total Debt', target_col)
            equity = get_value_safe(balance_sheet, 'Stockholders Equity', target_col)
            if total_debt and equity:
                de_ratio = total_debt / equity
                return f"Debt-to-Equity Ratio for FY{year%100:02d}: {de_ratio:.2f}"
        
        elif 'debt ratio' in ratio_lower:
            total_debt = get_value_safe(balance_sheet, 'Total Debt', target_col)
            total_assets = get_value_safe(balance_sheet, 'Total Assets', target_col)
            if total_debt and total_assets:
                debt_ratio = total_debt / total_assets
                return f"Debt Ratio for FY{year%100:02d}: {debt_ratio:.2f}"
        
        # Efficiency Ratios
        elif 'asset turnover' in ratio_lower:
            revenue = get_value_safe(financials, 'Total Revenue', target_col)
            total_assets = get_value_safe(balance_sheet, 'Total Assets', target_col)
            if revenue and total_assets:
                turnover = revenue / total_assets
                return f"Asset Turnover for FY{year%100:02d}: {turnover:.2f}"
        
        else:
            return f"I can calculate these ratios for FY{year%100:02d}: ROE, ROA, Net Margin, Gross Margin, Current Ratio, Quick Ratio, Debt-to-Equity, Debt Ratio, Asset Turnover. Please specify which one you'd like."
    
    except Exception as e:
        return f"Error calculating {ratio_name}: {str(e)}"

def get_value_safe(df, metric_name, column):
    """Safely get value from dataframe"""
    try:
        if metric_name in df.index:
            value = df.loc[metric_name, column]
            return float(value) if pd.notna(value) else None
        return None
    except:
        return None

def process_chat_query(query, company_data):
    """Process financial ratio queries"""
    if not company_data:
        return "Please select and analyze a company first before asking about financial ratios."
    
    query_lower = query.lower()
    
    # Extract FY year and quarter
    fy_year = None
    quarter = None
    
    # Look for FY patterns
    import re
    fy_pattern = r'fy\s*(\d{2,4})'
    fy_match = re.search(fy_pattern, query_lower)
    if fy_match:
        fy_year = f"FY{fy_match.group(1)}"
    
    # Look for quarter patterns
    quarter_pattern = r'q([1-4])'
    quarter_match = re.search(quarter_pattern, query_lower)
    if quarter_match:
        quarter = f"Q{quarter_match.group(1)}"
    
    if not fy_year:
        return "Please specify the Financial Year (e.g., FY23, FY24) in your question. For example: 'What is the ROE for FY23?'"
    
    # Extract ratio name from query
    ratio_keywords = {
        'roe': 'Return on Equity',
        'roa': 'Return on Assets', 
        'net margin': 'Net Profit Margin',
        'profit margin': 'Net Profit Margin',
        'gross margin': 'Gross Margin',
        'current ratio': 'Current Ratio',
        'quick ratio': 'Quick Ratio',
        'debt to equity': 'Debt-to-Equity Ratio',
        'debt equity': 'Debt-to-Equity Ratio',
        'debt ratio': 'Debt Ratio',
        'asset turnover': 'Asset Turnover'
    }
    
    ratio_name = None
    for keyword, full_name in ratio_keywords.items():
        if keyword in query_lower:
            ratio_name = full_name
            break
    
    if not ratio_name:
        available_ratios = list(ratio_keywords.values())
        return f"Please specify which financial ratio you want for {fy_year}. Available ratios: {', '.join(available_ratios)}"
    
    # Calculate the ratio
    result = calculate_financial_ratio(company_data, ratio_name, fy_year, quarter)
    return result

# Main Streamlit App
def main():
    st.title("🏦 NIFTY 50 Financial Analyst")
    st.markdown("*FCFF Analysis & Financial Ratio Chat Assistant*")
    
    # Create tabs
    analysis_tab, chat_tab = st.tabs(["📊 FCFF Analysis", "🧮 Financial Ratio Chat"])
    
    with analysis_tab:
        st.header("Free Cash Flow Analysis")
        
        # Stock selection
        col1, col2 = st.columns([3, 1])
        with col1:
            selected_stock = st.selectbox(
                "Select NIFTY 50 Stock:",
                options=list(NIFTY_50_STOCKS.keys()),
                format_func=lambda x: f"{NIFTY_50_STOCKS[x]} ({x})",
                key="stock_selector"
            )
        
        with col2:
            if st.button("📈 Analyze", type="primary"):
                with st.spinner("Fetching minimal data..."):
                    financial_data = get_minimal_financial_data(selected_stock)
                    if financial_data:
                        st.session_state.financial_cache[selected_stock] = financial_data
                        st.session_state.current_company_data = financial_data
                        st.success("✅ Data loaded!")
                    else:
                        st.error("❌ Failed to fetch data")
        
        # Display FCFF analysis
        if selected_stock in st.session_state.financial_cache:
            data = st.session_state.financial_cache[selected_stock]
            st.session_state.current_company_data = data
            
            st.subheader(f"📊 {data['company_name']}")
            st.write(f"**Sector:** {data['sector']} | **Ticker:** {data['ticker']}")
            
            # Calculate FCFF
            fcff_history = calculate_fcff_minimal(data)
            
            if fcff_history:
                # Display FCFF chart
                years = [f"Year {i+1}" for i in range(len(fcff_history))]
                fig = px.bar(
                    x=years,
                    y=[x/1e6 for x in fcff_history],  # Convert to millions
                    title="Free Cash Flow to Firm (₹ Millions)",
                    labels={'y': 'FCFF (₹ Millions)', 'x': 'Period'}
                )
                fig.update_traces(marker_color='green')
                st.plotly_chart(fig, use_container_width=True)
                
                # DCF Valuation
                st.subheader("🎯 DCF Valuation")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    growth_rate = st.slider("Growth Rate (%)", 0, 25, 10) / 100
                with col2:
                    discount_rate = st.slider("Discount Rate (%)", 8, 20, 12) / 100
                with col3:
                    terminal_growth = st.slider("Terminal Growth (%)", 1, 6, 3) / 100
                
                enterprise_value = simple_dcf_valuation(fcff_history, growth_rate, discount_rate, terminal_growth)
                
                st.metric(
                    "Enterprise Value", 
                    f"₹{enterprise_value/1e9:.2f}B",
                    delta=f"Based on {len(fcff_history)} years FCFF"
                )
                
            else:
                st.warning("Unable to calculate FCFF with available data")
    
    with chat_tab:
        st.header("🧮 Financial Ratio Assistant")
        st.markdown("*Ask about financial ratios for specific FY years*")
        
        # Show current company context
        if st.session_state.current_company_data:
            company_name = st.session_state.current_company_data.get('company_name', 'Unknown')
            st.info(f"📊 **Current Company:** {company_name}")
        else:
            st.warning("⚠️ Please select and analyze a company first in the FCFF Analysis tab")
        
        # Chat interface
        for i, (role, message) in enumerate(st.session_state.chat_history):
            if role == "user":
                st.chat_message("user").write(message)
            else:
                st.chat_message("assistant").write(message)
        
        # Chat input
        user_query = st.chat_input("Ask about financial ratios (e.g., 'What is ROE for FY23?')")
        
        if user_query:
            # Add user message
            st.session_state.chat_history.append(("user", user_query))
            
            # Process query
            response = process_chat_query(user_query, st.session_state.current_company_data)
            
            # Add assistant response
            st.session_state.chat_history.append(("assistant", response))
            
            st.rerun()
        
        # Example queries
        with st.expander("💡 Example Questions"):
            st.markdown("""
            **Financial Ratio Questions:**
            - What is the ROE for FY23?
            - Show me current ratio for FY24
            - What's the debt to equity ratio for FY22?
            - Calculate gross margin for FY23
            - What is asset turnover for FY24?
            
            **Available Ratios:**
            - ROE, ROA, Net Margin, Gross Margin
            - Current Ratio, Quick Ratio  
            - Debt-to-Equity, Debt Ratio, Asset Turnover
            """)
        
        # Chat controls
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    # Footer
    st.markdown("---")
    st.markdown(
        """<div style='text-align: center; color: #666;'>
        <p>🏦 NIFTY 50 Financial Analyst | Optimized for Minimal API Usage</p>
        </div>""", 
        unsafe_allow_html=True
    )

if __name__ == "__main__":
    main()
