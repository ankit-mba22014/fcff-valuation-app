import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import re
import subprocess
import sys
import base64
from datetime import datetime
from fpdf import FPDF
from langchain.chains import LLMChain
from langchain.llms import HuggingFaceHub
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from pinecone import Pinecone, ServerlessSpec
import os

# Auto-install dependencies
REQUIRED_PACKAGES = [
    'streamlit==1.28.0',
    'yfinance==0.2.28',
    'pandas==2.1.1',
    'numpy==1.26.0',
    'plotly==5.17.0',
    'pinecone-client==3.0.3',
    'langchain==0.0.346',
    'sentence-transformers==2.2.2',
    'scikit-learn==1.3.1',
    'fpdf2==2.7.5',
    'python-dotenv==1.0.0',
    'huggingface_hub==0.17.3',
    'tqdm==4.66.1',
    'torch==2.0.1',
    'transformers==4.32.0'
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in REQUIRED_PACKAGES:
    try:
        pkg_name = package.split('==')[0]
        __import__(pkg_name)
    except ImportError:
        install(package)

# Initialize services
try:
    # Pinecone setup
    pc = Pinecone(api_key=st.secrets['PINECONE']['API_KEY'])
    index_name = "financial-embeddings"
    dimension = 384
    
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=dimension,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)

    # Hugging Face setup
    embedder = HuggingFaceEmbeddings(
        model_name="BAAI/bge-small-en-v1.5",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    llm = HuggingFaceHub(
        repo_id="google/flan-t5-xxl",
        model_kwargs={"temperature":0.1, "max_length":256},
        huggingfacehub_api_token=st.secrets['HUGGINGFACE']['TOKEN']
    )

except Exception as e:
    st.error(f"Initialization failed: {str(e)}")
    st.stop()

# Financial Analysis Functions
@st.cache_data(ttl=3600)  # Cache for 1 hour (3600 seconds)
def get_financial_data(ticker):
    try:
        company = yf.Ticker(ticker)
        info = company.info
        financials = {
            'income_stmt': company.income_stmt,
            'cash_flow': company.cashflow,
            'balance_sheet': company.balance_sheet,
            'info': info
        }
        return financials
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None
        
        # Check required metrics
        required = ['EBIT', 'Tax Rate For Calcs', 
                   'Capital Expenditure', 'Change In Working Capital']
        for metric in required:
            if metric not in financials['income_stmt'].index and metric not in financials['cash_flow'].index:
                raise ValueError(f"Missing {metric}")
                
        return financials
    except Exception as e:
        st.error(f"Error fetching data: {str(e)}")
        return None

def calculate_historical_fcff(financial_data):
    """Calculate historical Free Cash Flow to Firm"""
    try:
        ebit = financial_data['income_stmt'].loc['EBIT']
        tax_rate = financial_data['income_stmt'].loc['Tax Rate For Calcs']
        capex = financial_data['cash_flow'].loc['Capital Expenditure']
        working_capital = financial_data['cash_flow'].loc['Change In Working Capital']
        
        # FCFF = EBIT(1-Tax) + D&A - Capex - ΔWorkingCapital
        nopat = ebit * (1 - tax_rate)
        fcff = nopat + capex + working_capital
        return fcff.dropna().tolist()
    except KeyError as e:
        st.error(f"Missing financial metric: {str(e)}")
        return None

def predict_growth_rates(history, years):
    """Use LLM to predict FCFF growth rates"""
    prompt_template = """Analyze the following FCFF history and predict {years} future growth rates as comma-separated percentages:
    {history}
    Return only numbers separated by commas, no commentary."""
    
    try:
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["history", "years"]
        )
        chain = LLMChain(llm=llm, prompt=prompt)
        prediction = chain.run(history=history, years=years)
        growth_rates = [float(x.strip())/100 for x in prediction.split(",")]
        return growth_rates[:years]
    except Exception as e:
        st.error(f"Prediction error: {str(e)}")
        return [0.02] * years  # Fallback

def dcf_valuation(fcff_projections, wacc, terminal_growth):
    """Calculate enterprise value using DCF"""
    try:
        terminal_year = len(fcff_projections)
        present_values = []
        
        # Discount projected cash flows
        for i, fcff in enumerate(fcff_projections):
            present_values.append(fcff / ((1 + wacc) ** (i + 1)))
        
        # Calculate terminal value
        terminal_value = (fcff_projections[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        present_terminal = terminal_value / ((1 + wacc) ** terminal_year)
        
        return sum(present_values) + present_terminal
    except ZeroDivisionError:
        st.error("Invalid WACC/Terminal Growth combination")
        return 0

def create_sensitivity_analysis(base_fcff, wacc_range, growth_range):
    """Generate sensitivity matrix"""
    sens_matrix = np.zeros((len(wacc_range), len(growth_range)))
    
    for i, wacc in enumerate(wacc_range):
        for j, growth in enumerate(growth_range):
            terminal_value = (base_fcff * (1 + growth)) / (wacc - growth)
            sens_matrix[i,j] = terminal_value
            
    return sens_matrix

def generate_pdf_report(ticker, info, valuation, assumptions):
    """Create PDF report"""
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # Header
    pdf.set_font('Arial', 'B', 16)
    pdf.cell(200, 10, txt=f"Valuation Report: {ticker}", ln=1, align='C')
    
    # Company Info
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Company Overview", ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.multi_cell(0, 10, txt=f"{info.get('longBusinessSummary', '')}")
    
    # Valuation
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Valuation Summary", ln=1)
    pdf.set_font('Arial', '', 12)
    pdf.cell(0, 10, txt=f"Enterprise Value: ${valuation/1e9:.2f}B", ln=1)
    
    # Assumptions
    pdf.ln(10)
    pdf.set_font('Arial', 'B', 14)
    pdf.cell(200, 10, txt="Key Assumptions", ln=1)
    for key, value in assumptions.items():
        pdf.cell(0, 10, txt=f"{key}: {value}", ln=1)
    
    # Footer
    pdf.ln(10)
    pdf.set_font('Arial', 'I', 10)
    pdf.cell(0, 10, txt=f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", align='C')
    
    return pdf.output(dest='S').encode('latin1')

# Streamlit UI
st.title("AI Financial Analyst")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):", value="AAPL")

if ticker:
    with st.spinner("Fetching financial data..."):
        financial_data = get_financial_data(ticker)
        
    if financial_data:
        info = financial_data['info']
        st.header(info.get('longName', ticker))
        
        # DCF Valuation Section
        st.subheader("DCF Valuation")
        historical_fcff = calculate_historical_fcff(financial_data)
        
        if historical_fcff:
            col1, col2 = st.columns(2)
            with col1:
                # Historical Performance
                st.plotly_chart(px.line(
                    x=range(len(historical_fcff)),
                    y=historical_fcff,
                    title="Historical FCFF",
                    labels={'x': 'Years', 'y': 'FCFF'}
                ))
            
            with col2:
                # User Inputs
                projection_years = st.slider("Projection Period (years)", 3, 10, 5)
                wacc = st.number_input("WACC (%)", 5.0, 20.0, 10.0) / 100
                terminal_growth = st.number_input("Terminal Growth (%)", 0.0, 5.0, 2.0) / 100
            
            if st.button("Calculate Valuation"):
                with st.spinner("Generating projections..."):
                    # Predict growth rates
                    history_str = ", ".join([f"${x/1e6:.1f}M" for x in historical_fcff[-3:]])
                    growth_rates = predict_growth_rates(history_str, projection_years)
                    
                    # Project FCFF
                    last_fcff = historical_fcff[-1]
                    projected_fcff = [last_fcff]
                    for rate in growth_rates:
                        projected_fcff.append(projected_fcff[-1] * (1 + rate))
                    projected_fcff = projected_fcff[1:]  # Remove initial value
                    
                    # Calculate valuation
                    enterprise_value = dcf_valuation(projected_fcff, wacc, terminal_growth)
                    
                    # Display results
                    st.success(f"Estimated Enterprise Value: ${enterprise_value/1e9:.2f}B")
                    
                    # Sensitivity Analysis
                    st.subheader("Sensitivity Analysis")
                    wacc_range = np.linspace(wacc-0.02, wacc+0.02, 5)
                    growth_range = np.linspace(terminal_growth-0.01, terminal_growth+0.01, 5)
                    sens_matrix = create_sensitivity_analysis(projected_fcff[-1], wacc_range, growth_range)
                    
                    fig = go.Figure(data=go.Heatmap(
                        z=sens_matrix,
                        x=[f"{g*100:.1f}%" for g in growth_range],
                        y=[f"{w*100:.1f}%" for w in wacc_range],
                        colorscale='Viridis'
                    ))
                    fig.update_layout(title="Terminal Value Sensitivity")
                    st.plotly_chart(fig)
                    
                    # Store in Pinecone
                    try:
                        financial_text = f"{info.get('longBusinessSummary', '')} Sector: {info.get('sector', '')}"
                        embedding = embedder.embed_query(financial_text)
                        index.upsert([(ticker, embedding, {
                            'ticker': ticker,
                            'valuation': enterprise_value,
                            'sector': info.get('sector', ''),
                            'date': datetime.now().isoformat()
                        })])
                        st.success("Analysis stored in Pinecone!")
                    except Exception as e:
                        st.error(f"Storage failed: {str(e)}")
                    
                    # PDF Export
                    assumptions = {
                        'WACC': f"{wacc*100:.1f}%",
                        'Terminal Growth': f"{terminal_growth*100:.1f}%",
                        'Projection Years': projection_years
                    }
                    pdf_report = generate_pdf_report(ticker, info, enterprise_value, assumptions)
                    b64 = base64.b64encode(pdf_report).decode()
                    href = f'<a href="data:application/pdf;base64,{b64}" download="{ticker}_valuation.pdf">Download PDF Report</a>'
                    st.markdown(href, unsafe_allow_html=True)

# System Health Check
with st.expander("System Status"):
    try:
        test_embed = embedder.embed_query("test")
        st.success(f"Embedding Model: Active (Dimension: {len(test_embed)})")
        stats = index.describe_index_stats()
        st.success(f"Pinecone: Connected (Vectors: {stats['total_vector_count']})")
    except Exception as e:
        st.error(f"System check failed: {str(e)}")

st.write("---")
st.caption("AI Financial Analyst v1.0 | Built with Streamlit, Pinecone, and Hugging Face")
