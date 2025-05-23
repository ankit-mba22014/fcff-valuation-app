# streamlit_app.py
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
    'tqdm==4.66.1'
]

def install(package):
    subprocess.check_call([sys.executable, "-m", "pip", "install", package])

for package in REQUIRED_PACKAGES:
    try:
        pkg_name = package.split('==')[0]
        __import__(pkg_name)
    except ImportError:
        install(package)

# Initialize services with Streamlit secrets
try:
    pc = Pinecone(api_key=st.secrets['PINECONE']['API_KEY'])
    index_name = "llama-text-embed-v2"
    
    # Create Pinecone index if not exists
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1024,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            )
        )
    index = pc.Index(index_name)
    
    # Initialize Hugging Face components
    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/llama-text-embed-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True},
        huggingfacehub_api_token=st.secrets['HUGGINGFACE']['TOKEN']
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
def get_financial_data(ticker):
    try:
        company = yf.Ticker(ticker)
        return {
            'income_stmt': company.income_stmt,
            'cash_flow': company.cashflow,
            'balance_sheet': company.balance_sheet,
            'info': company.info
        }
    except Exception as e:
        st.error(f"Data fetch error: {str(e)}")
        return None

def calculate_fcff(financial_data):
    try:
        ebit = financial_data['income_stmt'].loc['EBIT']
        tax_rate = financial_data['income_stmt'].loc['Tax Rate For Calcs']
        nopat = ebit * (1 - tax_rate)
        capex = financial_data['cash_flow'].loc['Capital Expenditure']
        working_capital = financial_data['cash_flow'].loc['Change In Working Capital']
        return (nopat + capex + working_capital).dropna().tolist()
    except KeyError as e:
        st.error(f"Missing metric: {str(e)}")
        return None

# LLM Functions
def predict_growth_rates(history, years):
    template = """Analyze FCFF history and predict {years} growth rates as comma-separated percentages:
    {history}
    """
    prompt = PromptTemplate(template=template, input_variables=["history", "years"])
    try:
        prediction = llm(prompt.format(history=history, years=years))
        return [float(x.strip())/100 for x in prediction.split(",")[:years]]
    except Exception as e:
        st.error(f"Prediction failed: {str(e)}")
        return [0.02] * years

def generate_recommendations(ratios):
    prompt = f"""Analyze financial ratios and provide recommendations:
    {ratios}
    Focus on liquidity, profitability, and risk factors.
    """
    try:
        return llm(prompt)
    except Exception as e:
        return "Recommendations unavailable due to API error"

# DCF Calculation
def dcf_valuation(fcff, wacc, terminal_growth):
    try:
        terminal_year = len(fcff)
        terminal_value = (fcff[-1] * (1 + terminal_growth)) / (wacc - terminal_growth)
        pv_cashflows = sum([cf/(1 + wacc)**(i+1) for i, cf in enumerate(fcff)])
        pv_terminal = terminal_value/(1 + wacc)**terminal_year
        return pv_cashflows + pv_terminal
    except ZeroDivisionError:
        st.error("Invalid growth rate/WACC combination")
        return 0

# Visualization
def plot_sensitivity(base_value, wacc_range, growth_range):
    matrix = np.zeros((len(wacc_range), len(growth_range)))
    for i, w in enumerate(wacc_range):
        for j, g in enumerate(growth_range):
            matrix[i,j] = (base_value * (1 + g))/(w - g)
    fig = go.Figure(go.Heatmap(
        z=matrix,
        x=[f"{g*100:.1f}%" for g in growth_range],
        y=[f"{w*100:.1f}%" for w in wacc_range],
        colorscale='Viridis'
    ))
    return fig

# Pinecone Integration
def store_analysis(ticker, data):
    try:
        text = f"{data['info']['longBusinessSummary']} Sector: {data['info']['sector']}"
        embedding = embedder.embed_query(text)
        index.upsert([(ticker, embedding, {
            'ticker': ticker,
            'valuation': data['valuation'],
            'industry': data['info'].get('industry', ''),
            'date': datetime.now().isoformat()
        })])
    except Exception as e:
        st.error(f"Storage failed: {str(e)}")

# Streamlit UI
st.title("AI Financial Analyst")
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL):")

if ticker:
    data = get_financial_data(ticker)
    if data and 'info' in data:
        st.header(data['info']['longName'])
        
        # DCF Valuation
        with st.expander("DCF Valuation"):
            fcff_history = calculate_fcff(data)
            if fcff_history:
                years = st.slider("Projection Years", 3, 10, 5)
                growth_rates = predict_growth_rates(fcff_history[-5:], years)
                projected_fcff = [fcff_history[-1] * (1 + gr) for gr in growth_rates]
                
                wacc = st.number_input("WACC (%)", 5.0, 20.0, 10.0)/100
                terminal_growth = st.number_input("Terminal Growth (%)", 0.0, 5.0, 2.0)/100
                
                valuation = dcf_valuation(projected_fcff, wacc, terminal_growth)
                st.metric("Enterprise Value", f"${valuation/1e9:.2f}B")
                
                # Sensitivity Analysis
                st.plotly_chart(plot_sensitivity(
                    projected_fcff[-1],
                    np.linspace(wacc-0.02, wacc+0.02, 5),
                    np.linspace(terminal_growth-0.01, terminal_growth+0.01, 5)
                ))
                
                store_analysis(ticker, {
                    'info': data['info'],
                    'valuation': valuation
                })

        # Ratio Analysis
        with st.expander("Financial Ratios"):
            try:
                ratios = {
                    'Current Ratio': data['balance_sheet']['Total Current Assets'] / data['balance_sheet']['Total Current Liabilities'],
                    'ROE': data['income_stmt']['Net Income'] / data['balance_sheet']['Total Stockholder Equity'],
                    'Debt/Equity': data['balance_sheet']['Total Liab'] / data['balance_sheet']['Total Stockholder Equity']
                }
                st.write(pd.DataFrame([ratios]).T)
                st.write("Recommendations:", generate_recommendations(ratios))
            except Exception as e:
                st.error(f"Ratio analysis failed: {str(e)}")

# Run with: streamlit run streamlit_app.py
