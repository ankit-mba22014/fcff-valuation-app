# AI Financial Analyst

![Demo](https://via.placeholder.com/800x400.png?text=Financial+Valuation+App+Demo)

An intelligent web app for automated DCF valuations and financial analysis powered by AI.

## Features
- **DCF Valuation**: FCFF-based discounted cash flow analysis
- **AI Projections**: LLM-powered growth rate predictions
- **Sensitivity Analysis**: Interactive heatmaps for scenario testing
- **PDF Reports**: Exportable valuation summaries
- **Vector Database**: Pinecone-powered company comparisons

## Tech Stack
- **Frontend**: Streamlit
- **AI/ML**: Hugging Face, LangChain
- **Database**: Pinecone
- **Data**: Yahoo Finance

## Setup

### 1. Prerequisites
- [Streamlit Account](https://share.streamlit.io/)
- [Pinecone Account](https://www.pinecone.io/)
- [Hugging Face Account](https://huggingface.co/)

### 2. Configuration

#### Pinecone
1. Create index:
   - **Name**: `financial-embeddings`
   - **Dimension**: `384`
   - **Metric**: `cosine`
   - **Cloud**: AWS
   - **Region**: `us-east-1`
2. Get API key from dashboard

#### Hugging Face
1. Create access token with `read` permissions
2. Accept model terms for:
   - [BAAI/bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
   - [flan-t5-xxl](https://huggingface.co/google/flan-t5-xxl)

### 3. Streamlit Secrets
Add these in Streamlit Cloud settings:
```toml
[pinecone]
API_KEY = "your-pinecone-key"

[huggingface]
TOKEN = "your-hf-token"
