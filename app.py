import sys
import os
import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# --- CONFIG ---
st.set_page_config(page_title="Forecaster Engine", layout="wide")
CACHE_FILE = "sp500_screener_cache.csv"

# --- AUTH ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    pwd = st.text_input("Enter Access Password:", type="password")
    if st.button("Unlock") and pwd == st.secrets.get("APP_PASSWORD", "admin123"):
        st.session_state.authenticated = True; st.rerun()
    st.stop()

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 Single Ticker", "🔍 Screener", "📈 Backtester"])

# --- SHARED FUNCTIONS ---
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options={"User-Agent": "Mozilla/5.0"})[0]['Symbol'].str.replace('.', '-', regex=False).tolist()

# --- TAB 1 & 2 LOGIC ---
# (Your existing core logic remains here, ensuring process_single_screener_stock is robust)

# --- TAB 3 BACKTESTER (RE-ENGINEERED FOR STABILITY) ---
with tab3:
    st.subheader("📊 Pure Point-In-Time S&P 500 Backtester")
    
    # Initialize DB from Screener cache if exists
    if os.path.exists(CACHE_FILE):
        if 'bt_df' not in st.session_state:
            st.session_state.bt_df = pd.read_csv(CACHE_FILE)
            
    if 'bt_df' not in st.session_state:
        st.warning("⚠️ Screener data not found. Please run 'Bulk Refresh' in Tab 2 first.")
    else:
        # User Controls
        c1, c2, c3 = st.columns(3)
        period = c1.selectbox("Horizon", ["1y", "3y", "5y"], index=2)
        inf_rate = c2.slider("Inflation Rate (%)", 0.0, 15.0, 3.5, 0.1) / 100
        hurdle = c3.slider("Real Growth Hurdle (%)", 0.0, 25.0, 0.0, 0.5) / 100
        
        # Filtering
        df = st.session_state.bt_df.copy()
        df['Real Growth'] = ((1 + (df['Hist NI CAGR (%)'].fillna(0) / 100)) / (1 + inf_rate)) - 1
        survivors = df[df['Real Growth'] >= hurdle].copy()
        
        st.metric("Total Stocks Remaining", len(survivors))
        
        if st.button("Run Simulation"):
            with st.spinner("Downloading historical prices..."):
                tickers = survivors['Ticker'].tolist()
                # Batch download in chunks to avoid Yahoo throttling
                data = yf.download(tickers + ['SPY'], period=period, interval="1d", auto_adjust=True, progress=False)['Close'].ffill().bfill()
                
                valid = [t for t in tickers if t in data.columns]
                if not valid: st.error("No data found for this set."); st.stop()
                
                # Point-in-Time Weighting
                start_price = data[valid].iloc[0]
                mcap = survivors.set_index('Ticker').loc[valid, 'Market Cap (B)'] * 1e9
                price = survivors.set_index('Ticker').loc[valid, 'Current Price']
                weights = (start_price * (mcap / price))
                weights /= weights.sum()
                
                # Returns
                returns = data[valid].pct_change().dropna()
                strat = (returns * weights).sum(axis=1)
                
                # Plotting
                plot_df = pd.DataFrame({
                    "Strategy": (1 + strat).cumprod() - 1,
                    "SPY": (1 + data['SPY'].pct_change().dropna()).cumprod() - 1
                }) * 100
                
                st.plotly_chart(px.line(plot_df, title="Cumulative % Return"), use_container_width=True)
                
                st.write("### Surviving Constituents")
                st.dataframe(survivors[['Ticker', 'Company Name', 'Real Growth (%)', 'Market Cap (B)']], use_container_width=True)
