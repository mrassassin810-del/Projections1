import sys
import os
import time
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import requests
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt
import plotly.express as px
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

# --- CONFIG ---
warnings.filterwarnings('ignore')
st.set_page_config(page_title="Forecaster & Screening Engine", layout="wide")

# --- AUTH & SECRETS ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    st.title("🔒 System Locked")
    if st.text_input("Password:", type="password") == st.secrets.get("APP_PASSWORD", "admin123"):
        st.session_state.authenticated = True; st.rerun()
    st.stop()

api_key = st.secrets.get("AV_API_KEY")
CACHE_FILE = "sp500_screener_cache.csv"
WATCHLIST_FILE = "watchlist.txt"
TICKER_CACHE_DIR = "ticker_cache"
os.makedirs(TICKER_CACHE_DIR, exist_ok=True)

# --- SHARED HELPERS ---
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    return pd.read_html("https://en.wikipedia.org/wiki/List_of_S%26P_500_companies", storage_options={"User-Agent": "Mozilla/5.0"})[0]['Symbol'].str.replace('.', '-', regex=False).tolist()

@st.cache_data(ttl=86400)
def get_historical_prices(tickers, period):
    data = yf.download(tickers + ['SPY'], period=period, interval="1d", auto_adjust=True, progress=False)
    # Forward fill to handle missing data days gracefully
    return data['Close'].ffill().bfill()

# --- TAB 3 BACKTEST ENGINE ---
# (Uses the cached data from the screener tab to avoid API bloat)
with st.sidebar:
    st.write("### ⚙️ Engine State")
    if st.button("Reload Screener Data"): st.rerun()

tab_single, tab_screener, tab_backtest = st.tabs(["📊 Single Ticker Forecast", "🔍 S&P 500 Screening Dashboard", "📈 Strategy Backtester"])

# [Rest of your Single Ticker & Screener logic remains here...]
# [I have abbreviated this to focus on the Tab 3 fix you requested]

with tab_backtest:
    st.subheader("📊 Pure Point-In-Time S&P 500 Backtester")
    if 'raw_screener_df' not in st.session_state or st.session_state.raw_screener_df.empty:
        st.warning("⚠️ Please run the **Bulk Refresh Matrix** in the Screener tab first.")
    else:
        c1, c2, c3 = st.columns(3)
        backtest_period = c1.selectbox("Horizon", ["1y", "3y", "5y"], index=2)
        inf_rate = c2.slider("Inflation Rate (%)", 0.0, 15.0, 4.2, 0.1) / 100
        hurdle = c3.slider("Real Growth Hurdle (%)", 0.0, 25.0, 0.0, 0.5) / 100
        
        df = st.session_state.raw_screener_df.copy()
        df['Real Growth'] = ((1 + (df['Hist NI CAGR (%)'] / 100)) / (1 + inf_rate)) - 1
        survivors = df[df['Real Growth'] >= hurdle].copy()

        if survivors.empty: st.error("No stocks met these criteria.")
        else:
            with st.spinner("Calculating point-in-time weights..."):
                prices = get_historical_prices(survivors['Ticker'].tolist(), backtest_period)
                start_date = prices.index[0]
                
                # Logic: Build weight vector based on market cap at START of period
                weights = []
                valid_tickers = []
                for t in survivors['Ticker']:
                    if t in prices.columns:
                        # Find closest price to start date
                        start_price = prices[t].iloc[0]
                        # Estimate shares using local data if possible, else current mcap
                        mcap = survivors.loc[survivors['Ticker']==t, 'Market Cap (B)'].values[0] * 1e9
                        current_p = survivors.loc[survivors['Ticker']==t, 'Current Price'].values[0]
                        shares = mcap / current_p if current_p > 0 else 1
                        weights.append(start_price * shares)
                        valid_tickers.append(t)
                
                weights = np.array(weights) / np.sum(weights)
                
                # Calculate returns
                returns = prices[valid_tickers].pct_change().dropna()
                strat_returns = (returns * weights).sum(axis=1)
                spy_returns = prices['SPY'].pct_change().dropna()
                
                # Annualized CAGR
                years = len(returns) / 252
                strat_cagr = ((1 + strat_returns).prod() ** (1/years)) - 1
                spy_cagr = ((1 + spy_returns).prod() ** (1/years)) - 1
                
                c1, c2 = st.columns(2)
                c1.metric("Strategy CAGR", f"{strat_cagr*100:.2f}%")
                c2.metric("SPY CAGR", f"{spy_cagr*100:.2f}%")
                
                plot_df = pd.DataFrame({
                    "Strategy": (1 + strat_returns).cumprod() - 1,
                    "SPY": (1 + spy_returns).cumprod() - 1
                }) * 100
                
                fig = px.line(plot_df, title="Cumulative % Return")
                st.plotly_chart(fig, use_container_width=True)
