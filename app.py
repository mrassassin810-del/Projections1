import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import os

# --- CONFIG & UI ---
st.set_page_config(layout="wide")
CACHE_FILE = "sp500_screener_cache.csv"

# --- TAB SETUP ---
tab1, tab2, tab3 = st.tabs(["📊 Single Ticker", "🔍 Screener", "📈 Backtester"])

# --- TAB 3: ROBUST BACKTESTER ---
with tab3:
    st.subheader("📈 Strategy Backtester")
    
    # 1. Load Data
    if not os.path.exists(CACHE_FILE):
        st.error("Please run the bulk scan in the 'Screener' tab first.")
    else:
        df = pd.read_csv(CACHE_FILE)
        
        # 2. Controls
        c1, c2, c3 = st.columns(3)
        period = c1.selectbox("Horizon", ["1y", "3y", "5y"], index=2)
        inf_rate = c2.slider("Inflation Rate (%)", 0.0, 15.0, 3.5, 0.1) / 100
        hurdle = c3.slider("Real Growth Hurdle (%)", 0.0, 25.0, 0.0, 0.5) / 100
        
        # 3. Filter
        # Use .get to avoid KeyErrors if columns are missing
        nominal_cagr = df.get('Hist NI CAGR (%)', 0) / 100
        df['Real Growth'] = ((1 + nominal_cagr) / (1 + inf_rate)) - 1
        survivors = df[df['Real Growth'] >= hurdle].copy()
        
        st.metric("Stocks Meeting Criteria", len(survivors))
        
        if st.button("Run Simulation"):
            with st.spinner("Processing historical prices..."):
                tickers = survivors['Ticker'].tolist()
                # Batch download prices
                data = yf.download(tickers + ['SPY'], period=period, interval="1d", auto_adjust=True, progress=False)['Close'].ffill().bfill()
                
                # Filter valid
                valid = [t for t in tickers if t in data.columns]
                if not valid: st.error("No valid price data found."); st.stop()
                
                # Weighting: Use cached Market Cap
                weights = survivors.set_index('Ticker').loc[valid, 'Market Cap (B)'].values
                weights = weights / np.sum(weights)
                
                # Returns
                rets = data[valid].pct_change().dropna()
                strat_rets = (rets * weights).sum(axis=1)
                spy_rets = data['SPY'].pct_change().dropna()
                
                # Results
                years = len(rets) / 252
                strat_cagr = ((1 + strat_rets).prod() ** (1/years)) - 1
                spy_cagr = ((1 + spy_rets).prod() ** (1/years)) - 1
                
                col1, col2 = st.columns(2)
                col1.metric("Strategy CAGR", f"{strat_cagr*100:.2f}%")
                col2.metric("SPY CAGR", f"{spy_cagr*100:.2f}%")
                
                plot_df = pd.DataFrame({
                    "Strategy": (1 + strat_rets).cumprod() - 1,
                    "SPY": (1 + spy_rets).cumprod() - 1
                }) * 100
                st.plotly_chart(px.line(plot_df, title="Cumulative Performance (%)"), use_container_width=True)
                
                st.write("### Portfolio Breakdown")
                st.dataframe(survivors[['Ticker', 'Company Name', 'Industry', 'Market Cap (B)', 'Real Growth (%)']], use_container_width=True)
