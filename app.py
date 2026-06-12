import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import yfinance as yf
import os

st.set_page_config(layout="wide")
CACHE_FILE = "sp500_screener_cache.csv"

# --- AUTH ---
if "authenticated" not in st.session_state: st.session_state.authenticated = False
if not st.session_state.authenticated:
    pwd = st.text_input("Password:", type="password")
    if st.button("Unlock") and pwd == st.secrets.get("APP_PASSWORD", "admin123"):
        st.session_state.authenticated = True; st.rerun()
    st.stop()

# --- BACKTESTER TAB ---
st.subheader("📈 Strategy Backtester")

if not os.path.exists(CACHE_FILE):
    st.warning("Please run a full scan in your Screener tab first.")
else:
    df = pd.read_csv(CACHE_FILE)
    
    # Ensure necessary columns exist for backtesting
    required_cols = ['Ticker', 'Hist NI CAGR (%)', 'Market Cap (B)', 'Current Price', 'Company Name', 'Industry']
    if not all(col in df.columns for col in required_cols):
        st.error(f"Cache missing data. Columns found: {df.columns.tolist()}")
    else:
        # UI Controls
        c1, c2, c3 = st.columns(3)
        period = c1.selectbox("Horizon", ["1y", "3y", "5y"], index=2)
        inf_rate = c2.slider("Inflation Rate (%)", 0.0, 15.0, 3.5, 0.1) / 100
        hurdle = c3.slider("Real Growth Hurdle (%)", 0.0, 25.0, 0.0, 0.5) / 100
        
        # Calculate Real Growth
        df['Real Growth'] = ((1 + (df['Hist NI CAGR (%)'] / 100)) / (1 + inf_rate)) - 1
        survivors = df[df['Real Growth'] >= hurdle].copy()
        
        st.metric("Total Stocks Remaining", len(survivors))
        
        if st.button("Run Simulation"):
            with st.spinner("Downloading historical pricing..."):
                # Pull prices only for survivors + SPY
                tickers = survivors['Ticker'].tolist()
                prices = yf.download(tickers + ['SPY'], period=period, interval="1d", auto_adjust=True, progress=False)['Close'].ffill().bfill()
                
                valid = [t for t in tickers if t in prices.columns]
                if not valid: st.error("No valid price data found."); st.stop()
                
                # Point-in-Time Weighting using cached Market Cap
                mcap = survivors.set_index('Ticker').loc[valid, 'Market Cap (B)'].values
                weights = mcap / np.sum(mcap)
                
                # Portfolio Returns
                rets = prices[valid].pct_change().dropna()
                strat_rets = (rets * weights).sum(axis=1)
                spy_rets = prices['SPY'].pct_change().dropna()
                
                # Metrics
                years = len(rets) / 252
                strat_cagr = ((1 + strat_rets).prod() ** (1/years)) - 1
                spy_cagr = ((1 + spy_rets).prod() ** (1/years)) - 1
                
                col1, col2 = st.columns(2)
                col1.metric("Strategy CAGR", f"{strat_cagr*100:.2f}%")
                col2.metric("SPY CAGR", f"{spy_cagr*100:.2f}%")
                
                # Plot
                plot_df = pd.DataFrame({
                    "Strategy": (1 + strat_rets).cumprod() - 1,
                    "SPY": (1 + spy_rets).cumprod() - 1
                }) * 100
                st.plotly_chart(px.line(plot_df, title="Cumulative Performance (%)"), use_container_width=True)
                
                # Breakdown
                st.write("### Surviving Constituents")
                st.dataframe(survivors[['Ticker', 'Company Name', 'Industry', 'Real Growth (%)', 'Market Cap (B)']].sort_values('Real Growth (%)', ascending=False), use_container_width=True)
