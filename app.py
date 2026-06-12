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

# [Keep your existing Password Gate and Secrets Management code block here]
# (Truncated for brevity to focus on the Backtester logic update)

# --- TAB 3: STRATEGY BACKTESTER (RE-ARCHITECTED) ---
with tab_backtest:
    st.subheader("📊 Pure Point-In-Time S&P 500 Backtester")
    
    if 'raw_screener_df' not in st.session_state or st.session_state.raw_screener_df.empty:
        st.warning("⚠️ Run 'Bulk Refresh Matrix' in the Screener tab first.")
    else:
        st.write("##### ⚙️ Strategy Parameters & Filters")
        
        # 1. Horizon & Macro
        c1, c2, c3 = st.columns(3)
        backtest_period = c1.selectbox("Backtest Horizon", ["1y", "3y", "5y"], index=2)
        inf_rate = c2.slider("Inflation Rate (%)", 0.0, 15.0, 3.5, 0.1) / 100
        
        # 2. Advanced Backtesting Filters
        c4, c5, c6 = st.columns(3)
        real_growth_hurdle = c4.slider("Real Growth Hurdle (%)", 0.0, 25.0, 5.0, 0.5) / 100
        fcf_yield_min = c5.slider("Min FCF Yield (%)", 0.0, 10.0, 0.0, 0.5) / 100
        max_de = c6.slider("Max Debt/Equity", 0.0, 500.0, 200.0, 10.0)

        df = st.session_state.raw_screener_df.copy()
        
        # Apply Logic
        df['Nominal CAGR'] = df['Hist NI CAGR (%)'] / 100
        df['Real Growth'] = ((1 + df['Nominal CAGR']) / (1 + inf_rate)) - 1
        
        survivors = df[
            (df['Real Growth'] >= real_growth_hurdle) & 
            (df['Debt/Equity'] <= max_de)
        ].copy()
        
        count = len(survivors)
        st.metric("Total Stocks Remaining", count)

        if survivors.empty:
            st.error("No companies met your criteria. Adjust parameters.")
        else:
            surviving_tickers = survivors['Ticker'].tolist()
            
            with st.spinner("Processing point-in-time pricing..."):
                prices = get_historical_prices(surviving_tickers, backtest_period)
                prices = prices.ffill().bfill()
                valid_tickers = [t for t in surviving_tickers if t in prices.columns]
                
            if not valid_tickers:
                st.error("Insufficient historical data.")
            else:
                # Weighted allocation logic
                weights = []
                for t in valid_tickers:
                    mcap = survivors.loc[survivors['Ticker']==t, 'Market Cap (B)'].values[0] * 1e9
                    weights.append(mcap)
                weights = np.array(weights) / np.sum(weights)
                
                # Returns calculation
                returns = prices[valid_tickers].pct_change().dropna()
                strat_returns = (returns * weights).sum(axis=1)
                spy_returns = prices['SPY'].pct_change().dropna()
                
                # Metrics
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
                
                st.plotly_chart(px.line(plot_df, title="Cumulative % Return", labels={'value': 'Return (%)'}), use_container_width=True)

                with st.expander("View Full Surviving List"):
                    st.dataframe(survivors[['Ticker', 'Company Name', 'Industry', 'Real Growth (%)', 'Debt/Equity']].sort_values('Real Growth (%)', ascending=False), use_container_width=True)
