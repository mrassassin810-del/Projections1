import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from concurrent.futures import ThreadPoolExecutor, as_completed
import warnings

warnings.filterwarnings('ignore')

# --- CONFIGURATION & MOBILE UI ---
st.set_page_config(page_title="S&P 500 Real Growth Backtester", layout="wide")

# Custom CSS for Pixel 7 readability
st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 1rem; }
    .stPlotlyChart { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

# --- HELPER FUNCTIONS ---
@st.cache_data(ttl=86400)
def fetch_sp500_tickers():
    """Pulls current S&P 500 constituents from Wikipedia."""
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    tickers = table['Symbol'].str.replace('.', '-', regex=False).tolist()
    return tickers

def calculate_ni_cagr(ticker):
    """Attempts to calculate 5-Year Net Income CAGR. Falls back to max available years."""
    try:
        stock = yf.Ticker(ticker)
        # yfinance typically limits free financial statements to 4 years.
        financials = stock.financials.T
        if 'Net Income' not in financials.columns or len(financials) < 2:
            return ticker, np.nan, np.nan
        
        ni_series = financials['Net Income'].dropna()
        if len(ni_series) < 2 or ni_series.iloc[-1] <= 0: # Cannot calculate CAGR on negative starting bases
            return ticker, np.nan, np.nan
            
        latest_ni = ni_series.iloc[0]
        oldest_ni = ni_series.iloc[-1]
        years = len(ni_series) - 1
        
        if oldest_ni <= 0:
            return ticker, np.nan, np.nan
            
        cagr = ((latest_ni / oldest_ni) ** (1 / years)) - 1
        mcap = stock.info.get('marketCap', np.nan)
        
        return ticker, cagr, mcap
    except:
        return ticker, np.nan, np.nan

@st.cache_data(ttl=86400)
def build_market_matrix(tickers):
    """Multithreaded data extraction for the entire S&P 500."""
    results = []
    with ThreadPoolExecutor(max_workers=20) as executor:
        futures = {executor.submit(calculate_ni_cagr, t): t for t in tickers}
        for future in as_completed(futures):
            t, cagr, mcap = future.result()
            if pd.notna(cagr) and pd.notna(mcap):
                results.append({"Ticker": t, "Nominal CAGR": cagr, "Market Cap": mcap})
    return pd.DataFrame(results)

def get_historical_prices(tickers, period):
    """Pulls historical adjusted close prices for backtesting."""
    data = yf.download(tickers + ['SPY'], period=period, interval="1d", auto_adjust=True, progress=False)
    return data['Close']

# --- UI & SIDEBAR ---
st.sidebar.title("⚙️ Strategy Parameters")
st.sidebar.markdown("Filter the S&P 500 by **Real Net Income Growth**.")

inf_rate = st.sidebar.slider("Inflation Rate (%)", min_value=0.0, max_value=15.0, value=4.2, step=0.1) / 100
hurdle_rate = st.sidebar.slider("Real Growth Hurdle (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.5) / 100
backtest_period = st.sidebar.selectbox("Backtest Horizon", options=["1y", "3y", "5y"], index=2)

run_engine = st.sidebar.button("🚀 Run Backtest", use_container_width=True)

# --- MAIN DASHBOARD ---
st.title("📊 Macro-Adjusted S&P 500 Screener")
st.markdown("Isolates companies outpacing inflation and backtests their historical market-cap weighted performance against the broader index.")

if run_engine:
    with st.spinner("Compiling fundamental matrix... (This takes ~30 seconds)"):
        # 1. Gather Constituents & Data
        tickers = fetch_sp500_tickers()
        df = build_market_matrix(tickers)
        
        # 2. The Fisher Equation: ((1 + Nominal) / (1 + Inflation)) - 1
        df['Real Growth'] = ((1 + df['Nominal CAGR']) / (1 + inf_rate)) - 1
        
        # 3. Apply Hurdle Filter
        survivors = df[df['Real Growth'] >= hurdle_rate].copy()
        
        if survivors.empty:
            st.error("No companies met the Real Growth hurdle rate. Lower your parameters.")
            st.stop()
            
        # 4. Weight by Market Cap
        total_mcap = survivors['Market Cap'].sum()
        survivors['Weight'] = survivors['Market Cap'] / total_mcap
        survivors = survivors.sort_values(by='Weight', ascending=False)
        
        st.success(f"Screening complete. **{len(survivors)} / {len(tickers)}** companies survived the hurdle.")

    # --- TREEMAP VISUALIZATION ---
    st.subheader("Current Portfolio Weights")
    
    # Format for Treemap
    survivors['Market Cap (B)'] = (survivors['Market Cap'] / 1e9).round(2)
    survivors['Real Growth (%)'] = (survivors['Real Growth'] * 100).round(2)
    
    fig_tree = px.treemap(
        survivors, 
        path=[px.Constant("Surviving Index"), 'Ticker'], 
        values='Weight',
        color='Real Growth (%)',
        color_continuous_scale='Greens',
        hover_data=['Market Cap (B)']
    )
    fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=400)
    st.plotly_chart(fig_tree, use_container_width=True)

    # --- BACKTESTING ENGINE ---
    with st.spinner(f"Downloading {backtest_period} historical price data..."):
        surviving_tickers = survivors['Ticker'].tolist()
        prices = get_historical_prices(surviving_tickers, backtest_period)
        
        # Drop tickers that don't have enough historical data to map against SPY
        prices = prices.dropna(axis=1, how='all')
        valid_tickers = [t for t in surviving_tickers if t in prices.columns]
        
        if not valid_tickers:
            st.error("Insufficient historical price data for the surviving basket.")
            st.stop()
            
        # Re-normalize weights based on valid tickers only (avoids math errors if a recent IPO is dropped)
        backtest_weights = survivors.set_index('Ticker').loc[valid_tickers, 'Weight']
        backtest_weights = backtest_weights / backtest_weights.sum()

        # Calculate daily returns
        daily_returns = prices.pct_change().dropna()
        
        # Portfolio Return = Sum of (Daily Returns * Weights)
        strat_returns = (daily_returns[valid_tickers] * backtest_weights.values).sum(axis=1)
        spy_returns = daily_returns['SPY']
        
        # Cumulative Returns (Base 100)
        cum_strat = (1 + strat_returns).cumprod() * 100
        cum_spy = (1 + spy_returns).cumprod() * 100
        
        # Combine for plotting
        plot_df = pd.DataFrame({
            "Custom Strategy": cum_strat,
            "SPY Benchmark": cum_spy
        })

    # --- LINE CHART VISUALIZATION ---
    st.subheader(f"Historical Performance ({backtest_period})")
    
    fig_line = px.line(
        plot_df, 
        labels={'value': 'Cumulative Return ($100 Base)', 'Date': 'Date'},
        color_discrete_map={"Custom Strategy": "#00FF00", "SPY Benchmark": "#808080"}
    )
    
    # Clean UI layout for Pixel 7 screens
    fig_line.update_layout(
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(t=20, l=10, r=10, b=10),
        xaxis_title="",
        yaxis_title="Portfolio Value ($)",
        height=350
    )
    st.plotly_chart(fig_line, use_container_width=True)

    # --- DATAFRAME VIEW ---
    with st.expander("View Surviving Constituents"):
        st.dataframe(
            survivors[['Ticker', 'Market Cap (B)', 'Nominal CAGR', 'Real Growth (%)', 'Weight']].style.format({
                'Nominal CAGR': '{:.2%}',
                'Weight': '{:.2%}'
            }),
            use_container_width=True
        )
