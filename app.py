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

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings('ignore')

# --- CONFIGURATION & MOBILE UI ---
st.set_page_config(page_title="Forecaster & Screening Engine", layout="wide")

st.markdown("""
    <style>
    .css-18e3th9 { padding-top: 1rem; }
    .stPlotlyChart { width: 100% !important; }
    </style>
""", unsafe_allow_html=True)

if not HAS_STATSMODELS:
    st.error("⚠️ **Missing Library:** Run: `pip install statsmodels`")
    st.stop()

# --- PASSWORD GATE ---
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

if not st.session_state.authenticated:
    st.title("🔒 System Locked")
    pwd = st.text_input("Enter Access Password:", type="password")
    if st.button("Unlock"):
        if pwd == st.secrets.get("APP_PASSWORD", "admin123"):
            st.session_state.authenticated = True
            st.rerun()
        else:
            st.error("Incorrect Password")
    st.stop() 

# --- SECRETS MANAGEMENT ---
try:
    api_key = st.secrets["AV_API_KEY"]
except:
    api_key = None

# --- GLOBAL CONFIG VARIABLES ---
CACHE_FILE = "sp500_screener_cache.csv"
WATCHLIST_FILE = "watchlist.txt"
TICKER_CACHE_DIR = "ticker_cache"
os.makedirs(TICKER_CACHE_DIR, exist_ok=True)

# Initialize Watchlist into session state
if 'watchlist' not in st.session_state:
    st.session_state.watchlist = []
    if os.path.exists(WATCHLIST_FILE):
        with open(WATCHLIST_FILE, "r") as f:
            st.session_state.watchlist = [line.strip() for line in f.readlines() if line.strip()]

def save_watchlist():
    with open(WATCHLIST_FILE, "w") as f:
        f.write("\n".join(st.session_state.watchlist))

drivers = ['Total Revenue', 'Cost Of Revenue', 'Operating Expense', 'Non-Op & Taxes', 'Shares Outstanding']
model_choices = ["Auto", "Linear", "Quadratic", "Derivative", "Logarithmic", "Holt-Winters", "ARIMA"]
display_order = ['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income', 'Non-Op & Taxes', 'Net Income', 'Shares Outstanding', 'EPS']

# --- SHARED HELPERS ---
@st.cache_data(ttl=86400)
def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url, storage_options={"User-Agent": "Mozilla/5.0"})[0]
    return table['Symbol'].str.replace('.', '-', regex=False).tolist()

def build_standard_df(rev, cogs, gp, op_inc, ni, shares):
    return pd.DataFrame({
        'Total Revenue': rev, 'Cost Of Revenue': cogs, 'Gross Profit': gp, 
        'Operating Expense': gp - op_inc, 'Operating Income': op_inc, 
        'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 
        'Shares Outstanding': shares, 'EPS': ni / shares
    }).dropna()

# --- DECOUPLED DATA PIPELINE WITH HYBRID STITCHING (TAB 1 & 2) ---
def fetch_financial_data(ticker, force_deep_dive=False, force_refresh=False):
    stock = yf.Ticker(ticker)
    df_yf, df_final = pd.DataFrame(), pd.DataFrame()
    data_source, api_warning = "None", None
    
    try:
        if force_deep_dive:
            df = stock.quarterly_income_stmt.T
            if len(df) < 8: df = stock.quarterly_financials.T if len(stock.quarterly_financials.T) > len(df) else df
            if len(df) < 8:
                try: df = stock.get_income_stmt(freq="quarterly").T if len(stock.get_income_stmt(freq="quarterly").T) > len(df) else df
                except: pass
        else:
            df = stock.income_stmt.T
            if df.empty: df = stock.financials.T

        if not df.empty and 'Total Revenue' in df.columns:
            df.index = pd.to_datetime(df.index)
            df = df.sort_index()
            df_raw = df / 1000
            rev = df_raw['Total Revenue']
            gp = df_raw.get('Gross Profit', rev)
            op_inc = df_raw.get('Operating Income', gp)
            ni = df_raw.get('Net Income', op_inc)
            shares = df_raw.get('Diluted Average Shares', df_raw.get('Basic Average Shares', pd.Series(1, index=df_raw.index)))
            df_yf = build_standard_df(rev, rev - gp, gp, op_inc, ni, shares)
    except: pass

    hist_1d = stock.history(period="1d")
    current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0
    analyst_target = stock.info.get('targetMeanPrice', np.nan)
    data_source, df_final = "Yahoo Finance (Standard)", df_yf

    if force_deep_dive:
        cache_path = os.path.join(TICKER_CACHE_DIR, f"{ticker}_deep.csv")
        if force_refresh or not os.path.exists(cache_path):
            if api_key:
                try:
                    r = requests.get(f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}').json()
                    if 'quarterlyReports' in r:
                        df_av = pd.DataFrame(r['quarterlyReports'])
                        df_av['fiscalDateEnding'] = pd.to_datetime(df_av['fiscalDateEnding'])
                        df_av = df_av.set_index('fiscalDateEnding').sort_index()
                        for col in ['totalRevenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome']:
                            df_av[col] = pd.to_numeric(df_av[col], errors='coerce').fillna(0) / 1000
                        rev, cogs, gp = df_av['totalRevenue'], df_av['costOfRevenue'], df_av['grossProfit']
                        op_inc, ni = df_av['operatingIncome'], df_av['netIncome']
                        shares = pd.Series(stock.info.get('sharesOutstanding', 100000) / 1000, index=df_av.index)
                        
                        df_av_clean = build_standard_df(rev, cogs, gp, op_inc, ni, shares)
                        df_av_clean.to_csv(cache_path) 
                        df_final, data_source = df_av_clean, "Alpha Vantage (Fresh Pull)"
                    else: api_warning = r.get('Information', r.get('Note', "Invalid API key or unknown limit reached."))
                except Exception as e: api_warning = f"Connection failed: {str(e)}"
            else: api_warning = "No API Key found in Streamlit Secrets."
        
        if os.path.exists(cache_path) and data_source != "Alpha Vantage (Fresh Pull)":
            df_vault = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            file_age_days = (time.time() - os.path.getmtime(cache_path)) / (60 * 60 * 24)
            if not df_yf.empty:
                df_final = pd.concat([df_vault[df_vault.index < df_yf.index.min()], df_yf]).sort_index()
                data_source = f"Hybrid Stitch: Vault ({int(file_age_days)}d old) + Live YF"
            else: df_final, data_source = df_vault, f"Alpha Vantage Vault ({int(file_age_days)}d old)"
                
    return df_final, current_price, analyst_target, data_source, api_warning

# --- CORE MATH ENGINE WITH REJECTION RULES ---
def calculate_metric_models(y_in, x_hist, x_fut, metric_name, force_conservative=False):
    y, n, results = np.asarray(y_in, dtype=float), len(y_in), {}
    floor_val = max(1, y[-1] * 0.5) if metric_name == 'Shares Outstanding' and n > 0 else 0

    lin_model = LinearRegression().fit(x_hist, y)
    results['Linear'] = {'forecast': np.maximum(floor_val, lin_model.predict(x_fut)), 'rmse': np.sqrt(mean_squared_error(y, lin_model.predict(x_hist)))}
    slope = lin_model.coef_[0]

    if n >= 4:
        poly_features, poly_fut_features = np.column_stack((x_hist, x_hist**2)), np.column_stack((x_fut, x_fut**2))
        poly_model = LinearRegression().fit(poly_features, y)
        results['Quadratic'] = {'forecast': np.maximum(floor_val, poly_model.predict(poly_fut_features)), 'rmse': np.sqrt(mean_squared_error(y, poly_model.predict(poly_features)))}
        
        diffs = np.diff(y)
        x_diff = np.arange(len(diffs)).reshape(-1, 1)
        deriv_model = LinearRegression().fit(x_diff, diffs)
        fut_diffs = deriv_model.predict(np.arange(len(diffs), len(diffs) + len(x_fut)).reshape(-1, 1))
        forecast_deriv, current_val = [], y[-1]
        for fd in fut_diffs:
            current_val = max(floor_val, current_val + fd)
            forecast_deriv.append(current_val)
        results['Derivative'] = {'forecast': np.array(forecast_deriv), 'rmse': np.sqrt(mean_squared_error(y[1:], y[:-1] + deriv_model.predict(x_diff)))}
    else: 
        results['Quadratic'] = {'forecast': None, 'rmse': float('inf')}
        results['Derivative'] = {'forecast': None, 'rmse': float('inf')}

    x_log_hist, x_log_fut = np.log(x_hist + 1), np.log(x_fut + 1)
    log_model = LinearRegression().fit(x_log_hist, y)
    results['Logarithmic'] = {'forecast': np.maximum(floor_val, log_model.predict(x_log_fut)), 'rmse': np.sqrt(mean_squared_error(y, log_model.predict(x_log_hist)))}

    for name, is_hw in [("Holt-Winters", True), ("ARIMA", False)]:
        try:
            if is_hw and n >= 8:
                model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=4, initialization_method="heuristic").fit()
                results[name] = {'forecast': np.maximum(floor_val, model.forecast(len(x_fut))), 'rmse': np.sqrt(mean_squared_error(y, model.fittedvalues))}
            elif not is_hw and n >= 6:
                model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
                results[name] = {'forecast': np.maximum(floor_val, model.forecast(len(x_fut))), 'rmse': np.sqrt(mean_squared_error(y, model.predict(start=0, end=len(y)-1)))}
            else: raise Exception
        except: results[name] = {'forecast': None, 'rmse': float('inf')}

    if metric_name == 'Non-Op & Taxes':
        for name in results:
            if results[name]['forecast'] is not None: results[name]['forecast'] = np.minimum(results[name]['forecast'], max(0, np.max(y)) * 1.5)

    valid_models = sorted([(name, data['rmse'], data['forecast']) for name, data in results.items() if data['forecast'] is not None and data['rmse'] != float('inf')], key=lambda x: x[1])
    if force_conservative: valid_models = [m for m in valid_models if m[0] in ["Linear", "Logarithmic"]]
    
    current_val, auto_choice = y[-1] if len(y) > 0 else 0, valid_models[0][0] if valid_models else "Linear"
    safe_models = []
    
    for name, rmse, forecast in valid_models:
        is_valid = True
        if current_val > 0:
            if metric_name == 'Total Revenue' and forecast[-1] > (current_val * 5.0): is_valid = False 
            elif metric_name in ['Cost Of Revenue', 'Operating Expense'] and forecast[-1] < (current_val * 0.2): is_valid = False 
            elif metric_name == 'Shares Outstanding':
                if forecast[-1] < (current_val * 0.5): is_valid = False
                if forecast[-1] > (current_val * 1.2): is_valid = False
            if metric_name in ['Shares Outstanding', 'Operating Expense', 'Cost Of Revenue'] and slope > 0 and forecast[-1] < (current_val * 0.95): is_valid = False
        if name in ["Quadratic", "Derivative", "ARIMA"] and current_val > 0 and forecast[-1] > (current_val * 3.5): is_valid = False
        if is_valid: safe_models.append(name)

    results['AutoChoice'] = safe_models[0] if safe_models else ("Logarithmic" if "Logarithmic" in [m[0] for m in valid_models] else "Linear")
    return results

def run_projections(norm_df, x_hist, x_fut, overrides=None, force_conservative=False):
    q_proj, rmse_tot, metric_results = {}, 0, {}
    for metric in drivers:
        res = calculate_metric_models(norm_df[metric].values, x_hist, x_fut, metric, force_conservative)
        metric_results[metric] = res
        act = overrides.get(metric, "Auto") if overrides else "Auto"
        if act not in res or act == "Auto": act = res.get('AutoChoice', 'Linear')
        model_data = res.get(act)
        if not model_data or model_data.get('forecast') is None:
            act, model_data = "Linear", res.get("Linear")
        if not model_data or model_data.get('forecast') is None:
            model_data = {'forecast': np.full(len(x_fut), norm_df[metric].values[-1] if len(norm_df[metric]) > 0 else 0), 'rmse': float('inf')}
        q_proj[metric] = model_data['forecast']
        rmse_tot += model_data.get('rmse', 0) if model_data.get('rmse') != float('inf') else 0

    q_proj['Gross Profit'] = q_proj['Total Revenue'] - q_proj['Cost Of Revenue']
    q_proj['Operating Income'] = q_proj['Gross Profit'] - q_proj['Operating Expense']
    q_proj['Net Income'] = q_proj['Operating Income'] + q_proj['Non-Op & Taxes']
    if 'Shares Outstanding' in q_proj: q_proj['Shares Outstanding'] = np.maximum(1, q_proj['Shares Outstanding'])
    return q_proj, rmse_tot, metric_results

# --- BACKGROUND WORKER FOR SCREENER (ANNUAL DATA ONLY) ---
def process_single_screener_stock(ticker):
    try:
        norm_df, current_p, _, _, _ = fetch_financial_data(ticker, force_deep_dive=False)
        if norm_df.empty or len(norm_df) < 2: return None
        
        stock = yf.Ticker(ticker)
        hist_5y = stock.history(period="5y")
        hist_5y.index = pd.to_datetime(hist_5y.index).tz_localize(None)
        current_p = hist_5y['Close'].iloc[-1] if not hist_5y.empty else 0.0

        x_hist, x_fut = np.arange(len(norm_df)).reshape(-1, 1), np.arange(len(norm_df), len(norm_df) + 5).reshape(-1, 1)

        proj, total_rmse, _ = run_projections(norm_df, x_hist, x_fut)
        eps_y1 = proj['Net Income'][0] / max(1, proj['Shares Outstanding'][0])
        
        info = stock.info
        f_eps = info.get('forwardEps', np.nan)
        max_hist_ni = max(1, norm_df['Net Income'].max())
        
        is_hallucinating = False
        if pd.notna(f_eps) and f_eps > 0 and eps_y1 > 0 and (eps_y1 > (f_eps * 2.5) or eps_y1 < (f_eps * 0.4)): is_hallucinating = True
        if proj['Net Income'][-1] > (max_hist_ni * 8.0): is_hallucinating = True

        if is_hallucinating: proj, total_rmse, _ = run_projections(norm_df, x_hist, x_fut, force_conservative=True)
        eps_y5 = proj['Net Income'][-1] / max(1, proj['Shares Outstanding'][-1])
        if current_p <= 0: return None
        
        # Calculate Historical 5-Year Average P/E dynamically
        avg_pe = np.nan
        if not hist_5y.empty and not norm_df.empty:
            pe_list = []
            for date, row in norm_df.iterrows():
                dt = pd.to_datetime(date).tz_localize(None)
                price_slice = hist_5y.loc[:dt]
                if not price_slice.empty:
                    price, eps = price_slice['Close'].iloc[-1], row['EPS']
                    if eps > 0: pe_list.append(price / eps)
            pe_list = [pe for pe in pe_list if pe < 300]
            if pe_list: avg_pe = np.mean(pe_list)
            
        # Robust Dynamic Falling Matrix for Historical Net Income CAGR
        oldest_ni = norm_df['Net Income'].iloc[0]
        latest_ni = norm_df['Net Income'].iloc[-1]
        years = len(norm_df) - 1
        
        if (oldest_ni <= 0 or latest_ni <= 0) and not stock.quarterly_financials.empty:
            try:
                q_ni = stock.quarterly_financials.T['Net Income'].dropna()
                if len(q_ni) >= 5:
                    latest_ni = q_ni.iloc[:4].sum() / 1000
                    oldest_ni = q_ni.iloc[-4:].sum() / 1000
                    years = (q_ni.index[0] - q_ni.index[-1]).days / 365.25
            except: pass

        hist_ni_cagr = np.nan
        if oldest_ni > 0 and latest_ni > 0 and years > 0:
            hist_ni_cagr = ((latest_ni / oldest_ni) ** (1 / years)) - 1

        return {
            "Ticker": ticker, "Company Name": info.get('shortName', info.get('longName', 'N/A')), "Industry": info.get('industry', 'N/A'),
            "Current Price": round(current_p, 2), "Year 5 EPS": eps_y5, "Avg Tracking Error (RMSE)": round(total_rmse / len(drivers), 2),
            "Market Cap (B)": info.get('marketCap', np.nan) / 1e9 if pd.notna(info.get('marketCap')) else np.nan,
            "Rev Growth (%)": (info.get('revenueGrowth', np.nan) * 100) if pd.notna(info.get('revenueGrowth')) else np.nan,
            "Current P/E": info.get('trailingPE', np.nan), "Forward P/E": info.get('forwardPE', np.nan), "5-Yr Avg P/E": avg_pe,
            "PEG Ratio": info.get('pegRatio', np.nan), "P/B Ratio": info.get('priceToBook', np.nan), "P/S Ratio": info.get('priceToSalesTrailing12Months', np.nan),
            "ROE (%)": (info.get('returnOnEquity', np.nan) * 100) if pd.notna(info.get('returnOnEquity')) else np.nan,
            "ROA (%)": (info.get('returnOnAssets', np.nan) * 100) if pd.notna(info.get('returnOnAssets')) else np.nan,
            "Debt/Equity": info.get('debtToEquity', np.nan), "Gross Margin (%)": (info.get('grossMargins', np.nan) * 100) if pd.notna(info.get('grossMargins')) else np.nan,
            "Profit Margin (%)": (info.get('profitMargins', np.nan) * 100) if pd.notna(info.get('profitMargins')) else np.nan,
            "Div Yield (%)": (info.get('dividendYield', np.nan) * 100) if pd.notna(info.get('dividendYield')) else 0.0,
            "Beta": info.get('beta', np.nan), "Short % Float": (info.get('shortPercentOfFloat', np.nan) * 100) if pd.notna(info.get('shortPercentOfFloat')) else np.nan,
            "Hist NI CAGR (%)": (hist_ni_cagr * 100) if pd.notna(hist_ni_cagr) else np.nan
        }
    except: return None

# --- UI APP TABS ---
tab_single, tab_screener, tab_backtest = st.tabs(["📊 Single Ticker Forecast", "🔍 S&P 500 Screening Dashboard", "📈 Strategy Backtester"])

# ================= TAB 1: SINGLE FORECASTER =================
with tab_single:
    for m in drivers:
        if f"ov_{m}" not in st.session_state: st.session_state[f"ov_{m}"] = "Auto"
    def reset_overrides():
        for m in drivers: st.session_state[f"ov_{m}"] = "Auto"

    col1, col2, col3 = st.columns([2, 1, 1])
    with col1: 
        ticker_input = st.text_input("Enter Ticker:", "PLTR", key="single_tick").upper()
        force_refresh_tab1 = st.checkbox("Force API Refresh", help="Bypass local vault and pull fresh AV data (uses 1 API credit).")
    with col2: lookback_input = st.number_input("Quarters back (0 = All):", min_value=0, max_value=40, value=0, step=1, key="single_lb")
    with col3:
        st.write(""); st.write("")
        if st.button("Fetch & Analyze", key="single_btn", use_container_width=True):
            with st.spinner(f"Executing Deep Data Mine for {ticker_input}..."):
                norm_df, current_price, analyst_target, data_source, api_warning = fetch_financial_data(ticker_input, force_deep_dive=True, force_refresh=force_refresh_tab1)
                if norm_df.empty: st.error(f"No financial data found for {ticker_input}."); st.stop()
                st.session_state.update({'norm_df': norm_df, 'current_price': current_price, 'analyst_target_tab1': analyst_target, 'ticker_analyzed': ticker_input, 'actual_lookback': lookback_input, 'data_source': data_source, 'api_warning': api_warning})

    if 'norm_df' in st.session_state and st.session_state.ticker_analyzed == ticker_input:
        if st.session_state.get('api_warning') and force_refresh_tab1: st.warning(f"⚠️ **API Notice:** {st.session_state.api_warning} (Safely fell back to standard Yahoo Finance data).")
        norm_df, current_price = st.session_state.norm_df, st.session_state.current_price
        df_reg = norm_df.tail(len(norm_df) if st.session_state.actual_lookback == 0 else st.session_state.actual_lookback)
        
        st.markdown(f"**Data Depth Indicator:** :{'green' if len(df_reg) >= 8 else 'red'}[{len(df_reg)} Quarters Loaded] via {st.session_state.data_source} *(Note: ARIMA/Holt-Winters require 6-8 minimum)*")
        
        x_hist, x