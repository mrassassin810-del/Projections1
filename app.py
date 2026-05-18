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
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Forecaster & Screening Engine", layout="wide")

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

# --- HELPER: STANDARDIZED DATAFRAME BUILDER ---
def build_standard_df(rev, cogs, gp, op_inc, ni, shares):
    return pd.DataFrame({
        'Total Revenue': rev, 'Cost Of Revenue': cogs, 'Gross Profit': gp, 
        'Operating Expense': gp - op_inc, 'Operating Income': op_inc, 
        'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 
        'Shares Outstanding': shares, 'EPS': ni / shares
    }).dropna()

# --- DECOUPLED DATA PIPELINE WITH HYBRID STITCHING ---
def fetch_financial_data(ticker, force_deep_dive=False, force_refresh=False):
    stock = yf.Ticker(ticker)
    df_yf, df_final = pd.DataFrame(), pd.DataFrame()
    data_source = "None"
    
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
                except: pass
        
        if os.path.exists(cache_path) and data_source != "Alpha Vantage (Fresh Pull)":
            df_vault = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            file_age_days = (time.time() - os.path.getmtime(cache_path)) / (60 * 60 * 24)
            
            if not df_yf.empty:
                df_final = pd.concat([df_vault[df_vault.index < df_yf.index.min()], df_yf]).sort_index()
                data_source = f"Hybrid Stitch: Vault ({int(file_age_days)}d old) + Live YF"
            else:
                df_final, data_source = df_vault, f"Alpha Vantage Vault ({int(file_age_days)}d old)"
                
    return df_final, current_price, analyst_target, data_source

# --- CORE MATH ENGINE WITH REJECTION RULES ---
def calculate_metric_models(y_in, x_hist, x_fut, metric_name, force_conservative=False):
    y = np.asarray(y_in, dtype=float)
    n = len(y)
    results = {}
    floor_val = max(1, y[-1] * 0.5) if metric_name == 'Shares Outstanding' and n > 0 else 0

    # 1. Base Linear
    lin_model = LinearRegression().fit(x_hist, y)
    results['Linear'] = {'forecast': np.maximum(floor_val, lin_model.predict(x_fut)), 'rmse': np.sqrt(mean_squared_error(y, lin_model.predict(x_hist)))}
    slope = lin_model.coef_[0]

    # 2. Quadratic
    if n >= 4:
        poly_features, poly_fut_features = np.column_stack((x_hist, x_hist**2)), np.column_stack((x_fut, x_fut**2))
        poly_model = LinearRegression().fit(poly_features, y)
        results['Quadratic'] = {'forecast': np.maximum(floor_val, poly_model.predict(poly_fut_features)), 'rmse': np.sqrt(mean_squared_error(y, poly_model.predict(poly_features)))}
    else: results['Quadratic'] = {'forecast': None, 'rmse': float('inf')}

    # 3. Derivative
    if n >= 4:
        diffs = np.diff(y)
        x_diff = np.arange(len(diffs)).reshape(-1, 1)
        deriv_model = LinearRegression().fit(x_diff, diffs)
        fut_diffs = deriv_model.predict(np.arange(len(diffs), len(diffs) + len(x_fut)).reshape(-1, 1))
        forecast_deriv, current_val = [], y[-1]
        for fd in fut_diffs:
            current_val = max(floor_val, current_val + fd)
            forecast_deriv.append(current_val)
        results['Derivative'] = {'forecast': np.array(forecast_deriv), 'rmse': np.sqrt(mean_squared_error(y[1:], y[:-1] + deriv_model.predict(x_diff)))}
    else: results['Derivative'] = {'forecast': None, 'rmse': float('inf')}

    # 4. Logarithmic
    x_log_hist, x_log_fut = np.log(x_hist + 1), np.log(x_fut + 1)
    log_model = LinearRegression().fit(x_log_hist, y)
    results['Logarithmic'] = {'forecast': np.maximum(floor_val, log_model.predict(x_log_fut)), 'rmse': np.sqrt(mean_squared_error(y, log_model.predict(x_log_hist)))}

    # 5. Holt-Winters & 6. ARIMA
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

# --- HELPER: UNIFIED PROJECTION RUNNER (BULLETPROOFED) ---
def run_projections(norm_df, x_hist, x_fut, overrides=None, force_conservative=False):
    q_proj, rmse_tot, metric_results = {}, 0, {}
    for metric in drivers:
        res = calculate_metric_models(norm_df[metric].values, x_hist, x_fut, metric, force_conservative)
        metric_results[metric] = res
        
        act = overrides.get(metric, "Auto") if overrides else "Auto"
        if act not in res or act == "Auto": act = res.get('AutoChoice', 'Linear')
            
        model_data = res.get(act)
        if not model_data or model_data.get('forecast') is None:
            act = "Linear"
            model_data = res.get("Linear")
            
        if not model_data or model_data.get('forecast') is None:
            flat_val = norm_df[metric].values[-1] if len(norm_df[metric]) > 0 else 0
            model_data = {'forecast': np.full(len(x_fut), flat_val), 'rmse': float('inf')}
            
        q_proj[metric] = model_data['forecast']
        rmse_tot += model_data.get('rmse', 0) if model_data.get('rmse') != float('inf') else 0

    q_proj['Gross Profit'] = q_proj['Total Revenue'] - q_proj['Cost Of Revenue']
    q_proj['Operating Income'] = q_proj['Gross Profit'] - q_proj['Operating Expense']
    q_proj['Net Income'] = q_proj['Operating Income'] + q_proj['Non-Op & Taxes']
    if 'Shares Outstanding' in
