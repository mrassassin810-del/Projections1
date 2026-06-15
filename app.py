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
        if norm_df is None or norm_df.empty or len(norm_df) < 2: return None
        
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
        
        x_hist, x_fut = np.arange(len(df_reg)).reshape(-1, 1), np.arange(len(df_reg), len(df_reg) + 20).reshape(-1, 1) 
        proj_quarterly_data, _, metric_results = run_projections(df_reg, x_hist, x_fut, overrides={m: st.session_state[f"ov_{m}"] for m in drivers})

        with st.expander("⚙️ Advanced: Override Projection Models & Explanations"):
            st.markdown("**Model Selection Guide:** \n* **Linear:** Stable value. \n* **Quadratic:** Hyper-growth/Cyclical. \n* **Logarithmic:** Maturing growth. \n* **Derivative:** Momentum shifts. \n* **Holt-Winters:** Seasonal. \n* **ARIMA:** Macro-driven.")
            st.write("---")
            st.button("🔄 Reset all to Auto", on_click=reset_overrides, key="reset_tab1")
            for metric in drivers: st.selectbox(metric, options=model_choices, format_func=lambda o, r=metric_results[metric]: f"Auto ({r.get('AutoChoice', 'Linear')})" if o == "Auto" else (f"{o} (RMSE: ±${int(r[o]['rmse']):,})" if r.get(o) and r[o].get('rmse', float('inf')) != float('inf') else f"{o} (N/A)"), key=f"ov_{metric}")

        proj_annual_data = {}
        for metric in display_order:
            if metric == 'Shares Outstanding': proj_annual_data[metric] = [np.mean(proj_quarterly_data[metric][i*4:(i+1)*4]) for i in range(5)]
            elif metric != 'EPS': proj_annual_data[metric] = [np.sum(proj_quarterly_data[metric][i*4:(i+1)*4]) for i in range(5)]
        proj_annual_data['EPS'] = (np.array(proj_annual_data['Net Income']) / np.array(proj_annual_data['Shares Outstanding'])).tolist()

        hist_labels, hist_data = [], {m: [] for m in display_order}
        for i in range(min(3, len(norm_df) // 4) or 1, 0, -1):
            chunk = norm_df.iloc[-4:] if i == 1 else norm_df.iloc[-(i*4):-((i-1)*4)]
            hist_labels.append("LTM (Current)" if i == 1 else f"LTM -{i-1}")
            for m in display_order: hist_data[m].append(chunk[m].mean() if m == 'Shares Outstanding' else (chunk['Net Income'].sum()/chunk['Shares Outstanding'].mean() if m == 'EPS' else chunk[m].sum()))

        proj_labels = [(norm_df.index[-1] + pd.DateOffset(months=12 * j)).strftime("LTM %b '%yE") for j in range(1, 6)]
        
        st.subheader(f"Historical & 5-Year Projections ({ticker_input})")
        md = f"\n\n| Metric | {' | '.join(hist_labels + proj_labels)} |\n|---{'|---'*len(hist_labels + proj_labels)}|\n"
        for metric in display_order:
            row = f"| **{metric}** |"
            comb = hist_data[metric] + proj_annual_data[metric]
            for idx, val in enumerate(comb):
                val_str = f"${val:,.2f}" if metric == 'EPS' else f"{val:,.0f}"
                if idx == 0: row += f" {val_str} |"
                else:
                    growth = (val - comb[idx-1]) / abs(comb[idx-1]) if comb[idx-1] else 0
                    color = "#1d9e75" if (growth > 0 and metric in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EPS']) or (growth < 0 and metric not in ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EPS']) else "#a32d2d"
                    row += f" {val_str} <span style='color:{color}; font-weight:600; font-size:0.85em;'>({growth:+.1%})</span> |"
            md += row + "\n"
        st.markdown(md, unsafe_allow_html=True)

        st.write("---")
        st.subheader("Implied Stock Price")
        col_val1, col_val2 = st.columns(2)
        col_val1.write(f"**Current Market Price:** ${current_price:,.2f}")
        col_val2.write(f"**Analyst Mean Target (1Y):** ${st.session_state.analyst_target_tab1:,.2f}" if isinstance(st.session_state.analyst_target_tab1, (int, float)) and pd.notna(st.session_state.analyst_target_tab1) else "**Analyst Mean Target (1Y):** N/A")
            
        t_pe = st.number_input("Target P/E Ratio:", value=25.0, step=1.0, key="pe_tab1")
        t_prices = [proj_annual_data['EPS'][j] * t_pe for j in range(5)]
        val_md = f"| Valuation | {' | '.join(proj_labels)} | 5-Yr CAGR |\n|---{'|---'*len(proj_labels)}|---|\n| **Target Price** |"
        for tp in t_prices: val_md += f" **${tp:,.2f}** |"
        cagr = (t_prices[-1] / current_price) ** (1/5) - 1 if current_price > 0 and t_prices[-1] > 0 else 0
        val_md += f" <span style='color:{'#1d9e75' if cagr > 0 else '#a32d2d'}; font-weight:600;'>{cagr:+.1%}</span> |"
        st.markdown(val_md, unsafe_allow_html=True)

        st.subheader("Visual Forecasts")
        combined_q_df = pd.concat([norm_df, pd.DataFrame(proj_quarterly_data, index=[norm_df.index[-1] + pd.DateOffset(months=3 * j) for j in range(1, 21)])])
        ttm_eps = (combined_q_df['Net Income'].rolling(window=4, min_periods=1).sum() * (4 / combined_q_df['Net Income'].rolling(window=4, min_periods=1).count())) / combined_q_df['Shares Outstanding'].rolling(window=4, min_periods=1).mean()
        
        c_df = pd.DataFrame(index=combined_q_df.index)
        c_df['Quarterly EPS'], c_df[f'Target Price (PE {t_pe:g})'] = combined_q_df['EPS'].round(2), (ttm_eps * t_pe).round(2)
        c_df_reset = c_df.reset_index().rename(columns={'index': 'Date'})

        base = alt.Chart(c_df_reset).encode(x=alt.X('Date:T', title=None, axis=alt.Axis(grid=True)))
        l_eps = base.mark_line(color="#1d9e75", point=alt.OverlayMarkDef(color="#1d9e75", size=60)).encode(y=alt.Y('Quarterly EPS:Q', title='Quarterly EPS ($)', axis=alt.Axis(titleColor='#1d9e75', grid=True, minExtent=40)), tooltip=['Date:T', 'Quarterly EPS'])
        l_prc = base.mark_line(color="#e8a329", point=alt.OverlayMarkDef(color="#e8a329", size=60)).encode(y=alt.Y(f'Target Price (PE {t_pe:g}):Q', title='Target Price ($)', axis=alt.Axis(titleColor='#e8a329', grid=False, minExtent=40)), tooltip=['Date:T', f'Target Price (PE {t_pe:g})'])
        st.altair_chart(alt.layer(l_eps, l_prc).resolve_scale(y='independent').properties(height=350).interactive(), use_container_width=True)

# ================= TAB 2: S&P 500 SCREENER =================
with tab_screener:
    st.subheader("S&P 500 Multi-Model Ranking Dashboard")
    
    if os.path.exists(CACHE_FILE) and 'raw_screener_df' not in st.session_state: 
        st.session_state.raw_screener_df = pd.read_csv(CACHE_FILE)
    
    st.markdown(f"**Data Last Loaded:** `{pd.to_datetime(os.path.getmtime(CACHE_FILE), unit='s').strftime('%B %d, %Y at %I:%M %p') if os.path.exists(CACHE_FILE) else 'Never'}`")
    
    st.write("### ⚡ Database & Watchlist Controls")
    c1, c2, c3 = st.columns([1, 1, 1.5])

    with c1:
        st.write("**Bulk Refresh Matrix:**")
        if st.button("🔄 Scan Entire S&P 500", use_container_width=True):
            with st.spinner("Fetching S&P 500 Roster & Executing Institutional Batched Scan..."):
                try: tickers = get_sp500_tickers()
                except Exception as e: st.error(f"Failed to fetch stock index list: {e}"); st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                screened_results = []
                
                # BATCH PIPELINE: Prevents Yahoo Finance from blocking your IP and returning only 4 stocks.
                BATCH_SIZE = 25
                for i in range(0, len(tickers), BATCH_SIZE):
                    batch = tickers[i:i+BATCH_SIZE]
                    with ThreadPoolExecutor(max_workers=5) as executor:
                        futures = [executor.submit(process_single_screener_stock, t) for t in batch]
                        for future in as_completed(futures):
                            res = future.result()
                            if res: screened_results.append(res)
                    
                    progress_bar.progress(min((i + BATCH_SIZE) / len(tickers), 1.0))
                    status_text.write(f"Scanned {len(screened_results)} / {len(tickers)} companies...")
                    time.sleep(1) # Institutional throttling to respect API limits

                status_text.success(f"Matrix complete! Modeled {len(screened_results)} companies.")
                raw_df = pd.DataFrame(screened_results)
                if 'Analyst Target' in raw_df.columns: raw_df = raw_df.drop(columns=['Analyst Target'])
                raw_df.to_csv(CACHE_FILE, index=False)
                st.session_state.raw_screener_df = raw_df
                st.rerun()

        if st.button("⭐ Refresh Watchlist Only", use_container_width=True):
            if not st.session_state.watchlist: st.warning("Your Watchlist is empty.")
            elif 'raw_screener_df' in st.session_state and not st.session_state.raw_screener_df.empty:
                with st.spinner("Updating Watchlist Models..."):
                    df_cache = st.session_state.raw_screener_df
                    for tick in st.session_state.watchlist:
                        if res := process_single_screener_stock(tick):
                            if 'Analyst Target' in res: del res['Analyst Target']
                            if tick in df_cache['Ticker'].values:
                                df_cache.loc[df_cache['Ticker'] == tick, list(res.keys())] = list(res.values())
                            else: df_cache = pd.concat([df_cache, pd.DataFrame([res])], ignore_index=True)
                    df_cache.to_csv(CACHE_FILE, index=False)
                    st.session_state.raw_screener_df = df_cache
                st.rerun()
            else: st.error("Cache is empty. Run a full scan first to build the database.")

    with c2:
        st.write("**Targeted Action:**")
        target_tick = st.text_input("Ticker", placeholder="e.g. NVDA", label_visibility="collapsed", key="screener_tgt_input").upper().strip()
        col_upd, col_wl = st.columns(2)
        
        if col_upd.button("Update Data", use_container_width=True) and target_tick:
            if 'raw_screener_df' in st.session_state:
                with st.spinner(f"Recalculating {target_tick}..."):
                    if res := process_single_screener_stock(target_tick):
                        df_cache = st.session_state.raw_screener_df
                        if 'Analyst Target' in res: del res['Analyst Target']
                        if target_tick in df_cache['Ticker'].values:
                            df_cache.loc[df_cache['Ticker'] == target_tick, list(res.keys())] = list(res.values())
                        else: df_cache = pd.concat([df_cache, pd.DataFrame([res])], ignore_index=True)
                        df_cache.to_csv(CACHE_FILE, index=False)
                        st.session_state.raw_screener_df = df_cache
                        st.rerun()
                    else: st.error(f"Could not calculate projections for {target_tick}.")
            else: st.error("Cache is empty. Run a full scan first to build the database.")
            
        if col_wl.button("⭐ Add/Drop", use_container_width=True) and target_tick:
            if target_tick in st.session_state.watchlist: 
                st.session_state.watchlist.remove(target_tick)
                st.success(f"Removed {target_tick}")
            else: 
                st.session_state.watchlist.append(target_tick)
                st.success(f"Added {target_tick}")
            save_watchlist()
            time.sleep(0.5)
            st.rerun()

    with c3:
        st.write("**Current Watchlist:**")
        st.caption(", ".join(st.session_state.watchlist) if st.session_state.watchlist else "Watchlist is empty. Add tickers via Targeted Action.")

    st.write("---")

    if 'raw_screener_df' not in st.session_state or st.session_state.raw_screener_df.empty:
        st.info("💡 **Database Status:** No structural metrics inside memory. Please click **🔄 Scan Entire S&P 500** to pull data and populate the dashboard.")
    else:
        df_base = st.session_state.raw_screener_df.copy()
        screener_pe = st.number_input("Universal Target P/E Multiple for Model:", value=25.0, step=1.0, key="pe_screener")

        df_base['Year 5 Target'] = df_base['Year 5 EPS'].fillna(0) * screener_pe
        df_base['5-Yr CAGR'] = np.where(df_base['Year 5 Target'] > 0, ((df_base['Year 5 Target'] / df_base['Current Price'].replace(0, np.nan)) ** (1/5) - 1) * 100, -100.0)
        df_base['5-Yr CAGR'] = df_base['5-Yr CAGR'].fillna(-100.0)

        def get_safe_bounds(series, d_min, d_max):
            cleaned = series.dropna()
            if cleaned.empty: return float(d_min), float(d_max)
            return float(cleaned.min()), float(cleaned.max())

        with st.expander("🔬 Deep Toggle Filters", expanded=True):
            search_ticker = st.text_input("🔍 Search Ticker:", "").upper()
            t_wl_filter = st.toggle("⭐ Show Watchlist Only", value=False)
            
            st.write("##### Enable specific filters to constrain the matrix:")
            f1, f2, f3, f4, f5 = st.columns(5)
            with f1:
                t_pe = st.toggle("Current P/E Range")
                b_pe = get_safe_bounds(df_base['Current P/E'], 0.0, 200.0)
                range_pe = st.slider("Current P/E", b_pe[0], max(b_pe[1], b_pe[0]+1), (max(b_pe[0], 10.0), min(b_pe[1], 50.0)), key="v_pe", label_visibility="collapsed") if t_pe else None
            with f2:
                t_fpe = st.toggle("Forward P/E Range")
                b_fpe = get_safe_bounds(df_base['Forward P/E'], 0.0, 150.0)
                range_fpe = st.slider("Forward P/E", b_fpe[0], max(b_fpe[1], b_fpe[0]+1), (max(b_fpe[0], 5.0), min(b_fpe[1], 35.0)), key="v_fpe", label_visibility="collapsed") if t_fpe else None
            with f3:
                t_avg_pe = st.toggle("5-Yr Avg P/E Range")
                b_ape = get_safe_bounds(df_base['5-Yr Avg P/E'], 0.0, 150.0)
                range_avg_pe = st.slider("5-Yr Avg P/E", b_ape[0], max(b_ape[1], b_ape[0]+1), (max(b_ape[0], 5.0), min(b_ape[1], 35.0)), key="v_avg_pe", label_visibility="collapsed") if t_avg_pe else None
            with f4:
                t_peg = st.toggle("PEG Ratio Range")
                b_peg = get_safe_bounds(df_base['PEG Ratio'], 0.0, 10.0)
                range_peg = st.slider("PEG Ratio", b_peg[0], max(b_peg[1], b_peg[0]+0.1), (max(b_peg[0], 0.0), min(b_peg[1], 3.0)), key="v_peg", label_visibility="collapsed") if t_peg else None
            with f5:
                t_ps = st.toggle("P/S Ratio Range")
                b_ps = get_safe_bounds(df_base['P/S Ratio'], 0.0, 50.0)
                range_ps = st.slider("P/S Ratio", b_ps[0], max(b_ps[1], b_ps[0]+1), (max(b_ps[0], 0.0), min(b_ps[1], 10.0)), key="v_ps", label_visibility="collapsed") if t_ps else None

            f6, f7, f8, f9 = st.columns(4)
            with f6:
                t_pb = st.toggle("P/B Ratio Range")
                b_pb = get_safe_bounds(df_base['P/B Ratio'], 0.0, 50.0)
                range_pb = st.slider("P/B Ratio", b_pb[0], max(b_pb[1], b_pb[0]+1), (max(b_pb[0], 0.0), min(b_pb[1], 15.0)), key="v_pb", label_visibility="collapsed") if t_pb else None
            with f7:
                t_roe = st.toggle("ROE (%) Range")
                b_roe = get_safe_bounds(df_base['ROE (%)'], -100.0, 200.0)
                range_roe = st.slider("ROE (%)", b_roe[0], max(b_roe[1], b_roe[0]+1), (max(b_roe[0], 10.0), min(b_roe[1], 200.0)), key="v_roe", label_visibility="collapsed") if t_roe else None
            with f8:
                t_pm = st.toggle("Profit Margin (%)")
                b_pm = get_safe_bounds(df_base['Profit Margin (%)'], -100.0, 100.0)
                range_pm = st.slider("Profit Margin (%)", b_pm[0], max(b_pm[1], b_pm[0]+1), (max(b_pm[0], 5.0), min(b_pm[1], 100.0)), key="v_pm", label_visibility="collapsed") if t_pm else None
            with f9:
                t_de = st.toggle("Debt/Equity Range")
                b_de = get_safe_bounds(df_base['Debt/Equity'], 0.0, 500.0)
                range_de = st.slider("Debt/Equity", b_de[0], max(b_de[1], b_de[0]+5), (max(b_de[0], 0.0), min(b_de[1], 200.0)), key="v_de", label_visibility="collapsed") if t_de else None

            f10, f11, f12, f13, f14 = st.columns(5)
            with f10:
                t_rg = st.toggle("Rev Growth (%)")
                b_rg = get_safe_bounds(df_base['Rev Growth (%)'], -50.0, 200.0)
                range_rg = st.slider("Rev Growth (%)", b_rg[0], max(b_rg[1], b_rg[0]+1), (max(b_rg[0], 5.0), min(b_rg[1], 200.0)), key="v_rg", label_visibility="collapsed") if t_rg else None
            with f11:
                t_dy = st.toggle("Div Yield (%)")
                b_dy = get_safe_bounds(df_base['Div Yield (%)'], 0.0, 20.0)
                range_dy = st.slider("Div Yield (%)", b_dy[0], max(b_dy[1], b_dy[0]+0.5), (max(b_dy[0], 1.0), min(b_dy[1], 20.0)), key="v_dy", label_visibility="collapsed") if t_dy else None
            with f12:
                t_beta = st.toggle("Beta Range")
                b_beta = get_safe_bounds(df_base['Beta'], 0.0, 5.0)
                range_beta = st.slider("Beta", b_beta[0], max(b_beta[1], b_beta[0]+0.1), (max(b_beta[0], 0.0), min(b_beta[1], 1.5)), key="v_beta", label_visibility="collapsed") if t_beta else None
            with f13:
                t_sh = st.toggle("Short % Float")
                b_sh = get_safe_bounds(df_base['Short % Float'], 0.0, 50.0)
                range_sh = st.slider("Short % Float", b_sh[0], max(b_sh[1], b_sh[0]+0.5), (max(b_sh[0], 0.0), min(b_sh[1], 10.0)), key="v_sh", label_visibility="collapsed") if t_sh else None
            with f14:
                t_mc = st.toggle("Market Cap (B)")
                b_mc = get_safe_bounds(df_base['Market Cap (B)'], 0.0, 3000.0)
                range_mc = st.slider("Market Cap (B)", b_mc[0], max(b_mc[1], b_mc[0]+5), (max(b_mc[0], 10.0), min(b_mc[1], 3000.0)), key="v_mc", label_visibility="collapsed") if t_mc else None

            st.write("##### Engine Confidence Limits")
            e1, e2 = st.columns(2)
            with e1:
                t_rmse = st.toggle("Max Tracking Error (RMSE)", value=True)
                b_rmse = get_safe_bounds(df_base['Avg Tracking Error (RMSE)'], 0.0, 1000000.0)
                max_rmse = st.slider("Max Tracking Error (RMSE):", b_rmse[0], max(b_rmse[1], b_rmse[0]+1), float(b_rmse[0] + (b_rmse[1] - b_rmse[0]) * 0.4), label_visibility="collapsed") if t_rmse else float('inf')
            with e2:
                t_cagr = st.toggle("5-Yr CAGR (%) Range", value=True)
                b_cagr = get_safe_bounds(df_base['5-Yr CAGR'], -50.0, 200.0)
                range_cagr = st.slider("5-Yr CAGR (%)", b_cagr[0], max(b_cagr[1], b_cagr[0]+1), (12.0, min(b_cagr[1], 200.0)), label_visibility="collapsed") if t_cagr else None

        filtered_df = df_base.copy()
        if search_ticker: filtered_df = filtered_df[filtered_df['Ticker'].str.contains(search_ticker, case=False, na=False)]
        if t_wl_filter: filtered_df = filtered_df[filtered_df['Ticker'].isin(st.session_state.watchlist)]
        
        if t_pe: filtered_df = filtered_df[filtered_df['Current P/E'].between(range_pe[0], range_pe[1]) | filtered_df['Current P/E'].isna()]
        if t_fpe: filtered_df = filtered_df[filtered_df['Forward P/E'].between(range_fpe[0], range_fpe[1]) | filtered_df['Forward P/E'].isna()]
        if t_avg_pe: filtered_df = filtered_df[filtered_df['5-Yr Avg P/E'].between(range_avg_pe[0], range_avg_pe[1]) | filtered_df['5-Yr Avg P/E'].isna()]
        if t_peg: filtered_df = filtered_df[filtered_df['PEG Ratio'].between(range_peg[0], range_peg[1]) | filtered_df['PEG Ratio'].isna()]
        if t_ps: filtered_df = filtered_df[filtered_df['P/S Ratio'].between(range_ps[0], range_ps[1]) | filtered_df['P/S Ratio'].isna()]
        if t_pb: filtered_df = filtered_df[filtered_df['P/B Ratio'].between(range_pb[0], range_pb[1]) | filtered_df['P/B Ratio'].isna()]
        if t_roe: filtered_df = filtered_df[filtered_df['ROE (%)'].between(range_roe[0], range_roe[1]) | filtered_df['ROE (%)'].isna()]
        if t_pm: filtered_df = filtered_df[filtered_df['Profit Margin (%)'].between(range_pm[0], range_pm[1]) | filtered_df['Profit Margin (%)'].isna()]
        if t_de: filtered_df = filtered_df[filtered_df['Debt/Equity'].between(range_de[0], range_de[1]) | filtered_df['Debt/Equity'].isna()]
        if t_rg: filtered_df = filtered_df[filtered_df['Rev Growth (%)'].between(range_rg[0], range_rg[1]) | filtered_df['Rev Growth (%)'].isna()]
        if t_dy: filtered_df = filtered_df[filtered_df['Div Yield (%)'].between(range_dy[0], range_dy[1]) | filtered_df['Div Yield (%)'].isna()]
        if t_beta: filtered_df = filtered_df[filtered_df['Beta'].between(range_beta[0], range_beta[1]) | filtered_df['Beta'].isna()]
        if t_sh: filtered_df = filtered_df[filtered_df['Short % Float'].between(range_sh[0], range_sh[1]) | filtered_df['Short % Float'].isna()]
        if t_mc: filtered_df = filtered_df[filtered_df['Market Cap (B)'].between(range_mc[0], range_mc[1]) | filtered_df['Market Cap (B)'].isna()]
        if t_rmse: filtered_df = filtered_df[filtered_df['Avg Tracking Error (RMSE)'] <= max_rmse]
        if t_cagr: filtered_df = filtered_df[filtered_df['5-Yr CAGR'].between(range_cagr[0], range_cagr[1])]

        filtered_df = filtered_df.sort_values(by="5-Yr CAGR", ascending=False).reset_index(drop=True)
        
        display_cols = [
            "Ticker", "Company Name", "Industry", "Current Price", "Year 5 Target", "5-Yr CAGR", 
            "Market Cap (B)", "Current P/E", "Forward P/E", "5-Yr Avg P/E", "PEG Ratio", "P/S Ratio", "P/B Ratio",
            "ROE (%)", "Debt/Equity", "Profit Margin (%)", "Rev Growth (%)", 
            "Div Yield (%)", "Avg Tracking Error (RMSE)"
        ]
        
        st.write(f"Showing **{len(filtered_df)}** matching profiles.")
        st.dataframe(filtered_df[display_cols].style.format({
            "Current Price": "${:,.2f}", "Year 5 Target": "${:,.2f}", "5-Yr CAGR": "{:+.1f}%", 
            "Market Cap (B)": "${:.2f}B", "Current P/E": "{:.2f}", "Forward P/E": "{:.2f}", "5-Yr Avg P/E": "{:.2f}", "PEG Ratio": "{:.2f}", 
            "P/S Ratio": "{:.2f}", "P/B Ratio": "{:.2f}", "ROE (%)": "{:.1f}%", 
            "Debt/Equity": "{:.2f}", "Profit Margin (%)": "{:.1f}%", "Rev Growth (%)": "{:.1f}%", 
            "Div Yield (%)": "{:.2f}%", "Avg Tracking Error (RMSE)": "±${:,.0f}"
        }, na_rep="N/A"), use_container_width=True)

# ================= TAB 3: STRATEGY BACKTESTER =================
with tab_backtest:
    st.subheader("📈 Institutional Strategy Backtester")
    st.markdown("Isolates companies outpacing inflation and evaluates their historical risk-adjusted performance using standard quantitative metrics.")
    
    if not os.path.exists(CACHE_FILE) or ('raw_screener_df' in st.session_state and st.session_state.raw_screener_df.empty):
        st.warning("⚠️ **Engine Uninitialized:** Please go to the **S&P 500 Screening Dashboard** tab and run the **Bulk Refresh Matrix** to populate the fundamental database.")
    else:
        df = pd.read_csv(CACHE_FILE)
        
        required_cols = ['Ticker', 'Hist NI CAGR (%)', 'Market Cap (B)', 'Current Price', 'Company Name', 'Industry']
        if not all(col in df.columns for col in required_cols):
            st.error(f"Your cache file is missing required columns. Please rerun the full scan in the Screener tab.")
        else:
            st.write("##### ⚙️ Strategy Parameters")
            c1, c2, c3 = st.columns(3)
            
            backtest_period = c1.selectbox("Backtest Horizon", options=["1y", "3y", "5y"], index=2, key="bt_horizon_select")
            historical_inflation_map = {"1y": 3.1, "3y": 4.5, "5y": 4.2}
            default_inf = historical_inflation_map.get(backtest_period, 4.2)
            
            inf_rate = c2.slider("Auto-Tracked Inflation Rate (%)", min_value=0.0, max_value=15.0, value=default_inf, step=0.1, key="bt_inf_slider") / 100
            hurdle_rate = c3.slider("Real Growth Hurdle (%)", min_value=0.0, max_value=25.0, value=5.0, step=0.5, key="bt_hurdle_slider") / 100
            
            # Fisher Equation with 0 fill to gracefully handle absent metrics
            df['Nominal CAGR'] = df['Hist NI CAGR (%)'].fillna(0) / 100
            df['Real Growth'] = ((1 + df['Nominal CAGR']) / (1 + inf_rate)) - 1
            survivors = df[df['Real Growth'] >= hurdle_rate].copy()
            
            st.metric("Total Stocks Remaining", len(survivors))

            if survivors.empty:
                st.error("No companies met the Real Growth hurdle rate. Lower your parameters.")
            else:
                if st.button("🚀 Run Institutional Simulation", use_container_width=True, key="bt_run_sim_btn"):
                    surviving_tickers = survivors['Ticker'].tolist()
                    
                    with st.spinner(f"Downloading historical pricing and calculating risk metrics for {len(survivors)} tickers..."):
                        # 'auto_adjust=True' accurately models total return (dividends/splits) vs. basic price return
                        prices = yf.download(surviving_tickers + ['SPY'], period=backtest_period, interval="1d", auto_adjust=True, progress=False)['Close']
                        
                        prices = prices.ffill().bfill()
                        valid_tickers = [t for t in surviving_tickers if t in prices.columns]
                        
                        if not valid_tickers:
                            st.error("Insufficient historical price data for the surviving basket.")
                            st.stop()

                        # Construct Weights safely using cached records
                        mcap = survivors.set_index('Ticker').loc[valid_tickers, 'Market Cap (B)'].fillna(1.0).values
                        if np.sum(mcap) <= 0: mcap = np.ones(len(valid_tickers))
                        weights = mcap / np.sum(mcap)
                        
                        # --- Treemap Allocation Grid ---
                        st.subheader("Day-One Portfolio Allocation Grid")
                        survivors_valid = survivors[survivors['Ticker'].isin(valid_tickers)].copy()
                        survivors_valid['Weight'] = weights
                        survivors_valid['Real Growth (%)'] = (survivors_valid['Real Growth'] * 100).round(2)
                        
                        fig_tree = px.treemap(
                            survivors_valid, path=[px.Constant("Custom Index"), 'Ticker'], values='Weight',
                            color='Real Growth (%)', color_continuous_scale='Greens', hover_data=['Market Cap (B)']
                        )
                        fig_tree.update_layout(margin=dict(t=20, l=10, r=10, b=10), height=400)
                        st.plotly_chart(fig_tree, use_container_width=True)

                        # --- Daily Returns ---
                        daily_returns = prices[valid_tickers].pct_change().dropna()
                        strat_returns = (daily_returns * weights).sum(axis=1)
                        
                        if 'SPY' in prices.columns:
                            spy_returns = prices['SPY'].pct_change().dropna()
                            
                            # --- Quantitative Performance Metrics ---
                            TRADING_DAYS = 252
                            RISK_FREE_RATE = 0.04
                            exact_years = len(daily_returns) / TRADING_DAYS
                            if exact_years <= 0: exact_years = 1.0
                            
                            # Cumulative Performance
                            strat_cumprod = (1 + strat_returns).cumprod()
                            spy_cumprod = (1 + spy_returns).cumprod()
                            
                            # CAGR
                            strat_cagr = (strat_cumprod.iloc[-1] ** (1 / exact_years)) - 1 if not strat_cumprod.empty else 0
                            spy_cagr = (spy_cumprod.iloc[-1] ** (1 / exact_years)) - 1 if not spy_cumprod.empty else 0
                            
                            # Annualized Volatility
                            strat_vol = strat_returns.std() * np.sqrt(TRADING_DAYS)
                            spy_vol = spy_returns.std() * np.sqrt(TRADING_DAYS)
                            
                            # Sharpe Ratio
                            strat_sharpe = (strat_cagr - RISK_FREE_RATE) / strat_vol if strat_vol > 0 else 0
                            spy_sharpe = (spy_cagr - RISK_FREE_RATE) / spy_vol if spy_vol > 0 else 0
                            
                            # Maximum Drawdown
                            strat_rolling_max = strat_cumprod.cummax()
                            strat_drawdown = (strat_cumprod / strat_rolling_max) - 1
                            strat_max_dd = strat_drawdown.min()
                            
                            spy_rolling_max = spy_cumprod.cummax()
                            spy_drawdown = (spy_cumprod / spy_rolling_max) - 1
                            spy_max_dd = spy_drawdown.min()
                            
                            # --- Display Metrics ---
                            st.write("---")
                            st.subheader("Quantitative Performance Analysis")
                            
                            m1, m2, m3, m4 = st.columns(4)
                            m1.metric("Strategy CAGR", f"{strat_cagr * 100:.2f}%", f"{(strat_cagr - spy_cagr) * 100:+.2f}% vs SPY")
                            m2.metric("Sharpe Ratio", f"{strat_sharpe:.2f}", f"{strat_sharpe - spy_sharpe:+.2f} vs SPY")
                            m3.metric("Annualized Volatility", f"{strat_vol * 100:.2f}%", delta=None)
                            m4.metric("Max Drawdown", f"{strat_max_dd * 100:.2f}%", delta=None, delta_color="inverse")

                            # --- Line Chart ---
                            plot_df = pd.DataFrame({
                                "Custom Strategy": (strat_cumprod - 1) * 100,
                                "SPY Benchmark": (spy_cumprod - 1) * 100
                            })

                            fig_line = px.line(plot_df, labels={'value': 'Cumulative Return (%)', 'Date': 'Date'}, color_discrete_map={"Custom Strategy": "#00FF00", "SPY Benchmark": "#808080"})
                            fig_line.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1), margin=dict(t=20, l=10, r=10, b=10), xaxis_title="", yaxis_title="Return (%)", height=350)
                            st.plotly_chart(fig_line, use_container_width=True)
                        else:
                            st.error("Failed to load SPY benchmark data.")

                    with st.expander("View Surviving Constituents"):
                        st.dataframe(
                            survivors_valid[['Ticker', 'Company Name', 'Industry', 'Market Cap (B)', 'Hist NI CAGR (%)', 'Real Growth (%)', 'Weight']].sort_values('Weight', ascending=False).style.format({
                                'Hist NI CAGR (%)': '{:.2f}%', 'Real Growth (%)': '{:.2f}%', 'Weight': '{:.2%}', 'Market Cap (B)': '${:.2f}B'
                            }), use_container_width=True
                        )
