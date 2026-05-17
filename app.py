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
TICKER_CACHE_DIR = "ticker_cache"
os.makedirs(TICKER_CACHE_DIR, exist_ok=True)

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
    if 'Shares Outstanding' in q_proj: q_proj['Shares Outstanding'] = np.maximum(1, q_proj['Shares Outstanding'])
    
    return q_proj, rmse_tot, metric_results

# --- BACKGROUND WORKER FOR SCREENER (ANNUAL DATA ONLY) ---
def process_single_screener_stock(ticker):
    try:
        norm_df, current_p, _, _ = fetch_financial_data(ticker, force_deep_dive=False)
        if norm_df.empty or len(norm_df) < 3: return None

        x_hist, x_fut = np.arange(len(norm_df)).reshape(-1, 1), np.arange(len(norm_df), len(norm_df) + 5).reshape(-1, 1)

        proj, total_rmse, _ = run_projections(norm_df, x_hist, x_fut)
        eps_y1 = proj['Net Income'][0] / max(1, proj['Shares Outstanding'][0])
        
        info = yf.Ticker(ticker).info
        try: f_eps = info.get('forwardEps', np.nan)
        except: f_eps = np.nan

        max_hist_ni = max(1, norm_df['Net Income'].max())
        is_hallucinating = False
        
        if pd.notna(f_eps) and f_eps > 0 and eps_y1 > 0 and (eps_y1 > (f_eps * 2.5) or eps_y1 < (f_eps * 0.4)): is_hallucinating = True
        if proj['Net Income'][-1] > (max_hist_ni * 8.0): is_hallucinating = True

        if is_hallucinating:
            proj, total_rmse, _ = run_projections(norm_df, x_hist, x_fut, force_conservative=True)

        eps_y5 = proj['Net Income'][-1] / max(1, proj['Shares Outstanding'][-1])
        if current_p <= 0: return None

        return {
            "Ticker": ticker, 
            "Current Price": round(current_p, 2), 
            "Year 5 EPS": eps_y5, 
            "Avg Tracking Error (RMSE)": round(total_rmse / len(drivers), 2),
            "Market Cap (B)": info.get('marketCap', np.nan) / 1e9 if pd.notna(info.get('marketCap')) else np.nan,
            "Rev Growth (%)": (info.get('revenueGrowth', np.nan) * 100) if pd.notna(info.get('revenueGrowth')) else np.nan,
            "Current P/E": info.get('trailingPE', np.nan),
            "Forward P/E": info.get('forwardPE', np.nan),
            "PEG Ratio": info.get('pegRatio', np.nan),
            "P/B Ratio": info.get('priceToBook', np.nan),
            "P/S Ratio": info.get('priceToSalesTrailing12Months', np.nan),
            "ROE (%)": (info.get('returnOnEquity', np.nan) * 100) if pd.notna(info.get('returnOnEquity')) else np.nan,
            "ROA (%)": (info.get('returnOnAssets', np.nan) * 100) if pd.notna(info.get('returnOnAssets')) else np.nan,
            "Debt/Equity": info.get('debtToEquity', np.nan),
            "Gross Margin (%)": (info.get('grossMargins', np.nan) * 100) if pd.notna(info.get('grossMargins')) else np.nan,
            "Profit Margin (%)": (info.get('profitMargins', np.nan) * 100) if pd.notna(info.get('profitMargins')) else np.nan,
            "Div Yield (%)": (info.get('dividendYield', np.nan) * 100) if pd.notna(info.get('dividendYield')) else 0.0,
            "Beta": info.get('beta', np.nan),
            "Short % Float": (info.get('shortPercentOfFloat', np.nan) * 100) if pd.notna(info.get('shortPercentOfFloat')) else np.nan
        }
    except: return None

# --- UI APP TABS ---
tab_single, tab_screener = st.tabs(["📊 Single Ticker Forecast", "🔍 S&P 500 Screening Dashboard"])

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
                norm_df, current_price, analyst_target, data_source = fetch_financial_data(ticker_input, force_deep_dive=True, force_refresh=force_refresh_tab1)
                if norm_df.empty: st.error(f"No financial data found for {ticker_input}."); st.stop()
                st.session_state.update({'norm_df': norm_df, 'current_price': current_price, 'analyst_target_tab1': analyst_target, 'ticker_analyzed': ticker_input, 'actual_lookback': lookback_input, 'data_source': data_source})

    if 'norm_df' in st.session_state and st.session_state.ticker_analyzed == ticker_input:
        norm_df = st.session_state.norm_df
        current_price = st.session_state.current_price
        df_reg = norm_df.tail(len(norm_df) if st.session_state.actual_lookback == 0 else st.session_state.actual_lookback)
        
        st.markdown(f"**Data Depth Indicator:** :{'green' if len(df_reg) >= 8 else 'red'}[{len(df_reg)} Quarters Loaded] via {st.session_state.data_source} *(Note: ARIMA/Holt-Winters require 6-8 minimum)*")
        if not api_key: st.info("💡 **Want deeper data?** Add a free Alpha Vantage API key to your Streamlit Secrets.")
        
        x_hist, x_fut = np.arange(len(df_reg)).reshape(-1, 1), np.arange(len(df_reg), len(df_reg) + 20).reshape(-1, 1) 
        
        overrides = {m: st.session_state[f"ov_{m}"] for m in drivers}
        proj_quarterly_data, _, metric_results = run_projections(df_reg, x_hist, x_fut, overrides=overrides)

        with st.expander("⚙️ Advanced: Override Projection Models & Explanations"):
            st.markdown("""
            **Model Selection Guide (When to override the Auto-Picker):**
            * **Linear:** Best for mature, stable value stocks with consistent trajectories (e.g., KO, JNJ).
            * **Quadratic:** Best for identifying accelerating hyper-growth or cyclical supercycles (e.g., NVDA, PLTR).
            * **Logarithmic:** Best for growth companies that are maturing and hitting market saturation (e.g., NFLX, PYPL).
            * **Derivative:** Best for weighting recent momentum and sudden earnings trajectory shifts over long-term history.
            * **Holt-Winters:** Best for tracking highly seasonal businesses with predictable intra-year swings (e.g., Retailers like TGT, Travel like DAL).
            * **ARIMA:** Best for complex, macro-driven trajectories that do not fit clean geometric curves.
            """)
            st.write("---")
            st.button("🔄 Reset all to Auto", on_click=reset_overrides, key="reset_tab1")
            for metric in drivers:
                res = metric_results[metric]
                st.selectbox(metric, options=model_choices, format_func=lambda o, r=res: f"Auto ({r.get('AutoChoice', 'Linear')})" if o == "Auto" else (f"{o} (RMSE: ±${int(r[o]['rmse']):,})" if r.get(o) and r[o].get('rmse', float('inf')) != float('inf') else f"{o} (N/A)"), key=f"ov_{metric}")

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
        md = f"| Metric | {' | '.join(hist_labels + proj_labels)} |\n|---{'|---'*len(hist_labels + proj_labels)}|\n"
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
    
    if os.path.exists(CACHE_FILE) and 'raw_screener_df' not in st.session_state: st.session_state.raw_screener_df = pd.read_csv(CACHE_FILE)
    st.markdown(f"**Data Last Loaded:** `{pd.to_datetime(os.path.getmtime(CACHE_FILE), unit='s').strftime('%B %d, %Y at %I:%M %p') if os.path.exists(CACHE_FILE) else 'Never'}`")
    
    st.write("### ⚡ Data Refresh Controls")
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.write("Update the entire S&P 500 matrix (takes ~1 minute).")
        if st.button("🔄 Force Refresh All (Annual API Scan)", use_container_width=True):
            with st.spinner("Fetching S&P 500 Roster & Executing Fast Scan..."):
                try: tickers = [t.replace('.', '-') for t in pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options={"User-Agent": "Mozilla/5.0"})[0]['Symbol'].tolist()]
                except Exception as e: st.error(f"Failed to fetch stock index list: {e}"); st.stop()

                progress_bar, status_text, screened_results, completed = st.progress(0), st.empty(), [], 0
                with ThreadPoolExecutor(max_workers=15) as executor:
                    for future in as_completed({executor.submit(process_single_screener_stock, t): t for t in tickers}):
                        completed += 1
                        if res := future.result(): screened_results.append(res)
                        if completed % 15 == 0 or completed == len(tickers):
                            progress_bar.progress(completed / len(tickers))
                            status_text.write(f"Scanned {completed}/{len(tickers)}...")

                status_text.success(f"Matrix complete! Modeled {len(screened_results)} companies.")
                raw_df = pd.DataFrame(screened_results)
                if 'Analyst Target' in raw_df.columns: raw_df = raw_df.drop(columns=['Analyst Target'])
                raw_df.to_csv(CACHE_FILE, index=False)
                st.session_state.raw_screener_df = raw_df
                st.rerun()

    with c2:
        st.write("Targeted refresh for a single stock.")
        refresh_tick = st.text_input("Ticker", placeholder="e.g. NVDA", label_visibility="collapsed").upper().strip()

    with c3:
        st.write("") 
        if st.button("Targeted Update", use_container_width=True):
            if refresh_tick and 'raw_screener_df' in st.session_state:
                with st.spinner(f"Recalculating {refresh_tick}..."):
                    if res := process_single_screener_stock(refresh_tick):
                        df_cache = st.session_state.raw_screener_df
                        if 'Analyst Target' in res: del res['Analyst Target']
                        if refresh_tick in df_cache['Ticker'].values:
                            for k, v in res.items(): df_cache.at[df_cache.index[df_cache['Ticker'] == refresh_tick][0], k] = v
                        else: df_cache = pd.concat([df_cache, pd.DataFrame([res])], ignore_index=True)
                        if 'Analyst Target' in df_cache.columns: df_cache = df_cache.drop(columns=['Analyst Target'])
                        df_cache.to_csv(CACHE_FILE, index=False)
                        st.session_state.raw_screener_df = df_cache
                        st.rerun()
                    else: st.error(f"Could not calculate projections for {refresh_tick}. Requires 3+ years of public data.")
            elif not refresh_tick: st.warning("Please enter a ticker symbol.")
            else: st.error("Cache is empty. Run a full scan first to build the database.")

    if 'raw_screener_df' in st.session_state:
        df_base = st.session_state.raw_screener_df.copy()
        
        # Patch for older caches to prevent NA crash
        expected_cols = [
            "Market Cap (B)", "Rev Growth (%)", "Current P/E", "Forward P/E", 
            "PEG Ratio", "P/B Ratio", "P/S Ratio", "ROE (%)", "ROA (%)", 
            "Debt/Equity", "Gross Margin (%)", "Profit Margin (%)", 
            "Div Yield (%)", "Beta", "Short % Float"
        ]
        for col in expected_cols:
            if col not in df_base.columns: df_base[col] = np.nan
            
        st.write("---")
        
        with st.expander("🔬 Deep Toggle Filters", expanded=True):
            col_search, col_pe = st.columns([1, 2])
            search_ticker = col_search.text_input("🔍 Search Ticker:", "").upper()
            screener_pe = col_pe.number_input("Universal Target P/E Multiple for Model:", value=25.0, step=1.0, key="pe_screener")
            
            st.write("##### Enable specific filters to constrain the matrix:")
            
            # Row 1: Valuations
            f1, f2, f3, f4 = st.columns(4)
            with f1:
                t_pe = st.toggle("Max Current P/E")
                if t_pe: max_pe_filter = st.number_input("Value:", value=50.0, key="v_pe")
            with f2:
                t_fpe = st.toggle("Max Forward P/E")
                if t_fpe: max_fpe_filter = st.number_input("Value:", value=35.0, key="v_fpe")
            with f3:
                t_peg = st.toggle("Max PEG Ratio")
                if t_peg: max_peg_filter = st.number_input("Value:", value=3.0, key="v_peg")
            with f4:
                t_ps = st.toggle("Max P/S Ratio")
                if t_ps: max_ps_filter = st.number_input("Value:", value=10.0, key="v_ps")

            # Row 2: Profitability & Health
            f5, f6, f7, f8 = st.columns(4)
            with f5:
                t_pb = st.toggle("Max P/B Ratio")
                if t_pb: max_pb_filter = st.number_input("Value:", value=15.0, key="v_pb")
            with f6:
                t_roe = st.toggle("Min ROE (%)")
                if t_roe: min_roe_filter = st.number_input("Value:", value=10.0, key="v_roe")
            with f7:
                t_pm = st.toggle("Min Profit Margin (%)")
                if t_pm: min_pm_filter = st.number_input("Value:", value=5.0, key="v_pm")
            with f8:
                t_de = st.toggle("Max Debt/Equity")
                if t_de: max_de_filter = st.number_input("Value:", value=200.0, key="v_de")

            # Row 3: Momentum & Yield
            f9, f10, f11, f12 = st.columns(4)
            with f9:
                t_rg = st.toggle("Min Rev Growth (%)")
                if t_rg: min_rg_filter = st.number_input("Value:", value=5.0, key="v_rg")
            with f10:
                t_dy = st.toggle("Min Div Yield (%)")
                if t_dy: min_dy_filter = st.number_input("Value:", value=1.0, key="v_dy")
            with f11:
                t_beta = st.toggle("Max Beta")
                if t_beta: max_beta_filter = st.number_input("Value:", value=1.5, key="v_beta")
            with f12:
                t_sh = st.toggle("Max Short %")
                if t_sh: max_sh_filter = st.number_input("Value:", value=10.0, key="v_sh")

            st.write("##### Engine Confidence Limits")
            e1, e2 = st.columns(2)
            with e1:
                t_rmse = st.toggle("Max Tracking Error (RMSE)", value=True)
                if t_rmse: max_rmse = st.slider("Max Tracking Error (RMSE):", float(df_base['Avg Tracking Error (RMSE)'].min()), float(df_base['Avg Tracking Error (RMSE)'].max()), float(df_base['Avg Tracking Error (RMSE)'].max() * 0.4), label_visibility="collapsed")
            with e2:
                t_cagr = st.toggle("Min 5-Yr CAGR (%)", value=True)
                if t_cagr: min_cagr = st.slider("Min Acceptable 5-Yr CAGR (%):", float(df_base['5-Yr CAGR'].min()) if '5-Yr CAGR' in df_base.columns else -50.0, 100.0, 12.0, label_visibility="collapsed")

        if search_ticker: df_base = df_base[df_base['Ticker'].str.contains(search_ticker, case=False, na=False)]
        
        # Safe CAGR Calculation for Negative Projections
        df_base['Year 5 Target'] = df_base['Year 5 EPS'] * screener_pe
        df_base['5-Yr CAGR'] = np.where(
            df_base['Year 5 Target'] > 0,
            ((df_base['Year 5 Target'] / df_base['Current Price']) ** (1/5) - 1) * 100,
            -100.0 
        )
        
        filtered_df = df_base.copy()
        if t_pe: filtered_df = filtered_df[(filtered_df['Current P/E'] <= max_pe_filter) & pd.notna(filtered_df['Current P/E'])]
        if t_fpe: filtered_df = filtered_df[(filtered_df['Forward P/E'] <= max_fpe_filter) & pd.notna(filtered_df['Forward P/E'])]
        if t_peg: filtered_df = filtered_df[(filtered_df['PEG Ratio'] <= max_peg_filter) & pd.notna(filtered_df['PEG Ratio'])]
        if t_ps: filtered_df = filtered_df[(filtered_df['P/S Ratio'] <= max_ps_filter) & pd.notna(filtered_df['P/S Ratio'])]
        if t_pb: filtered_df = filtered_df[(filtered_df['P/B Ratio'] <= max_pb_filter) & pd.notna(filtered_df['P/B Ratio'])]
        if t_roe: filtered_df = filtered_df[(filtered_df['ROE (%)'] >= min_roe_filter) & pd.notna(filtered_df['ROE (%)'])]
        if t_pm: filtered_df = filtered_df[(filtered_df['Profit Margin (%)'] >= min_pm_filter) & pd.notna(filtered_df['Profit Margin (%)'])]
        if t_de: filtered_df = filtered_df[(filtered_df['Debt/Equity'] <= max_de_filter) & pd.notna(filtered_df['Debt/Equity'])]
        if t_rg: filtered_df = filtered_df[(filtered_df['Rev Growth (%)'] >= min_rg_filter) & pd.notna(filtered_df['Rev Growth (%)'])]
        if t_dy: filtered_df = filtered_df[(filtered_df['Div Yield (%)'] >= min_dy_filter) & pd.notna(filtered_df['Div Yield (%)'])]
        if t_beta: filtered_df = filtered_df[(filtered_df['Beta'] <= max_beta_filter) & pd.notna(filtered_df['Beta'])]
        if t_mc: filtered_df = filtered_df[(filtered_df['Market Cap (B)'] >= min_mc_filter) & pd.notna(filtered_df['Market Cap (B)'])]
        if t_sh: filtered_df = filtered_df[(filtered_df['Short % Float'] <= max_sh_filter) & pd.notna(filtered_df['Short % Float'])]
        
        if t_rmse: filtered_df = filtered_df[filtered_df['Avg Tracking Error (RMSE)'] <= max_rmse]
        if t_cagr: filtered_df = filtered_df[filtered_df['5-Yr CAGR'] >= min_cagr]

        filtered_df = filtered_df.sort_values(by="5-Yr CAGR", ascending=False).reset_index(drop=True)
        
        # Updated to include Forward P/E and Debt/Equity
        display_cols = [
            "Ticker", "Current Price", "Year 5 Target", "5-Yr CAGR", 
            "Current P/E", "Forward P/E", "PEG Ratio", "P/S Ratio", "P/B Ratio",
            "ROE (%)", "Debt/Equity", "Profit Margin (%)", "Rev Growth (%)", 
            "Div Yield (%)", "Avg Tracking Error (RMSE)"
        ]
        
        st.write(f"Showing **{len(filtered_df)}** matching profiles.")
        st.dataframe(filtered_df[display_cols].style.format({
            "Current Price": "${:,.2f}", "Year 5 Target": "${:,.2f}", "5-Yr CAGR": "{:+.1f}%", 
            "Current P/E": "{:.2f}", "Forward P/E": "{:.2f}", "PEG Ratio": "{:.2f}", 
            "P/S Ratio": "{:.2f}", "P/B Ratio": "{:.2f}", "ROE (%)": "{:.1f}%", 
            "Debt/Equity": "{:.2f}", "Profit Margin (%)": "{:.1f}%", "Rev Growth (%)": "{:.1f}%", 
            "Div Yield (%)": "{:.2f}%", "Avg Tracking Error (RMSE)": "±${:,.0f}"
        }, na_rep="N/A"), use_container_width=True)
