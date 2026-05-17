import sys
import os
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
drivers = ['Total Revenue', 'Cost Of Revenue', 'Operating Expense', 'Non-Op & Taxes', 'Shares Outstanding']
model_choices = ["Auto", "Linear", "Quadratic", "Derivative", "Logarithmic", "Holt-Winters", "ARIMA"]
display_order = ['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income', 'Non-Op & Taxes', 'Net Income', 'Shares Outstanding', 'EPS']

# --- DECOUPLED DATA PIPELINE ---
def fetch_financial_data(ticker, force_deep_dive=False):
    stock = yf.Ticker(ticker)
    use_yf = True
    df_final = pd.DataFrame()
    data_source = "None"
    
    hist_1d = stock.history(period="1d")
    current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0
    analyst_target = stock.info.get('targetMeanPrice', np.nan)

    if force_deep_dive and api_key:
        try:
            r = requests.get(f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker}&apikey={api_key}').json()
            if 'quarterlyReports' in r:
                use_yf = False
                df_av = pd.DataFrame(r['quarterlyReports'])
                df_av['fiscalDateEnding'] = pd.to_datetime(df_av['fiscalDateEnding'])
                df_av = df_av.set_index('fiscalDateEnding').sort_index()
                for col in ['totalRevenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome']:
                    df_av[col] = pd.to_numeric(df_av[col], errors='coerce').fillna(0) / 1000
                rev, cogs, gp, op_inc, ni = df_av['totalRevenue'], df_av['costOfRevenue'], df_av['grossProfit'], df_av['operatingIncome'], df_av['netIncome']
                shares = pd.Series(stock.info.get('sharesOutstanding', 100000) / 1000, index=df_av.index)
                df_final = pd.DataFrame({'Total Revenue': rev, 'Cost Of Revenue': cogs, 'Gross Profit': gp, 'Operating Expense': gp - op_inc, 'Operating Income': op_inc, 'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 'Shares Outstanding': shares, 'EPS': ni / shares}).dropna()
                data_source = "Alpha Vantage (Deep History)"
        except: pass

    if use_yf:
        try:
            df = stock.quarterly_income_stmt.T
            if len(df) < 8:
                df_alt = stock.quarterly_financials.T
                if len(df_alt) > len(df): df = df_alt
            if len(df) < 8:
                try:
                    df_get = stock.get_income_stmt(freq="quarterly").T
                    if len(df_get) > len(df): df = df_get
                except: pass
            if not df.empty and 'Total Revenue' in df.columns:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()
                df_raw = df / 1000
                rev = df_raw['Total Revenue']
                gp = df_raw['Gross Profit'] if 'Gross Profit' in df_raw.columns else rev
                op_inc = df_raw['Operating Income'] if 'Operating Income' in df_raw.columns else gp
                ni = df_raw['Net Income'] if 'Net Income' in df_raw.columns else op_inc
                if 'Diluted Average Shares' in df_raw.columns: shares = df_raw['Diluted Average Shares']
                elif 'Basic Average Shares' in df_raw.columns: shares = df_raw['Basic Average Shares']
                else: shares = pd.Series(1, index=df_raw.index)
                df_final = pd.DataFrame({'Total Revenue': rev, 'Cost Of Revenue': rev - gp, 'Gross Profit': gp, 'Operating Expense': gp - op_inc, 'Operating Income': op_inc, 'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 'Shares Outstanding': shares, 'EPS': ni / shares}).dropna()
                data_source = "Yahoo Finance (Standard)"
        except: pass
        
    return df_final, current_price, analyst_target, data_source

# --- CORE MATH ENGINE WITH REJECTION RULES ---
def calculate_metric_models(y_in, x_hist, x_fut, metric_name):
    y = np.asarray(y_in, dtype=float)
    n = len(y)
    results = {}
    floor_val = max(1, y[-1] * 0.5) if metric_name == 'Shares Outstanding' and n > 0 else 0

    # 1. Base Linear
    lin_model = LinearRegression().fit(x_hist, y)
    pred_lin_fut = np.maximum(floor_val, lin_model.predict(x_fut))
    rmse_lin = np.sqrt(mean_squared_error(y, lin_model.predict(x_hist)))
    results['Linear'] = {'forecast': pred_lin_fut, 'rmse': rmse_lin}

    # 2. Quadratic (DISABLED for < 6 data points to prevent parabolic hyper-inflation)
    if n >= 6:
        poly_features = np.column_stack((x_hist, x_hist**2))
        poly_model = LinearRegression().fit(poly_features, y)
        poly_fut_features = np.column_stack((x_fut, x_fut**2))
        pred_poly_fut = np.maximum(floor_val, poly_model.predict(poly_fut_features))
        rmse_poly = np.sqrt(mean_squared_error(y, poly_model.predict(poly_features)))
        results['Quadratic'] = {'forecast': pred_poly_fut, 'rmse': rmse_poly}
    else:
        results['Quadratic'] = {'forecast': None, 'rmse': float('inf')}

    # 3. Derivative (Requires 5 points to build 4 stable differentials)
    if n >= 5:
        diffs = np.diff(y)
        x_diff = np.arange(len(diffs)).reshape(-1, 1)
        deriv_model = LinearRegression().fit(x_diff, diffs)
        x_diff_fut = np.arange(len(diffs), len(diffs) + 20).reshape(-1, 1)
        fut_diffs = deriv_model.predict(x_diff_fut)
        forecast_deriv, current_val = [], y[-1]
        for fd in fut_diffs:
            current_val = max(floor_val, current_val + fd)
            forecast_deriv.append(current_val)
        rmse_deriv = np.sqrt(mean_squared_error(y[1:], y[:-1] + deriv_model.predict(x_diff)))
        results['Derivative'] = {'forecast': np.array(forecast_deriv), 'rmse': rmse_deriv}
    else:
        results['Derivative'] = {'forecast': None, 'rmse': float('inf')}

    # 4. Logarithmic
    x_log_hist = np.log(x_hist + 1)
    x_log_fut = np.log(x_fut + 1)
    log_model = LinearRegression().fit(x_log_hist, y)
    pred_log_fut = np.maximum(floor_val, log_model.predict(x_log_fut))
    rmse_log = np.sqrt(mean_squared_error(y, log_model.predict(x_log_hist)))
    results['Logarithmic'] = {'forecast': pred_log_fut, 'rmse': rmse_log}

    # 5. Holt-Winters
    if n >= 8: 
        try:
            hw_model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=4, initialization_method="heuristic").fit()
            results['Holt-Winters'] = {'forecast': np.maximum(floor_val, hw_model.forecast(20)), 'rmse': np.sqrt(mean_squared_error(y, hw_model.fittedvalues))}
        except Exception: 
            results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}
    else: results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}

    # 6. ARIMA
    if n >= 6:
        try:
            arima_model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
            results['ARIMA'] = {'forecast': np.maximum(floor_val, arima_model.forecast(20)), 'rmse': np.sqrt(mean_squared_error(y, arima_model.predict(start=0, end=len(y)-1)))}
        except Exception: 
            results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}
    else: results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}

    if metric_name == 'Non-Op & Taxes':
        upper_bound = max(0, np.max(y)) * 1.5
        for name in results:
            if results[name]['forecast'] is not None:
                results[name]['forecast'] = np.minimum(results[name]['forecast'], upper_bound)

    valid_models = [ (name, data['rmse'], data['forecast']) for name, data in results.items() if data['forecast'] is not None and data['rmse'] != float('inf') ]
    valid_models.sort(key=lambda x: x[1]) 
    
    current_val = y[-1] if len(y) > 0 else 0
    safe_models = []
    
    for name, rmse, forecast in valid_models:
        is_valid = True
        
        if current_val > 0:
            if metric_name == 'Total Revenue' and forecast[-1] > (current_val * 5.0):
                is_valid = False 
            elif metric_name in ['Cost Of Revenue', 'Operating Expense'] and forecast[-1] < (current_val * 0.2):
                is_valid = False 
            elif metric_name == 'Shares Outstanding' and forecast[-1] < (current_val * 0.5):
                is_valid = False 
                
        if name in ["Quadratic", "Derivative", "ARIMA"] and current_val > 0:
            if forecast[-1] > (current_val * 3.5):
                is_valid = False
                
        if is_valid:
            safe_models.append(name)

    # Fallback to pure stability if all models flag as dangerously volatile
    if safe_models:
        auto_choice = safe_models[0]
    else:
        if "Logarithmic" in [m[0] for m in valid_models]: auto_choice = "Logarithmic"
        else: auto_choice = "Linear"

    results['AutoChoice'] = auto_choice
    return results

# --- BACKGROUND WORKER FOR SCREENER ---
def process_single_screener_stock(ticker):
    norm_df, current_p, analyst_target, _ = fetch_financial_data(ticker, force_deep_dive=False)
    
    # Lowered threshold to 4 to prevent Yahoo Finance free-tier rejections
    if norm_df.empty or len(norm_df) < 4: return None

    x_hist, x_fut = np.arange(len(norm_df)).reshape(-1, 1), np.arange(len(norm_df), len(norm_df) + 20).reshape(-1, 1)
    total_rmse = 0
    q_proj = {}

    for metric in drivers:
        y = norm_df[metric].values
        res = calculate_metric_models(y, x_hist, x_fut, metric)
        winning_model = res['AutoChoice']
        total_rmse += res[winning_model]['rmse']
        q_proj[metric] = res[winning_model]['forecast']

    q_proj['Gross Profit'] = q_proj['Total Revenue'] - q_proj['Cost Of Revenue']
    q_proj['Operating Income'] = q_proj['Gross Profit'] - q_proj['Operating Expense']
    q_proj['Net Income'] = q_proj['Operating Income'] + q_proj['Non-Op & Taxes']

    eps_5y_avg = np.sum(q_proj['Net Income'][16:20]) / np.mean(q_proj['Shares Outstanding'][16:20])
    
    if current_p <= 0 or eps_5y_avg <= 0: return None

    return {
        "Ticker": ticker, 
        "Current Price": round(current_p, 2), 
        "Analyst Target": round(analyst_target, 2) if pd.notna(analyst_target) else None,
        "Year 5 EPS": eps_5y_avg, 
        "Avg Tracking Error (RMSE)": round(total_rmse / len(drivers), 2)
    }

# --- UI APP TABS ---
tab_single, tab_screener = st.tabs(["📊 Single Ticker Forecast", "🔍 S&P 500 Screening Dashboard"])

# ================= TAB 1: SINGLE FORECASTER =================
with tab_single:
    for m in drivers:
        if f"ov_{m}" not in st.session_state: st.session_state[f"ov_{m}"] = "Auto"

    def reset_overrides():
        for m in drivers: st.session_state[f"ov_{m}"] = "Auto"

    col1, col2, col3 = st.columns([2, 2, 1])
    with col1: ticker_input = st.text_input("Enter Ticker:", "PLTR", key="single_tick").upper()
    with col2: lookback_input = st.number_input("Quarters back (0 = All):", min_value=0, max_value=40, value=0, step=1, key="single_lb")
    with col3:
        st.write(""); st.write("")
        analyze_btn = st.button("Fetch & Analyze", key="single_btn", use_container_width=True)

    if analyze_btn:
        with st.spinner(f"Executing Deep Data Mine for {ticker_input}..."):
            norm_df, current_price, analyst_target, data_source = fetch_financial_data(ticker_input, force_deep_dive=True)
            if norm_df.empty:
                st.error(f"No financial data found for {ticker_input}.")
                st.stop()
            else:
                st.session_state.norm_df = norm_df
                st.session_state.current_price = current_price
                st.session_state.analyst_target_tab1 = analyst_target
                st.session_state.ticker_analyzed = ticker_input
                st.session_state.actual_lookback = lookback_input
                st.session_state.data_source = data_source

    if 'norm_df' in st.session_state and st.session_state.ticker_analyzed == ticker_input:
        norm_df = st.session_state.norm_df
        current_price = st.session_state.current_price
        df_reg = norm_df.tail(len(norm_df) if st.session_state.actual_lookback == 0 else st.session_state.actual_lookback)
        
        depth_color = "green" if len(df_reg) >= 8 else "red"
        st.markdown(f"**Data Depth Indicator:** :{depth_color}[{len(df_reg)} Quarters Loaded] via {st.session_state.data_source} *(Note: ARIMA/Holt-Winters require 6-8 minimum)*")
        if not api_key: st.info("💡 **Want deeper data?** Add a free Alpha Vantage API key to your Streamlit Secrets.")
        
        x_historical, x_future = np.arange(len(df_reg)).reshape(-1, 1), np.arange(len(df_reg), len(df_reg) + 20).reshape(-1, 1) 
        
        metric_results = {}
        for metric in drivers:
            metric_results[metric] = calculate_metric_models(df_reg[metric].values, x_historical, x_future, metric)

        with st.expander("⚙️ Advanced: Override Projection Models"):
            st.button("🔄 Reset all to Auto", on_click=reset_overrides, key="reset_tab1")
            for metric in drivers:
                res = metric_results[metric]
                st.selectbox(metric, options=model_choices, format_func=lambda o, r=res: f"Auto ({r['AutoChoice']})" if o == "Auto" else (f"{o} (RMSE: ±${int(r[o]['rmse']):,})" if r[o]['rmse'] != float('inf') else f"{o} (N/A)"), key=f"ov_{metric}")

        proj_annual_data, proj_quarterly_data, errors, methods = {}, {}, {}, {}
        for metric in drivers:
            res = metric_results[metric]
            act = res['AutoChoice'] if st.session_state[f"ov_{metric}"] == "Auto" else st.session_state[f"ov_{metric}"]
            if res[act]['forecast'] is None: act = "Linear"
            
            proj_quarterly_data[metric] = res[act]['forecast']
            errors[metric], methods[metric] = res[act]['rmse'], f"Auto: {act}" if st.session_state[f"ov_{metric}"] == "Auto" else f"Manual: {act}"

        proj_quarterly_data['Gross Profit'] = proj_quarterly_data['Total Revenue'] - proj_quarterly_data['Cost Of Revenue']
        proj_quarterly_data['Operating Income'] = proj_quarterly_data['Gross Profit'] - proj_quarterly_data['Operating Expense']
        proj_quarterly_data['Net Income'] = proj_quarterly_data['Operating Income'] + proj_quarterly_data['Non-Op & Taxes']

        for metric in display_order:
            if metric == 'Shares Outstanding': proj_annual_data[metric] = [np.mean(proj_quarterly_data[metric][i*4:(i+1)*4]) for i in range(5)]
            elif metric == 'EPS': pass 
            else: proj_annual_data[metric] = [np.sum(proj_quarterly_data[metric][i*4:(i+1)*4]) for i in range(5)]
        proj_annual_data['EPS'] = (np.array(proj_annual_data['Net Income']) / np.array(proj_annual_data['Shares Outstanding'])).tolist()

        hist_labels, hist_data = [], {m: [] for m in display_order}
        num_q = len(norm_df)
        ltm_c = min(3, num_q // 4) or 1
        for i in range(ltm_c, 0, -1):
            chunk = norm_df.iloc[-4:] if i == 1 else norm_df.iloc[-(i*4):-((i-1)*4)]
            hist_labels.append("LTM (Current)" if i == 1 else f"LTM -{i-1}")
            for m in display_order:
                if m == 'Shares Outstanding': hist_data[m].append(chunk[m].mean())
                elif m == 'EPS': hist_data[m].append(chunk['Net Income'].sum()/chunk['Shares Outstanding'].mean() if chunk['Shares Outstanding'].mean() else 0)
                else: hist_data[m].append(chunk[m].sum())

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
            
        st.markdown(f'<div style="overflow-x: auto; max-width: 100%;">{st.markdown(md, unsafe_allow_html=True)}</div>', unsafe_allow_html=True)

        st.write("---")
        st.subheader("Implied Stock Price")
        
        col_val1, col_val2 = st.columns(2)
        with col_val1:
            st.write(f"**Current Market Price:** ${current_price:,.2f}")
        with col_val2:
            analyst_str = f"${st.session_state.analyst_target_tab1:,.2f}" if isinstance(st.session_state.analyst_target_tab1, (int, float)) and pd.notna(st.session_state.analyst_target_tab1) else "N/A"
            st.write(f"**Analyst Mean Target (1Y):** {analyst_str}")
            
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
        c_df['Quarterly EPS'] = combined_q_df['EPS'].round(2)
        c_df[f'Target Price (PE {t_pe:g})'] = (ttm_eps * t_pe).round(2)
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

    last_updated = "Never"
    if os.path.exists(CACHE_FILE):
        timestamp = os.path.getmtime(CACHE_FILE)
        last_updated = pd.to_datetime(timestamp, unit='s').strftime('%B %d, %Y at %I:%M %p')
        
    st.markdown(f"**Data Last Loaded:** `{last_updated}`")
    
    # --- TARGETED CACHE INJECTION CONTROLS ---
    st.write("### ⚡ Data Refresh Controls")
    c1, c2, c3 = st.columns([2, 1, 1])

    with c1:
        st.write("Update the entire S&P 500 matrix (takes ~1 minute).")
        if st.button("🔄 Force Refresh All (Full API Scan)", use_container_width=True):
            with st.spinner("Fetching S&P 500 Roster & Executing Fast Scan..."):
                try:
                    wiki_headers = {"User-Agent": "Mozilla/5.0"}
                    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies', storage_options=wiki_headers)[0]
                    tickers = [t.replace('.', '-') for t in sp500_table['Symbol'].tolist()]
                except Exception as e:
                    st.error(f"Failed to fetch stock index list: {e}")
                    st.stop()

                progress_bar = st.progress(0)
                status_text = st.empty()
                screened_results, completed, total_stocks = [], 0, len(tickers)

                with ThreadPoolExecutor(max_workers=15) as executor:
                    future_to_ticker = {executor.submit(process_single_screener_stock, t): t for t in tickers}
                    for future in as_completed(future_to_ticker):
                        completed += 1
                        res = future.result()
                        if res: screened_results.append(res)
                        if completed % 15 == 0 or completed == total_stocks:
                            progress_bar.progress(completed / total_stocks)
                            status_text.write(f"Scanned {completed}/{total_stocks}...")

                status_text.success(f"Matrix complete! Modeled {len(screened_results)} companies.")
                raw_df = pd.DataFrame(screened_results)
                raw_df.to_csv(CACHE_FILE, index=False)
                st.session_state.raw_screener_df = raw_df
                st.rerun()

    with c2:
        st.write("Targeted refresh for a single stock.")
        refresh_tick = st.text_input("Ticker", placeholder="e.g. NVDA", label_visibility="collapsed").upper().strip()

    with c3:
        st.write("") 
        if st.button("Targeted Update", use_container_width=True):
            if refresh_tick:
                if 'raw_screener_df' in st.session_state:
                    with st.spinner(f"Recalculating {refresh_tick}..."):
                        res = process_single_screener_stock(refresh_tick)
                        if res:
                            df_cache = st.session_state.raw_screener_df
                            if 'Analyst Target' not in df_cache.columns: df_cache['Analyst Target'] = np.nan

                            if refresh_tick in df_cache['Ticker'].values:
                                idx = df_cache.index[df_cache['Ticker'] == refresh_tick][0]
                                for k, v in res.items():
                                    df_cache.at[idx, k] = v
                            else:
                                df_cache = pd.concat([df_cache, pd.DataFrame([res])], ignore_index=True)

                            df_cache.to_csv(CACHE_FILE, index=False)
                            st.session_state.raw_screener_df = df_cache
                            st.rerun()
                        else:
                            st.error(f"Could not calculate projections for {refresh_tick}.")
                else:
                    st.error("Cache is empty. Run a full scan first to build the database.")
            else:
                st.warning("Please enter a ticker symbol.")

    if 'raw_screener_df' in st.session_state:
        df_base = st.session_state.raw_screener_df.copy()
        if 'Analyst Target' not in df_base.columns: df_base['Analyst Target'] = np.nan
        
        st.write("---")
        st.subheader("🎛️ Filter Opportunities")
        
        screener_pe = st.number_input("Universal Target P/E Multiple for Screen:", value=25.0, step=1.0, key="pe_screener")
        
        df_base['Year 5 Target'] = df_base['Year 5 EPS'] * screener_pe
        df_base['5-Yr CAGR'] = ((df_base['Year 5 Target'] / df_base['Current Price']) ** (1/5) - 1) * 100
        
        max_rmse = st.slider(
            "Forecast Confidence Filter (Max Historical Tracking Error):", 
            min_value=float(df_base['Avg Tracking Error (RMSE)'].min()), 
            max_value=float(df_base['Avg Tracking Error (RMSE)'].max()), 
            value=float(df_base['Avg Tracking Error (RMSE)'].max() * 0.4),
            help="Acts as a confidence interval. Lowering this strictness filters out unpredictable stocks."
        )
        min_cagr = st.slider("Minimum Acceptable 5-Yr CAGR (%):", float(df_base['5-Yr CAGR'].min()), float(df_base['5-Yr CAGR'].max()), 12.0)

        filtered_df = df_base[(df_base['Avg Tracking Error (RMSE)'] <= max_rmse) & (df_base['5-Yr CAGR'] >= min_cagr)].sort_values(by="5-Yr CAGR", ascending=False).reset_index(drop=True)
        
        display_cols = ["Ticker", "Current Price", "Analyst Target", "Year 5 Target", "5-Yr CAGR", "Avg Tracking Error (RMSE)"]
        
        st.write(f"Showing **{len(filtered_df)}** matching profiles.")
        st.dataframe(filtered_df[display_cols].style.format({
            "Current Price": "${:,.2f}", 
            "Analyst Target": lambda x: f"${x:,.2f}" if pd.notna(x) else "N/A",
            "Year 5 Target": "${:,.2f}", 
            "5-Yr CAGR": "{:+.1f}%", 
            "Avg Tracking Error (RMSE)": "±${:,.0f}"
        }), use_container_width=True)
