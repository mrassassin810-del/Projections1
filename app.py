import sys
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
drivers = ['Total Revenue', 'Cost Of Revenue', 'Operating Expense', 'Non-Op & Taxes', 'Shares Outstanding']
model_choices = ["Auto", "Linear", "Derivative", "Logarithmic", "Holt-Winters", "ARIMA"]
display_order = ['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income', 'Non-Op & Taxes', 'Net Income', 'Shares Outstanding', 'EPS']

# --- CORE MATH ENGINE ---
def calculate_metric_models(y_in, x_hist, x_fut, is_expense=False, is_shares=False, is_non_op=False):
    y = np.asarray(y_in, dtype=float)
    n = len(y)
    results = {}
    floor_val = 1 if is_shares else 0

    # 1. Base Linear
    lin_model = LinearRegression().fit(x_hist, y)
    pred_lin_fut = np.maximum(floor_val, lin_model.predict(x_fut))
    rmse_lin = np.sqrt(mean_squared_error(y, lin_model.predict(x_hist)))
    results['Linear'] = {'forecast': pred_lin_fut, 'rmse': rmse_lin}
    slope = lin_model.coef_[0]

    # 2. Derivative
    if n >= 4:
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

    # 3. Logarithmic
    x_log_hist = np.log(x_hist + 1)
    x_log_fut = np.log(x_fut + 1)
    log_model = LinearRegression().fit(x_log_hist, y)
    pred_log_fut = np.maximum(floor_val, log_model.predict(x_log_fut))
    rmse_log = np.sqrt(mean_squared_error(y, log_model.predict(x_log_hist)))
    results['Logarithmic'] = {'forecast': pred_log_fut, 'rmse': rmse_log}

    # 4. Holt-Winters
    if n >= 8: 
        try:
            hw_model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=4, initialization_method="heuristic").fit()
            results['Holt-Winters'] = {'forecast': np.maximum(floor_val, hw_model.forecast(20)), 'rmse': np.sqrt(mean_squared_error(y, hw_model.fittedvalues))}
        except Exception: 
            results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}
    else: results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}

    # 5. ARIMA
    if n >= 6:
        try:
            arima_model = ARIMA(y, order=(1, 1, 1), enforce_stationarity=False, enforce_invertibility=False).fit()
            results['ARIMA'] = {'forecast': np.maximum(floor_val, arima_model.forecast(20)), 'rmse': np.sqrt(mean_squared_error(y, arima_model.predict(start=0, end=len(y)-1)))}
        except Exception: 
            results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}
    else: results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}

    if is_non_op:
        upper_bound = max(0, np.max(y)) * 1.2
        for name in results:
            if results[name]['forecast'] is not None:
                results[name]['forecast'] = np.minimum(results[name]['forecast'], upper_bound)

    valid_models = [ (name, data['rmse'], data['forecast']) for name, data in results.items() if data['forecast'] is not None and data['rmse'] != float('inf') ]
    valid_models.sort(key=lambda x: x[1]) 
    
    auto_choice = "Linear" 
    for name, rmse, forecast in valid_models:
        if is_expense and slope > 0 and forecast[-1] < y[-1] and name != "Linear": continue
        auto_choice = name
        break

    results['AutoChoice'] = auto_choice
    return results

# --- BACKGROUND WORKER FOR SCREENER ---
def process_single_screener_stock(ticker, target_pe):
    try:
        stock = yf.Ticker(ticker)
        df = stock.quarterly_income_stmt.T
        if df.empty: df = stock.quarterly_financials.T
        if df.empty or 'Total Revenue' not in df.columns: return None

        df = df.sort_index()
        df_raw = df / 1000

        rev = df_raw['Total Revenue']
        gp = df_raw['Gross Profit'] if 'Gross Profit' in df_raw.columns else rev
        cogs = rev - gp
        op_inc = df_raw['Operating Income'] if 'Operating Income' in df_raw.columns else gp
        opex = gp - op_inc
        ni = df_raw['Net Income'] if 'Net Income' in df_raw.columns else op_inc
        non_op_taxes = ni - op_inc

        if 'Diluted Average Shares' in df_raw.columns: shares = df_raw['Diluted Average Shares']
        elif 'Basic Average Shares' in df_raw.columns: shares = df_raw['Basic Average Shares']
        else: return None

        norm_df = pd.DataFrame({'Total Revenue': rev, 'Cost Of Revenue': cogs, 'Operating Expense': opex, 'Non-Op & Taxes': non_op_taxes, 'Shares Outstanding': shares, 'Net Income': ni}).dropna()
        if len(norm_df) < 5: return None

        x_hist, x_fut = np.arange(len(norm_df)).reshape(-1, 1), np.arange(len(norm_df), len(norm_df) + 20).reshape(-1, 1)
        total_rmse, proj_5y = 0, {}

        for metric in drivers:
            y = norm_df[metric].values
            res = calculate_metric_models(y, x_hist, x_fut, metric in ['Cost Of Revenue', 'Operating Expense'], metric == 'Shares Outstanding', metric == 'Non-Op & Taxes')
            winning_model = res['AutoChoice']
            total_rmse += res[winning_model]['rmse']
            q_forecast = res[winning_model]['forecast']
            proj_5y[metric] = np.mean(q_forecast[16:20]) if metric == 'Shares Outstanding' else np.sum(q_forecast[16:20])

        net_income_5y = (proj_5y['Total Revenue'] - proj_5y['Cost Of Revenue']) - proj_5y['Operating Expense'] + proj_5y['Non-Op & Taxes']
        eps_5y = net_income_5y / proj_5y['Shares Outstanding'] if proj_5y['Shares Outstanding'] else 0
        target_price_5y = eps_5y * target_pe

        hist_1d = stock.history(period="1d")
        current_p = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0

        if current_p <= 0 or target_price_5y <= 0: return None
        cagr = (target_price_5y / current_p) ** (1/5) - 1

        return {"Ticker": ticker, "Current Price": round(current_p, 2), "Year 5 Target": round(target_price_5y, 2), "5-Yr CAGR": round(cagr * 100, 2), "Avg Tracking Error (RMSE)": round(total_rmse / len(drivers), 2)}
    except: return None

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
        with st.spinner(f"Mining data for {ticker_input}..."):
            error_found, use_yf = False, True
            stock = yf.Ticker(ticker_input)
            
            # --- ALPHA VANTAGE SECRETS PULL ---
            if api_key:
                try:
                    r = requests.get(f'https://www.alphavantage.co/query?function=INCOME_STATEMENT&symbol={ticker_input}&apikey={api_key}').json()
                    if 'quarterlyReports' in r:
                        use_yf = False
                        df_av = pd.DataFrame(r['quarterlyReports'])
                        df_av['fiscalDateEnding'] = pd.to_datetime(df_av['fiscalDateEnding'])
                        df_av = df_av.set_index('fiscalDateEnding').sort_index()

                        for col in ['totalRevenue', 'costOfRevenue', 'grossProfit', 'operatingIncome', 'netIncome']:
                            df_av[col] = pd.to_numeric(df_av[col], errors='coerce').fillna(0) / 1000

                        rev, cogs, gp, op_inc, ni = df_av['totalRevenue'], df_av['costOfRevenue'], df_av['grossProfit'], df_av['operatingIncome'], df_av['netIncome']
                        shares = pd.Series(stock.info.get('sharesOutstanding', 100000) / 1000, index=df_av.index)
                        
                        st.session_state.norm_df = pd.DataFrame({
                            'Total Revenue': rev, 'Cost Of Revenue': cogs, 'Gross Profit': gp, 'Operating Expense': gp - op_inc,
                            'Operating Income': op_inc, 'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 'Shares Outstanding': shares, 'EPS': ni / shares
                        }).dropna()
                        
                        hist_1d = stock.history(period="1d")
                        st.session_state.current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0
                        st.session_state.ticker_analyzed = ticker_input
                        st.session_state.actual_lookback = lookback_input
                        st.session_state.data_source = "Alpha Vantage (Deep History)"
                    else: st.warning("Alpha Vantage limit reached. Falling back to Yahoo Finance...")
                except: st.warning(f"Alpha Vantage Error. Falling back to Yahoo Finance...")

            # --- YFINANCE FALLBACK ---
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
                    
                    if df.empty or 'Total Revenue' not in df.columns:
                        st.error(f"No financial data found for {ticker_input}.")
                        error_found = True
                    else:
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

                        st.session_state.norm_df = pd.DataFrame({
                            'Total Revenue': rev, 'Cost Of Revenue': rev - gp, 'Gross Profit': gp, 'Operating Expense': gp - op_inc,
                            'Operating Income': op_inc, 'Non-Op & Taxes': ni - op_inc, 'Net Income': ni, 'Shares Outstanding': shares, 'EPS': ni / shares
                        }).dropna()
                        
                        hist_1d = stock.history(period="1d")
                        st.session_state.current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0
                        st.session_state.ticker_analyzed = ticker_input
                        st.session_state.actual_lookback = lookback_input
                        st.session_state.data_source = "Yahoo Finance (Standard)"
                except Exception as e:
                    st.error(f"Error processing data: {e}")
                    error_found = True
            
            if error_found: st.stop()

    if 'norm_df' in st.session_state and st.session_state.ticker_analyzed == ticker_input:
        norm_df = st.session_state.norm_df
        current_price = st.session_state.current_price
        df_reg = norm_df.tail(len(norm_df) if st.session_state.actual_lookback == 0 else st.session_state.actual_lookback)
        
        depth_color = "green" if len(df_reg) >= 8 else "red"
        st.markdown(f"**Data Depth Indicator:** :{depth_color}[{len(df_reg)} Quarters Loaded] via {st.session_state.data_source} *(Note: ARIMA/Holt-Winters require 6-8 minimum)*")
        
        x_historical, x_future = np.arange(len(df_reg)).reshape(-1, 1), np.arange(len(df_reg), len(df_reg) + 20).reshape(-1, 1) 
        
        metric_results = {}
        for metric in drivers:
            metric_results[metric] = calculate_metric_models(df_reg[metric].values, x_historical, x_future, metric in ['Cost Of Revenue', 'Operating Expense'], metric == 'Shares Outstanding', metric == 'Non-Op & Taxes')

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
            proj_annual_data[metric] = [np.mean(res[act]['forecast'][i*4:(i+1)*4]) if metric == 'Shares Outstanding' else np.sum(res[act]['forecast'][i*4:(i+1)*4]) for i in range(5)]

        proj_annual_data['Gross Profit'] = (np.array(proj_annual_data['Total Revenue']) - np.array(proj_annual_data['Cost Of Revenue'])).tolist()
        proj_annual_data['Operating Income'] = (np.array(proj_annual_data['Gross Profit']) - np.array(proj_annual_data['Operating Expense'])).tolist()
        proj_annual_data['Net Income'] = (np.array(proj_annual_data['Operating Income']) + np.array(proj_annual_data['Non-Op & Taxes'])).tolist()
        proj_annual_data['EPS'] = (np.array(proj_annual_data['Net Income']) / np.array(proj_annual_data['Shares Outstanding'])).tolist()

        proj_quarterly_data['Gross Profit'] = proj_quarterly_data['Total Revenue'] - proj_quarterly_data['Cost Of Revenue']
        proj_quarterly_data['Operating Income'] = proj_quarterly_data['Gross Profit'] - proj_quarterly_data['Operating Expense']
        proj_quarterly_data['Net Income'] = proj_quarterly_data['Operating Income'] + proj_quarterly_data['Non-Op & Taxes']
        proj_quarterly_data['EPS'] = proj_quarterly_data['Net Income'] / proj_quarterly_data['Shares Outstanding']

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
        st.write(f"**Current Market Price:** ${current_price:,.2f}")
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
    screener_pe = st.number_input("Universal Target P/E Multiple for Screen:", value=25.0, step=1.0, key="pe_screener")
    
    start_screen = st.button("🚀 Start S&P 500 Matrix Scan", use_container_width=True)
        
    if start_screen:
        with st.spinner("Fetching S&P 500 Roster..."):
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
            future_to_ticker = {executor.submit(process_single_screener_stock, t, screener_pe): t for t in tickers}
            for future in as_completed(future_to_ticker):
                completed += 1
                res = future.result()
                if res: screened_results.append(res)
                if completed % 15 == 0 or completed == total_stocks:
                    progress_bar.progress(completed / total_stocks)
                    status_text.write(f"Scanned {completed}/{total_stocks}...")

        status_text.success(f"Matrix complete! Modeled {len(screened_results)} companies.")
        st.session_state.screener_df = pd.DataFrame(screened_results)

    if 'screener_df' in st.session_state:
        df_display = st.session_state.screener_df.copy()
        st.write("---")
        st.subheader("🎛️ Filter Opportunities")
        
        max_rmse = st.slider("Max Avg Tracking Error Allowed (RMSE):", float(df_display['Avg Tracking Error (RMSE)'].min()), float(df_display['Avg Tracking Error (RMSE)'].max()), float(df_display['Avg Tracking Error (RMSE)'].max() * 0.4))
        min_cagr = st.slider("Minimum Acceptable 5-Yr CAGR (%):", float(df_display['5-Yr CAGR'].min()), float(df_display['5-Yr CAGR'].max()), 12.0)

        filtered_df = df_display[(df_display['Avg Tracking Error (RMSE)'] <= max_rmse) & (df_display['5-Yr CAGR'] >= min_cagr)].sort_values(by="5-Yr CAGR", ascending=False).reset_index(drop=True)
        st.write(f"Showing **{len(filtered_df)}** matching profiles.")
        st.dataframe(filtered_df.style.format({"Current Price": "${:,.2f}", "Year 5 Target": "${:,.2f}", "5-Yr CAGR": "{:+.1f}%", "Avg Tracking Error (RMSE)": "±${:,.0f}"}), use_container_width=True)
