import sys
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import altair as alt
import warnings

# Attempt to load advanced stats models
try:
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.tsa.arima.model import ARIMA
    HAS_STATSMODELS = True
except ImportError:
    HAS_STATSMODELS = False

warnings.filterwarnings('ignore')

st.set_page_config(page_title="Forecaster & Valuation Model", layout="wide")
st.title("Advanced Income Statement Forecaster")

if not HAS_STATSMODELS:
    st.error("⚠️ **Missing Library:** To use ARIMA and Holt-Winters, please open your terminal and run: `pip install statsmodels`")
    st.stop()

st.write("Calculates **Linear**, **Derivative**, **Logarithmic**, **Holt-Winters**, and **ARIMA** models. Auto-selects the lowest historical error (RMSE). Values in **Thousands ($000s)**.")

drivers = ['Total Revenue', 'Cost Of Revenue', 'Operating Expense', 'Non-Op & Taxes', 'Shares Outstanding']
model_choices = ["Auto", "Linear", "Derivative", "Logarithmic", "Holt-Winters", "ARIMA"]

for m in drivers:
    if f"ov_{m}" not in st.session_state:
        st.session_state[f"ov_{m}"] = "Auto"

def reset_overrides():
    for m in drivers:
        st.session_state[f"ov_{m}"] = "Auto"

# --- UI CONTROLS ---
col1, col2, col3 = st.columns([2, 2, 1])
with col1:
    ticker_input = st.text_input("Enter Ticker (e.g., PLTR, CRM, AMD):", "PLTR").upper()
with col2:
    lookback_input = st.number_input("Quarters back (0 = All):", min_value=0, max_value=40, value=0, step=1)
with col3:
    st.write("") 
    st.write("")
    analyze_btn = st.button("Fetch & Analyze")

# --- DATA FETCHING ---
if analyze_btn:
    with st.spinner(f"Fetching data for {ticker_input}..."):
        error_found = False
        try:
            stock = yf.Ticker(ticker_input)
            df = stock.quarterly_income_stmt.T
            if df.empty: df = stock.quarterly_financials.T
            
            if df.empty:
                st.error(f"No financial data found for {ticker_input} on Yahoo Finance.")
                error_found = True
            else:
                df.index = pd.to_datetime(df.index)
                df = df.sort_index()

                if 'Total Revenue' not in df.columns:
                    st.error("Total Revenue data missing.")
                    error_found = True
                else:
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
                    else: shares = pd.Series(1, index=df_raw.index)

                    norm_df = pd.DataFrame({
                        'Total Revenue': rev,
                        'Cost Of Revenue': cogs,
                        'Gross Profit': gp,
                        'Operating Expense': opex,
                        'Operating Income': op_inc,
                        'Non-Op & Taxes': non_op_taxes,
                        'Net Income': ni,
                        'Shares Outstanding': shares,
                        'EPS': ni / shares
                    }).dropna()
                    
                    hist_1d = stock.history(period="1d")
                    current_price = hist_1d['Close'].iloc[-1] if not hist_1d.empty else 0.0

                    st.session_state.norm_df = norm_df
                    st.session_state.ticker_analyzed = ticker_input
                    st.session_state.actual_lookback = lookback_input
                    st.session_state.current_price = current_price

        except Exception as e:
            st.error(f"Error processing data: {e}")
            error_found = True
            
        if error_found:
            st.stop()

# --- MATH ENGINE ---
def calculate_metric_models(y, x_hist, x_fut, is_expense=False, is_shares=False, is_non_op=False):
    n = len(y)
    results = {}
    floor_val = 1 if is_shares else 0

    # 1. Base Linear Model
    lin_model = LinearRegression().fit(x_hist, y)
    pred_lin_fut = np.maximum(floor_val, lin_model.predict(x_fut))
    pred_lin_hist = lin_model.predict(x_hist)
    slope = lin_model.coef_[0]
    rmse_lin = np.sqrt(mean_squared_error(y, pred_lin_hist))
    results['Linear'] = {'forecast': pred_lin_fut, 'rmse': rmse_lin}

    # 2. Derivative Model (Velocity)
    if n >= 4:
        diffs = np.diff(y)
        x_diff = np.arange(len(diffs)).reshape(-1, 1)
        deriv_model = LinearRegression().fit(x_diff, diffs)
        x_diff_fut = np.arange(len(diffs), len(diffs) + 20).reshape(-1, 1)
        fut_diffs = deriv_model.predict(x_diff_fut)

        forecast_deriv = []
        current_val = y[-1]
        for fd in fut_diffs:
            current_val = max(floor_val, current_val + fd)
            forecast_deriv.append(current_val)
            
        pred_deriv_hist = y[:-1] + deriv_model.predict(x_diff)
        rmse_deriv = np.sqrt(mean_squared_error(y[1:], pred_deriv_hist))
        results['Derivative'] = {'forecast': np.array(forecast_deriv), 'rmse': rmse_deriv}
    else:
        results['Derivative'] = {'forecast': None, 'rmse': float('inf')}

    # 3. Logarithmic Model
    x_log_hist = np.log(x_hist + 1)
    x_log_fut = np.log(x_fut + 1)
    log_model = LinearRegression().fit(x_log_hist, y)
    pred_log_fut = np.maximum(floor_val, log_model.predict(x_log_fut))
    pred_log_hist = log_model.predict(x_log_hist)
    rmse_log = np.sqrt(mean_squared_error(y, pred_log_hist))
    results['Logarithmic'] = {'forecast': pred_log_fut, 'rmse': rmse_log}

    # 4. Holt-Winters
    if n >= 8: 
        try:
            hw_model = ExponentialSmoothing(y, trend='add', seasonal='add', seasonal_periods=4, initialization_method="estimated").fit()
            pred_hw_fut = np.maximum(floor_val, hw_model.forecast(20))
            rmse_hw = np.sqrt(mean_squared_error(y, hw_model.fittedvalues))
            results['Holt-Winters'] = {'forecast': pred_hw_fut, 'rmse': rmse_hw}
        except:
            results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}
    else:
        results['Holt-Winters'] = {'forecast': None, 'rmse': float('inf')}

    # 5. ARIMA 
    if n >= 6:
        try:
            arima_model = ARIMA(y, order=(1, 1, 1)).fit()
            pred_arima_fut = np.maximum(floor_val, arima_model.forecast(20))
            pred_arima_hist = arima_model.predict(start=0, end=len(y)-1)
            rmse_arima = np.sqrt(mean_squared_error(y, pred_arima_hist))
            results['ARIMA'] = {'forecast': pred_arima_fut, 'rmse': rmse_arima}
        except:
            results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}
    else:
        results['ARIMA'] = {'forecast': None, 'rmse': float('inf')}

    # --- NON-OPERATING REALITY CLAMP ---
    if is_non_op:
        upper_bound = max(0, np.max(y)) * 1.5
        for name in results:
            if results[name]['forecast'] is not None:
                results[name]['forecast'] = np.minimum(results[name]['forecast'], upper_bound)

    # --- AUTO-FIT SELECTION LOGIC ---
    valid_models = []
    for name, data in results.items():
        if data['forecast'] is not None and data['rmse'] != float('inf'):
            valid_models.append((name, data['rmse'], data['forecast']))
    
    valid_models.sort(key=lambda x: x[1]) 
    
    auto_choice = "Linear" 
    for name, rmse, forecast in valid_models:
        if is_expense and slope > 0 and forecast[-1] < y[-1] and name != "Linear":
            continue
        auto_choice = name
        break

    results['AutoChoice'] = auto_choice
    return results

if 'norm_df' in st.session_state:
    norm_df = st.session_state.norm_df
    ticker = st.session_state.ticker_analyzed
    lookback = st.session_state.actual_lookback
    current_price = st.session_state.current_price
    
    actual_lookback = len(norm_df) if lookback == 0 or lookback > len(norm_df) else lookback
    df_reg = norm_df.tail(actual_lookback)
    
    x_historical = np.arange(len(df_reg)).reshape(-1, 1)
    x_future = np.arange(len(df_reg), len(df_reg) + 20).reshape(-1, 1) 
    
    st.write("---")
    
    metric_results = {}
    for metric in drivers:
        y = df_reg[metric].values
        is_exp = metric in ['Cost Of Revenue', 'Operating Expense']
        is_sh = metric == 'Shares Outstanding'
        is_nonop = metric == 'Non-Op & Taxes'
        metric_results[metric] = calculate_metric_models(y, x_historical, x_future, is_expense=is_exp, is_shares=is_sh, is_non_op=is_nonop)

    with st.expander("⚙️ Advanced: Override Projection Models", expanded=False):
        st.write("Force specific math models for each line item. Auto-updates instantly.")
        st.button("🔄 Reset all to Auto", on_click=reset_overrides)
        cols = st.columns(5)
        for i, metric in enumerate(drivers):
            with cols[i]:
                res = metric_results[metric]
                
                def fmt(o, r=res):
                    if o == "Auto": return f"Auto (Picked {r['AutoChoice']})"
                    if r[o]['rmse'] != float('inf'):
                        return f"{o} (RMSE: ±${int(r[o]['rmse']):,})"
                    return f"{o} (N/A - Insufficient Data)"
                
                st.selectbox(metric, options=model_choices, format_func=fmt, key=f"ov_{metric}")

    proj_annual_data = {}
    proj_quarterly_data = {}
    errors = {}
    methods = {}
    
    for metric in drivers:
        res = metric_results[metric]
        choice = st.session_state[f"ov_{metric}"]
        
        if choice == "Auto":
            active_method = res['AutoChoice']
            method_label = f"Auto picked: {active_method}"
        else:
            active_method = choice
            method_label = f"Manual: {active_method}"
            
        if res[active_method]['forecast'] is None:
            active_method = "Linear"
            method_label = f"Forced Linear ({choice} Failed)"
            
        q_forecast = res[active_method]['forecast']
        rmse = res[active_method]['rmse']
        
        errors[metric] = rmse
        methods[metric] = method_label
        proj_quarterly_data[metric] = q_forecast
        
        is_sh = metric == 'Shares Outstanding'
        if is_sh: proj_annual_data[metric] = [np.mean(q_forecast[i*4:(i+1)*4]) for i in range(5)]
        else: proj_annual_data[metric] = [np.sum(q_forecast[i*4:(i+1)*4]) for i in range(5)]

    proj_annual_data['Gross Profit'] = (np.array(proj_annual_data['Total Revenue']) - np.array(proj_annual_data['Cost Of Revenue'])).tolist()
    proj_annual_data['Operating Income'] = (np.array(proj_annual_data['Gross Profit']) - np.array(proj_annual_data['Operating Expense'])).tolist()
    proj_annual_data['Net Income'] = (np.array(proj_annual_data['Operating Income']) + np.array(proj_annual_data['Non-Op & Taxes'])).tolist()
    proj_annual_data['EPS'] = (np.array(proj_annual_data['Net Income']) / np.array(proj_annual_data['Shares Outstanding'])).tolist()

    proj_quarterly_data['Gross Profit'] = proj_quarterly_data['Total Revenue'] - proj_quarterly_data['Cost Of Revenue']
    proj_quarterly_data['Operating Income'] = proj_quarterly_data['Gross Profit'] - proj_quarterly_data['Operating Expense']
    proj_quarterly_data['Net Income'] = proj_quarterly_data['Operating Income'] + proj_quarterly_data['Non-Op & Taxes']
    proj_quarterly_data['EPS'] = proj_quarterly_data['Net Income'] / proj_quarterly_data['Shares Outstanding']

    display_order = ['Total Revenue', 'Cost Of Revenue', 'Gross Profit', 'Operating Expense', 'Operating Income', 'Non-Op & Taxes', 'Net Income', 'Shares Outstanding', 'EPS']
    
    hist_labels = []
    hist_data = {m: [] for m in display_order}
    num_q = len(norm_df)
    ltm_count = min(3, num_q // 4)
    if ltm_count == 0: ltm_count = 1 

    for i in range(ltm_count, 0, -1):
        if i == 1:
            chunk = norm_df.iloc[-4:] if num_q >= 4 else norm_df
            hist_labels.append("LTM (Current)")
        else:
            chunk = norm_df.iloc[-(i*4):-((i-1)*4)]
            hist_labels.append(f"LTM -{i-1}")
            
        for metric in display_order:
            if metric == 'Shares Outstanding': hist_data[metric].append(chunk[metric].mean())
            elif metric == 'EPS':
                ni = chunk['Net Income'].sum()
                sh = chunk['Shares Outstanding'].mean()
                hist_data[metric].append(ni/sh if sh else 0)
            else: hist_data[metric].append(chunk[metric].sum())

    last_date = norm_df.index[-1]
    proj_labels = [(last_date + pd.DateOffset(months=12 * i)).strftime("LTM %b '%yE") for i in range(1, 6)]
    all_labels = hist_labels + proj_labels

    st.subheader(f"Historical & 5-Year Projections ({ticker})")
    md = f"| Metric | {' | '.join(all_labels)} |\n|---{'|---'*len(all_labels)}|\n"
    
    for metric in display_order:
        row = f"| **{metric}** |"
        combined_vals = hist_data[metric] + proj_annual_data[metric]
        
        for i in range(len(combined_vals)):
            val = combined_vals[i]
            prev_val = combined_vals[i-1] if i > 0 else val 
            growth = (val - prev_val) / abs(prev_val) if prev_val and prev_val != 0 else 0
            
            val_str = f"${val:,.2f}" if metric == 'EPS' else f"{val:,.0f}"
            
            if i == 0:
                row += f" {val_str} |"
            else:
                growth_str = f"{growth:+.1%}"
                good_up = ['Total Revenue', 'Gross Profit', 'Operating Income', 'Net Income', 'EPS']
                is_good = (growth > 0 and metric in good_up) or (growth < 0 and metric not in good_up)
                color = "#1d9e75" if is_good else "#a32d2d"
                row += f" {val_str} <span style='color:{color}; font-weight:600; font-size:0.9em;'>({growth_str})</span> |"
        md += row + "\n"

    st.markdown(md, unsafe_allow_html=True)
    
    st.write("---")
    st.subheader(f"Implied Stock Price")
    st.write(f"**Current Market Price ({ticker}):** ${current_price:,.2f}")
    
    target_pe = st.number_input("Target P/E Ratio (Adjust to recalculate targets instantly):", value=25.0, step=1.0)
    
    val_md = f"| Valuation | {' | '.join(proj_labels)} | 5-Yr CAGR |\n|---{'|---'*len(proj_labels)}|---|\n| **Target Price** |"
    target_prices = []
    
    for i in range(5):
        implied_price = proj_annual_data['EPS'][i] * target_pe
        target_prices.append(implied_price)
        val_md += f" **${implied_price:,.2f}** |"
        
    final_price = target_prices[-1]

    if current_price > 0 and final_price > 0:
        cagr = (final_price / current_price) ** (1/5) - 1
        cagr_str = f"{cagr:+.1%}"
        color_cagr = "#1d9e75" if cagr > 0 else "#a32d2d"
        val_md += f" <span style='color:{color_cagr}; font-weight:600;'>{cagr_str}</span> |"
    else:
        val_md += f" <span style='color:gray;'>N/A</span> |"
        
    st.markdown(val_md, unsafe_allow_html=True)
    
    st.write("---")
    
    st.subheader("Winning Model & Historical Error (RMSE)")
    st.write("RMSE reflects the average raw dollar deviation (in thousands) of the trendline against actual history. Lower error wins.")
    cols = st.columns(len(drivers))
    for i, (m) in enumerate(drivers):
        err = errors[m]
        method = methods[m]
        with cols[i]:
            err_disp = f"±${int(err):,}" if err != float('inf') else "N/A"
            st.markdown(f"**{m}**<br>:{'green'}[{err_disp}]<br><span style='font-size:0.85em;color:gray;'>{method}</span>", unsafe_allow_html=True)

    with st.expander("View Raw Quarterly History & Velocity"):
        velocity_df = norm_df.diff(periods=4) 
        hist_format = {col: "{:,.0f}" for col in norm_df.columns}
        hist_format['EPS'] = "${:,.2f}"
        
        vcol1, vcol2 = st.columns(2)
        with vcol1:
            st.caption("Historical Quarterly Totals")
            st.dataframe(norm_df.style.format(hist_format))
        with vcol2:
            st.caption("Quarterly YoY Differences")
            st.dataframe(velocity_df.dropna(how='all').style.format(hist_format))

    st.subheader("Visual Forecasts")
    future_dates = [last_date + pd.DateOffset(months=3 * i) for i in range(1, 21)]
    q_proj_df = pd.DataFrame(proj_quarterly_data, index=future_dates)
    
    chart_df_totals = pd.concat([norm_df[['Total Revenue', 'Net Income']], q_proj_df[['Total Revenue', 'Net Income']]])
    st.caption("Quarterly Revenue vs Net Income ($000s)")
    st.line_chart(chart_df_totals, color=["#1f77b4", "#2ca02c"]) 

    combined_q_df = pd.concat([norm_df, q_proj_df])
    ttm_ni = combined_q_df['Net Income'].rolling(window=4, min_periods=1).sum()
    run_rate_multiplier = 4 / combined_q_df['Net Income'].rolling(window=4, min_periods=1).count()
    ttm_ni = ttm_ni * run_rate_multiplier
    
    ttm_shares = combined_q_df['Shares Outstanding'].rolling(window=4, min_periods=1).mean()
    ttm_eps = ttm_ni / ttm_shares
    
    price_col = f"Target Price (PE {target_pe:g})"
    
    chart_df_eps = pd.DataFrame(index=combined_q_df.index)
    chart_df_eps['Quarterly EPS'] = combined_q_df['EPS'].round(2)
    chart_df_eps[price_col] = (ttm_eps * target_pe).round(2)
    
    chart_df_eps_reset = chart_df_eps.reset_index().rename(columns={'index': 'Date'})

    st.caption("Quarterly Earnings Per Share (EPS) & Implied Stock Price (Based on TTM EPS)")
    
    base = alt.Chart(chart_df_eps_reset).encode(
        x=alt.X('Date:T', title=None)
    )

    line_eps = base.mark_line(color="#1d9e75").encode(
        y=alt.Y('Quarterly EPS:Q', title='Quarterly EPS ($)', axis=alt.Axis(titleColor='#1d9e75')),
        tooltip=[alt.Tooltip('Date:T', format='%b %Y', title='Date'), 'Quarterly EPS']
    )

    line_price = base.mark_line(color="#e8a329").encode(
        y=alt.Y(f'{price_col}:Q', title='Target Price ($)', axis=alt.Axis(titleColor='#e8a329')),
        tooltip=[alt.Tooltip('Date:T', format='%b %Y', title='Date'), f'{price_col}']
    )

    dual_chart = alt.layer(line_eps, line_price).resolve_scale(
        y='independent'
    ).properties(
        height=400
    ).configure_axis(
        grid=False 
    )

    st.altair_chart(dual_chart, use_container_width=True)
