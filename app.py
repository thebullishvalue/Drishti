"""
AARAMBH — Nifty50 Market Evaluation Dashboard
Streamlit app replicating the Aarambh.xlsx workbook.
"""
import datetime
import io
import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from engine import compute_all, get_regression_details, get_latest_summary, DEFAULT_PARAMS, REGRESSION_VARS_PE
from fetcher import auto_fetch_today

# ──────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="AARAMBH",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# Custom CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    .main .block-container { padding-top: 1rem; max-width: 100%; }
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border-radius: 10px; padding: 15px; margin: 5px 0;
        border-left: 4px solid #0f3460; color: #e0e0e0;
    }
    .metric-card .value { font-size: 1.5rem; font-weight: 700; color: #00d4ff; }
    .metric-card .label { font-size: 0.8rem; color: #a0a0a0; text-transform: uppercase; }
    .signal-buy { background: #0d4d0d; border-left-color: #00ff41; }
    .signal-sell { background: #4d0d0d; border-left-color: #ff4141; }
    .signal-neutral { background: #1a1a2e; border-left-color: #666; }
    div[data-testid="stMetricValue"] { font-size: 1.1rem; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1a1a2e; border-radius: 6px;
        padding: 8px 16px; color: #e0e0e0;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# Data Loading
# ──────────────────────────────────────────────
@st.cache_data
def load_historical_data():
    """Load historical CSV data."""
    try:
        df = pd.read_csv('data/historical.csv', parse_dates=['date'])
        return df
    except FileNotFoundError:
        st.error("Historical data file not found. Please ensure data/historical.csv exists.")
        return pd.DataFrame()


def get_session_data():
    """Get data from session state, initializing if needed."""
    if 'df_raw' not in st.session_state:
        st.session_state.df_raw = load_historical_data()
    return st.session_state.df_raw


# ──────────────────────────────────────────────
# Sidebar
# ──────────────────────────────────────────────
def render_sidebar():
    with st.sidebar:
        st.markdown("## ⚙️ AARAMBH")
        st.caption("Nifty50 Market Evaluation System")

        # Parameters
        with st.expander("📐 Model Parameters", expanded=False):
            params = DEFAULT_PARAMS.copy()
            params['ema_period'] = st.number_input("EMA Period (Breadth)", value=10, min_value=2, max_value=50)
            params['bol_upper_mult'] = st.number_input("Bollinger Upper σ", value=2.0, min_value=0.5, max_value=5.0, step=0.1)
            params['bol_lower_mult'] = st.number_input("Bollinger Lower σ", value=2.0, min_value=0.5, max_value=5.0, step=0.1)
            params['osc_threshold'] = st.number_input("Oscillator Threshold", value=1.0, min_value=0.1, max_value=5.0, step=0.1)
            st.session_state.params = params

        # Data Entry
        with st.expander("📝 Add New Day's Data", expanded=True):
            st.caption("Auto-fetch or manually enter today's data")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🔄 Auto-Fetch", use_container_width=True):
                    with st.spinner("Fetching..."):
                        fetched = auto_fetch_today()
                        st.session_state.fetched_data = fetched
                        if fetched:
                            st.success(f"Fetched {len(fetched)} fields")
                        else:
                            st.warning("Could not fetch data")
            with col2:
                if st.button("➕ Add Row", use_container_width=True, type="primary"):
                    _add_new_row()

            fetched = st.session_state.get('fetched_data', {})

            new_date = st.date_input("Date", value=datetime.date.today())
            st.session_state.new_date = new_date

            c1, c2 = st.columns(2)
            with c1:
                st.session_state.new_nifty = st.number_input("NIFTY Close", value=fetched.get('nifty', 0.0), format="%.2f")
                st.session_state.new_ad = st.number_input("AD Ratio", value=fetched.get('ad_ratio', 0.0), format="%.2f")
                st.session_state.new_pe = st.number_input("NIFTY PE", value=fetched.get('nifty_pe', 0.0), format="%.2f")
                st.session_state.new_dy = st.number_input("NIFTY DY", value=fetched.get('nifty_dy', 0.0), format="%.2f")
                st.session_state.new_pb = st.number_input("NIFTY PB", value=fetched.get('nifty_pb', 0.0), format="%.2f")
                st.session_state.new_in10y = st.number_input("IN 10Y", value=fetched.get('in10y', 0.0), format="%.3f")
                st.session_state.new_in02y = st.number_input("IN 02Y", value=fetched.get('in02y', 0.0), format="%.3f")
                st.session_state.new_in30y = st.number_input("IN 30Y", value=fetched.get('in30y', 0.0), format="%.3f")
            with c2:
                st.session_state.new_iniryy = st.number_input("IN IRYY", value=fetched.get('iniryy', 0.0), format="%.3f")
                st.session_state.new_repo = st.number_input("REPO", value=fetched.get('repo', 0.0), format="%.2f")
                st.session_state.new_crr = st.number_input("CRR", value=fetched.get('crr', 0.0), format="%.2f")
                st.session_state.new_us02y = st.number_input("US 02Y", value=fetched.get('us02y', 0.0), format="%.3f")
                st.session_state.new_us10y = st.number_input("US 10Y", value=fetched.get('us10y', 0.0), format="%.3f")
                st.session_state.new_us30y = st.number_input("US 30Y", value=fetched.get('us30y', 0.0), format="%.3f")
                st.session_state.new_usfed = st.number_input("US FED", value=fetched.get('us_fed', 0.0), format="%.2f")

        # Export
        with st.expander("💾 Export Data", expanded=False):
            if st.button("📥 Download CSV", use_container_width=True):
                st.session_state.download_ready = True

    return st.session_state.get('params', DEFAULT_PARAMS)


def _add_new_row():
    """Add a new data row from sidebar inputs."""
    df = get_session_data()
    new_row = {
        'date': pd.Timestamp(st.session_state.get('new_date', datetime.date.today())),
        'ad_ratio': st.session_state.get('new_ad', 0),
        'nifty': st.session_state.get('new_nifty', 0),
        'nifty_pe': st.session_state.get('new_pe', 0),
        'nifty_dy': st.session_state.get('new_dy', 0),
        'nifty_pb': st.session_state.get('new_pb', 0),
        'in10y': st.session_state.get('new_in10y', 0) or None,
        'in02y': st.session_state.get('new_in02y', 0) or None,
        'in30y': st.session_state.get('new_in30y', 0) or None,
        'iniryy': st.session_state.get('new_iniryy', 0) or None,
        'repo': st.session_state.get('new_repo', 0) or None,
        'crr': st.session_state.get('new_crr', 0) or None,
        'us02y': st.session_state.get('new_us02y', 0) or None,
        'us10y': st.session_state.get('new_us10y', 0) or None,
        'us30y': st.session_state.get('new_us30y', 0) or None,
        'us_fed': st.session_state.get('new_usfed', 0) or None,
    }

    # Replace zero values with None for optional fields
    for k in ['in10y','in02y','in30y','iniryy','us02y','us10y','us30y']:
        if new_row[k] == 0:
            new_row[k] = None

    if new_row['nifty'] > 0:
        new_df = pd.DataFrame([new_row])
        # Remove existing row for same date if any
        df = df[df['date'] != new_row['date']]
        df = pd.concat([df, new_df], ignore_index=True)
        df = df.sort_values('date').reset_index(drop=True)
        st.session_state.df_raw = df
        st.toast(f"✅ Added data for {new_row['date'].strftime('%Y-%m-%d')}", icon="✅")
    else:
        st.toast("⚠️ NIFTY close must be > 0", icon="⚠️")


# ──────────────────────────────────────────────
# Dashboard Tab
# ──────────────────────────────────────────────
def render_dashboard(df):
    s = get_latest_summary(df)
    if not s or s.get('nifty') is None:
        st.warning("No data available")
        return

    # Top row: Key metrics
    cols = st.columns(6)
    with cols[0]:
        chg = s.get('nifty_change')
        delta_str = f"{chg:+.2f}%" if pd.notna(chg) else None
        st.metric("NIFTY 50", f"{s['nifty']:,.2f}", delta_str)
    with cols[1]:
        st.metric("RSI (14)", f"{s['rsi']:.1f}" if pd.notna(s['rsi']) else "—")
    with cols[2]:
        b = s.get('breadth')
        st.metric("BREADTH", f"{b:.3f}" if pd.notna(b) else "—")
    with cols[3]:
        st.metric("PE", f"{s['nifty_pe']:.2f}" if pd.notna(s['nifty_pe']) else "—")
    with cols[4]:
        dev = s.get('pe_dev')
        st.metric("PE Deviation", f"{dev:+.2f}" if pd.notna(dev) else "—")
    with cols[5]:
        # Signal
        btd = s.get('btd', 0)
        stt = s.get('stt', 0)
        if btd == 1:
            st.markdown('<div class="metric-card signal-buy"><div class="label">SIGNAL</div><div class="value">🟢 BTD</div></div>', unsafe_allow_html=True)
        elif stt == -1:
            st.markdown('<div class="metric-card signal-sell"><div class="label">SIGNAL</div><div class="value">🔴 STT</div></div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="metric-card signal-neutral"><div class="label">SIGNAL</div><div class="value">⚪ NEUTRAL</div></div>', unsafe_allow_html=True)

    st.markdown("---")

    # Second row
    cols2 = st.columns(6)
    with cols2[0]:
        st.metric("AD Ratio", f"{s['ad_ratio']:.2f}" if pd.notna(s['ad_ratio']) else "—")
    with cols2[1]:
        st.metric("Oscillator Count", f"{s['count']:.1f}" if pd.notna(s['count']) else "—")
    with cols2[2]:
        st.metric("Spread 90", f"{s['spread90']:.2f}%" if pd.notna(s['spread90']) else "—")
    with cols2[3]:
        st.metric("Spread 200", f"{s['spread200']:.2f}%" if pd.notna(s['spread200']) else "—")
    with cols2[4]:
        boll = s.get('bol_lower')
        bolu = s.get('bol_upper')
        st.metric("Bol. Lower", f"{boll:,.0f}" if pd.notna(boll) else "—")
    with cols2[5]:
        st.metric("Bol. Upper", f"{bolu:,.0f}" if pd.notna(bolu) else "—")

    st.markdown(f"**Last Updated:** {s['date'].strftime('%d %b %Y') if hasattr(s['date'], 'strftime') else s['date']}")


# ──────────────────────────────────────────────
# Charts Tab
# ──────────────────────────────────────────────
def render_charts(df):
    # Date range selector
    col1, col2 = st.columns([1, 3])
    with col1:
        lookback = st.selectbox("Period", ["3M", "6M", "1Y", "2Y", "5Y", "MAX"], index=2)
    lbmap = {"3M": 63, "6M": 126, "1Y": 252, "2Y": 504, "5Y": 1260, "MAX": len(df)}
    n = min(lbmap[lookback], len(df))
    dfc = df.tail(n).copy()

    # Chart 1: NIFTY with Bollinger Bands
    fig1 = make_subplots(rows=4, cols=1, shared_xaxes=True,
                         vertical_spacing=0.03,
                         row_heights=[0.45, 0.18, 0.18, 0.19],
                         subplot_titles=("NIFTY 50 + Bollinger Bands", "RSI (14)",
                                         "Breadth (EMA)", "Oscillator Count"))

    # NIFTY + Bollinger
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty'], name='NIFTY',
                              line=dict(color='#00d4ff', width=1.5)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_ma20'], name='MA20',
                              line=dict(color='#ffa500', width=1, dash='dot')), row=1, col=1)
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['bol_upper'], name='Upper BB',
                              line=dict(color='rgba(255,65,65,0.5)', width=0.8)), row=1, col=1)
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['bol_lower'], name='Lower BB',
                              line=dict(color='rgba(0,255,65,0.5)', width=0.8),
                              fill='tonexty', fillcolor='rgba(100,100,100,0.1)'), row=1, col=1)
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_ma90'], name='MA90',
                              line=dict(color='#ff00ff', width=1, dash='dash')), row=1, col=1)

    # BTD/STT markers
    btd_pts = dfc[dfc['btd'] == 1]
    stt_pts = dfc[dfc['stt'] == -1]
    if len(btd_pts) > 0:
        fig1.add_trace(go.Scatter(x=btd_pts['date'], y=btd_pts['nifty'], mode='markers',
                                  name='BTD', marker=dict(color='lime', size=10, symbol='triangle-up')), row=1, col=1)
    if len(stt_pts) > 0:
        fig1.add_trace(go.Scatter(x=stt_pts['date'], y=stt_pts['nifty'], mode='markers',
                                  name='STT', marker=dict(color='red', size=10, symbol='triangle-down')), row=1, col=1)

    # RSI
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['rsi'], name='RSI',
                              line=dict(color='#00d4ff', width=1.2)), row=2, col=1)
    fig1.add_hline(y=70, line_dash="dash", line_color="red", opacity=0.5, row=2, col=1)
    fig1.add_hline(y=40, line_dash="dash", line_color="green", opacity=0.5, row=2, col=1)
    fig1.add_hline(y=50, line_dash="dot", line_color="gray", opacity=0.3, row=2, col=1)

    # Breadth
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['breadth'], name='Breadth',
                              line=dict(color='#ffa500', width=1.2)), row=3, col=1)
    fig1.add_trace(go.Scatter(x=dfc['date'], y=dfc['rel_breadth'], name='Rel Breadth',
                              line=dict(color='#00ff88', width=1, dash='dot')), row=3, col=1)
    fig1.add_hline(y=0.5, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    # Oscillator Count
    colors = ['#00ff41' if v > 0 else '#ff4141' for v in dfc['count'].fillna(0)]
    fig1.add_trace(go.Bar(x=dfc['date'], y=dfc['count'], name='Count',
                          marker_color=colors), row=4, col=1)

    fig1.update_layout(
        height=900, template='plotly_dark',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=60, r=20, t=60, b=40),
        showlegend=True,
    )
    fig1.update_xaxes(rangeslider_visible=False)
    st.plotly_chart(fig1, use_container_width=True)


def render_valuation_charts(df):
    """PE/EY valuation charts."""
    col1, col2 = st.columns([1, 3])
    with col1:
        lookback = st.selectbox("Valuation Period", ["1Y", "2Y", "5Y", "10Y", "MAX"], index=4, key='val_period')
    lbmap = {"1Y": 252, "2Y": 504, "5Y": 1260, "10Y": 2520, "MAX": len(df)}
    n = min(lbmap[lookback], len(df))
    dfc = df.tail(n).copy()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        row_heights=[0.4, 0.3, 0.3],
                        subplot_titles=("NIFTY PE vs Corrected PE", "PE Deviation",
                                        "Earnings Yield vs Corrected EY"))

    # PE
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_pe'], name='Actual PE',
                             line=dict(color='#00d4ff', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['cor_pe'], name='Corrected PE',
                             line=dict(color='#ffa500', width=1.2)), row=1, col=1)
    mean_pe = dfc['nifty_pe'].mean()
    fig.add_hline(y=mean_pe, line_dash="dot", line_color="gray", opacity=0.5, row=1, col=1,
                  annotation_text=f"Mean: {mean_pe:.1f}")

    # PE Deviation
    dev_colors = ['#00ff41' if v < 0 else '#ff4141' for v in dfc['pe_dev'].fillna(0)]
    fig.add_trace(go.Bar(x=dfc['date'], y=dfc['pe_dev'], name='PE Dev',
                         marker_color=dev_colors), row=2, col=1)

    # EY
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_ey'], name='Actual EY',
                             line=dict(color='#00d4ff', width=1.2)), row=3, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['cor_ey'], name='Corrected EY',
                             line=dict(color='#ffa500', width=1.2)), row=3, col=1)

    fig.update_layout(
        height=800, template='plotly_dark',
        legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
        margin=dict(l=60, r=20, t=60, b=40),
    )
    st.plotly_chart(fig, use_container_width=True)


def render_spread_charts(df):
    """MA Spread charts."""
    n = min(1260, len(df))
    dfc = df.tail(n).copy()

    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.08,
                        subplot_titles=("NIFTY vs MA90 & MA200(lag)", "Spreads (%)"))

    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty'], name='NIFTY',
                             line=dict(color='#00d4ff', width=1.2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_ma90'], name='MA90',
                             line=dict(color='#ffa500', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['nifty_ma200_lag'], name='MA200(lag)',
                             line=dict(color='#ff00ff', width=1)), row=1, col=1)

    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['spread90'], name='Spread90',
                             line=dict(color='#ffa500', width=1)), row=2, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['spread200'], name='Spread200',
                             line=dict(color='#ff00ff', width=1)), row=2, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=2, col=1)

    fig.update_layout(height=600, template='plotly_dark',
                      legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                      margin=dict(l=60, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Regression Tab
# ──────────────────────────────────────────────
def render_regression(df):
    details = get_regression_details(df)
    if details is None:
        st.warning("Insufficient data for regression analysis")
        return

    st.subheader("Correlation-Weighted Regression Model")
    st.caption("Model: Y_pred = B₀ + Σ(corr_i × x_i), where B₀ = mean_Y − Σ(corr_i × mean_xi)")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**PE Regression Coefficients**")
        pe_df = details[['Variable', 'Mean', 'PE_Correl', 'PE_Slope']].copy()
        pe_df.columns = ['Variable', 'Mean', 'Correlation', 'Slope']
        pe_df['Correlation'] = pe_df['Correlation'].round(4)
        pe_df['Slope'] = pe_df['Slope'].round(4)
        pe_df['Mean'] = pe_df['Mean'].round(4)
        st.dataframe(pe_df, use_container_width=True, hide_index=True)

    with col2:
        st.markdown("**EY Regression Coefficients**")
        ey_df = details[['Variable', 'Mean', 'EY_Correl', 'EY_Slope']].copy()
        ey_df.columns = ['Variable', 'Mean', 'Correlation', 'Slope']
        ey_df['Correlation'] = ey_df['Correlation'].round(4)
        ey_df['Slope'] = ey_df['Slope'].round(4)
        ey_df['Mean'] = ey_df['Mean'].round(4)
        st.dataframe(ey_df, use_container_width=True, hide_index=True)

    # Correlation heatmap
    st.markdown("---")
    st.subheader("Variable Correlation Matrix")
    var_cols = [v[0] for v in REGRESSION_VARS_PE if v[0] in df.columns]
    var_cols_with_pe = ['nifty_pe', 'nifty_ey'] + var_cols
    corr_data = df[var_cols_with_pe].dropna()
    if len(corr_data) > 50:
        corr_matrix = corr_data.corr()
        labels = ['PE', 'EY'] + [v[1] for v in REGRESSION_VARS_PE if v[0] in df.columns]
        fig = go.Figure(data=go.Heatmap(
            z=corr_matrix.values, x=labels, y=labels,
            colorscale='RdBu_r', zmid=0, text=np.round(corr_matrix.values, 2),
            texttemplate='%{text}', textfont={"size": 9}
        ))
        fig.update_layout(height=500, template='plotly_dark',
                          margin=dict(l=60, r=20, t=40, b=40))
        st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Data Tab
# ──────────────────────────────────────────────
def render_data(df):
    st.subheader(f"Historical Data — {len(df)} rows")

    # Column groups
    col_groups = {
        "📊 Core": ['date', 'nifty', 'pct_change', 'ad_ratio', 'rsi', 'breadth'],
        "📈 Bollinger": ['date', 'nifty', 'nifty_ma20', 'bol_lower', 'bol_upper', 'nifty_std20'],
        "🔢 AD Features": ['date', 'ad_ratio', 'ad_normalized', 'ad_diff', 'ad_2d', 'ad_3d', 'ad_5d', 'rel_ad_ratio'],
        "📉 RSI": ['date', 'nifty', 'gain', 'loss', 'avg_gain', 'avg_loss', 'rs', 'rsi'],
        "🌊 Breadth": ['date', 'breadth', 'breadth_2d', 'breadth_3d', 'breadth_5d', 'breadth_8d', 'breadth_13d', 'breadth_21d', 'rel_breadth'],
        "💰 Valuation": ['date', 'nifty_pe', 'nifty_ey', 'cor_pe', 'pe_dev', 'cor_ey', 'ey_dev'],
        "📏 Spreads": ['date', 'nifty', 'nifty_ma90', 'nifty_ma200_lag', 'spread90', 'spread200'],
        "🎯 Signals": ['date', 'nifty', 'rsi', 'breadth', 'bol_lower', 'bol_upper', 'btd', 'stt', 'count', 'osc'],
        "🏛️ Macro Input": ['date', 'in10y', 'in02y', 'in30y', 'iniryy', 'repo', 'crr', 'us02y', 'us10y', 'us30y', 'us_fed'],
        "📋 All Columns": list(df.columns),
    }

    selected_group = st.selectbox("Column Group", list(col_groups.keys()))
    cols_to_show = [c for c in col_groups[selected_group] if c in df.columns]

    display_df = df[cols_to_show].tail(500).sort_values('date', ascending=False)

    # Format numeric columns
    for c in display_df.select_dtypes(include=[np.number]).columns:
        display_df[c] = display_df[c].round(4)

    st.dataframe(display_df, use_container_width=True, height=600, hide_index=True)

    # Download
    csv_buf = io.StringIO()
    df.to_csv(csv_buf, index=False)
    st.download_button("📥 Download Full Dataset (CSV)", csv_buf.getvalue(),
                       file_name=f"aarambh_data_{datetime.date.today()}.csv",
                       mime="text/csv")


# ──────────────────────────────────────────────
# Signal History Tab
# ──────────────────────────────────────────────
def render_signals(df):
    st.subheader("BTD / STT Signal History")
    st.caption("BTD: NIFTY < Lower BB AND RSI < 40 AND Breadth > 0 | STT: NIFTY > Upper BB AND RSI > 70 AND Breadth > 0")

    btd_df = df[df['btd'] == 1][['date', 'nifty', 'rsi', 'breadth', 'bol_lower', 'pct_change']].copy()
    stt_df = df[df['stt'] == -1][['date', 'nifty', 'rsi', 'breadth', 'bol_upper', 'pct_change']].copy()

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"### 🟢 Buy-The-Dip ({len(btd_df)} signals)")
        if len(btd_df) > 0:
            st.dataframe(btd_df.sort_values('date', ascending=False).round(2),
                         use_container_width=True, hide_index=True, height=400)
    with col2:
        st.markdown(f"### 🔴 Sell-The-Top ({len(stt_df)} signals)")
        if len(stt_df) > 0:
            st.dataframe(stt_df.sort_values('date', ascending=False).round(2),
                         use_container_width=True, hide_index=True, height=400)


# ──────────────────────────────────────────────
# AD Features Tab
# ──────────────────────────────────────────────
def render_ad_features(df):
    n = min(504, len(df))
    dfc = df.tail(n).copy()

    fig = make_subplots(rows=3, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05,
                        subplot_titles=("AD Ratio & Normalized", "AD Diff Multi-Day MAs",
                                        "Relative AD Ratio"))

    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['ad_ratio'], name='AD Ratio',
                             line=dict(color='#00d4ff', width=1)), row=1, col=1)
    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['ad_normalized'], name='A/(A+D)',
                             line=dict(color='#ffa500', width=1)), row=1, col=1)

    for col, name, color in [('ad_2d','2D','#ff6b6b'), ('ad_3d','3D','#ffa500'),
                              ('ad_5d','5D','#00ff88')]:
        fig.add_trace(go.Scatter(x=dfc['date'], y=dfc[col], name=name,
                                 line=dict(width=1, color=color)), row=2, col=1)

    fig.add_trace(go.Scatter(x=dfc['date'], y=dfc['rel_ad_ratio'], name='Rel AD',
                             line=dict(color='#00d4ff', width=1.2)), row=3, col=1)
    fig.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5, row=3, col=1)

    fig.update_layout(height=700, template='plotly_dark',
                      legend=dict(orientation='h', y=1.02, x=0.5, xanchor='center'),
                      margin=dict(l=60, r=20, t=60, b=40))
    st.plotly_chart(fig, use_container_width=True)


# ──────────────────────────────────────────────
# Main App
# ──────────────────────────────────────────────
def main():
    params = render_sidebar()

    # Load and compute
    df_raw = get_session_data()
    if df_raw.empty:
        st.error("No data loaded. Please check data/historical.csv")
        return

    with st.spinner("Computing all indicators..."):
        df = compute_all(df_raw, params)

    # Title
    st.markdown("# 📊 AARAMBH")
    st.caption("Nifty50 Market Evaluation System — Complete Excel Replica")

    # Tabs
    tabs = st.tabs(["🏠 Dashboard", "📈 Price & Technicals", "💰 Valuation",
                     "📏 Spreads", "🔬 Regression", "🎯 Signals",
                     "🔄 AD Features", "📋 Data"])

    with tabs[0]:
        render_dashboard(df)
    with tabs[1]:
        render_charts(df)
    with tabs[2]:
        render_valuation_charts(df)
    with tabs[3]:
        render_spread_charts(df)
    with tabs[4]:
        render_regression(df)
    with tabs[5]:
        render_signals(df)
    with tabs[6]:
        render_ad_features(df)
    with tabs[7]:
        render_data(df)


if __name__ == "__main__":
    main()
