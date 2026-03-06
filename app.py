"""
TATTVA (तत्त्व) - MLR Engine | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Multivariate Linear Regression & Collinearity Diagnostics Engine.
Calculates true partial regression coefficients and eliminates overlapping macroeconomic noise.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import warnings

warnings.filterwarnings('ignore')

# --- Dependencies ---
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    STATSMODELS_AVAILABLE = False

# --- Constants ---
VERSION = "v2.0.0"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

# --- Page Config ---
st.set_page_config(
    page_title="TATTVA | MLR Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Premium CSS (Hemrek Design System) ---
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --bg-card: #1A1A1A;
        --bg-elevated: #2A2A2A;
        --text-primary: #EAEAEA;
        --text-secondary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --border-light: #3A3A3A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --warning-amber: #f59e0b;
        --info-cyan: #06b6d4;
        --purple: #8b5cf6;
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    /* Sidebar toggle button - always visible */
    [data-testid="collapsedControl"] {
        display: flex !important;
        visibility: visible !important;
        opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important;
        border-radius: 8px !important;
        padding: 10px !important;
        margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important;
        position: fixed !important;
        top: 14px !important;
        left: 14px !important;
        width: 40px !important;
        height: 40px !important;
        align-items: center !important;
        justify-content: center !important;
    }
    
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important;
        transform: scale(1.05);
    }
    
    [data-testid="collapsedControl"] svg {
        stroke: var(--primary-color) !important;
        width: 20px !important;
        height: 20px !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color);
        position: relative;
        overflow: hidden;
        margin-top: 1rem;
    }
    
    .premium-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%);
        pointer-events: none;
    }
    
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    
    .metric-card {
        background-color: var(--bg-card);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
        min-height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px;}
    .metric-card h3 { color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.purple h2 { color: var(--purple); }

    .signal-card { background: var(--bg-card); border-radius: 16px; border: 2px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }
    .signal-card.fair { border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }
    .signal-card.danger { border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }
    
    .guide-box { background: rgba(var(--primary-rgb), 0.05); border-left: 3px solid var(--primary-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: var(--text-secondary); font-size: 0.9rem; }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MULTIVARIATE LINEAR REGRESSION ENGINE
# ============================================================================

class MLREngine:
    """
    Multivariate Linear Regression & Collinearity Engine.
    Computes true partial slopes and measures Variance Inflation Factors (VIF).
    """
    
    def __init__(self, df, target, features):
        self.df = df.copy()
        self.target = target
        self.features = features
        self.model = None
        self.vif_data = None
        self.coef_df = None
        
        # Prepare Data
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_with_const = sm.add_constant(self.X)
        
    def fit(self):
        """Fit the OLS model and calculate VIFs."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is required for the MLR Engine.")
            
        # Fit OLS
        self.model = sm.OLS(self.y, self.X_with_const).fit()
        
        # Extract Coefficients into clean DataFrame
        self.coef_df = pd.DataFrame({
            'Variable': self.model.params.index,
            'Coefficient (Slope)': self.model.params.values,
            'Standard Error': self.model.bse.values,
            't-Statistic': self.model.tvalues.values,
            'p-Value': self.model.pvalues.values
        })
        
        # Compute VIF
        self._compute_vif()
        return self
        
    def _compute_vif(self):
        """Calculate Variance Inflation Factor for each independent variable."""
        vif_df = pd.DataFrame()
        vif_df["Variable"] = self.X.columns
        
        vifs = []
        for i in range(len(self.X.columns)):
            try:
                # Catch perfect collinearity warnings/errors
                v = variance_inflation_factor(self.X.values, i)
                vifs.append(v)
            except Exception:
                vifs.append(np.inf)
                
        vif_df["VIF Score"] = vifs
        # Map interpretations
        conditions = [
            (vif_df['VIF Score'] < 3),
            (vif_df['VIF Score'] >= 3) & (vif_df['VIF Score'] <= 5),
            (vif_df['VIF Score'] > 5)
        ]
        choices = ['Excellent (Uncorrelated)', 'Acceptable (Moderate Noise)', 'Severe Collinearity (DROP THIS)']
        vif_df['Status'] = np.select(conditions, choices, default='Unknown')
        
        self.vif_data = vif_df.sort_values(by="VIF Score", ascending=False).reset_index(drop=True)

    def get_predictions(self):
        return self.model.predict(self.X_with_const)

# ============================================================================
# DATA UTILITIES
# ============================================================================

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)

def clean_data(df, target, features):
    cols = [target] + features
    data = df[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    return data.reset_index(drop=True)

def update_chart_theme(fig):
    fig.update_layout(
        template="plotly_dark", plot_bgcolor="#1A1A1A", paper_bgcolor="#1A1A1A",
        font=dict(family="Inter", color="#EAEAEA"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12)
    )
    return fig

# ============================================================================
# UI RENDERERS
# ============================================================================

def render_landing_page():
    """Renders the landing page content when no data is loaded."""
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>📐 Partial Coefficients</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Solves the "double-counting" trap. Calculates the true, isolated impact of a variable on your target by holding all other variables constant.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card danger' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--danger-red); margin-bottom: 0.5rem;'>⚠️ Multicollinearity Trap</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Adding simple slopes together breaks your model. If you use the 10Y Yield, 30Y Yield, and Repo Rate simultaneously, your model exaggerates moves because they all overlap.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>🔍 VIF Diagnostics</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                The Variance Inflation Factor (VIF) mathematically identifies overlapping signals. Drop variables with a VIF > 5 to purify your forecasting engine.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 How to use this engine:</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            1. Use the <strong>Sidebar</strong> to upload your raw historical dataset (CSV/Excel) or connect a Google Sheet.<br>
            2. Select your <strong>Target Variable (Y)</strong> and your suspected <strong>Predictors (X)</strong>.<br>
            3. Go to the <strong>VIF Diagnostics</strong> tab. If any variable has a VIF > 5, remove it from the sidebar.<br>
            4. Once VIFs are clean, use the slopes in the <strong>Regression Output</strong> tab for your live Excel model.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer():
    """Render dynamic footer with IST time"""
    utc_now = datetime.utcnow()
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

def highlight_vif(val):
    """Pandas styler for VIF column"""
    if isinstance(val, (int, float)):
        if val > 5:
            return 'background-color: rgba(239, 68, 68, 0.2); color: #ef4444; font-weight: bold;'
        elif val > 3:
            return 'background-color: rgba(245, 158, 11, 0.2); color: #f59e0b;'
        else:
            return 'color: #10b981;'
    return ''

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    if not STATSMODELS_AVAILABLE:
        st.error("Critical Dependency Missing: `statsmodels` library is required. Please install it.")
        return

    # --- Sidebar Configuration ---
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">TATTVA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">तत्त्व | MLR Engine</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        data_source = st.radio("Source", ["📤 Upload", "📊 Google Sheets"], horizontal=True, label_visibility="collapsed")
        
        df = None
        
        if data_source == "📤 Upload":
            uploaded_file = st.file_uploader("CSV/Excel", type=['csv', 'xlsx'], label_visibility="collapsed")
            if uploaded_file:
                try:
                    df = pd.read_csv(uploaded_file) if uploaded_file.name.endswith('.csv') else pd.read_excel(uploaded_file)
                except Exception as e:
                    st.error(f"Error: {e}")
                    return
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
            sheet_url = st.text_input("Sheet URL", value=default_url, label_visibility="collapsed")
            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed: {error}")
                        return
                    if 'mlr_cache' in st.session_state:
                        del st.session_state.mlr_cache
                    st.session_state['data'] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if 'data' in st.session_state:
                df = st.session_state['data']
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Show landing page if no data
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <h1>TATTVA : MLR Engine</h1>
            <div class="tagline">Multivariate Linear Regression & Collinearity Diagnostics</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return
    
    # --- Model Configuration (Sidebar) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns to perform regression.")
        return
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Model Configuration</div>', unsafe_allow_html=True)
        
        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        target_col = st.selectbox("Dependent Variable (Y)", numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
        
        available = [c for c in numeric_cols if c != target_col]
        
        # User selection for X variables
        feature_cols = st.multiselect("Independent Variables (X)", available, default=available[:3])
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> OLS statsmodels
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    if not feature_cols:
        st.markdown("""
        <div class="premium-header">
            <h1>TATTVA : MLR Engine</h1>
            <div class="tagline">Multivariate Linear Regression & Collinearity Diagnostics</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Please select Independent Variables (X) from the sidebar to generate the model.")
        render_footer()
        return

    # --- Run Model ---
    data = clean_data(df, target_col, feature_cols)
    if len(data) < len(feature_cols) + 2:
        st.error("Not enough data points relative to the number of features selected.")
        return

    cache_key = f"mlr_{target_col}_{'-'.join(sorted(feature_cols))}_{len(data)}"
    
    if 'mlr_cache' not in st.session_state or st.session_state.mlr_cache_key != cache_key:
        with st.spinner("Computing Partial Coefficients and VIFs..."):
            engine = MLREngine(data, target_col, feature_cols)
            engine.fit()
            st.session_state.mlr_engine = engine
            st.session_state.mlr_cache_key = cache_key
            
    engine = st.session_state.mlr_engine

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY METRICS
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    
    max_vif = engine.vif_data['VIF Score'].max()
    r_squared = engine.model.rsquared
    adj_r_squared = engine.model.rsquared_adj
    f_pvalue = engine.model.f_pvalue
    
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f'<div class="metric-card primary"><h4>Model Fit (R²)</h4><h2>{r_squared:.4f}</h2><div class="sub-metric">Adj R²: {adj_r_squared:.4f}</div></div>', unsafe_allow_html=True)
    
    with c2:
        vif_color = "success" if max_vif < 3 else "warning" if max_vif <= 5 else "danger"
        st.markdown(f'<div class="metric-card {vif_color}"><h4>Max VIF</h4><h2>{max_vif:.2f}</h2><div class="sub-metric">Target: < 5.0</div></div>', unsafe_allow_html=True)
    
    with c3:
        p_color = "success" if f_pvalue < 0.05 else "danger"
        st.markdown(f'<div class="metric-card {p_color}"><h4>Model F-Test (p-val)</h4><h2>{f_pvalue:.4f}</h2><div class="sub-metric">Significance of full model</div></div>', unsafe_allow_html=True)
        
    with c4:
        st.markdown(f'<div class="metric-card info"><h4>Observations</h4><h2>{len(engine.y)}</h2><div class="sub-metric">Rows analyzed</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TABS
    # ═══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3 = st.tabs([
        "**📐 Partial Coefficients (Slopes)**",
        "**🔍 Collinearity Diagnostics (VIF)**",
        "**📊 Visualizations**"
    ])
    
    # --- TAB 1: Coefficients ---
    with tab1:
        st.markdown("##### True Partial Regression Coefficients")
        st.markdown("""<p style="color: #888; font-size: 0.9rem;">
        These are your mathematically isolated slopes. They have been stripped of the "double-counting" noise. 
        Use the <b>Coefficient (Slope)</b> column in your Excel formulas.</p>""", unsafe_allow_html=True)
        
        # Style the coefficient dataframe
        styled_coef = engine.coef_df.style.format({
            'Coefficient (Slope)': "{:.5f}",
            'Standard Error': "{:.5f}",
            't-Statistic': "{:.3f}",
            'p-Value': "{:.4f}"
        }).applymap(lambda x: 'color: #ef4444;' if isinstance(x, float) and x > 0.05 else 'color: #10b981;', subset=['p-Value'])
        
        st.dataframe(styled_coef, width=1200, height=400)
        
        st.markdown("""
        <div class="guide-box success">
            <strong>How to read p-Values:</strong><br>
            If a variable's p-Value is <span style="color: #10b981;">Green (< 0.05)</span>, it is a statistically significant predictor.<br>
            If a variable's p-Value is <span style="color: #ef4444;">Red (> 0.05)</span>, it adds no predictive value in the presence of the other variables and should likely be removed.
        </div>
        """, unsafe_allow_html=True)
        
        with st.expander("Raw Statsmodels Output"):
            st.text(engine.model.summary().as_text())

    # --- TAB 2: VIF Diagnostics ---
    with tab2:
        st.markdown("##### Variance Inflation Factor (VIF)")
        
        if max_vif > 5:
            st.markdown("""
            <div class="signal-card danger" style="padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: var(--danger-red); margin: 0 0 0.5rem 0;">⚠️ Multicollinearity Detected</h4>
                <p style="margin: 0; font-size: 0.9rem;">One or more variables have a VIF score greater than 5. Your model is currently unstable due to overlapping signals. <b>Remove the variable with the highest VIF score using the sidebar, and let the model recalculate.</b></p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="signal-card fair" style="padding: 1rem; margin-bottom: 1rem;">
                <h4 style="color: var(--primary-color); margin: 0 0 0.5rem 0;">✅ Clean Signal Geometry</h4>
                <p style="margin: 0; font-size: 0.9rem;">All variables have a VIF score under 5. Your model is stable and free of severe multicollinearity.</p>
            </div>
            """, unsafe_allow_html=True)
            
        styled_vif = engine.vif_data.style.format({
            'VIF Score': "{:.2f}"
        }).applymap(highlight_vif, subset=['VIF Score'])
        
        st.dataframe(styled_vif, width=1000)

    # --- TAB 3: Visualizations ---
    with tab3:
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            st.markdown("##### Actual vs Predicted")
            preds = engine.get_predictions()
            
            fig_pred = go.Figure()
            fig_pred.add_trace(go.Scatter(
                x=engine.y, y=preds, mode='markers', name='Predictions',
                marker=dict(color='#FFC300', size=6, opacity=0.7)
            ))
            
            # Perfect fit line
            min_val = min(engine.y.min(), preds.min())
            max_val = max(engine.y.max(), preds.max())
            fig_pred.add_trace(go.Scatter(
                x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit',
                line=dict(color='#06b6d4', dash='dash')
            ))
            
            fig_pred.update_layout(height=400, xaxis_title=f'Actual {target_col}', yaxis_title='Predicted')
            update_chart_theme(fig_pred)
            st.plotly_chart(fig_pred, use_container_width=True)
            
        with col_chart2:
            st.markdown("##### Feature Correlation Heatmap")
            corr_matrix = engine.df[[target_col] + feature_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto", 
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=400)
            update_chart_theme(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=True)
            
        st.markdown("---")
        st.markdown("##### Residuals Distribution")
        residuals = engine.model.resid
        fig_resid = px.histogram(
            residuals, nbins=50, title="Residual Histogram",
            color_discrete_sequence=['#8b5cf6']
        )
        fig_resid.update_layout(height=350, xaxis_title="Residual Value", yaxis_title="Frequency")
        update_chart_theme(fig_resid)
        st.plotly_chart(fig_resid, use_container_width=True)

    render_footer()

if __name__ == "__main__":
    main()
