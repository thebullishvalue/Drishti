"""
TATTVA (तत्त्व) - Essence Matrix | Powered by Nirnay Design System
A Hemrek Capital Product

Advanced Quant Engine — Dream Implementation
All values rounded to 2 decimals.
Includes Recommended Feature Utilization Summary.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
from scipy.stats import spearmanr
from sklearn.model_selection import KFold
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
from sklearn.cluster import SpectralClustering
import warnings

warnings.filterwarnings("ignore")

# =============================================================================
# NIRNAY DESIGN SYSTEM CSS (EXACT COPY - 100% identical look & feel)
# =============================================================================

st.set_page_config(
    page_title="TATTVA | Essence Matrix",
    layout="wide",
    page_icon="✦",
    initial_sidebar_state="expanded"
)

VERSION = "v5.1-DreamEngine"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

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
        --neutral: #888888;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
    [data-testid="collapsedControl"] {
        display: flex !important; visibility: visible !important; opacity: 1 !important;
        background-color: var(--secondary-background-color) !important;
        border: 2px solid var(--primary-color) !important; border-radius: 8px !important;
        padding: 10px !important; margin: 12px !important;
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.4) !important;
        z-index: 999999 !important; position: fixed !important; top: 14px !important; left: 14px !important;
        width: 40px !important; height: 40px !important; align-items: center !important; justify-content: center !important;
    }
    [data-testid="collapsedControl"]:hover {
        background-color: rgba(var(--primary-rgb), 0.2) !important;
        box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.6) !important; transform: scale(1.05);
    }
    [data-testid="collapsedControl"] svg { stroke: var(--primary-color) !important; width: 20px !important; height: 20px !important; }
    
    .premium-header {
        background: var(--secondary-background-color); padding: 1.25rem 2rem; border-radius: 16px;
        margin-bottom: 1.5rem; box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.1);
        border: 1px solid var(--border-color); position: relative; overflow: hidden; margin-top: 1rem;
    }
    .premium-header::before {
        content: ''; position: absolute; top: 0; left: 0; right: 0; bottom: 0;
        background: radial-gradient(circle at 20% 50%, rgba(var(--primary-rgb),0.08) 0%, transparent 50%); pointer-events: none;
    }
    .premium-header h1 { margin: 0; font-size: 2rem; font-weight: 700; color: var(--text-primary); letter-spacing: -0.50px; position: relative; }
    .premium-header .tagline { color: var(--text-muted); font-size: 0.9rem; margin-top: 0.25rem; font-weight: 400; position: relative; }
    
    .metric-card {
        background-color: var(--bg-card); padding: 1.25rem; border-radius: 12px; border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); margin-bottom: 0.5rem;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); position: relative; overflow: hidden;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.neutral h2 { color: var(--neutral); }
    
    .signal-card { background: var(--bg-card); border-radius: 12px; border: 1px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }
    .signal-card.buy::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--success-green); }
    .signal-card.sell::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; background: var(--danger-red); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# DREAM ENGINE (your exact code)
# =============================================================================

def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / mad

def bootstrap_mean_std(func, x, y, n=50):
    vals = []
    N = len(x)
    for _ in range(n):
        idx = np.random.choice(N, N, replace=True)
        vals.append(func(x[idx], y[idx]))
    return np.mean(vals), np.std(vals)

def distance_correlation(x, y):
    x = x.reshape(-1,1)
    y = y.reshape(-1,1)
    A = squareform(pdist(x))
    B = squareform(pdist(y))
    A -= A.mean(axis=0)
    B -= B.mean(axis=0)
    dcov = np.sqrt((A * B).mean())
    dvarx = np.sqrt((A * A).mean())
    dvary = np.sqrt((B * B).mean())
    return 0 if dvarx*dvary == 0 else dcov/np.sqrt(dvarx*dvary)

def stability_selection(X, y, n_runs=30):
    N, p = X.shape
    scores = np.zeros(p)
    for _ in range(n_runs):
        idx = np.random.choice(N, int(N*0.7), replace=False)
        model = RandomForestRegressor(n_estimators=100, max_depth=6)
        model.fit(X[idx], y[idx])
        scores += model.feature_importances_
    return scores / n_runs

def cross_val_predictive_power(X, y):
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    rf_scores = []
    ridge_scores = []
    for tr, te in kf.split(X):
        rf = RandomForestRegressor(n_estimators=200, max_depth=8)
        ridge = Ridge(alpha=1.0)
        rf.fit(X[tr], y[tr])
        ridge.fit(X[tr], y[tr])
        rf_scores.append(r2_score(y[te], rf.predict(X[te])))
        ridge_scores.append(r2_score(y[te], ridge.predict(X[te])))
    return np.mean(rf_scores), np.mean(ridge_scores)

def redundancy_clustering(X):
    corr = np.corrcoef(X.T)
    clustering = SpectralClustering(
        n_clusters=min(5, X.shape[1]),
        affinity='precomputed',
        random_state=42
    )
    labels = clustering.fit_predict(np.abs(corr))
    return labels

class TattvaEngine:
    def __init__(self, data, target_col, feature_cols, date_col=None):
        self.data = data.copy()
        self.target = target_col
        self.features = feature_cols
        self.date_col = date_col
        self.scaler = StandardScaler()
        self.X = self.scaler.fit_transform(self.data[self.features])
        self.y = robust_zscore(self.data[self.target].values)
        self.results = []

    def analyze(self, progress_callback=None):
        if progress_callback: progress_callback(10)
        rf_cv, ridge_cv = cross_val_predictive_power(self.X, self.y)
        if progress_callback: progress_callback(25)
        stability = stability_selection(self.X, self.y)
        if progress_callback: progress_callback(40)
        clusters = redundancy_clustering(self.X)
        if progress_callback: progress_callback(55)
        for i, feat in enumerate(self.features):
            x = self.X[:, i]
            pearson = np.corrcoef(x, self.y)[0,1]
            spear = spearmanr(x, self.y).correlation
            dcor = distance_correlation(x, self.y)
            boot_mean, boot_std = bootstrap_mean_std(
                lambda a,b: np.corrcoef(a,b)[0,1], x, self.y
            )
            nonlinear_bias = dcor - abs(pearson)
            cluster_penalty = 1 / (1 + np.sum(clusters == clusters[i]))
            predictive_strength = (
                abs(pearson)*0.15 +
                abs(spear)*0.10 +
                dcor*0.25 +
                stability[i]*0.25 +
                cluster_penalty*0.10 +
                rf_cv*0.10 +
                ridge_cv*0.05
            )
            uncertainty_penalty = np.exp(-boot_std)
            composite = predictive_strength * uncertainty_penalty
            self.results.append({
                "Feature": feat,
                "Pearson": round(pearson,2),
                "Spearman": round(spear,2),
                "Distance_Corr": round(dcor,2),
                "Stability": round(stability[i],2),
                "Cluster_Label": int(clusters[i]),
                "NonLinear_Bias": round(nonlinear_bias,2),
                "Uncertainty": round(boot_std,3),
                "Composite_Score": round(composite*100,2)
            })
        if progress_callback: progress_callback(90)
        self.res_df = pd.DataFrame(self.results)
        self.res_df = self.res_df.sort_values("Composite_Score", ascending=False).reset_index(drop=True)
        if progress_callback: progress_callback(100)

    def get_insights(self):
        df = self.res_df
        hidden = df[
            (df["NonLinear_Bias"] > 0.2) &
            (df["Composite_Score"] > df["Composite_Score"].median())
        ]["Feature"].tolist()
        redundant = df.groupby("Cluster_Label")["Feature"].apply(list)
        redundant = [group[1:] for group in redundant if len(group) > 1]
        redundant = [f for sub in redundant for f in sub]
        return {
            "top_feature": df.iloc[0]["Feature"],
            "top_score": df.iloc[0]["Composite_Score"],
            "hidden_nonlinear": hidden[:3],
            "redundant_features": redundant[:5]
        }


# =============================================================================
# DATA UTILITIES (unchanged)
# =============================================================================

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
        gid = re.search(r'gid=(\d+)', sheet_url).group(1) if re.search(r'gid=(\d+)', sheet_url) else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        return pd.read_csv(csv_url), None
    except Exception as e:
        return None, str(e)

def clean_data(df, target, features, date_col=None):
    cols = [target] + features
    if date_col and date_col != "None": cols.append(date_col)
    data = df[cols].copy()
    for c in [target] + features:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    data = data.dropna()
    data = data[np.isfinite(data[[target] + features]).all(axis=1)]
    if date_col and date_col != "None":
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col]).sort_values(date_col)
        except:
            pass
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


# =============================================================================
# MAIN APPLICATION — NIRNAY LOOK & FEEL
# =============================================================================

def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">TATTVA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">तत्त्व | Essence Matrix</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">📁 DATA SOURCE</div>', unsafe_allow_html=True)
        data_source = st.radio("Source", ["📤 Upload CSV/Excel", "📊 Google Sheets"], horizontal=True)

        df = None
        if data_source == "📤 Upload CSV/Excel":
            uploaded = st.file_uploader("Choose file", type=['csv', 'xlsx'])
            if uploaded:
                df = pd.read_csv(uploaded) if uploaded.name.endswith('.csv') else pd.read_excel(uploaded)
        else:
            url = st.text_input("Google Sheet URL")
            if st.button("LOAD SHEET"):
                df, err = load_google_sheet(url)
                if err:
                    st.error(err)

        if df is None:
            st.info("Upload data or load sheet to begin")
            return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 TARGET & FEATURES</div>', unsafe_allow_html=True)
        target = st.selectbox("Target Variable (Y)", numeric_cols)
        features = st.multiselect("Features (X)", [c for c in numeric_cols if c != target], default=[c for c in numeric_cols if c != target][:10])

        date_col = st.selectbox("Date Column (optional)", ["None"] + list(df.columns), index=0)

        run_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)

    if not run_btn:
        st.markdown("""
        <div class="premium-header">
            <h1>TATTVA : Essence Matrix</h1>
            <div class="tagline">Revealing the fundamental predictive truth of your features.</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("Configure target/features in sidebar and click RUN ANALYSIS")
        return

    data = clean_data(df, target, features, date_col if date_col != "None" else None)
    if len(data) < 30:
        st.error("Need at least 30 clean rows")
        return

    with st.spinner("Computing Dream Engine..."):
        progress_bar = st.progress(0)
        def cb(p): progress_bar.progress(p)
        engine = TattvaEngine(data, target, features, date_col if date_col != "None" else None)
        engine.analyze(cb)
        progress_bar.empty()

    res_df = engine.res_df
    insights = engine.get_insights()

    # Top metrics (Nirnay style)
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card primary"><h4>Primary Essence</h4><h2>{insights["top_feature"]}</h2><div class="sub-metric">Score: {insights["top_score"]:.2f}</div></div>', unsafe_allow_html=True)
    with c2:
        n_hidden = len(insights["hidden_nonlinear"])
        st.markdown(f'<div class="metric-card info"><h4>Hidden Gems</h4><h2>{n_hidden}</h2><div class="sub-metric">Non-linear value</div></div>', unsafe_allow_html=True)
    with c3:
        n_red = len(insights["redundant_features"])
        st.markdown(f'<div class="metric-card danger"><h4>Redundant</h4><h2>{n_red}</h2><div class="sub-metric">Cluster groups</div></div>', unsafe_allow_html=True)
    with c4:
        st.markdown(f'<div class="metric-card neutral"><h4>Stability</h4><h2>{res_df["Stability"].mean():.2f}</h2><div class="sub-metric">Avg stability</div></div>', unsafe_allow_html=True)

    # Recommended Summary (Nirnay style)
    st.markdown("### Recommended Feature Utilization")
    colA, colB = st.columns([3,2])
    with colA:
        keep = res_df[res_df['Composite_Score'] > 70]['Feature'].tolist()[:8]
        if keep:
            st.markdown("**Keep** (high essence):  \n" + " • " + "  \n • ".join(keep))
        if insights["hidden_nonlinear"]:
            st.markdown("\n**Non-linear gems**:  \n" + " • " + "  \n • ".join(insights["hidden_nonlinear"]))
    with colB:
        if insights["redundant_features"]:
            st.markdown("**Drop candidates** (redundant clusters):  \n" + " • " + "  \n • ".join(insights["redundant_features"][:6]))
        else:
            st.markdown("**No redundancy detected**")

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # Tabs (Nirnay style)
    tab1, tab2 = st.tabs(["**🎯 Essence Ranking**", "**📋 Full Table**"])

    with tab1:
        fig = px.bar(res_df, x='Composite_Score', y='Feature', orientation='h',
                     color='Composite_Score', color_continuous_scale=['#ef4444','#f59e0b','#10b981','#FFC300'])
        fig.update_layout(height=500, yaxis={'categoryorder':'total ascending'})
        update_chart_theme(fig)
        st.plotly_chart(fig, width='stretch')

    with tab2:
        disp = res_df.copy()
        for c in disp.select_dtypes(include=['float64']).columns:
            disp[c] = disp[c].round(2)
        st.dataframe(disp, width='stretch', hide_index=True, height=600)

        csv = disp.to_csv(index=False).encode()
        st.download_button("📥 Download Essence Matrix", csv, "tattva_essence.csv", "text/csv")

    # Footer (Nirnay style)
    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {ist.strftime('%Y-%m-%d %H:%M:%S IST')}")


if __name__ == "__main__":
    main()
