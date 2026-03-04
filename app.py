"""
TATTVA (तत्त्व) - Essence Matrix | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced mathematical and physics-based feature essence extraction.
Reveals fundamental predictive truth using Information Theory, 
Energy Statistics (Distance Correlation), Game Theory (SHAP), 
Hybrid Topology, and Directional Causality.

v4.0-Ultimate Edition — SHAP + Hybrid Graph + Granger + dCor core
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from scipy.spatial.distance import pdist, squareform
import warnings

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# Dependencies
# ────────────────────────────────────────────────────────────────────────────
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.inspection import permutation_importance
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False

# ────────────────────────────────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────────────────────────────────
VERSION = "v4.0-Ultimate"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

st.set_page_config(
    page_title="TATTVA | Essence Matrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ────────────────────────────────────────────────────────────────────────────
# Premium CSS (Nirnay Design System) — unchanged
# ────────────────────────────────────────────────────────────────────────────
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
        min-height: 160px; display: flex; flex-direction: column; justify-content: center;
    }
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; min-height: 30px; display: flex; align-items: center; }
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
    .metric-card.neutral h2 { color: var(--neutral); }

    .guide-box { background: rgba(var(--primary-rgb), 0.05); border-left: 3px solid var(--primary-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: var(--text-secondary); font-size: 0.9rem; }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }
    
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


# ────────────────────────────────────────────────────────────────────────────
# Utilities
# ────────────────────────────────────────────────────────────────────────────

def distance_correlation(x, y):
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = len(x)
    if n > 1500:
        idx = np.random.choice(n, 1500, replace=False)
        x = x[idx]
        y = y[idx]
        n = 1500
    A = squareform(pdist(x[:, None]))
    B = squareform(pdist(y[:, None]))
    A_mean_row = np.mean(A, axis=1, keepdims=True)
    A_mean_col = np.mean(A, axis=0, keepdims=True)
    A_mean = np.mean(A)
    A_cent = A - A_mean_row - A_mean_col + A_mean
    B_mean_row = np.mean(B, axis=1, keepdims=True)
    B_mean_col = np.mean(B, axis=0, keepdims=True)
    B_mean = np.mean(B)
    B_cent = B - B_mean_row - B_mean_col + B_mean
    dcov2 = np.sum(A_cent * B_cent) / (n**2)
    dvarx2 = np.sum(A_cent * A_cent) / (n**2)
    dvary2 = np.sum(B_cent * B_cent) / (n**2)
    if dvarx2 > 0 and dvary2 > 0:
        dcor = np.sqrt(dcov2 / np.sqrt(dvarx2 * dvary2))
        return float(np.clip(dcor, 0.0, 1.0))
    return 0.0


def pagerank_centrality(adj, damping=0.85, max_iter=100, tol=1e-6):
    n = adj.shape[0]
    deg = adj.sum(axis=0, keepdims=True)
    deg[deg == 0] = 1
    P = adj / deg
    pr = np.ones(n) / n
    for _ in range(max_iter):
        new_pr = damping * (P @ pr) + (1 - damping) / n
        if np.max(np.abs(new_pr - pr)) < tol:
            break
        pr = new_pr
    return pr / (np.sum(pr) + 1e-12)


def granger_causality_score(x, y, maxlag=4):
    if len(x) < 30 or not STATSMODELS_AVAILABLE:
        return 0.0
    try:
        dfg = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(dfg) < 30:
            return 0.0
        results = grangercausalitytests(dfg[['y', 'x']], maxlag=maxlag, verbose=False)
        pvals = [results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
        min_p = min(pvals)
        return float(np.clip(-np.log10(min_p + 1e-12) / 4.0, 0.0, 1.0))
    except:
        return 0.0


# ────────────────────────────────────────────────────────────────────────────
# Core Engine
# ────────────────────────────────────────────────────────────────────────────

class TattvaEngine:
    def __init__(self, data, target_col, feature_cols, date_col=None):
        self.data = data.copy()
        self.target = target_col
        self.features = feature_cols
        self.date_col = date_col

        if SKLEARN_AVAILABLE:
            self.scaler = StandardScaler()
            self.X_scaled = self.scaler.fit_transform(self.data[self.features])
            self.y_scaled = self.scaler.fit_transform(self.data[[self.target]]).flatten()
        else:
            self.X_scaled = self.data[self.features].values
            self.y_scaled = self.data[self.target].values

        self.results = []
        self.vif_data = {}
        self.corr_matrix = None
        self.shap_importance = None
        self.topo_scores = None

    def analyze(self):
        y = self.data[self.target].values
        X_df = self.data[self.features]

        self.corr_matrix = self.data[[self.target] + self.features].corr(method='pearson')

        if STATSMODELS_AVAILABLE:
            try:
                X_with_const = sm.add_constant(X_df)
                for i, col in enumerate(self.features):
                    self.vif_data[col] = variance_inflation_factor(X_with_const.values, i+1)
            except:
                self.vif_data = {f: np.nan for f in self.features}
        else:
            self.vif_data = {f: 1.0 for f in self.features}

        mi_scores = mutual_info_regression(self.X_scaled, self.y_scaled) if SKLEARN_AVAILABLE else np.zeros(len(self.features))

        rf_importance = np.zeros(len(self.features))
        if SKLEARN_AVAILABLE:
            rf = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            rf.fit(self.X_scaled, self.y_scaled)
            rf_importance = rf.feature_importances_

        # SHAP / Permutation
        adv_imp = np.zeros(len(self.features))
        if SKLEARN_AVAILABLE:
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(rf)
                    shap_vals = explainer.shap_values(self.X_scaled)
                    adv_imp = np.abs(shap_vals).mean(0)
                except:
                    pass
            if np.all(adv_imp == 0):
                try:
                    perm = permutation_importance(rf, self.X_scaled, self.y_scaled, n_repeats=10, random_state=42)
                    adv_imp = perm.importances_mean
                except:
                    adv_imp = rf_importance.copy()

        # Hybrid Topology
        if len(self.features) > 1:
            dcor_mat = np.eye(len(self.features))
            for i in range(len(self.features)):
                for j in range(i+1, len(self.features)):
                    d = distance_correlation(self.X_scaled[:,i], self.X_scaled[:,j])
                    dcor_mat[i,j] = d
                    dcor_mat[j,i] = d
            try:
                eigenvalues, eigenvectors = np.linalg.eig(dcor_mat)
                idx = np.argmax(np.real(eigenvalues))
                eigen_c = np.abs(np.real(eigenvectors[:, idx]))
                eigen_c /= np.sum(eigen_c) + 1e-12

                pr_c = pagerank_centrality(dcor_mat)

                self.topo_scores = (eigen_c + pr_c) / 2
            except:
                self.topo_scores = np.ones(len(self.features)) / len(self.features)
        else:
            self.topo_scores = np.array([1.0])

        # Collect results
        for i, feat in enumerate(self.features):
            x = self.data[feat].values
            pearson = np.corrcoef(x, y)[0,1] if np.std(x) > 0 else 0
            spearman = pd.Series(x).corr(pd.Series(y), method='spearman')
            dcor = distance_correlation(self.X_scaled[:,i], self.y_scaled)
            granger = granger_causality_score(x, y) if self.date_col else 0.0

            self.results.append({
                'Feature': feat,
                'Pearson': pearson,
                'Abs_Pearson': abs(pearson),
                'Spearman': spearman,
                'Distance_Corr': dcor,
                'Mutual_Info': mi_scores[i],
                'RF_Importance': rf_importance[i],
                'Advanced_Importance': adv_imp[i],
                'VIF': self.vif_data.get(feat, np.nan),
                'Topological_Centrality': self.topo_scores[i],
                'Granger_Score': granger
            })

        self.res_df = pd.DataFrame(self.results)
        self._calculate_composite_score()

    def _calculate_composite_score(self):
        df = self.res_df
        def norm(col):
            mx, mn = df[col].max(), df[col].min()
            return np.zeros(len(df)) if mx == mn else (df[col] - mn) / (mx - mn)

        score = (
            norm('Abs_Pearson')        * 0.05 +
            norm('Spearman')           * 0.05 +
            norm('Distance_Corr')      * 0.18 +
            norm('Mutual_Info')        * 0.15 +
            norm('RF_Importance')      * 0.05 +
            norm('Advanced_Importance')* 0.25 +
            norm('Topological_Centrality') * 0.15 +
            norm('Granger_Score')      * 0.12
        ) * 100

        vif_penalty = np.where(df['VIF'] > 10, 0.8, 1.0)
        vif_penalty = np.where(df['VIF'] > 50, 0.5, vif_penalty)

        df['Composite_Score'] = score * vif_penalty
        self.res_df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)

    def get_insights(self):
        df = self.res_df
        top_feat = df.iloc[0]['Feature'] if not df.empty else "None"
        def norm(a): return np.zeros_like(a) if a.max()==a.min() else (a-a.min())/(a.max()-a.min())
        df['NonLinear_Bias'] = (norm(df['Distance_Corr']) + norm(df['Mutual_Info'])) / 2 - norm(df['Abs_Pearson'])
        hidden = df[df['NonLinear_Bias'] > 0.3]['Feature'].tolist()
        redundant = df[df['VIF'] > 10]['Feature'].tolist()
        return {
            'top_feature': top_feat,
            'top_score': df.iloc[0]['Composite_Score'] if not df.empty else 0,
            'hidden_nonlinear': hidden[:3],
            'redundant_features': redundant
        }


# ────────────────────────────────────────────────────────────────────────────
# Data helpers
# ────────────────────────────────────────────────────────────────────────────

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
    if date_col and date_col != "None":
        cols.append(date_col)
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
        template="plotly_dark",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(family="Inter", color="#EAEAEA"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        margin=dict(t=40, l=20, r=20, b=20),
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12)
    )
    return fig


def render_landing_page():
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🌌 Information Theory</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Measures the actual "bits" of predictive truth each feature carries.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Methodology:</strong><br>
                • Non-parametric density estimation<br>
                • K-Nearest Neighbors entropy
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>🧬 Energy Statistics</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Brownian Distance Correlation — true independence when dCor = 0.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Key:</strong><br>
                • Captures any dependence<br>
                • Physics-based energy distance
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>🕸️ Network Topology</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Maps redundancy & influence using VIF and full dCor graph centrality.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Outputs:</strong><br>
                • Redundancy flags<br>
                • Hybrid centrality ranking
            </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("### 🎯 Signal Interpretation")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""
        <div style='background: rgba(16,185,129,0.1); border:1px solid var(--success-green); border-radius:12px; padding:1.25rem;'>
            <h4 style='color:var(--success-green);'>🟢 High Essence Score</h4>
            <p style='color:var(--text-muted); font-size:0.85rem;'>Score > 75</p>
            <p style='color:var(--text-secondary); font-size:0.85rem; margin-top:0.5rem;'>
                Core predictive truth — must keep in models.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c2:
        st.markdown("""
        <div style='background: rgba(6,182,212,0.1); border:1px solid var(--info-cyan); border-radius:12px; padding:1.25rem;'>
            <h4 style='color:var(--info-cyan);'>🔵 Hidden Non-Linear Value</h4>
            <p style='color:var(--text-muted); font-size:0.85rem;'>Low Pearson, High dCor/MI</p>
            <p style='color:var(--text-secondary); font-size:0.85rem; margin-top:0.5rem;'>
                Invisible to linear models — powerful in trees/NNs.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with c3:
        st.markdown("""
        <div style='background: rgba(239,68,68,0.1); border:1px solid var(--danger-red); border-radius:12px; padding:1.25rem;'>
            <h4 style='color:var(--danger-red);'>🔴 Redundant / Toxic</h4>
            <p style='color:var(--text-muted); font-size:0.85rem;'>VIF > 10</p>
            <p style='color:var(--text-secondary); font-size:0.85rem; margin-top:0.5rem;'>
                High collinearity — usually drop.
            </p>
        </div>
        """, unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# Main Application
# ────────────────────────────────────────────────────────────────────────────

def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">TATTVA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">तत्त्व | Essence Matrix</div>
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
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
            sheet_url = st.text_input("Sheet URL", value=default_url, label_visibility="collapsed")
            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed: {error}")
                        return
                    st.session_state['data'] = df
                    if 'tattva_engine' in st.session_state: del st.session_state.tattva_engine
                    if 'tattva_cache' in st.session_state: del st.session_state.tattva_cache
                    st.toast("Data loaded!", icon="✅")
            if 'data' in st.session_state:
                df = st.session_state['data']

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    if df is None:
        st.markdown("""
        <div class="premium-header">
            <h1>TATTVA : Essence Matrix</h1>
            <div class="tagline">Revealing the fundamental predictive truth of your features.</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()

    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Target Configuration</div>', unsafe_allow_html=True)
        target_col = st.selectbox("Target Variable (Y)", numeric_cols, index=0)

        available = [c for c in numeric_cols if c != target_col]
        feature_cols = st.multiselect("Predictors (X)", available, default=available[:min(10, len(available))])

        if not feature_cols:
            st.info("Select at least one predictor to begin.")
            render_footer()
            return

        st.markdown('<div class="sidebar-title">📅 Context (Optional)</div>', unsafe_allow_html=True)
        date_candidates = [c for c in all_cols if 'date' in c.lower() or 'time' in c.lower()]
        date_col = st.selectbox("Time Axis", ["None"] + all_cols,
                                index=all_cols.index(date_candidates[0]) + 1 if date_candidates else 0)

        st.markdown('<br>', unsafe_allow_html=True)
        if st.button("🚀 RUN ANALYSIS", type="primary"):
            st.session_state.is_analyzed = True

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> SHAP • Hybrid Topology • dCor • Granger
            </p>
        </div>
        """, unsafe_allow_html=True)

    if 'is_analyzed' not in st.session_state:
        st.session_state.is_analyzed = False

    if not st.session_state.is_analyzed:
        st.markdown("""
        <div class="premium-header">
            <h1>TATTVA : Essence Matrix</h1>
            <div class="tagline">Revealing the fundamental predictive truth of your features.</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return

    data = clean_data(df, target_col, feature_cols, date_col if date_col != "None" else None)
    if len(data) < 30:
        st.error("Not enough clean rows (need ≥ 30 after cleaning).")
        return

    cache_key = f"{target_col}_{sorted(feature_cols)}_{len(data)}"
    if 'tattva_cache' not in st.session_state or st.session_state.tattva_cache != cache_key:
        with st.spinner("Extracting feature essence — SHAP, topology, causality..."):
            engine = TattvaEngine(data, target_col, feature_cols, date_col if date_col != "None" else None)
            engine.analyze()
            st.session_state.tattva_engine = engine
            st.session_state.tattva_cache = cache_key

    engine = st.session_state.tattva_engine
    res_df = engine.res_df
    insights = engine.get_insights()

    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.markdown(f'<div class="metric-card primary"><h4>Primary Essence</h4><h2>{insights["top_feature"]}</h2><div class="sub-metric">Score: {insights["top_score"]:.1f}/100</div></div>', unsafe_allow_html=True)
    with c2:
        n_hidden = len(insights["hidden_nonlinear"])
        txt = ", ".join(insights["hidden_nonlinear"])[:25] + "..." if n_hidden else "None"
        col = "info" if n_hidden > 0 else "neutral"
        st.markdown(f'<div class="metric-card {col}"><h4>Hidden Non-Linear Gems</h4><h2>{n_hidden}</h2><div class="sub-metric">{txt}</div></div>', unsafe_allow_html=True)
    with c3:
        n_red = len(insights["redundant_features"])
        col = "danger" if n_red > 0 else "success"
        st.markdown(f'<div class="metric-card {col}"><h4>Redundant (VIF>10)</h4><h2>{n_red}</h2><div class="sub-metric">Consider dropping</div></div>', unsafe_allow_html=True)
    with c4:
        avg_mi = res_df["Mutual_Info"].mean()
        col = "purple" if avg_mi > 0.1 else "warning"
        st.markdown(f'<div class="metric-card {col}"><h4>Information Gain</h4><h2>{avg_mi:.3f}</h2><div class="sub-metric">Average MI</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**🎯 Essence Ranking**",
        "**🧬 Correlations**",
        "**🌌 Advanced Dependence**",
        "**🕸️ Topology & VIF**",
        "**📋 Full Table**"
    ])

    with tab1:
        st.markdown("##### Composite Essence Score")
        fig_bar = px.bar(
            res_df, x='Composite_Score', y='Feature', orientation='h',
            color='Composite_Score', color_continuous_scale=['#ef4444','#f59e0b','#10b981','#FFC300'],
            hover_data=['Pearson','Distance_Corr','VIF']
        )
        fig_bar.update_layout(
            height=400 + min(len(feature_cols)*15, 400),
            yaxis={'categoryorder':'total ascending'},
            showlegend=False
        )
        update_chart_theme(fig_bar)
        st.plotly_chart(fig_bar, width='stretch')

        col1, col2 = st.columns(2)
        with col1:
            if insights["redundant_features"]:
                st.markdown(f"""
                <div class="guide-box danger">
                    <strong>⚠️ High Collinearity Detected</strong><br>
                    Features: {", ".join(insights["redundant_features"])}<br>
                    Pick highest scoring one — drop others.
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="guide-box success">
                    <strong>✓ Clean structure</strong><br>
                    No severe multicollinearity detected.
                </div>
                """, unsafe_allow_html=True)
        with col2:
            if insights["hidden_nonlinear"]:
                st.markdown(f"""
                <div class="guide-box info">
                    <strong>🌌 Non-linear value detected</strong><br>
                    Features: {", ".join(insights["hidden_nonlinear"])}<br>
                    Strong in trees / neural nets.
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Pearson Correlation Heatmap")
        fig_heat = go.Figure(go.Heatmap(
            z=engine.corr_matrix.values,
            x=engine.corr_matrix.columns,
            y=engine.corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(engine.corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        fig_heat.update_layout(height=600, width=600)
        update_chart_theme(fig_heat)
        st.plotly_chart(fig_heat, width='stretch')

    with tab3:
        st.markdown("##### Physics vs Linear Dependence")
        fig_scatter = px.scatter(
            res_df, x='Abs_Pearson', y='Distance_Corr', text='Feature',
            size='Mutual_Info', color='Composite_Score',
            color_continuous_scale='Viridis',
            hover_data=['Spearman']
        )
        maxv = max(res_df['Abs_Pearson'].max(), res_df['Distance_Corr'].max()) + 0.1
        fig_scatter.add_trace(go.Scatter(x=[0,maxv], y=[0,maxv], mode='lines', line=dict(dash='dash', color='#888')))
        fig_scatter.update_layout(height=600, xaxis_title="Abs Pearson", yaxis_title="Distance Correlation")
        update_chart_theme(fig_scatter)
        st.plotly_chart(fig_scatter, width='stretch')

    with tab4:
        st.markdown("##### Variance Inflation Factor (VIF)")
        vif_df = res_df[['Feature','VIF']].sort_values('VIF', ascending=False)
        colors = ['#ef4444' if v>10 else '#f59e0b' if v>5 else '#10b981' for v in vif_df['VIF']]
        fig_vif = go.Figure(go.Bar(
            x=vif_df['Feature'], y=vif_df['VIF'],
            marker_color=colors,
            text=np.round(vif_df['VIF'],1),
            textposition='auto'
        ))
        fig_vif.add_hline(y=10, line_dash="dash", line_color="rgba(239,68,68,0.8)", annotation_text="Critical (10)")
        fig_vif.add_hline(y=5, line_dash="dash", line_color="rgba(245,158,11,0.8)", annotation_text="Warning (5)")
        fig_vif.update_layout(height=450, yaxis_title="VIF", xaxis_title="Feature")
        update_chart_theme(fig_vif)
        st.plotly_chart(fig_vif, width='stretch')

    with tab5:
        st.markdown("##### Complete Essence Table")
        disp = res_df.copy()
        for c in disp.columns:
            if disp[c].dtype == 'float64':
                disp[c] = disp[c].round(4)
        disp = disp[['Feature','Composite_Score','Pearson','Spearman','Distance_Corr','Mutual_Info','RF_Importance','VIF']]
        st.dataframe(disp, width='stretch', hide_index=True, height=500)

        csv = disp.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Essence Matrix", csv, f"tattva_{target_col}.csv", "text/csv")

    render_footer()


def render_footer():
    from datetime import datetime, timezone, timedelta
    now_utc = datetime.now(timezone.utc)
    now_ist = now_utc + timedelta(hours=5, minutes=30)
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {now_ist.strftime('%Y-%m-%d %H:%M:%S IST')}")


if __name__ == "__main__":
    main()
