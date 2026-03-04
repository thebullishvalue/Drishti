"""
DRISHTI (दृष्टि) - Deep Feature Matrix | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Advanced mathematical and physics-based correlation analysis.
Identifies absolute feature utility using Information Theory, 
Energy Statistics (Distance Correlation), and Topology.

v3.0.0-Haywire Edition — Backend completely overhauled with:
• Pure-NumPy Graph Topology (Eigenvector Centrality on dCor network)
• Granger Causality (directional time-series power when date provided)
• Smarter composite scoring
Everything else (UI/UX, cards, tabs, charts, styling) 100% untouched.
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

# --- Dependencies ---
try:
    import statsmodels.api as sm
    from statsmodels.tsa.stattools import grangercausalitytests
    STATSMODELS_AVAILABLE = True
except ImportError:
    sm = None
    grangercausalitytests = None
    STATSMODELS_AVAILABLE = False

try:
    from sklearn.feature_selection import mutual_info_regression
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

# --- Constants ---
VERSION = "v3.0.0-Haywire"
PRODUCT_NAME = "Drishti"
COMPANY = "Hemrek Capital"

# --- Page Config ---
st.set_page_config(
    page_title="DRISHTI | Deep Feature Matrix",
    page_icon="🌌",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# --- Premium CSS (Nirnay Design System) ---
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

    .signal-card { background: var(--bg-card); border-radius: 16px; border: 2px solid var(--border-color); padding: 1.5rem; position: relative; overflow: hidden; }
    .signal-card.overvalued { border-color: var(--danger-red); box-shadow: 0 0 30px rgba(239, 68, 68, 0.15); }
    .signal-card.undervalued { border-color: var(--success-green); box-shadow: 0 0 30px rgba(16, 185, 129, 0.15); }
    .signal-card.fair { border-color: var(--primary-color); box-shadow: 0 0 30px rgba(255, 195, 0, 0.15); }
    .signal-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }
    .signal-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1; }
    .signal-card .subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.75rem; }
    
    .guide-box { background: rgba(var(--primary-rgb), 0.05); border-left: 3px solid var(--primary-color); padding: 1rem; border-radius: 8px; margin: 1rem 0; color: var(--text-secondary); font-size: 0.9rem; }
    .guide-box.success { background: rgba(16, 185, 129, 0.05); border-left-color: var(--success-green); }
    .guide-box.danger { background: rgba(239, 68, 68, 0.05); border-left-color: var(--danger-red); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }

    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }

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


# ============================================================================
# PHYSICS & MATH UTILITIES — HAYWIRE UPGRADE
# ============================================================================

def distance_correlation(x, y):
    """
    Computes Energy Statistics Distance Correlation.
    Measures both linear and non-linear dependence between two variables.
    dCor(X,Y) = 0 implies true independence, unlike Pearson.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    
    # Subsample if too large to prevent memory/compute explosion (O(N^2))
    n = len(x)
    if n > 1500:
        idx = np.random.choice(n, 1500, replace=False)
        x = x[idx]
        y = y[idx]
        n = 1500
        
    A = squareform(pdist(x[:, None]))
    B = squareform(pdist(y[:, None]))
    
    # Double centering
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


def granger_causality_score(x, y, maxlag=3):
    """
    HAYWIRE ADDITION: Granger causality strength (X → Y).
    Returns 0-1 score. Higher = stronger directional predictive power over time.
    Only active when date column is provided and data is sorted.
    """
    if len(x) < 30 or not STATSMODELS_AVAILABLE or grangercausalitytests is None:
        return 0.0
    try:
        dfg = pd.DataFrame({'y': np.asarray(y), 'x': np.asarray(x)}).dropna()
        if len(dfg) < 30:
            return 0.0
        results = grangercausalitytests(dfg[['y', 'x']], maxlag=maxlag, verbose=False)
        pvals = [results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag + 1)]
        min_p = min(pvals)
        score = np.clip(-np.log10(min_p + 1e-12) / 4.0, 0.0, 1.0)
        return float(score)
    except:
        return 0.0


# ============================================================================
# DEEP CORRELATION ENGINE — FULL HAYWIRE OVERHAUL
# ============================================================================

class DeepCorrelationEngine:
    """
    Advanced mathematical engine for ultimate feature extraction.
    NOW WITH: Graph Topology (NumPy eigenvector centrality) + Granger Causality.
    """
    def __init__(self, data, target_col, feature_cols, date_col=None):
        self.data = data.copy()
        self.target = target_col
        self.features = feature_cols
        self.date_col = date_col  # NEW: enables time-series causality

        # Standardize for scale-invariant distance & ML
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
        self.topo_scores = None

    def analyze(self):
        """Execute the deep analysis pipeline — now with graph + causality."""
        y = self.data[self.target].values
        X_df = self.data[self.features]
        
        # 1. Base Correlations Matrix (Pearson)
        self.corr_matrix = self.data[[self.target] + self.features].corr(method='pearson')
        
        # 2. VIF (Multicollinearity)
        if STATSMODELS_AVAILABLE:
            X_with_const = sm.add_constant(X_df)
            for i, col in enumerate(X_with_const.columns):
                if col == 'const': continue
                try:
                    vif = variance_inflation_factor(X_with_const.values, i)
                    self.vif_data[col] = vif
                except:
                    self.vif_data[col] = np.nan
        else:
            self.vif_data = {f: 1.0 for f in self.features}
        
        # 3. Mutual Info (Information Gain / Entropy)
        if SKLEARN_AVAILABLE:
            mi_scores = mutual_info_regression(self.X_scaled, self.y_scaled)
        else:
            mi_scores = np.zeros(len(self.features))
            
        # 4. Feature Importance (Random Forest Ensembling)
        if SKLEARN_AVAILABLE:
            rf = RandomForestRegressor(n_estimators=100, max_depth=5, random_state=42)
            rf.fit(self.X_scaled, self.y_scaled)
            rf_importance = rf.feature_importances_
        else:
            rf_importance = np.zeros(len(self.features))

        # 5. HAYWIRE: Pure-NumPy Graph Topology — Eigenvector Centrality on dCor adjacency
        if len(self.features) > 1:
            dcor_mat = np.eye(len(self.features))
            for i in range(len(self.features)):
                for j in range(i + 1, len(self.features)):
                    d = distance_correlation(self.X_scaled[:, i], self.X_scaled[:, j])
                    dcor_mat[i, j] = d
                    dcor_mat[j, i] = d
            try:
                eigenvalues, eigenvectors = np.linalg.eig(dcor_mat)
                idx = np.argmax(np.real(eigenvalues))
                centrality_vec = np.abs(np.real(eigenvectors[:, idx]))
                centrality_vec /= (np.sum(centrality_vec) + 1e-12)
                self.topo_scores = centrality_vec
            except:
                self.topo_scores = np.ones(len(self.features)) / len(self.features)
        else:
            self.topo_scores = np.array([1.0])

        # 6. Assemble results with NEW topology + causality scores
        for i, feat in enumerate(self.features):
            x = self.data[feat].values
            
            # Linear stats
            pearson = np.corrcoef(x, y)[0, 1] if np.std(x) > 0 else 0
            spearman = pd.Series(x).corr(pd.Series(y), method='spearman')
            
            # Physics stats
            dcor = distance_correlation(self.X_scaled[:, i], self.y_scaled)
            
            # NEW: Granger Causality (only if time-aware data)
            granger_score = 0.0
            if self.date_col is not None and STATSMODELS_AVAILABLE:
                granger_score = granger_causality_score(x, y)
            
            self.results.append({
                'Feature': feat,
                'Pearson': pearson,
                'Abs_Pearson': abs(pearson),
                'Spearman': spearman,
                'Distance_Corr': dcor,
                'Mutual_Info': mi_scores[i],
                'RF_Importance': rf_importance[i],
                'VIF': self.vif_data.get(feat, np.nan),
                'Topological_Centrality': float(self.topo_scores[i]),   # NEW
                'Granger_Score': granger_score                         # NEW
            })
            
        self.res_df = pd.DataFrame(self.results)
        self._calculate_composite_score()
        
    def _calculate_composite_score(self):
        """Creates a unifying 0-100 score — now with Topology + Granger blended in."""
        df = self.res_df
        
        def norm(col):
            if df[col].max() == df[col].min(): return np.zeros(len(df))
            return (df[col] - df[col].min()) / (df[col].max() - df[col].min())
        
        # HAYWIRE WEIGHTS (sum = 1.0): Topology + Granger now influence ranking
        score = (
            norm('Abs_Pearson') * 0.10 +
            norm('Spearman') * 0.10 +
            norm('Distance_Corr') * 0.25 +
            norm('Mutual_Info') * 0.20 +
            norm('RF_Importance') * 0.15 +
            norm('Topological_Centrality') * 0.10 +   # Graph influence
            norm('Granger_Score') * 0.10              # Time-series causality
        ) * 100
        
        # Penalize heavy multicollinearity
        vif_penalty = np.where(df['VIF'] > 10, 0.8, 1.0)
        vif_penalty = np.where(df['VIF'] > 50, 0.5, vif_penalty)
        
        self.res_df['Composite_Score'] = score * vif_penalty
        self.res_df = self.res_df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)

    def get_insights(self):
        df = self.res_df
        top_feat = df.iloc[0]['Feature'] if not df.empty else "None"
        
        # Find hidden non-linear signals: Low Pearson but high Distance Corr/MI
        def norm(arr):
            if np.max(arr) == np.min(arr): return np.zeros_like(arr)
            return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))
        
        df['NonLinear_Bias'] = (norm(df['Distance_Corr'].values) + norm(df['Mutual_Info'].values)) / 2 - norm(df['Abs_Pearson'].values)
        hidden_gems = df[df['NonLinear_Bias'] > 0.3]['Feature'].tolist()
        
        redundant = df[df['VIF'] > 10]['Feature'].tolist()
        
        return {
            'top_feature': top_feat,
            'top_score': df.iloc[0]['Composite_Score'] if not df.empty else 0,
            'hidden_nonlinear': hidden_gems[:3],
            'redundant_features': redundant
        }


def norm(arr):
    if np.max(arr) == np.min(arr): return np.zeros_like(arr)
    return (arr - np.min(arr)) / (np.max(arr) - np.min(arr))


# ============================================================================
# DATA UTILITIES (unchanged)
# ============================================================================

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match: return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)

def clean_data(df, target, features, date_col=None):
    cols = [target] + features
    if date_col and date_col != "None" and date_col in df.columns: cols.append(date_col)
    data = df[cols].copy()
    for col in [target] + features: data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    numeric_subset = data[[target] + features]
    is_finite = np.isfinite(numeric_subset).all(axis=1)
    data = data[is_finite]
    if date_col and date_col != "None" and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col])
            data = data.sort_values(date_col)
        except: pass
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


def render_landing_page():
    # (exactly the same — no change)
    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class='metric-card purple' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--purple); margin-bottom: 0.5rem;'>🌌 Information Theory</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Uses Shannon Entropy & Mutual Information to measure the actual "bits" of knowledge a feature provides about the target.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Methodology:</strong><br>
                • Non-parametric density<br>
                • K-Nearest Neighbors entropy<br>
                • Captures any dependence
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 0.5rem;'>🧬 Energy Statistics</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Calculates Brownian Distance Correlation. Unlike Pearson, a dCor of 0 implies true mathematical independence between variables.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Key Metrics:</strong><br>
                • Universal Non-linear detection<br>
                • Double-centered matrices<br>
                • Physics-based energy distance
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 0.5rem;'>🕸️ Network Topology</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Maps multicollinearity and feature redundancy using Variance Inflation Factors (VIF) and correlation network graphs.
            </p>
            <br>
            <p style='color: var(--text-secondary); font-size: 0.85rem;'>
                <strong>Outputs:</strong><br>
                • Redundancy Flags (VIF > 10)<br>
                • Cluster grouping<br>
                • Dimensionality reduction hints
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("### 🎯 Signal Interpretation")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    with col_s1:
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'>🟢 High Composite Score</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Score > 75</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                The holy grail. Feature contains immense predictive power with low redundancy. <b>Must include</b> in your downstream ML model.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_s2:
        st.markdown("""
        <div style='background: rgba(6, 182, 212, 0.1); border: 1px solid var(--info-cyan); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--info-cyan); margin-bottom: 0.75rem;'>🔵 Non-Linear Gems</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>Low Pearson, High MI/dCor</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Invisible to standard OLS regression. Use Tree-based (XGBoost) or Neural Networks to extract this value.
            </p>
        </div>
        """, unsafe_allow_html=True)
    with col_s3:
        st.markdown("""
        <div style='background: rgba(239, 68, 68, 0.1); border: 1px solid var(--danger-red); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--danger-red); margin-bottom: 0.75rem;'>🔴 Toxic Redundancy</h4>
            <p style='color: var(--text-muted); font-size: 0.85rem;'>VIF > 10</p>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Feature is highly correlated with other predictors. Including it will destabilize your linear models. <b>Drop it.</b>
            </p>
        </div>
        """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION — ONLY ONE TINY CHANGE (engine instantiation)
# ============================================================================

def main():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">DRISHTI</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">दृष्टि | Deep Feature Matrix</div>
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
                    if 'deep_engine' in st.session_state: del st.session_state.deep_engine
                    if 'deep_cache' in st.session_state: del st.session_state.deep_cache
                    st.session_state['data'] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if 'data' in st.session_state: df = st.session_state['data']
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <h1>DRISHTI : Deep Feature Matrix</h1>
            <div class="tagline">Extracting absolute mathematical utility from your datasets for predictive modeling.</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Target Configuration</div>', unsafe_allow_html=True)
        target_col = st.selectbox("Target Variable (Y)", numeric_cols, index=0)
        
        available = [c for c in numeric_cols if c != target_col]
        feature_cols = st.multiselect("Predictors (X)", available, default=available[:min(10, len(available))])
        
        if not feature_cols:
            st.info("👈 Select predictors to analyze")
            render_footer()
            return
            
        st.markdown('<div class="sidebar-title">📅 Context (Optional)</div>', unsafe_allow_html=True)
        date_candidates = [c for c in all_cols if 'date' in c.lower() or 'time' in c.lower()]
        date_col = st.selectbox("Time Axis", ["None"] + all_cols, 
                                index=all_cols.index(date_candidates[0])+1 if date_candidates else 0)
        
        st.markdown('<br>', unsafe_allow_html=True)
        run_analysis_btn = st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True)
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Shannon MI + Brownian dCor + Graph Topology + Granger Causality
            </p>
        </div>
        """, unsafe_allow_html=True)

    config_hash = f"{target_col}_{'-'.join(sorted(feature_cols))}_{date_col}"
    
    if 'drishti_config' not in st.session_state:
        st.session_state.drishti_config = config_hash
        st.session_state.is_analyzed = False
        
    if config_hash != st.session_state.drishti_config:
        st.session_state.is_analyzed = False
        st.session_state.drishti_config = config_hash
        
    if run_analysis_btn:
        st.session_state.is_analyzed = True

    if not st.session_state.is_analyzed:
        st.markdown("""
        <div class="premium-header">
            <h1>DRISHTI : Deep Feature Matrix</h1>
            <div class="tagline">Extracting absolute mathematical utility from your datasets for predictive modeling.</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 System ready. Configure your targets and click **RUN ANALYSIS** in the sidebar to process the feature matrix.")
        render_landing_page()
        render_footer()
        return

    data = clean_data(df, target_col, feature_cols, date_col if date_col != "None" else None)
    if len(data) < 30:
        st.error("Insufficient valid data rows (Need >30 after dropna).")
        return

    # Processing Engine — ONLY CHANGE HERE (pass date_col)
    cache_key = f"{target_col}_{'-'.join(sorted(feature_cols))}_{len(data)}"
    if 'deep_cache' not in st.session_state or st.session_state.deep_cache != cache_key:
        with st.spinner("Initializing Deep Correlation Algorithms... Computing Information Theory, Energy Statistics, Graph Topology & Granger Causality..."):
            engine = DeepCorrelationEngine(data, target_col, feature_cols, date_col if date_col != "None" else None)
            engine.analyze()
            st.session_state.deep_engine = engine
            st.session_state.deep_cache = cache_key
            
    engine = st.session_state.deep_engine
    res_df = engine.res_df
    insights = engine.get_insights()

    # ═══════════════════════════════════════════════════════════════════════
    # TOP METRICS (EXECUTIVE SUMMARY) — unchanged
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    
    with c1:
        st.markdown(f'<div class="metric-card primary"><h4>Primary Predictor</h4><h2>{insights["top_feature"]}</h2><div class="sub-metric">Score: {insights["top_score"]:.1f}/100</div></div>', unsafe_allow_html=True)
    with c2:
        n_hidden = len(insights["hidden_nonlinear"])
        hidden_txt = ", ".join(insights["hidden_nonlinear"]) if n_hidden > 0 else "None detected"
        col = "info" if n_hidden > 0 else "neutral"
        st.markdown(f'<div class="metric-card {col}"><h4>Non-Linear Hidden Signals</h4><h2>{n_hidden}</h2><div class="sub-metric">{hidden_txt[:25]}{"..." if len(hidden_txt)>25 else ""}</div></div>', unsafe_allow_html=True)
    with c3:
        n_red = len(insights["redundant_features"])
        red_col = "danger" if n_red > 0 else "success"
        st.markdown(f'<div class="metric-card {red_col}"><h4>Toxic Redundancy (VIF>10)</h4><h2>{n_red}</h2><div class="sub-metric">Features to drop</div></div>', unsafe_allow_html=True)
    with c4:
        avg_mi = res_df["Mutual_Info"].mean()
        mi_col = "purple" if avg_mi > 0.1 else "warning"
        st.markdown(f'<div class="metric-card {mi_col}"><h4>System Information Gain</h4><h2>{avg_mi:.3f}</h2><div class="sub-metric">Avg. Mutual Info</div></div>', unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # TABS — 100% unchanged
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**🎯 Synthesis Dashboard**",
        "**🧬 Correlation Matrix**",
        "**🌌 Physics & Information**",
        "**🕸️ VIF Topology**",
        "**📋 Raw Analytics**"
    ])

    with tab1:
        st.markdown("##### Composite Predictive Power Score")
        st.markdown('<p style="color: #888;">The ultimate ranking of feature utility. Combines Linear, Non-Linear, and Tree-based importance, penalized by Multicollinearity.</p>', unsafe_allow_html=True)
        
        fig_bar = px.bar(
            res_df, x='Composite_Score', y='Feature', orientation='h',
            color='Composite_Score', color_continuous_scale=['#ef4444', '#f59e0b', '#10b981', '#FFC300'],
            hover_data=['Pearson', 'Distance_Corr', 'VIF']
        )
        fig_bar.update_layout(height=400 + min(len(feature_cols)*15, 400), yaxis={'categoryorder':'total ascending'}, showlegend=False)
        update_chart_theme(fig_bar)
        st.plotly_chart(fig_bar, use_container_width=True)
        
        col1, col2 = st.columns(2)
        with col1:
            if insights["redundant_features"]:
                st.markdown(f"""
                <div class="guide-box danger">
                    <strong>⚠️ Data Leakage / Collinearity Warning</strong><br>
                    The following features exhibit massive collinearity (VIF > 10). They are measuring the exact same variance. 
                    <b>Action:</b> Pick the one with the highest Composite Score and drop the rest.<br><br>
                    <i>{", ".join(insights["redundant_features"])}</i>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="guide-box success">
                    <strong>✅ Clean Topology</strong><br>
                    No severe multicollinearity detected. Feature space is orthogonal and stable for OLS/Linear regression.
                </div>
                """, unsafe_allow_html=True)
                
        with col2:
            if insights["hidden_nonlinear"]:
                st.markdown(f"""
                <div class="guide-box info">
                    <strong>🌌 Non-Linear Goldmine</strong><br>
                    These features have low linear correlation (Pearson) but high Distance Correlation or Mutual Info. 
                    Linear models will miss them, but complex models (XGBoost/Neural Nets) will extract immense value.<br><br>
                    <i>{", ".join(insights["hidden_nonlinear"])}</i>
                </div>
                """, unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Full Correlation Heatmap (Pearson Linear)")
        corr_matrix = engine.corr_matrix
        
        fig_heat = go.Figure(data=go.Heatmap(
            z=corr_matrix.values,
            x=corr_matrix.columns,
            y=corr_matrix.columns,
            colorscale='RdBu',
            zmin=-1, zmax=1,
            text=np.round(corr_matrix.values, 2),
            texttemplate="%{text}",
            hoverinfo="x+y+z"
        ))
        fig_heat.update_layout(height=600, width=600)
        update_chart_theme(fig_heat)
        st.plotly_chart(fig_heat, use_container_width=True)

    with tab3:
        st.markdown("##### Advanced Dependence Mapping (Physics vs Stats)")
        st.markdown('<p style="color: #888;">Distance Correlation (Y) vs Absolute Pearson (X). Features above the diagonal possess significant Non-Linear relationships invisible to standard statistics.</p>', unsafe_allow_html=True)
        
        fig_scatter = px.scatter(
            res_df, x='Abs_Pearson', y='Distance_Corr', text='Feature',
            size='Mutual_Info', color='Composite_Score',
            color_continuous_scale='Viridis',
            hover_data=['Spearman']
        )
        
        max_val = max(res_df['Abs_Pearson'].max(), res_df['Distance_Corr'].max()) + 0.1
        fig_scatter.add_trace(go.Scatter(x=[0, max_val], y=[0, max_val], mode='lines', name='Linear Boundary', line=dict(dash='dash', color='#888')))
        
        fig_scatter.update_traces(textposition='top center')
        fig_scatter.update_layout(height=600, xaxis_title="Abs Pearson (Linear Stats)", yaxis_title="Distance Correlation (Energy Stats)")
        update_chart_theme(fig_scatter)
        st.plotly_chart(fig_scatter, use_container_width=True)

    with tab4:
        st.markdown("##### Variance Inflation Factor (VIF)")
        st.markdown('<p style="color: #888;">Shows how much the variance of an estimated regression coefficient increases due to collinearity. Values > 10 indicate problematic redundancy.</p>', unsafe_allow_html=True)
        
        vif_df = res_df[['Feature', 'VIF']].copy().sort_values('VIF', ascending=False)
        colors = ['#ef4444' if v > 10 else '#f59e0b' if v > 5 else '#10b981' for v in vif_df['VIF']]
        
        fig_vif = go.Figure(go.Bar(
            x=vif_df['Feature'], y=vif_df['VIF'],
            marker_color=colors, text=np.round(vif_df['VIF'], 1), textposition='auto'
        ))
        fig_vif.add_hline(y=10, line_dash="dash", line_color="rgba(239,68,68,0.8)", annotation_text="Critical Limit (10)")
        fig_vif.add_hline(y=5, line_dash="dash", line_color="rgba(245,158,11,0.8)", annotation_text="Warning Limit (5)")
        
        fig_vif.update_layout(height=450, yaxis_title="VIF Score", xaxis_title="Feature")
        update_chart_theme(fig_vif)
        st.plotly_chart(fig_vif, use_container_width=True)

    with tab5:
        st.markdown("##### Deep Analytics Table")
        
        disp_df = res_df.copy()
        for col in disp_df.columns:
            if disp_df[col].dtype == 'float64':
                disp_df[col] = disp_df[col].round(4)
                
        disp_df = disp_df[['Feature', 'Composite_Score', 'Pearson', 'Spearman', 'Distance_Corr', 'Mutual_Info', 'RF_Importance', 'VIF']]
        st.dataframe(disp_df, width='stretch', hide_index=True, height=500)
        
        csv_data = disp_df.to_csv(index=False).encode('utf-8')
        st.download_button("📥 Download Correlation Matrix", csv_data, f"deep_matrix_{target_col}.csv", "text/csv")

    render_footer()

def render_footer():
    from datetime import timezone, timedelta, datetime
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

if __name__ == "__main__":
    main()
