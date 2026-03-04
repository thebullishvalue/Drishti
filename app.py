"""
TATTVA (तत्त्व) – Essence Matrix
Advanced Feature Truth Engine | A Pragyam Product Family Member

Refined premium dark-mode UI adopting the Nirnay design system.
Dream Engine backend – Stability • Clustering • Bootstrap • CV R²
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
import datetime

warnings.filterwarnings("ignore")

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIGURATION
# ══════════════════════════════════════════════════════════════════════════════

st.set_page_config(
    page_title="TATTVA | Essence Matrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERSION = "v5.3.0 - Pragyam Unified"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

# ══════════════════════════════════════════════════════════════════════════════
# PRAGYAM DESIGN SYSTEM CSS (Exact Nirnay Implementation)
# ══════════════════════════════════════════════════════════════════════════════

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
    
    button[kind="header"] { z-index: 999999 !important; }
    
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
    }
    
    .metric-card:hover { transform: translateY(-2px); box-shadow: 0 8px 30px rgba(0,0,0,0.3); border-color: var(--border-light); }
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    
    .signal-card {
        background-color: var(--bg-card);
        padding: 1.5rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08);
        margin-bottom: 1rem;
        position: relative;
        overflow: hidden;
    }
    
    .signal-card::before { content: ''; position: absolute; top: 0; left: 0; width: 4px; height: 100%; }
    .signal-card.buy::before { background: var(--success-green); }
    .signal-card.sell::before { background: var(--danger-red); }
    .signal-card.neutral::before { background: var(--info-cyan); }
    
    .signal-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
    .signal-card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(6, 182, 212, 0.15); color: var(--info-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .stButton>button { border: 2px solid var(--primary-color); background: transparent; color: var(--primary-color); font-weight: 700; border-radius: 12px; padding: 0.75rem 2rem; transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1); text-transform: uppercase; letter-spacing: 0.5px; }
    .stButton>button:hover { box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6); background: var(--primary-color); color: #1A1A1A; transform: translateY(-2px); }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-score { font-weight: 700; font-size: 0.9rem; }
    
    .conviction-meter { height: 8px; background: var(--bg-elevated); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    .stTextInput > div > div > input, .stSelectbox > div > div > div { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus, .stSelectbox > div > div > div:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }

    .stProgress > div > div > div { background-color: var(--primary-color) !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# DREAM ENGINE CORE (Audited & Fortified)
# ══════════════════════════════════════════════════════════════════════════════

def robust_zscore(x):
    med = np.median(x)
    mad = np.median(np.abs(x - med)) + 1e-9
    return (x - med) / mad

def bootstrap_mean_std(func, x, y, n=50):
    vals = []
    N = len(x)
    for _ in range(n):
        idx = np.random.choice(N, N, replace=True)
        try:
            val = func(x[idx], y[idx])
            vals.append(val)
        except:
            pass
    # Audit Fix: Handle NaN propagation
    return np.nanmean(vals) if vals else 0.0, np.nanstd(vals) if vals else 0.0

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
    if X.shape[1] == 0:
        return np.array([])
    N, p = X.shape
    scores = np.zeros(p)
    for _ in range(n_runs):
        idx = np.random.choice(N, int(N*0.7), replace=False)
        model = RandomForestRegressor(n_estimators=100, max_depth=6)
        model.fit(X[idx], y[idx])
        scores += model.feature_importances_
    return scores / n_runs

def cross_val_predictive_power(X, y):
    if X.shape[1] == 0:
        return 0.0, 0.0
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
    # Audit Fix: Prevent clustering crash if < 2 features are selected
    if X.shape[1] < 2:
        return np.zeros(X.shape[1])
        
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
        
        # Audit Fix: Drop zero-variance features to prevent division-by-zero crashes
        valid_features = []
        for f in feature_cols:
            if self.data[f].std() > 1e-6:
                valid_features.append(f)
        self.features = valid_features
        
        if len(self.features) == 0:
            raise ValueError("All selected features have zero variance. Analysis cannot proceed.")
            
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
            
            # Safe Pearson and Spearman
            pearson = np.corrcoef(x, self.y)[0,1]
            if np.isnan(pearson): pearson = 0.0
                
            spear = spearmanr(x, self.y).correlation
            if np.isnan(spear): spear = 0.0
                
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
            
            uncertainty_penalty = np.exp(-boot_std) if not np.isnan(boot_std) else 0.5
            composite = predictive_strength * uncertainty_penalty
            
            self.results.append({
                "Feature": feat,
                "Pearson": round(pearson,3),
                "Spearman": round(spear,3),
                "Distance_Corr": round(dcor,3),
                "Stability": round(stability[i],3),
                "Cluster_Label": int(clusters[i]),
                "NonLinear_Bias": round(nonlinear_bias,3),
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
            (df["NonLinear_Bias"] > 0.15) &
            (df["Composite_Score"] > df["Composite_Score"].median())
        ]["Feature"].tolist()
        
        redundant = df.groupby("Cluster_Label")["Feature"].apply(list)
        redundant = [group[1:] for group in redundant if len(group) > 1]
        redundant = [f for sub in redundant for f in sub]
        
        return {
            "top_feature": df.iloc[0]["Feature"] if not df.empty else "None",
            "top_score": df.iloc[0]["Composite_Score"] if not df.empty else 0,
            "hidden_nonlinear": hidden[:4],
            "redundant_features": redundant[:5]
        }


# ══════════════════════════════════════════════════════════════════════════════
# DATA UTILITIES
# ══════════════════════════════════════════════════════════════════════════════

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
    if date_col and date_col != "None" and date_col in df.columns:
        cols.append(date_col)
        
    data = df[cols].copy()
    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
        
    data = data.dropna(subset=[target] + features)
    data = data[np.isfinite(data[[target] + features]).all(axis=1)]
    
    if date_col and date_col != "None" and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col]).sort_values(date_col)
        except:
            pass
            
    return data.reset_index(drop=True)


# ══════════════════════════════════════════════════════════════════════════════
# CHART UTILITIES (Nirnay-Adopted Aesthetics)
# ══════════════════════════════════════════════════════════════════════════════

def create_gauge_chart(value, title="Composite Score"):
    color = '#10b981' if value > 70 else '#f59e0b' if value > 40 else '#ef4444'
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=value,
        number=dict(font=dict(size=32, color=color, family='Inter'), suffix=""),
        gauge=dict(
            axis=dict(range=[0, 100], tickwidth=1, tickcolor='#3A3A3A', tickvals=[0, 25, 50, 75, 100], tickfont=dict(size=10, color='#888888')),
            bar=dict(color=color, thickness=0.3), bgcolor='#1A1A1A', borderwidth=2, bordercolor='#2A2A2A',
            steps=[dict(range=[0, 40], color='rgba(239,68,68,0.15)'), dict(range=[40, 70], color='rgba(245,158,11,0.15)'), dict(range=[70, 100], color='rgba(16,185,129,0.15)')],
            threshold=dict(line=dict(color='white', width=2), thickness=0.8, value=value)
        )
    ))
    fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', height=220, margin=dict(l=20, r=20, t=10, b=20), font=dict(family='Inter', color='#EAEAEA'))
    return fig

def create_scatter_chart(df):
    fig = go.Figure()
    
    # Identify non-linear gems to highlight them
    gems = df[(df["NonLinear_Bias"] > 0.15) & (df["Composite_Score"] > df["Composite_Score"].median())]
    regular = df[~df['Feature'].isin(gems['Feature'])]
    
    fig.add_trace(go.Scatter(
        x=regular['Pearson'].abs(), y=regular['Distance_Corr'], mode='markers',
        marker=dict(size=10, color='#888888', line=dict(color='#2A2A2A', width=1), opacity=0.6),
        name="Standard Features",
        text=regular['Feature'], hovertemplate="<b>%{text}</b><br>Linear: %{x:.2f}<br>Non-Linear: %{y:.2f}<extra></extra>"
    ))
    
    if not gems.empty:
        fig.add_trace(go.Scatter(
            x=gems['Pearson'].abs(), y=gems['Distance_Corr'], mode='markers+text',
            marker=dict(size=14, color='#06b6d4', line=dict(color='#EAEAEA', width=1.5), opacity=0.9),
            name="Non-Linear Gems",
            text=gems['Feature'], textposition='top center', textfont=dict(size=10, color='#06b6d4'),
            hovertemplate="<b>%{text}</b><br>Linear: %{x:.2f}<br>Non-Linear: %{y:.2f}<extra></extra>"
        ))
        
    fig.add_shape(type="line", x0=0, y0=0, x1=1, y1=1, line=dict(color="rgba(255,195,0,0.3)", dash="dash"))
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
        margin=dict(l=40, r=10, t=30, b=40),
        xaxis=dict(title=dict(text='Linear Correlation (|Pearson|)', font=dict(size=11, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[0, 1]),
        yaxis=dict(title=dict(text='Non-Linear Correlation (Distance)', font=dict(size=11, color='#888888')), showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[0, 1]),
        font=dict(family='Inter', color='#EAEAEA'), legend=dict(orientation="h", y=1.05, x=0)
    )
    return fig

def create_ranking_chart(df, top_n=15):
    sorted_df = df.head(top_n).sort_values('Composite_Score', ascending=True)
    colors = ['#10b981' if v > 70 else '#f59e0b' if v > 40 else '#888888' for v in sorted_df['Composite_Score']]
    
    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=sorted_df['Feature'], x=sorted_df['Composite_Score'], orientation='h',
        marker=dict(color=colors, line=dict(color='#2A2A2A', width=1)),
        text=[f"{v:.1f}" for v in sorted_df['Composite_Score']], textposition='outside', textfont=dict(size=10, color='#888888')
    ))
    
    fig.update_layout(
        template='plotly_dark', paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='#1A1A1A', height=400,
        margin=dict(l=80, r=50, t=10, b=10),
        xaxis=dict(showgrid=True, gridcolor='rgba(42,42,42,0.5)', range=[0, 100]),
        yaxis=dict(showgrid=False, tickfont=dict(size=10)),
        font=dict(family='Inter', color='#EAEAEA')
    )
    return fig


# ══════════════════════════════════════════════════════════════════════════════
# UI LAYOUT & MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def render_sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align: center; padding: 1rem 0; margin-bottom: 1rem;">
            <div style="font-size: 1.75rem; font-weight: 800; color: #FFC300;">TATTVA</div>
            <div style="color: #888888; font-size: 0.75rem; margin-top: 0.25rem;">तत्त्व | Essence Matrix</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        st.markdown('<div class="sidebar-title">📁 Data Source</div>', unsafe_allow_html=True)
        source = st.radio("", ["Upload CSV/Excel", "Google Sheets"], horizontal=True, label_visibility="collapsed")

        df = None
        if source == "Upload CSV/Excel":
            file = st.file_uploader("", type=["csv", "xlsx"])
            if file:
                # Clear run state when new file is uploaded
                if 'last_file' not in st.session_state or st.session_state['last_file'] != file.name:
                    st.session_state['run'] = False
                    st.session_state['last_file'] = file.name
                
                try:
                    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                except Exception as e:
                    st.error(f"File error: {e}")
        else:
            url = st.text_input("Google Sheet URL", placeholder="https://docs.google.com/spreadsheets/d/...")
            if st.button("Load Sheet", use_container_width=True):
                with st.spinner("Fetching..."):
                    df, err = load_google_sheet(url)
                if err:
                    st.error(err)
                else:
                    st.toast("Sheet loaded successfully!", icon="✅")
                    st.session_state['run'] = False

        if df is not None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric) < 2:
                st.error("Dataset needs at least 2 numeric columns.")
                st.stop()

            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">⚙️ Analysis Setup</div>', unsafe_allow_html=True)
            
            target = st.selectbox("Target (Y)", numeric)
            features = st.multiselect("Features (X)", [c for c in numeric if c != target],
                                     default=[c for c in numeric if c != target][:12])
            
            date_col = st.selectbox("Date Column (optional)", ["None"] + list(df.columns))

            st.markdown("<br>", unsafe_allow_html=True)
            if st.button("◈ RUN TRUTH ENGINE", type="primary", use_container_width=True):
                st.session_state['run'] = True
                st.session_state['df'] = df
                st.session_state['target'] = target
                st.session_state['features'] = features
                st.session_state['date_col'] = date_col

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> Stability • Clustering • Bootstrap<br>
                <strong>Focus:</strong> Feature Truth Discovery
            </p>
        </div>
        """, unsafe_allow_html=True)

    return df


def render_header():
    st.markdown("""
    <div class="premium-header">
        <h1>TATTVA : Feature Truth Engine</h1>
        <div class="tagline">Multivariate Feature Selection & Structural Analysis System</div>
    </div>
    """, unsafe_allow_html=True)


def main():
    df = render_sidebar()
    render_header()

    if 'run' not in st.session_state or not st.session_state['run']:
        st.markdown("""
        <div class='info-box' style='text-align:center; padding: 4rem 2rem;'>
            <h3 style='color: var(--text-muted);'>Configure Data Source in Sidebar</h3>
            <p style='color: var(--text-muted);'>Upload a dataset and select your target and features to reveal the Essence Matrix.</p>
        </div>
        """, unsafe_allow_html=True)
        return

    # Retrieve from session state
    df = st.session_state['df']
    target = st.session_state['target']
    features = st.session_state['features']
    date_col = st.session_state['date_col']

    if not features:
        st.warning("Please select at least one feature.")
        return

    # ── Main Processing Flow ───────────────────────────────────────────────────
    with st.spinner(""):
        status_text = st.empty()
        status_text.markdown("**⏳ Extracting feature essence...**")
        progress = st.progress(0)
        
        def update(pct): 
            progress.progress(pct / 100)
            if pct < 25: status_text.markdown("**⏳ Computing Cross-Validation Power...**")
            elif pct < 50: status_text.markdown("**⏳ Bootstrapping Stability Selection...**")
            elif pct < 75: status_text.markdown("**⏳ Redundancy Spectral Clustering...**")
            elif pct < 95: status_text.markdown("**⏳ Synthesizing Non-Linear Bias...**")
            else: status_text.markdown("**✅ Finalizing Essence Matrix...**")

        try:
            data = clean_data(df, target, features, date_col if date_col != "None" else None)
            if len(data) < 30:
                raise ValueError(f"Need ≥ 30 clean rows after dropping missing values. Found {len(data)}.")

            engine = TattvaEngine(data, target, features)
            engine.analyze(update)
            progress.empty()
            status_text.empty()
            st.toast("Analysis Complete!", icon="✅")

        except Exception as e:
            progress.empty()
            status_text.empty()
            st.error(f"Analysis failed: {str(e)}")
            st.info("Possible causes: Zero variance features, too few rows, non-numeric data. Try fewer features or cleaner input.")
            return

    res_df = engine.res_df
    insights = engine.get_insights()

    # ── Top Metrics Cards ──────────────────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="metric-card primary">
            <h4>Primary Essence</h4>
            <h2>{insights['top_feature']}</h2>
            <div class="sub-metric">Composite Score: {insights['top_score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        n = len(insights['hidden_nonlinear'])
        st.markdown(f"""
        <div class="metric-card info">
            <h4>Hidden Gems</h4>
            <h2>{n}</h2>
            <div class="sub-metric">High Non-Linear Value</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        n = len(insights['redundant_features'])
        st.markdown(f"""
        <div class="metric-card danger">
            <h4>Redundant</h4>
            <h2>{n}</h2>
            <div class="sub-metric">Identified Cluster Groups</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        avg_stab = res_df['Stability'].mean()
        st.markdown(f"""
        <div class="metric-card neutral">
            <h4>Avg Stability</h4>
            <h2>{avg_stab:.2f}</h2>
            <div class="sub-metric">Bootstrap Stability Index</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ── Nirnay Styled Tabs ────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["**🎯 Primary Signals**", "**📈 Linear vs Non-Linear**", "**📊 Clusters & Ranking**", "**📋 Truth Matrix**"])

    with tab1:
        col_s1, col_s2 = st.columns(2)
        
        with col_s1:
            st.markdown('<div class="signal-card buy"><div class="signal-card-header"><span class="signal-card-title">🟢 Strong Predictors</span></div>', unsafe_allow_html=True)
            
            st.markdown("##### Primary Feature Score")
            st.plotly_chart(create_gauge_chart(insights['top_score'], insights['top_feature']), width="stretch", config={'displayModeBar': False})
            
            keep = res_df[res_df['Composite_Score'] > 50].head(5)
            if not keep.empty:
                st.markdown('<span class="status-badge buy">HIGH CONVICTION FEATURES</span><br><br>', unsafe_allow_html=True)
                for _, row in keep.iterrows():
                    pct = row["Composite_Score"]
                    color = '#10b981' if pct > 70 else '#f59e0b'
                    st.markdown(f'''
                    <div style="margin-bottom: 0.85rem;">
                        <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                            <span style="color: #EAEAEA; font-weight: 600;">{row["Feature"]}</span>
                            <span style="color: {color}; font-weight: 600;">{pct:.1f}</span>
                        </div>
                        <div class="conviction-meter">
                            <div class="conviction-fill" style="width: {pct}%; background: {color};"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: var(--text-muted);">No strong dominant predictors found.</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)
            
        with col_s2:
            st.markdown('<div class="signal-card neutral"><div class="signal-card-header"><span class="signal-card-title">💎 Non-Linear Hidden Gems</span></div>', unsafe_allow_html=True)
            if insights['hidden_nonlinear']:
                st.markdown('<p style="color: var(--text-muted); font-size: 0.85rem;">These features show weak linear correlation (Pearson) but strong structural relationships via Distance Correlation.</p>', unsafe_allow_html=True)
                st.markdown('<span class="status-badge neutral">STRUCTURAL VALUE DETECTED</span><br><br>', unsafe_allow_html=True)
                for f in insights['hidden_nonlinear']:
                    nl_bias = res_df[res_df['Feature'] == f]['NonLinear_Bias'].values[0]
                    st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{f}</span></div><span class="symbol-score" style="color: var(--info-cyan);">+{nl_bias:.2f} Bias</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: var(--text-muted);">No significant non-linear structural outliers detected in this dataset.</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

            st.markdown('<div class="signal-card sell"><div class="signal-card-header"><span class="signal-card-title">🔴 Redundancy Warning</span></div>', unsafe_allow_html=True)
            if insights['redundant_features']:
                st.markdown('<p style="color: var(--text-muted); font-size: 0.85rem;">Features mapping to identical spectral clusters. Consider dropping to reduce dimensionality.</p>', unsafe_allow_html=True)
                st.markdown('<span class="status-badge sell">REDUNDANT CO-CLUSTERS</span><br><br>', unsafe_allow_html=True)
                for f in insights['redundant_features'][:6]:
                    st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{f}</span></div><span class="symbol-score" style="color: var(--danger-red);">Drop Candidate</span></div>', unsafe_allow_html=True)
            else:
                st.markdown('<p style="color: var(--text-muted);">Features appear highly independent.</p>', unsafe_allow_html=True)
            st.markdown("</div>", unsafe_allow_html=True)

    with tab2:
        st.markdown("##### Linear vs Non-Linear Relationship Matrix")
        st.markdown('<p style="color: #888888; font-size: 0.85rem;">Features above the dashed diagonal have stronger non-linear relationships than linear ones.</p>', unsafe_allow_html=True)
        st.plotly_chart(create_scatter_chart(res_df), width="stretch", config={'displayModeBar': False})

    with tab3:
        col_c1, col_c2 = st.columns(2)
        with col_c1:
            st.markdown("##### Composite Score Ranking")
            st.plotly_chart(create_ranking_chart(res_df), width="stretch", config={'displayModeBar': False})
        
        with col_c2:
            st.markdown("##### Spectral Cluster Distribution")
            cluster_counts = res_df['Cluster_Label'].value_counts()
            fig_cluster = go.Figure(go.Pie(
                labels=[f"Cluster {int(i)}" for i in cluster_counts.index], 
                values=cluster_counts.values, hole=0.5,
                marker=dict(line=dict(color='#1A1A1A', width=2)),
                textinfo='label+percent', textfont=dict(size=11, color='white')
            ))
            fig_cluster.update_layout(paper_bgcolor='rgba(0,0,0,0)', font=dict(family='Inter', color='#EAEAEA'), height=400, margin=dict(l=20, r=20, t=30, b=20), showlegend=False)
            st.plotly_chart(fig_cluster, width="stretch", config={'displayModeBar': False})

    with tab4:
        st.markdown(f"##### Complete Feature Truth Matrix ({len(res_df)} features)")
        
        disp = res_df.copy()
        for c in disp.select_dtypes(include='number').columns:
            disp[c] = disp[c].round(3)
        
        st.dataframe(
            disp[['Feature','Composite_Score','Pearson','Spearman','Distance_Corr','Stability','Cluster_Label','NonLinear_Bias','Uncertainty']],
            use_container_width=True,
            height=500,
            hide_index=True
        )

        st.markdown("<br>", unsafe_allow_html=True)
        csv = disp.to_csv(index=False).encode()
        st.download_button("📥 Download Essence Matrix (CSV)", csv, f"tattva_matrix_{datetime.date.today().strftime('%Y%m%d')}.csv", "text/csv")

    # ── Footer ────────────────────────────────────────────────────────────────
    utc_now = datetime.datetime.now(datetime.timezone.utc)
    ist_now = utc_now + datetime.timedelta(hours=5, minutes=30)
    current_time_ist = ist_now.strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

if __name__ == "__main__":
    main()
