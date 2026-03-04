"""
TATTVA (तत्त्व) - Essence Matrix | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
v5.0-Frontier Edition

Reveals fundamental predictive truth using:
• Hilbert-Schmidt Independence Criterion (HSIC)
• SHAP Game Theory
• Brownian Distance Correlation (dCor)
• Hybrid Graph Topology (PageRank + Eigenvector)
• Granger Causality (when time axis provided)
• Mutual Information
• Variance Inflation Factor (VIF)

Strictly button-triggered analysis + real-time progress bar
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from scipy.spatial.distance import pdist, squareform
import warnings
from datetime import datetime, timezone, timedelta

warnings.filterwarnings('ignore')

# ────────────────────────────────────────────────────────────────────────────
# Dependencies with graceful fallback
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
VERSION = "v5.0-Frontier"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

st.set_page_config(
    page_title="TATTVA | Essence Matrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ────────────────────────────────────────────────────────────────────────────
# CSS (minimal version for clarity — you can paste full previous CSS here)
# ────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    :root {
        --primary-color: #FFC300;
        --primary-rgb: 255, 195, 0;
        --background-color: #0F0F0F;
        --secondary-background-color: #1A1A1A;
        --text-primary: #EAEAEA;
        --text-muted: #888888;
        --border-color: #2A2A2A;
        --success-green: #10b981;
        --danger-red: #ef4444;
        --info-cyan: #06b6d4;
        --purple: #8b5cf6;
    }
    * { font-family: 'Inter', sans-serif; }
    .main { background-color: var(--background-color); color: var(--text-primary); }
    .block-container { padding-top: 3.5rem; max-width: 90%; }
    .premium-header {
        background: var(--secondary-background-color);
        padding: 1.25rem 2rem;
        border-radius: 16px;
        margin-bottom: 1.5rem;
        border: 1px solid var(--border-color);
    }
    .metric-card {
        background: var(--secondary-background-color);
        padding: 1.25rem;
        border-radius: 12px;
        border: 1px solid var(--border-color);
        min-height: 160px;
    }
    .guide-box {
        padding: 1rem;
        border-radius: 8px;
        border-left: 3px solid var(--primary-color);
        background: rgba(var(--primary-rgb), 0.05);
    }
    .guide-box.success { border-left-color: var(--success-green); background: rgba(16,185,129,0.05); }
    .guide-box.danger  { border-left-color: var(--danger-red);   background: rgba(239,68,68,0.05);   }
    .stButton>button {
        border: 2px solid var(--primary-color);
        background: transparent;
        color: var(--primary-color);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.75rem 2rem;
    }
    .stButton>button:hover { background: var(--primary-color); color: #000; }
</style>
""", unsafe_allow_html=True)

# ────────────────────────────────────────────────────────────────────────────
# Math / Physics Functions
# ────────────────────────────────────────────────────────────────────────────

def distance_correlation(x, y):
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = len(x)
    if n > 1500:
        idx = np.random.choice(n, 1500, replace=False)
        x, y = x[idx], y[idx]
        n = 1500
    A = squareform(pdist(x))
    B = squareform(pdist(y))
    A_cent = A - A.mean(1, keepdims=True) - A.mean(0, keepdims=True) + A.mean()
    B_cent = B - B.mean(1, keepdims=True) - B.mean(0, keepdims=True) + B.mean()
    dcov2 = (A_cent * B_cent).sum() / n**2
    dvarx2 = (A_cent * A_cent).sum() / n**2
    dvary2 = (B_cent * B_cent).sum() / n**2
    if dvarx2 > 0 and dvary2 > 0:
        return float(np.sqrt(dcov2 / np.sqrt(dvarx2 * dvary2)))
    return 0.0


def hsic(x, y):
    """Hilbert-Schmidt Independence Criterion — kernel-based dependence"""
    x = np.asarray(x, dtype=float).reshape(-1, 1)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    n = len(x)
    if n > 1200:
        idx = np.random.choice(n, 1200, replace=False)
        x, y = x[idx], y[idx]
        n = 1200

    def rbf(z, sigma=1.0):
        dist = squareform(pdist(z, 'sqeuclidean'))
        return np.exp(-dist / (2 * sigma**2))

    K = rbf(x)
    L = rbf(y)
    H = np.eye(n) - np.ones((n,n)) / n
    Kc = H @ K @ H
    Lc = H @ L @ H
    hs = np.trace(Kc @ Lc) / n**2
    return float(np.sqrt(max(hs, 0)))


def pagerank_centrality(adj, damping=0.85, max_iter=100):
    n = adj.shape[0]
    deg = adj.sum(axis=0, keepdims=True)
    deg[deg == 0] = 1
    P = adj / deg
    pr = np.ones(n) / n
    for _ in range(max_iter):
        new_pr = damping * P @ pr + (1 - damping) / n
        if np.max(np.abs(new_pr - pr)) < 1e-6:
            break
        pr = new_pr
    return pr / (pr.sum() + 1e-12)


def granger_causality_score(x, y, maxlag=4):
    if len(x) < 30 or not STATSMODELS_AVAILABLE:
        return 0.0
    try:
        dfg = pd.DataFrame({'y': y, 'x': x}).dropna()
        if len(dfg) < 30:
            return 0.0
        results = grangercausalitytests(dfg[['y', 'x']], maxlag=maxlag, verbose=False)
        pvals = [results[lag][0]['ssr_ftest'][1] for lag in range(1, maxlag+1)]
        return float(np.clip(-np.log10(min(pvals) + 1e-12) / 4.0, 0.0, 1.0))
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
        self.topo_scores = None

    def analyze(self, progress_callback=None):
        def prog(pct, msg):
            if progress_callback:
                progress_callback(pct, msg)

        prog(5, "Preparing scaled data...")
        y = self.data[self.target].values
        X_df = self.data[self.features]

        prog(15, "Pearson correlations & VIF...")
        self.corr_matrix = self.data[[self.target] + self.features].corr(method='pearson')

        if STATSMODELS_AVAILABLE:
            try:
                Xc = sm.add_constant(X_df)
                for i, col in enumerate(self.features):
                    self.vif_data[col] = variance_inflation_factor(Xc.values, i+1)
            except:
                self.vif_data = {f: np.nan for f in self.features}
        else:
            self.vif_data = {f: 1.0 for f in self.features}

        prog(30, "Mutual Information & Random Forest...")
        mi_scores = mutual_info_regression(self.X_scaled, self.y_scaled) if SKLEARN_AVAILABLE else np.zeros(len(self.features))

        rf_importance = np.zeros(len(self.features))
        rf_model = None
        if SKLEARN_AVAILABLE:
            rf_model = RandomForestRegressor(n_estimators=200, max_depth=8, random_state=42)
            rf_model.fit(self.X_scaled, self.y_scaled)
            rf_importance = rf_model.feature_importances_

        prog(45, "Advanced importance (SHAP / Permutation)...")
        adv_imp = np.zeros(len(self.features))
        if SKLEARN_AVAILABLE and rf_model is not None:
            if SHAP_AVAILABLE:
                try:
                    explainer = shap.TreeExplainer(rf_model)
                    shap_vals = explainer.shap_values(self.X_scaled)
                    adv_imp = np.abs(shap_vals).mean(0)
                except:
                    pass
            if np.all(adv_imp == 0):
                try:
                    perm = permutation_importance(rf_model, self.X_scaled, self.y_scaled,
                                                  n_repeats=10, random_state=42, n_jobs=-1)
                    adv_imp = perm.importances_mean
                except:
                    adv_imp = rf_importance.copy()

        prog(60, "dCor graph & hybrid topology...")
        if len(self.features) > 1:
            dcor_mat = np.eye(len(self.features))
            step = 35 / (len(self.features) * (len(self.features)-1) // 2)
            cnt = 0
            for i in range(len(self.features)):
                for j in range(i+1, len(self.features)):
                    d = distance_correlation(self.X_scaled[:,i], self.X_scaled[:,j])
                    dcor_mat[i,j] = dcor_mat[j,i] = d
                    cnt += 1
                    if cnt % 5 == 0:
                        prog(60 + step * cnt, "Building dependence graph...")
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

        prog(80, "HSIC & Granger causality...")
        for i, feat in enumerate(self.features):
            x_raw = self.data[feat].values
            pearson = np.corrcoef(x_raw, y)[0,1] if np.std(x_raw) > 0 else 0
            spearman = pd.Series(x_raw).corr(pd.Series(y), method='spearman')
            dcor = distance_correlation(self.X_scaled[:,i], self.y_scaled)
            hsic_val = hsic(self.X_scaled[:,i], self.y_scaled)
            granger_val = granger_causality_score(x_raw, y) if self.date_col else 0.0

            self.results.append({
                'Feature': feat,
                'Pearson': pearson,
                'Abs_Pearson': abs(pearson),
                'Spearman': spearman,
                'Distance_Corr': dcor,
                'HSIC': hsic_val,
                'Mutual_Info': mi_scores[i],
                'RF_Importance': rf_importance[i],
                'Advanced_Importance': adv_imp[i],
                'VIF': self.vif_data.get(feat, np.nan),
                'Topological_Centrality': self.topo_scores[i],
                'Granger_Score': granger_val
            })

        prog(95, "Computing final essence scores...")
        self.res_df = pd.DataFrame(self.results)
        self._calculate_composite_score()

        prog(100, "Tattva analysis complete ✓")
        return self.res_df


    def _calculate_composite_score(self):
        df = self.res_df
        def norm(col):
            mx, mn = df[col].max(), df[col].min()
            return np.zeros(len(df)) if mx == mn else (df[col] - mn) / (mx - mn)

        score = (
            norm('Abs_Pearson')         * 0.05 +
            norm('Spearman')            * 0.05 +
            norm('Distance_Corr')       * 0.14 +
            norm('HSIC')                * 0.16 +
            norm('Mutual_Info')         * 0.14 +
            norm('Advanced_Importance') * 0.20 +
            norm('Topological_Centrality') * 0.15 +
            norm('Granger_Score')       * 0.11
        ) * 100

        vif_penalty = np.where(df['VIF'] > 10, 0.8, 1.0)
        vif_penalty = np.where(df['VIF'] > 50, 0.5, vif_penalty)
        df['Composite_Score'] = score * vif_penalty
        self.res_df = df.sort_values('Composite_Score', ascending=False).reset_index(drop=True)


    def get_insights(self):
        df = self.res_df
        top_feat = df.iloc[0]['Feature'] if not df.empty else "None"
        def norm(a): return np.zeros_like(a) if a.max()==a.min() else (a-a.min())/(a.max()-a.min())
        df['NonLinear_Bias'] = (norm(df['Distance_Corr']) + norm(df['HSIC']) + norm(df['Mutual_Info'])) / 3 - norm(df['Abs_Pearson'])
        hidden = df[df['NonLinear_Bias'] > 0.3]['Feature'].tolist()
        redundant = df[df['VIF'] > 10]['Feature'].tolist()
        return {
            'top_feature': top_feat,
            'top_score': df.iloc[0]['Composite_Score'] if not df.empty else 0,
            'hidden_nonlinear': hidden[:3],
            'redundant_features': redundant
        }


# ────────────────────────────────────────────────────────────────────────────
# Data loading & cleaning
# ────────────────────────────────────────────────────────────────────────────

def load_google_sheet(sheet_url):
    try:
        import re
        sheet_id = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url).group(1)
        gid = re.search(r'gid=(\d+)', sheet_url).group(1) if re.search(r'gid=', sheet_url) else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url)
        return df, None
    except Exception as e:
        return None, str(e)


def clean_data(df, target, features, date_col=None):
    cols = [target] + features
    if date_col and date_col != "None" and date_col in df.columns:
        cols.append(date_col)
    data = df[cols].copy()
    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    numeric_cols = [target] + features
    data = data[np.isfinite(data[numeric_cols]).all(axis=1)]
    if date_col and date_col != "None" and date_col in data.columns:
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col])
            data = data.sort_values(date_col)
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


# ────────────────────────────────────────────────────────────────────────────
# Landing page
# ────────────────────────────────────────────────────────────────────────────

def render_landing_page():
    st.markdown("""
    <div class="premium-header">
        <h1>TATTVA : Essence Matrix</h1>
        <div class="tagline">Revealing the fundamental predictive truth of your features.</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid var(--purple);">
            <h3 style="color: var(--purple);">Information & Kernel Theory</h3>
            <p>Mutual Information + HSIC → captures any dependence structure</p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid var(--info-cyan);">
            <h3 style="color: var(--info-cyan);">Energy & Game Theory</h3>
            <p>dCor + SHAP → physics-based independence + fair contribution</p>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown("""
        <div class="metric-card" style="border-left: 4px solid var(--primary-color);">
            <h3 style="color: var(--primary-color);">Topology & Causality</h3>
            <p>Hybrid centrality + Granger → network influence & time direction</p>
        </div>
        """, unsafe_allow_html=True)

    st.info("Configure target, features and optional time axis in the sidebar, then click **RUN ANALYSIS**")


# ────────────────────────────────────────────────────────────────────────────
# Main Application Logic
# ────────────────────────────────────────────────────────────────────────────

def main():
    # Sidebar
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2rem; font-weight:800; color:#FFC300;">TATTVA</div>
            <div style="color:#888; font-size:0.9rem;">तत्त्व | Essence Matrix</div>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Data Source")
        source = st.radio("", ["Upload CSV/Excel", "Google Sheets"], horizontal=True)

        df = None
        if source == "Upload CSV/Excel":
            file = st.file_uploader("", type=["csv", "xlsx"])
            if file:
                try:
                    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                    st.session_state["data"] = df
                except Exception as e:
                    st.error(f"File reading error: {e}")
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
            url = st.text_input("Sheet URL", value=default_url)
            if st.button("Load Google Sheet"):
                with st.spinner("Fetching sheet..."):
                    df, err = load_google_sheet(url)
                    if err:
                        st.error(err)
                    else:
                        st.session_state["data"] = df
                        st.success("Sheet loaded")

        if "data" in st.session_state:
            df = st.session_state["data"]

    if df is None:
        render_landing_page()
        return

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Dataset needs at least 2 numeric columns.")
        return

    with st.sidebar:
        st.markdown("### Analysis Setup")
        target = st.selectbox("Target variable (Y)", numeric_cols)
        candidates = [c for c in numeric_cols if c != target]
        features = st.multiselect("Feature columns (X)", candidates, default=candidates[:min(10, len(candidates))])

        time_cols = [c for c in df.columns if "date" in c.lower() or "time" in c.lower()]
        time_axis = st.selectbox("Time column (for Granger)", ["None"] + df.columns.tolist(),
                                 index=df.columns.tolist().index(time_cols[0]) if time_cols else 0)

        if st.button("🚀 RUN ANALYSIS", type="primary", use_container_width=True):
            st.session_state["run_triggered"] = True
            st.session_state["current_config"] = (target, tuple(sorted(features)), time_axis)

    # Only run when button is pressed and config changed
    current_config = (target, tuple(sorted(features)), time_axis)
    if not st.session_state.get("run_triggered", False) or \
       st.session_state.get("last_config") == current_config:
        if not st.session_state.get("run_triggered", False):
            render_landing_page()
        return

    st.session_state["last_config"] = current_config

    # Clean data
    cleaned = clean_data(df, target, list(features), time_axis if time_axis != "None" else None)
    if len(cleaned) < 30:
        st.error("After cleaning & type conversion — fewer than 30 valid rows remain.")
        return

    # Progress UI
    progress = st.progress(0)
    status_text = st.empty()

    def progress_update(pct: int, message: str):
        progress.progress(pct / 100)
        status_text.text(f"{pct}% — {message}")

    # Cache key
    cache_key = f"{target}_{sorted(features)}_{len(cleaned)}_{time_axis}"

    if st.session_state.get("cache_key") != cache_key or "engine" not in st.session_state:
        progress_update(0, "Starting Tattva computation...")
        engine = TattvaEngine(cleaned, target, features, time_axis if time_axis != "None" else None)
        engine.analyze(progress_update)
        st.session_state["engine"] = engine
        st.session_state["cache_key"] = cache_key
    else:
        progress_update(100, "Using cached results")
        engine = st.session_state["engine"]

    progress.empty()
    status_text.empty()

    res = engine.res_df
    insight = engine.get_insights()

    # ───── Dashboard ─────
    st.markdown(f"""
    <div class="premium-header">
        <h1>TATTVA Results — {target}</h1>
        <div class="tagline">{len(features)} features analyzed • {len(res)} valid rows</div>
    </div>
    """, unsafe_allow_html=True)

    cols = st.columns(4)
    with cols[0]:
        st.metric("Top Feature", insight["top_feature"], f"{insight['top_score']:.1f}/100")
    with cols[1]:
        st.metric("Hidden Gems", len(insight["hidden_nonlinear"]), ",".join(insight["hidden_nonlinear"])[:30])
    with cols[2]:
        st.metric("Redundant (VIF>10)", len(insight["redundant_features"]))
    with cols[3]:
        st.metric("Avg HSIC", f"{res['HSIC'].mean():.4f}")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Ranking", "Correlations", "Dependence", "Topology", "Table"
    ])

    with tab1:
        fig = px.bar(res, x="Composite_Score", y="Feature", orientation="h",
                     color="Composite_Score", color_continuous_scale="RdYlGn")
        fig.update_layout(height=600, yaxis={'categoryorder':'total ascending'})
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        fig = go.Figure(go.Heatmap(
            z=engine.corr_matrix.values,
            x=engine.corr_matrix.columns,
            y=engine.corr_matrix.columns,
            colorscale="RdBu_r",
            zmin=-1, zmax=1,
            text=np.round(engine.corr_matrix.values, 2),
            texttemplate="%{text}"
        ))
        fig.update_layout(height=700)
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab3:
        fig = px.scatter(res, x="Abs_Pearson", y="HSIC", size="Mutual_Info",
                         color="Composite_Score", hover_name="Feature",
                         color_continuous_scale="Viridis")
        fig.update_layout(height=600)
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab4:
        fig = px.bar(res.sort_values("VIF", ascending=False),
                     x="Feature", y="VIF", color="VIF",
                     color_continuous_scale=["green","yellow","red"])
        fig.add_hline(y=10, line_dash="dash", line_color="red")
        fig.update_layout(height=500)
        update_chart_theme(fig)
        st.plotly_chart(fig, use_container_width=True)

    with tab5:
        show_cols = ["Feature", "Composite_Score", "Pearson", "Spearman",
                     "Distance_Corr", "HSIC", "Mutual_Info", "VIF"]
        st.dataframe(res[show_cols].round(4), use_container_width=True, height=600)

        csv = res.to_csv(index=False).encode('utf-8')
        st.download_button("Download full results", csv, f"tattva_{target}.csv", "text/csv")


def render_footer():
    now_ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.caption(f"© {now_ist.year} {PRODUCT_NAME} | {COMPANY} | {VERSION} | {now_ist.strftime('%Y-%m-%d %H:%M IST')}")


if __name__ == "__main__":
    main()
    render_footer()
