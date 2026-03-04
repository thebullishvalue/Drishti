"""
TATTVA (तत्त्व) – Essence Matrix
Advanced Feature Truth Engine | Hemrek Capital

Refined premium dark-mode UI inspired by Nirnay.
Dream Engine backend – Stability • Clustering • Bootstrap • CV R²
All values rounded to 2 decimals.
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
# PAGE & THEME – Clean, elegant, Nirnay-inspired but refined
# =============================================================================

st.set_page_config(
    page_title="TATTVA • Essence Matrix",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded"
)

VERSION = "v5.1 – Polished Edition"
PRODUCT_NAME = "Tattva"
COMPANY = "Hemrek Capital"

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

    :root {
        --primary: #FFC300;
        --primary-dark: #D4A017;
        --bg: #0A0A0A;
        --bg-card: #111111;
        --bg-elevated: #181818;
        --text: #F5F5F5;
        --text-muted: #A0A0A0;
        --border: #222222;
        --border-light: #333333;
        --green: #22C55E;
        --red: #EF4444;
        --amber: #F59E0B;
        --cyan: #06B6D4;
    }

    * { font-family: 'Inter', system-ui, sans-serif; }

    .stApp { background: var(--bg); color: var(--text); }

    section[data-testid="stSidebar"] {
        background: var(--bg-elevated);
        border-right: 1px solid var(--border);
    }

    /* Elegant sidebar toggle */
    [data-testid="collapsedControl"] {
        background: var(--bg-card) !important;
        border: 1px solid var(--primary) !important;
        border-radius: 12px !important;
        box-shadow: 0 4px 16px rgba(255,195,0,0.12) !important;
        top: 20px !important;
        left: 20px !important;
        width: 48px !important;
        height: 48px !important;
        transition: all 0.25s ease !important;
    }
    [data-testid="collapsedControl"]:hover {
        transform: scale(1.1) !important;
        box-shadow: 0 8px 24px rgba(255,195,0,0.25) !important;
    }

    /* Premium header */
    .header {
        background: linear-gradient(135deg, #111111 0%, #0A0A0A 100%);
        padding: 2rem 3rem;
        border-radius: 20px;
        margin: 1.5rem 0 2.5rem;
        border: 1px solid var(--border-light);
        box-shadow: 0 10px 40px rgba(0,0,0,0.7);
        position: relative;
        overflow: hidden;
    }
    .header::before {
        content: '';
        position: absolute;
        inset: -50%;
        background: radial-gradient(circle at 30% 20%, rgba(255,195,0,0.08) 0%, transparent 70%);
        pointer-events: none;
    }
    .header h1 {
        margin: 0;
        font-size: 3rem;
        font-weight: 900;
        letter-spacing: -1.2px;
        background: linear-gradient(90deg, #fff, var(--primary), #fff);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
    }
    .header .tagline {
        color: var(--text-muted);
        font-size: 1.1rem;
        margin-top: 0.6rem;
        font-weight: 400;
    }

    /* Cards – refined depth & hover */
    .card {
        background: var(--bg-card);
        border: 1px solid var(--border);
        border-radius: 16px;
        padding: 1.6rem;
        box-shadow: 0 8px 24px rgba(0,0,0,0.5);
        transition: all 0.3s cubic-bezier(0.4,0,0.2,1);
        backdrop-filter: blur(6px);
    }
    .card:hover {
        transform: translateY(-8px);
        box-shadow: 0 20px 48px rgba(255,195,0,0.15);
        border-color: rgba(255,195,0,0.35);
    }
    .card h4 {
        color: var(--text-muted);
        font-size: 0.85rem;
        font-weight: 600;
        letter-spacing: 1px;
        text-transform: uppercase;
        margin-bottom: 0.8rem;
    }
    .card h2 {
        font-size: 2.4rem;
        font-weight: 800;
        margin: 0;
        line-height: 1;
    }
    .card .sub {
        font-size: 0.92rem;
        color: var(--text-muted);
        margin-top: 0.6rem;
    }

    .card.primary h2 { color: var(--primary); }
    .card.success h2 { color: var(--green); }
    .card.danger h2 { color: var(--red); }
    .card.warning h2 { color: var(--amber); }
    .card.info h2 { color: var(--cyan); }

    /* Tabs – modern & elegant */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2.5rem;
        background: transparent;
        border-bottom: 1px solid var(--border);
        padding-bottom: 0.8rem;
        margin-bottom: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        color: var(--text-muted);
        font-weight: 600;
        font-size: 1.1rem;
        padding: 0.9rem 2rem;
        border-radius: 12px 12px 0 0;
        transition: all 0.25s ease;
    }
    .stTabs [aria-selected="true"] {
        color: var(--primary);
        background: rgba(255,195,0,0.07);
        border-bottom: 4px solid var(--primary);
    }

    /* Button */
    .stButton > button {
        background: transparent;
        border: 1.5px solid var(--primary);
        color: var(--primary);
        font-weight: 700;
        border-radius: 12px;
        padding: 0.9rem 2.2rem;
        transition: all 0.25s ease;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .stButton > button:hover {
        background: var(--primary);
        color: #000;
        box-shadow: 0 0 30px rgba(255,195,0,0.45);
        transform: translateY(-3px);
    }

    /* Progress bar – golden */
    .stProgress > div > div > div {
        background: var(--primary) !important;
    }

    /* Divider */
    .divider {
        height: 1px;
        background: linear-gradient(90deg, transparent, var(--border-light), transparent);
        margin: 3rem 0;
    }

    /* Scrollbar */
    ::-webkit-scrollbar { width: 7px; height: 7px; }
    ::-webkit-scrollbar-track { background: var(--bg); }
    ::-webkit-scrollbar-thumb { background: var(--border-light); border-radius: 4px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--primary); }

</style>
""", unsafe_allow_html=True)


# =============================================================================
# DREAM ENGINE (your exact code – no modifications)
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
            "top_feature": df.iloc[0]["Feature"] if not df.empty else "None",
            "top_score": df.iloc[0]["Composite_Score"] if not df.empty else 0,
            "hidden_nonlinear": hidden[:3],
            "redundant_features": redundant[:5]
        }


# =============================================================================
# DATA UTILITIES
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
    if date_col and date_col != "None":
        cols.append(date_col)
    data = df[cols].copy()
    for col in [target] + features:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    data = data.dropna()
    data = data[np.isfinite(data[[target] + features]).all(axis=1)]
    if date_col and date_col != "None":
        try:
            data[date_col] = pd.to_datetime(data[date_col], errors='coerce')
            data = data.dropna(subset=[date_col]).sort_values(date_col)
        except:
            pass
    return data.reset_index(drop=True)


# =============================================================================
# MAIN APP – Polished UI
# =============================================================================

def main():
    # ── Header ───────────────────────────────────────────────────────────────
    st.markdown(f"""
    <div class="header">
        <h1>TATTVA</h1>
        <div class="tagline">तत्त्व • Essence Matrix • Feature Truth Engine</div>
        <div style="margin-top:0.8rem; font-size:0.95rem; color:var(--text-muted);">
            v{VERSION} • {COMPANY}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### Data Source")
        source = st.radio("", ["Upload CSV/Excel", "Google Sheets"], horizontal=True)

        df = None
        if source == "Upload CSV/Excel":
            file = st.file_uploader("", type=["csv", "xlsx"])
            if file:
                try:
                    df = pd.read_csv(file) if file.name.endswith(".csv") else pd.read_excel(file)
                except Exception as e:
                    st.error(f"File error: {e}")
        else:
            url = st.text_input("Google Sheet URL")
            if st.button("Load Sheet", use_container_width=True):
                df, err = load_google_sheet(url)
                if err:
                    st.error(err)
                else:
                    st.success("Sheet loaded")

        if df is not None:
            numeric = df.select_dtypes(include=[np.number]).columns.tolist()
            if len(numeric) < 2:
                st.error("Need at least 2 numeric columns")
                st.stop()

            st.markdown("### Analysis Setup")
            target = st.selectbox("Target (Y)", numeric)
            features = st.multiselect("Features (X)", [c for c in numeric if c != target],
                                     default=[c for c in numeric if c != target][:12])

            date_col = st.selectbox("Date Column (optional)", ["None"] + list(df.columns))

            if st.button("RUN ANALYSIS", type="primary", use_container_width=True):
                st.session_state['run'] = True

    # ── Main flow ────────────────────────────────────────────────────────────
    if 'run' not in st.session_state or not st.session_state['run']:
        st.info("Configure target/features in sidebar → click RUN ANALYSIS")
        return

    with st.spinner("Extracting feature essence..."):
        progress = st.progress(0)
        def update(pct): progress.progress(pct / 100)

        try:
            data = clean_data(df, target, features, date_col if date_col != "None" else None)
            if len(data) < 30:
                raise ValueError("Need ≥ 30 clean rows after processing")

            engine = TattvaEngine(data, target, features)
            engine.analyze(update)
            progress.empty()

        except Exception as e:
            st.error(f"Analysis failed: {str(e)}")
            st.info("Possible causes: too few rows, non-numeric data, singular matrix in clustering. Try fewer features or cleaner input.")
            return

    res_df = engine.res_df
    insights = engine.get_insights()

    # ── Top cards – elegant layout ──────────────────────────────────────────
    cols = st.columns(4)
    with cols[0]:
        st.markdown(f"""
        <div class="card primary">
            <h4>Primary Essence</h4>
            <h2>{insights['top_feature']}</h2>
            <div class="sub">Score: {insights['top_score']:.2f}</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[1]:
        n = len(insights['hidden_nonlinear'])
        st.markdown(f"""
        <div class="card info">
            <h4>Hidden Gems</h4>
            <h2>{n}</h2>
            <div class="sub">Non-linear value</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[2]:
        n = len(insights['redundant_features'])
        st.markdown(f"""
        <div class="card danger">
            <h4>Redundant</h4>
            <h2>{n}</h2>
            <div class="sub">Cluster groups</div>
        </div>
        """, unsafe_allow_html=True)

    with cols[3]:
        avg_stab = res_df['Stability'].mean()
        st.markdown(f"""
        <div class="card neutral">
            <h4>Avg Stability</h4>
            <h2>{avg_stab:.2f}</h2>
            <div class="sub">Bootstrap stability</div>
        </div>
        """, unsafe_allow_html=True)

    # ── Guidance panel ──────────────────────────────────────────────────────
    st.markdown("### Feature Utilization Guidance")
    with st.container():
        st.markdown('<div style="background:rgba(255,195,0,0.05); border:1px solid rgba(255,195,0,0.3); border-radius:16px; padding:1.8rem;">', unsafe_allow_html=True)

        c1, c2 = st.columns([3, 2])

        with c1:
            keep = res_df[res_df['Composite_Score'] > 70]['Feature'].head(7).tolist()
            if keep:
                st.markdown("**Strongly recommended to keep** (score > 70)")
                st.markdown("\n".join([f"• {f}" for f in keep]))
            if insights['hidden_nonlinear']:
                st.markdown("\n**Non-linear hidden value**")
                st.markdown("\n".join([f"• {f}" for f in insights['hidden_nonlinear']]))

        with c2:
            if insights['redundant_features']:
                st.markdown("**Likely redundant** (same cluster)")
                st.markdown("\n".join([f"• {f}" for f in insights['redundant_features'][:6]]))
            else:
                st.markdown("**No strong redundancy** — features appear independent")

        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────────────
    tab1, tab2 = st.tabs(["Essence Ranking", "Full Results"])

    with tab1:
        fig = px.bar(
            res_df,
            x='Composite_Score',
            y='Feature',
            orientation='h',
            color='Composite_Score',
            color_continuous_scale=['#ef4444', '#f59e0b', '#22C55E', '#FFC300'],
            hover_data={
                'Pearson': ':.2f',
                'Spearman': ':.2f',
                'Distance_Corr': ':.2f',
                'Stability': ':.2f',
                'NonLinear_Bias': ':.2f',
                'Uncertainty': ':.3f'
            }
        )
        fig.update_layout(
            height=600,
            yaxis={'categoryorder':'total ascending'},
            margin=dict(l=20,r=20,t=40,b=20),
            xaxis_title="Composite Essence Score",
            template="plotly_dark"
        )
        st.plotly_chart(fig, use_container_width=True)

    with tab2:
        disp = res_df.copy()
        for c in disp.select_dtypes(include='number').columns:
            disp[c] = disp[c].round(2)
        st.dataframe(
            disp[['Feature','Composite_Score','Pearson','Spearman','Distance_Corr','Stability','Cluster_Label','NonLinear_Bias','Uncertainty']],
            use_container_width=True,
            height=700
        )

        csv = disp.to_csv(index=False).encode()
        st.download_button("Download Full Results", csv, "tattva_results.csv", "text/csv")

    # ── Footer ──────────────────────────────────────────────────────────────
    from datetime import datetime, timezone, timedelta
    ist = datetime.now(timezone.utc) + timedelta(hours=5, minutes=30)
    st.markdown(f"""
    <div style="text-align:center; color:var(--text-muted); margin:4rem 0 2rem; font-size:0.95rem;">
        {PRODUCT_NAME} • {COMPANY} • {VERSION} • {ist.strftime('%d %b %Y %H:%M IST')}
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
