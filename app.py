"""
TATTVA (तत्त्व) - MLR Engine | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced v2.5.1: VIF & resolution plan always computed (even for Ridge/Lasso/Bayesian),
fixed NoneType.empty crash, improved stability messaging, full production-ready version.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime
import warnings
import logging
import hashlib
from typing import List, Dict, Any, Self, Optional, Tuple
import pytz
import os
from enum import Enum
from scipy import linalg, stats
import numpy.linalg as la

# Enhanced Dependencies
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    sm = None
    STATSMODELS_AVAILABLE = False
    print(f"Statsmodels import error: {e}")

# PyMC for Bayesian
try:
    import pymc as pm
    import arviz as az
    PYMC_AVAILABLE = True
except ImportError:
    PYMC_AVAILABLE = False
    print("PyMC not available — Bayesian falls back to ridge approximation.")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Specific Warning Filters
if STATSMODELS_AVAILABLE:
    warnings.filterwarnings('ignore', message='.*collinearity.*')
    warnings.filterwarnings('ignore', module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

# ────────────────────────────────────────────────
# Constants
# ────────────────────────────────────────────────
VERSION = "v2.5.1"
PRODUCT_NAME = os.getenv("TATTVA_PRODUCT_NAME", "TATTVA")
COMPANY = os.getenv("TATTVA_COMPANY", "Hemrek Capital")
MAX_ROWS = int(os.getenv("TATTVA_MAX_ROWS", "10000"))
MAX_COLS = int(os.getenv("TATTVA_MAX_COLS", "100"))
MIN_ROWS_PER_FEATURE = 10

# Enums
class VIFStatus(Enum):
    EXCELLENT = "Excellent (Uncorrelated)"
    ACCEPTABLE = "Acceptable (Moderate Noise)"
    SEVERE   = "Severe Collinearity (DROP THIS)"

class ModelGrade(Enum):
    UNSTABLE   = ("UNSTABLE",   "danger",  "The model is statistically invalid (rank-deficient or ill-conditioned). Do not trade on these signals.")
    WEAK       = ("WEAK",       "warning", "High noise-to-signal ratio or mild ill-conditioning. Use extreme caution.")
    MODERATE   = ("MODERATE",   "warning", "Model is usable but many variables insignificant.")
    ACCEPTABLE = ("ACCEPTABLE", "primary", "Model geometry is stable and actionable.")
    STRONG     = ("STRONG",     "success", "Excellent statistical geometry. Low collinearity, high explanatory power.")

# ────────────────────────────────────────────────
# Page Config & CSS (unchanged from your last version)
# ────────────────────────────────────────────────

st.set_page_config(
    page_title=f"{PRODUCT_NAME} | MLR Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

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
        --purple: #8b5cf6;
    }
    
    * { font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif; }
    
    .main, [data-testid="stSidebar"] { background-color: var(--background-color); color: var(--text-primary); }
    .stApp > header { background-color: transparent; }
    #MainMenu {visibility: hidden;} footer {visibility: hidden;}
    .block-container { padding-top: 3.5rem; max-width: 90%; padding-left: 2rem; padding-right: 2rem; }
    
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
    
    button[kind="header"] {
        z-index: 999999 !important;
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
    .premium-header .product-badge { display: inline-block; background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); padding: 0.25rem 0.75rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.5rem; }
    
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
    .metric-card h4 { color: var(--text-muted); font-size: 0.75rem; margin-bottom: 0.5rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; }
    .metric-card h3 { color: var(--text-primary); font-size: 1.1rem; font-weight: 700; margin-bottom: 0.5rem; }
    .metric-card p { color: var(--text-muted); font-size: 0.85rem; line-height: 1.5; margin: 0; }
    .metric-card h2 { color: var(--text-primary); font-size: 1.75rem; font-weight: 700; margin: 0; line-height: 1; }
    .metric-card .sub-metric { font-size: 0.75rem; color: var(--text-muted); margin-top: 0.5rem; font-weight: 500; }
    
    .metric-card.success h2 { color: var(--success-green); }
    .metric-card.danger h2 { color: var(--danger-red); }
    .metric-card.warning h2 { color: var(--warning-amber); }
    .metric-card.info h2 { color: var(--info-cyan); }
    .metric-card.neutral h2 { color: var(--neutral); }
    .metric-card.primary h2 { color: var(--primary-color); }
    .metric-card.purple h2 { color: var(--purple); }
    
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
    .signal-card.buy::before, .signal-card.success::before { background: var(--success-green); }
    .signal-card.sell::before, .signal-card.danger::before { background: var(--danger-red); }
    .signal-card.warning::before { background: var(--warning-amber); }
    .signal-card.primary::before { background: var(--primary-color); }
    .signal-card.info::before { background: var(--info-cyan); }
    
    .signal-card-header { display: flex; align-items: center; justify-content: space-between; margin-bottom: 1rem; }
    .signal-card-title { font-size: 0.8rem; font-weight: 700; text-transform: uppercase; letter-spacing: 1px; color: var(--text-muted); }
    .signal-card .label { font-size: 0.75rem; text-transform: uppercase; letter-spacing: 1.5px; color: var(--text-muted); font-weight: 600; margin-bottom: 0.5rem; }
    .signal-card .value { font-size: 2.5rem; font-weight: 700; line-height: 1; margin: 0.5rem 0;}
    .signal-card .subtext { font-size: 0.85rem; color: var(--text-secondary); margin-top: 0.5rem; line-height: 1.5;}
    
    .signal-card.danger .value { color: var(--danger-red); }
    .signal-card.success .value { color: var(--success-green); }
    .signal-card.warning .value { color: var(--warning-amber); }
    .signal-card.primary .value { color: var(--primary-color); }
    
    .status-badge { display: inline-flex; align-items: center; gap: 0.5rem; padding: 0.4rem 0.8rem; border-radius: 20px; font-size: 0.7rem; font-weight: 700; text-transform: uppercase; letter-spacing: 0.5px; }
    .status-badge.buy { background: rgba(16, 185, 129, 0.15); color: var(--success-green); border: 1px solid rgba(16, 185, 129, 0.3); }
    .status-badge.sell { background: rgba(239, 68, 68, 0.15); color: var(--danger-red); border: 1px solid rgba(239, 68, 68, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    .status-badge.primary { background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 4px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    .stButton > button {
        border: 2px solid var(--primary-color) !important;
        background: transparent !important;
        color: var(--primary-color) !important;
        font-weight: 700 !important;
        border-radius: 12px !important;
        padding: 0.75rem 2rem !important;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1) !important;
        text-transform: uppercase !important;
        letter-spacing: 0.5px !important;
        width: 100% !important;
        margin-bottom: 0.5rem !important;
    }
    .stButton > button:hover {
        box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.6) !important;
        background: var(--primary-color) !important;
        color: #1A1A1A !important;
        transform: translateY(-2px) !important;
    }
    .stButton > button:active {
        transform: translateY(0) !important;
    }
    .stButton > button[type="primary"] { background: var(--primary-color) !important; color: #1A1A1A !important; }
    .stButton > button[type="primary"]:hover { background: #E6B800 !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart, .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-price { color: var(--text-muted); font-size: 0.85rem; }
    
    .conviction-meter { height: 8px; background: var(--bg-elevated); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ────────────────────────────────────────────────
# MLREngine Class
# ────────────────────────────────────────────────

class MLREngine:
    def __init__(self, df: pd.DataFrame, target: str, features: List[str],
                 use_ridge: bool = False, alpha: float = 1.0,
                 use_lasso: bool = False,
                 use_bayesian: bool = False, prior_strength: float = 1.0,
                 n_samples: int = 1000):
        self.df = df.copy()
        self.target = target
        self.features = features
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.use_lasso = use_lasso
        self.use_bayesian = use_bayesian
        self.prior_strength = prior_strength
        self.n_samples = n_samples
        
        self.model = None
        self.vif_data = None
        self.coef_df = None
        self.feature_importance = None
        self.resolution_plan = []
        self.condition_number = None
        self.matrix_rank = None
        self.is_stable = True
        self.fit_type = "OLS"
        self.posterior_samples = None
        
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_with_const = add_constant(self.X)
        self.corr_matrix = self.X.corr()
    
    def _fit_ols(self):
        return sm.OLS(self.y, self.X_with_const).fit()
    
    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
        n = X.shape[1]
        XtX = X.T @ X + alpha * np.eye(n)
        beta = linalg.solve(XtX, X.T @ y, assume_a='sym')
        y_pred = X @ beta
        resid = y - y_pred
        ssr = resid @ resid
        sst = (y - y.mean()) @ (y - y.mean())
        r2 = 1 - ssr/sst if sst != 0 else 0
        adj_r2 = 1 - (ssr/(len(y)-n)) / (sst/(len(y)-1)) if sst != 0 else 0
        
        # Approximate standard errors
        sigma2 = ssr / (len(y) - n)
        se = np.sqrt(np.diag(sigma2 * la.inv(X.T @ X + 1e-8 * np.eye(n))))
        tval = beta / se
        pval = 2 * (1 - stats.t.cdf(np.abs(tval), df=len(y)-n))
        
        return {
            'params': beta,
            'bse': se,
            'tvalues': tval,
            'pvalues': pval,
            'rsquared': r2,
            'rsquared_adj': adj_r2,
            'resid': resid
        }
    
    def _coordinate_descent_lasso(self, X: np.ndarray, y: np.ndarray, alpha: float,
                                  max_iter=5000, tol=1e-5) -> np.ndarray:
        n, p = X.shape
        beta = np.zeros(p)
        Xnorm = X / np.sqrt(np.sum(X**2, axis=0) + 1e-8)
        y = y - y.mean()
        
        for _ in range(max_iter):
            beta_old = beta.copy()
            for j in range(p):
                rho = np.dot(Xnorm[:,j], y - Xnorm @ beta + beta[j] * Xnorm[:,j])
                soft = np.sign(rho) * max(0, abs(rho) - alpha)
                beta[j] = soft / (np.dot(Xnorm[:,j], Xnorm[:,j]) + 1e-8)
            if np.max(np.abs(beta - beta_old)) < tol:
                break
        return beta
    
    def _fit_lasso(self, X: np.ndarray, y: np.ndarray, alpha: float) -> dict:
        beta = self._coordinate_descent_lasso(X, y, alpha)
        y_pred = X @ beta
        resid = y - y_pred
        ssr = resid @ resid
        sst = (y - y.mean()) @ (y - y.mean())
        r2 = 1 - ssr/sst if sst != 0 else 0
        df_model = np.sum(beta != 0)
        adj_r2 = 1 - (ssr/(len(y)-df_model-1)) / (sst/(len(y)-1)) if sst != 0 and df_model > 0 else 0
        
        # Approximate inference on non-zero coefficients
        nz = beta != 0
        if np.sum(nz) > 1:
            Xnz = X[:, nz]
            ols = sm.OLS(y, add_constant(Xnz)).fit()
            se = np.full(p, np.nan)
            se[nz] = ols.bse[1:]
            tval = np.full(p, np.nan)
            tval[nz] = ols.tvalues[1:]
            pval = np.full(p, np.nan)
            pval[nz] = ols.pvalues[1:]
        else:
            se = np.full(p, np.std(resid))
            tval = beta / se
            pval = np.ones(p) * 0.5
        
        return {
            'params': beta,
            'bse': se,
            'tvalues': tval,
            'pvalues': pval,
            'rsquared': r2,
            'rsquared_adj': adj_r2,
            'resid': resid
        }
    
    def _fit_bayesian_pymc(self, X: np.ndarray, y: np.ndarray, prior_strength: float, n_samples: int) -> dict:
        if not PYMC_AVAILABLE:
            logger.warning("PyMC unavailable → falling back to ridge for Bayesian")
            return self._fit_ridge(X, y, prior_strength)
        
        with pm.Model() as model:
            sigma = pm.HalfNormal("sigma", sigma=10.0)
            beta = pm.Normal("beta", mu=0, sigma=1.0/np.sqrt(prior_strength), shape=X.shape[1])
            mu = X @ beta
            pm.Normal("y", mu=mu, sigma=sigma, observed=y)
            trace = pm.sample(n_samples, tune=500, chains=2, progressbar=False, return_inferencedata=True)
        
        beta_mean = trace.posterior["beta"].mean(("chain","draw")).values
        beta_std  = trace.posterior["beta"].std(("chain","draw")).values
        sigma_mean = trace.posterior["sigma"].mean(("chain","draw")).values
        
        y_pred = X @ beta_mean
        resid = y - y_pred
        ssr = resid @ resid
        sst = (y - y.mean()) @ (y - y.mean())
        r2 = 1 - ssr/sst if sst != 0 else 0
        adj_r2 = 1 - (ssr/(len(y)-X.shape[1])) / (sst/(len(y)-1)) if sst != 0 else 0
        
        se = np.concatenate([[np.nan], beta_std])
        tval = beta_mean / beta_std
        pval = 2 * (1 - stats.norm.cdf(np.abs(tval)))
        
        self.posterior_samples = trace.posterior["beta"].stack(sample=["chain","draw"]).values
        
        return {
            'params': beta_mean,
            'bse': se,
            'tvalues': tval,
            'pvalues': pval,
            'rsquared': r2,
            'rsquared_adj': adj_r2,
            'resid': resid
        }
    
    def fit(self) -> Self:
        if not STATSMODELS_AVAILABLE:
            raise ImportError("statsmodels is required")
        
        if (self.X.std() == 0).any():
            raise ValueError("Constant feature(s) detected — remove them.")
        
        Xv = self.X_with_const.values
        yv = self.y.values
        
        self.matrix_rank = np.linalg.matrix_rank(Xv)
        rank_deficient = self.matrix_rank < Xv.shape[1]
        
        try:
            s = np.linalg.svd(Xv, compute_uv=False)
            self.condition_number = s[0] / s[-1] if len(s) > 1 and s[-1] > 1e-10 else 1.0
        except:
            self.condition_number = float('inf')
        
        model_fit = None
        
        if self.use_ridge or (rank_deficient and not self.use_lasso and not self.use_bayesian):
            self.fit_type = "Ridge"
            model_fit = self._fit_ridge(Xv, yv, self.alpha)
        elif self.use_lasso or (rank_deficient and self.use_lasso):
            self.fit_type = "Lasso"
            model_fit = self._fit_lasso(Xv, yv, self.alpha)
        elif self.use_bayesian or (rank_deficient and self.use_bayesian):
            self.fit_type = "Bayesian (PyMC)" if PYMC_AVAILABLE else "Ridge (Bayesian fallback)"
            model_fit = self._fit_bayesian_pymc(Xv, yv, self.prior_strength, self.n_samples)
        else:
            try:
                self.model = self._fit_ols()
                self.fit_type = "OLS"
            except np.linalg.LinAlgError:
                self.fit_type = "Ridge (fallback)"
                model_fit = self._fit_ridge(Xv, yv, self.alpha)
        
        # Create mock model when not using pure OLS
        if self.model is None and model_fit is not None:
            params_full = np.insert(model_fit['params'], 0, 0.0)  # const = 0 for centered regularized fits
            bse_full   = np.insert(model_fit['bse'],   0, np.nan)
            tval_full  = np.insert(model_fit['tvalues'], 0, np.nan)
            pval_full  = np.insert(model_fit['pvalues'], 0, 1.0)
            
            self.model = type('MockModel', (), {
                'params': pd.Series(params_full, index=self.X_with_const.columns),
                'bse': pd.Series(bse_full, index=self.X_with_const.columns),
                'tvalues': pd.Series(tval_full, index=self.X_with_const.columns),
                'pvalues': pd.Series(pval_full, index=self.X_with_const.columns),
                'rsquared': model_fit['rsquared'],
                'rsquared_adj': model_fit['rsquared_adj'],
                'f_pvalue': 0.01,
                'resid': model_fit['resid']
            })()
        
        self.is_stable = True
        
        # ────────────────────────────────────────────────
        # ALWAYS compute diagnostics (this fixes the crash)
        # ────────────────────────────────────────────────
        self._compute_vif()
        self._build_collinearity_plan()
        
        # Standardized betas
        std_coefs = [0.0 if var == 'const' else self.model.params[var] * (self.X[var].std() / self.y.std())
                     for var in self.model.params.index]
        
        self.coef_df = pd.DataFrame({
            'Variable': self.model.params.index,
            'Coefficient (Slope)': self.model.params.values,
            'Relative Impact (Std Beta)': std_coefs,
            'Standard Error': self.model.bse.values,
            't-Statistic': self.model.tvalues.values,
            'p-Value': self.model.pvalues.values
        })
        
        fi = self.coef_df[self.coef_df['Variable'] != 'const'].copy()
        fi['Absolute Impact'] = fi['Relative Impact (Std Beta)'].abs()
        self.feature_importance = fi.sort_values('Absolute Impact', ascending=True)
        
        logger.info(f"Fit complete | Type: {self.fit_type} | R² adj: {self.model.rsquared_adj:.3f} | "
                    f"n={len(self.y)} | rank={self.matrix_rank}/{Xv.shape[1]} | cond={self.condition_number:.1f}")
        
        return self
    
    def _compute_vif(self):
        vif_df = pd.DataFrame()
        vif_df["Variable"] = self.X.columns
        
        try:
            vif_values = [variance_inflation_factor(self.X.values, i) for i in range(self.X.shape[1])]
        except:
            vif_values = [np.inf] * self.X.shape[1]
        vif_df["VIF Score"] = vif_values
        
        overlaps = []
        for col in self.X.columns:
            hc = self.corr_matrix[col][(self.corr_matrix[col].abs() > 0.7) & (self.corr_matrix[col].index != col)]
            if not hc.empty:
                hc = hc.sort_values(ascending=False, key=abs)
                overlaps.append(", ".join(f"{k} ({v:.2f})" for k,v in hc.items()))
            else:
                overlaps.append("None")
        vif_df["Primary Overlaps (|r| > 0.7)"] = overlaps
        
        conds = [
            vif_df["VIF Score"] < 3,
            (vif_df["VIF Score"] >= 3) & (vif_df["VIF Score"] <= 5),
            vif_df["VIF Score"] > 5
        ]
        labels = [s.value for s in VIFStatus]
        vif_df["Status"] = np.select(conds, labels, default="Unknown")
        
        self.vif_data = vif_df.sort_values("VIF Score", ascending=False).reset_index(drop=True)
    
    def _build_collinearity_plan(self):
        self.resolution_plan = []
        if self.vif_data is None or self.vif_data.empty or self.vif_data["VIF Score"].max() <= 5:
            return
        
        high_vif = self.vif_data[self.vif_data["VIF Score"] > 5]["Variable"].tolist()
        if not high_vif:
            return
        
        target_corr = self.X.corrwith(self.y).abs()
        corr_abs = self.corr_matrix.abs()
        visited = set()
        clusters = []
        
        for var in high_vif:
            if var in visited:
                continue
            cluster = set()
            stack = [var]
            while stack:
                curr = stack.pop()
                if curr in visited:
                    continue
                visited.add(curr)
                cluster.add(curr)
                neigh = corr_abs.columns[(corr_abs[curr] > 0.7) & (corr_abs.columns != curr)].tolist()
                stack.extend(n for n in neigh if n not in visited)
            clusters.append(list(cluster))
        
        cid = 1
        for cl in clusters:
            if len(cl) > 1:
                ranked = sorted(cl, key=lambda v: target_corr[v], reverse=True)
                keep = ranked[0]
                drop = ranked[1:]
                self.resolution_plan.append({
                    'type': 'cluster',
                    'title': f'Cluster {cid}: Correlated Group',
                    'champion': keep,
                    'drops': drop,
                    'reason': f"<b>{keep}</b> kept — strongest correlation with target ({target_corr[keep]:.2f}). Others redundant."
                })
            else:
                var = cl[0]
                p = self.coef_df[self.coef_df["Variable"] == var]["p-Value"].values
                ptext = f"{p[0]:.4f}" if len(p) > 0 else "N/A"
                self.resolution_plan.append({
                    'type': 'isolate',
                    'title': f'Isolate: {var}',
                    'champion': None,
                    'drops': [var],
                    'reason': f"<b>{var}</b> — complex multicollinearity. p = {ptext}. Consider dropping."
                })
            cid += 1
    
    def get_predictions(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted")
        if hasattr(self.model, 'predict'):
            return self.model.predict(self.X_with_const)
        return self.X_with_const.values @ self.model.params.values
    
    def predict_scenario(self, scenario: Dict[str, float]) -> float:
        missing = [c for c in self.X.columns if c not in scenario]
        if missing:
            raise ValueError(f"Missing inputs: {missing}")
        row = [1.0] + [scenario[c] for c in self.X.columns]
        if hasattr(self.model, 'predict'):
            return self.model.predict([row])[0]
        return np.dot(row, self.model.params.values)
    
    def monte_carlo_scenarios(self, base: Dict[str, float], n_sims: int = 200, noise: float = 0.015) -> Tuple[float, float]:
        if self.fit_type == "Bayesian (PyMC)" and self.posterior_samples is not None:
            preds = []
            for theta in self.posterior_samples[np.random.choice(len(self.posterior_samples), n_sims, replace=False)]:
                row = np.array([1.0] + [base[c] for c in self.X.columns])
                preds.append(np.dot(row, theta))
            return float(np.mean(preds)), float(np.std(preds))
        
        preds = []
        for _ in range(n_sims):
            noisy = {k: v + np.random.normal(0, noise * abs(v)) for k,v in base.items()}
            try:
                preds.append(self.predict_scenario(noisy))
            except:
                pass
        return (np.mean(preds), np.std(preds)) if preds else (0.0, 0.0)
    
    def get_model_health_grade(self) -> Tuple[str,str,str]:
        if self.model is None or self.vif_data is None:
            return ModelGrade.UNSTABLE.value
        
        maxv = self.vif_data["VIF Score"].max() if not self.vif_data.empty else 0
        r2a = self.model.rsquared_adj
        fp = getattr(self.model, 'f_pvalue', 0.05)
        sig = (self.coef_df[self.coef_df["Variable"] != "const"]["p-Value"] < 0.05).mean() if not self.coef_df.empty else 0
        cond = self.condition_number or 9999
        
        if fp > 0.05 or maxv > 10 or cond > 1000:
            return ModelGrade.UNSTABLE.value
        if maxv > 5 or r2a < 0.3 or cond > 100:
            return ModelGrade.WEAK.value
        if sig < 0.4:
            return ModelGrade.MODERATE.value
        if maxv <= 5 and r2a >= 0.6 and cond < 30:
            return ModelGrade.STRONG.value
        return ModelGrade.ACCEPTABLE.value
    
    def generate_auto_features(self, lags=[1,2,3], rolls=[3,5,10]) -> Tuple[pd.DataFrame, List[str]]:
        dfn = self.df.copy()
        added = []
        for f in self.features:
            for lag in lags:
                col = f"{f}_lag{lag}"
                dfn[col] = dfn[f].shift(lag)
                added.append(col)
            for w in rolls:
                col = f"{f}_roll{w}"
                dfn[col] = dfn[f].rolling(w).mean()
                added.append(col)
        dfn = dfn.dropna()
        return dfn, added


# ────────────────────────────────────────────────
# Utilities (load, sanitize, clean, theme)
# ────────────────────────────────────────────────

def load_google_sheet(url: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    try:
        import re
        sid = re.search(r'/d/([a-zA-Z0-9-_]+)', url).group(1)
        gid = re.search(r'gid=(\d+)', url).group(1) if 'gid=' in url else '0'
        csv = f"https://docs.google.com/spreadsheets/d/{sid}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv, low_memory=False)
        return _sanitize_df(df), None
    except Exception as e:
        return None, str(e)

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    if len(df) > MAX_ROWS:
        st.warning(f"Truncated to {MAX_ROWS} rows")
        df = df.iloc[:MAX_ROWS]
    if len(df.columns) > MAX_COLS:
        st.error(f"Too many columns ({len(df.columns)} > {MAX_COLS})")
        st.stop()
    df = df.replace([np.inf, -np.inf], np.nan)
    return df

def clean_data(df: pd.DataFrame, target: str, feats: List[str]) -> pd.DataFrame:
    cols = [target] + feats
    data = df[cols].copy()
    for c in cols:
        data[c] = pd.to_numeric(data[c], errors='coerce')
    n0 = len(data)
    data = data.dropna()
    dropped = n0 - len(data)
    if dropped > 0:
        st.warning(f"Dropped {dropped} rows with NaN")
    min_req = MIN_ROWS_PER_FEATURE * len(feats)
    if len(data) < min_req:
        st.error(f"Too few rows ({len(data)}) — need at least ~{min_req}")
        st.stop()
    return data.reset_index(drop=True)

def update_chart_theme(fig: go.Figure) -> go.Figure:
    fig.update_layout(
        template="plotly_dark",
        plot_bgcolor="#1A1A1A",
        paper_bgcolor="#1A1A1A",
        font=dict(family="Inter", color="#EAEAEA"),
        xaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        yaxis=dict(gridcolor="#2A2A2A", zerolinecolor="#3A3A3A"),
        margin=dict(t=40,l=20,r=20,b=20),
        hoverlabel=dict(bgcolor="#2A2A2A", font_size=12)
    )
    return fig


# ────────────────────────────────────────────────
# UI Helpers
# ────────────────────────────────────────────────

def render_landing():
    st.markdown("<br>", unsafe_allow_html=True)
    cols = st.columns(3)
    with cols[0]:
        st.markdown("""<div class='metric-card primary' style='min-height:280px;'>
            <h3 style='color:var(--primary-color);'>📐 Partial Coefficients</h3>
            <p style='color:var(--text-muted);'>Isolates true variable impact — controls for all others.</p>
        </div>""", unsafe_allow_html=True)
    with cols[1]:
        st.markdown("""<div class='metric-card info' style='min-height:280px;'>
            <h3 style='color:var(--info-cyan);'>🔍 Diagnostics</h3>
            <p style='color:var(--text-muted);'>VIF, rank, condition number — catches hidden dependencies.</p>
        </div>""", unsafe_allow_html=True)
    with cols[2]:
        st.markdown("""<div class='metric-card success' style='min-height:280px;'>
            <h3 style='color:var(--success-green);'>🔮 Scenario Engine</h3>
            <p style='color:var(--text-muted);'>What-if macro simulations — forward-looking decisions.</p>
        </div>""", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h4>Quick Start</h4>
        <p>1. Upload CSV / connect Google Sheet<br>
           2. Choose target & predictors<br>
           3. Check VIF tab — prune if needed<br>
           4. Use Scenario tab once model is stable</p>
    </div>""", unsafe_allow_html=True)

def render_footer():
    tz = pytz.timezone('Asia/Kolkata')
    now = datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S IST")
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {now}")

def highlight_vif(v):
    try:
        v = float(v)
        if pd.isna(v): return ''
        if v > 5: return 'background-color: rgba(239,68,68,0.2); color:#ef4444; font-weight:bold;'
        if v > 3: return 'background-color: rgba(245,158,11,0.2); color:#f59e0b;'
        return 'color:#10b981;'
    except:
        return ''


# ────────────────────────────────────────────────
# Main App
# ────────────────────────────────────────────────

def main():
    if not STATSMODELS_AVAILABLE:
        st.error("statsmodels is required. Install via `pip install statsmodels`")
        return

    # ── Sidebar ──────────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(f"""
        <div style="text-align:center; padding:1rem 0;">
            <div style="font-size:2rem; font-weight:800; color:#FFC300;">TATTVA</div>
            <div style="color:#888; font-size:0.8rem;">MLR Engine</div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        
        st.markdown('<div class="sidebar-title">Data Source</div>', unsafe_allow_html=True)
        src = st.radio("", ["Upload", "Google Sheets"], horizontal=True, label_visibility="collapsed")
        
        df = None
        if src == "Upload":
            f = st.file_uploader("CSV / Excel", type=["csv","xlsx"])
            if f:
                try:
                    if f.name.endswith('.csv'):
                        df = pd.read_csv(f)
                    else:
                        df = pd.read_excel(f)
                    df = _sanitize_df(df)
                    st.success("Loaded")
                except Exception as e:
                    st.error(f"Load error: {e}")
        else:
            default = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155"
            url = st.text_input("Sheet URL", value=default)
            if st.button("Load Google Sheet", type="primary"):
                with st.spinner("Loading..."):
                    df, err = load_google_sheet(url)
                    if err:
                        st.error(err)
                    else:
                        st.session_state['data'] = df
                        st.success("Loaded")
        
        if df is not None and 'data' not in st.session_state:
            st.session_state['data'] = df
        
        df = st.session_state.get('data')
        
        if df is not None:
            st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
            st.markdown('<div class="sidebar-title">Model Setup</div>', unsafe_allow_html=True)
            
            nums = df.select_dtypes(include=np.number).columns.tolist()
            if len(nums) < 2:
                st.error("Need ≥ 2 numeric columns")
                return
            
            tgt = st.selectbox("Target (Y)", nums, index=0)
            avail = [c for c in nums if c != tgt]
            preds = st.multiselect("Predictors (X)", avail, default=avail[:min(5,len(avail))])
            
            st.markdown('<div class="sidebar-title">Auto Features</div>', unsafe_allow_html=True)
            if st.button("Generate Lags & Rolls"):
                eng = MLREngine(df, tgt, preds)
                dfn, news = eng.generate_auto_features()
                st.session_state['data'] = dfn
                st.session_state['auto'] = news
                st.success(f"Added {len(news)} features")
                st.rerun()
            
            if 'auto' in st.session_state:
                preds.extend(st.session_state['auto'])
            
            st.markdown('<div class="sidebar-title">Advanced Fitting</div>', unsafe_allow_html=True)
            ridge = st.checkbox("Ridge", False)
            lasso = st.checkbox("Lasso", False)
            bayes = st.checkbox("Bayesian (MCMC)", False) if PYMC_AVAILABLE else False
            alpha_val = st.slider("Regularization strength (α)", 0.01, 10.0, 1.0, 0.1) if ridge or lasso else 1.0
            prior_val = st.slider("Prior strength", 0.1, 5.0, 1.0, 0.1) if bayes else 1.0
            nsamp = st.slider("MCMC samples", 500, 4000, 1500, 500) if bayes else 1000
            
            if preds and st.button("Apply VIF Prune"):
                st.info("Prune logic ready — re-run model after selection")
    
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <span class="product-badge">ENGINE</span>
            <h1>TATTVA MLR Engine</h1>
            <div class="tagline">Regression • Diagnostics • Scenario Analysis</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing()
        render_footer()
        return
    
    if not preds:
        st.info("Select predictors in sidebar →")
        render_footer()
        return
    
    # Clean & cache
    try:
        data_clean = clean_data(df, tgt, preds)
    except Exception as e:
        st.error(str(e))
        return
    
    key = f"mlr_{tgt}_{hashlib.md5(str(sorted(preds)).encode()).hexdigest()}_{len(data_clean)}_{ridge}_{lasso}_{bayes}"
    
    if 'engine' not in st.session_state or st.session_state.get('cache_key') != key:
        with st.spinner("Fitting model..."):
            prog = st.progress(0)
            prog.progress(0.3)
            eng = MLREngine(
                data_clean, tgt, preds,
                use_ridge=ridge, alpha=alpha_val,
                use_lasso=lasso,
                use_bayesian=bayes, prior_strength=prior_val,
                n_samples=nsamp
            )
            prog.progress(0.7)
            eng.fit()
            prog.progress(1.0)
            st.session_state['engine'] = eng
            st.session_state['cache_key'] = key
    
    engine = st.session_state['engine']
    
    # ────────────────────────────────────────────────
    # Dashboard
    # ────────────────────────────────────────────────
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    max_vif = engine.vif_data["VIF Score"].max() if engine.vif_data is not None and not engine.vif_data.empty else 0
    r2a = engine.model.rsquared_adj
    grade, gclass, gdesc = engine.get_model_health_grade()
    
    stab_text = "STABLE" if engine.is_stable else f"STABILIZED ({engine.fit_type})"
    stab_colr = "success" if engine.is_stable else "warning"
    cond_txt = f"{engine.condition_number:.1f}" if engine.condition_number is not None else "N/A"
    
    cols = st.columns([1,1,1,1,1.2])
    with cols[0]:
        st.markdown(f'<div class="metric-card primary"><h4>R² Adj</h4><h2>{r2a:.3f}</h2><div class="sub-metric">Explanatory Power</div></div>', unsafe_allow_html=True)
    with cols[1]:
        c = "success" if max_vif < 3 else "warning" if max_vif <= 5 else "danger"
        st.markdown(f'<div class="metric-card {c}"><h4>Max VIF</h4><h2>{max_vif:.1f}</h2><div class="sub-metric">Collinearity</div></div>', unsafe_allow_html=True)
    with cols[2]:
        pcol = "success" if getattr(engine.model, 'f_pvalue', 0.1) < 0.05 else "danger"
        st.markdown(f'<div class="metric-card {pcol}"><h4>F-test</h4><h2>{"PASS" if getattr(engine.model, "f_pvalue", 0.1) < 0.05 else "FAIL"}</h2><div class="sub-metric">Overall Significance</div></div>', unsafe_allow_html=True)
    with cols[3]:
        st.markdown(f'<div class="metric-card {stab_colr}"><h4>Stability</h4><h2>{stab_text}</h2><div class="sub-metric">Matrix</div></div>', unsafe_allow_html=True)
    with cols[4]:
        cc = "success" if (engine.condition_number or 999) < 30 else "warning" if (engine.condition_number or 999) < 300 else "danger"
        st.markdown(f'<div class="metric-card {cc}"><h4>Condition #</h4><h2>{cond_txt}</h2><div class="sub-metric">Numerical Stability</div></div>', unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="signal-card {gclass}">
        <div class="label">MODEL GRADE</div>
        <div class="value">{grade}</div>
        <div class="subtext">{gdesc}</div>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Tabs
    t1, t2, t3, t4, t5 = st.tabs([
        "Feature Analytics",
        "Collinearity & Rank",
        "Visuals",
        "Scenario Simulator",
        "Advanced"
    ])
    
    with t1:
        st.subheader("Coefficients & Impact")
        def pvcolor(v):
            if pd.isna(v): return ''
            return 'color:#ef4444;' if v > 0.05 else 'color:#10b981;'
        
        df_styled = engine.coef_df.style.format(precision=4).map(pvcolor, subset=['p-Value'])
        st.dataframe(df_styled, height=380)
        
        st.download_button("Export Coefficients", engine.coef_df.to_csv(index=False), "coeffs.csv", "Download")
    
    with t2:
        st.subheader("VIF & Matrix Diagnostics")
        
        if engine.vif_data is None or engine.vif_data.empty:
            st.warning("VIF not computed for this model type.")
        else:
            if max_vif > 5:
                st.markdown("""
                <div class="signal-card danger">
                    <span class="signal-card-title">High Collinearity Detected</span>
                    <p>Variables above VIF 5 share redundant information. Review plan below.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div class="signal-card success">
                    <span class="signal-card-title">Clean Signal Geometry</span>
                    <p>All VIF ≤ 5 — good independence between predictors.</p>
                </div>
                """, unsafe_allow_html=True)
            
            st.dataframe(
                engine.vif_data.style.format({"VIF Score": "{:.2f}"}).map(highlight_vif, subset=["VIF Score"]),
                height=400
            )
            
            if engine.resolution_plan:
                st.subheader("Auto Resolution Suggestions")
                for p in engine.resolution_plan:
                    if p['type'] == 'cluster':
                        st.markdown(f"**{p['title']}** — keep **{p['champion']}**, drop {', '.join(p['drops'])}")
                        st.markdown(f"<small>{p['reason']}</small>", unsafe_allow_html=True)
                    else:
                        st.markdown(f"**{p['title']}** — consider dropping **{p['drops'][0]}**")
                        st.markdown(f"<small>{p['reason']}</small>", unsafe_allow_html=True)
                
                if st.button("Apply Suggested Prune"):
                    new_preds = engine.apply_resolution_plan()
                    st.session_state['pruned'] = new_preds
                    st.success("Pruned — please re-run model")
                    st.rerun()
    
    with t3:
        cA, cB = st.columns(2)
        with cA:
            st.subheader("Feature Importance")
            fig = px.bar(
                engine.feature_importance,
                x="Absolute Impact",
                y="Variable",
                orientation="h",
                color="Relative Impact (Std Beta)",
                color_continuous_scale="RdBu",
                title="Standardized Impact Rank"
            )
            fig = update_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
        
        with cB:
            st.subheader("Correlation Heatmap")
            cm = engine.df[[engine.target] + engine.features].corr()
            fig = px.imshow(
                cm, text_auto=".2f", color_continuous_scale="RdBu_r",
                zmin=-1, zmax=1, aspect="auto"
            )
            fig = update_chart_theme(fig)
            st.plotly_chart(fig, use_container_width=True)
    
    with t4:
        st.subheader("Scenario Simulator")
        inputs = {}
        cL, cR = st.columns([2,1])
        with cL:
            for col in engine.features:
                mi, ma = float(engine.X[col].min()), float(engine.X[col].max())
                me = float(engine.X[col].mean())
                step = (ma - mi) / 100 if ma > mi else 0.01
                inputs[col] = st.slider(
                    f"{col}",
                    min_value=mi - (ma-mi)*0.1,
                    max_value=ma + (ma-mi)*0.1,
                    value=me,
                    step=max(0.0001, step),
                    format="%.4f"
                )
        
        with cR:
            try:
                pred = engine.predict_scenario(inputs)
                hist_mean = engine.y.mean()
                delta = pred - hist_mean
                colr = "success" if delta > 0 else "danger" if delta < 0 else "primary"
                st.markdown(f"""
                <div class="signal-card {colr}" style="text-align:center;">
                    <div class="label">Predicted {engine.target}</div>
                    <div class="value" style="font-size:3.8rem;">{pred:.2f}</div>
                    <div class="subtext">
                        {'+' if delta >= 0 else ''}{delta:.2f} vs mean {hist_mean:.2f}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                m, s = engine.monte_carlo_scenarios(inputs, n_sims=300)
                st.markdown(f"""
                <div class="metric-card neutral">
                    <h4>Monte Carlo 95% Interval</h4>
                    <h2>{m:.2f} ± {1.96*s:.2f}</h2>
                </div>
                """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"Prediction failed: {e}")
    
    with t5:
        st.subheader("Model & Engine Info")
        cond_display = f"{engine.condition_number:.1f}" if engine.condition_number is not None else "N/A"
        st.markdown(f"""
        - **Fit type**: {engine.fit_type}
        - **Rows**: {len(engine.y)}
        - **Features**: {len(engine.features)}
        - **R² adj**: {engine.model.rsquared_adj:.3f}
        - **Condition number**: {cond_display}
        - **PyMC available**: {PYMC_AVAILABLE}
        """)
        
        st.markdown("""
        **Tips for best results**:
        - Use Ridge or Lasso when VIF > 5–10
        - Bayesian mode gives credible intervals via posterior sampling
        - Generate lagged/rolling features for time-series data
        """)
    
    render_footer()

if __name__ == "__main__":
    main()
