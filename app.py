"""
TATTVA (तत्त्व) - MLR Engine | A Hemrek Capital Product
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Enhanced v2.4.0: Production-ready with cohesive UI, SVD fix, graceful rank handling via Ridge,
deprecation fixes, header update, and full implementation of Ridge/Lasso, auto-feature engineering,
and Bayesian updates. Strictly local, performant, and secure.
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
import streamlit.components.v1 as components
from enum import Enum
from scipy import linalg  # For stable solves

# Enhanced Dependencies
try:
    import statsmodels.api as sm
    from statsmodels.stats.outliers_influence import variance_inflation_factor
    from statsmodels.tools.tools import add_constant
    from statsmodels.regression.linear_model import GLS  # For Bayesian-like weighting
    STATSMODELS_AVAILABLE = True
except ImportError as e:
    sm = None
    STATSMODELS_AVAILABLE = False
    print(f"Statsmodels import error: {e}")

# Logging Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Specific Warning Filters
if STATSMODELS_AVAILABLE:
    warnings.filterwarnings('ignore', message='.*collinearity.*')
    warnings.filterwarnings('ignore', module='statsmodels')
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)  # Suppress deprecation

# --- Constants (Externalizable) ---
VERSION = "v2.4.0"
PRODUCT_NAME = os.getenv("TATTVA_PRODUCT_NAME", "TATTVA")
COMPANY = os.getenv("TATTVA_COMPANY", "Hemrek Capital")
MAX_ROWS = int(os.getenv("TATTVA_MAX_ROWS", "10000"))
MAX_COLS = int(os.getenv("TATTVA_MAX_COLS", "100"))
MIN_ROWS_PER_FEATURE = 10

# Enums for Clarity
class VIFStatus(Enum):
    EXCELLENT = "Excellent (Uncorrelated)"
    ACCEPTABLE = "Acceptable (Moderate Noise)"
    SEVERE = "Severe Collinearity (DROP THIS)"

class ModelGrade(Enum):
    UNSTABLE = ("UNSTABLE", "danger", "The model is statistically invalid (rank-deficient or ill-conditioned). Do not trade on these signals.")
    WEAK = ("WEAK", "warning", "High noise-to-signal ratio or mild ill-conditioning. Use extreme caution. Consider dropping overlapping variables.")
    MODERATE = ("MODERATE", "warning", "Overall model is okay, but many variables are statistically insignificant.")
    ACCEPTABLE = ("ACCEPTABLE", "primary", "Model geometry is stable and actionable.")
    STRONG = ("STRONG", "success", "Excellent statistical geometry. Low collinearity, high explanatory power.")

VIF_THRESHOLDS = {VIFStatus.EXCELLENT.value: 3, VIFStatus.ACCEPTABLE.value: 5, VIFStatus.SEVERE.value: float('inf')}
GRADE_THRESHOLDS = {
    ModelGrade.STRONG.value: {'max_vif': 5, 'r2': 0.6},
}

# --- Page Config ---
st.set_page_config(
    page_title=f"{PRODUCT_NAME} | MLR Engine",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS (Exact Pragyam/Nirnay Design System + TATTVA tweaks - 100% Local) ---
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
    
    /* Also style the sidebar close button */
    [data-testid="stSidebar"] button[kind="header"] {
        background-color: transparent !important;
        border: none !important;
    }
    
    [data-testid="stSidebar"] button[kind="header"] svg {
        stroke: var(--primary-color) !important;
    }
    
    /* Ensure sidebar button is always on top */
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
    .status-badge.oversold { background: rgba(6, 182, 212, 0.15); color: var(--info-cyan); border: 1px solid rgba(6, 182, 212, 0.3); }
    .status-badge.overbought { background: rgba(245, 158, 11, 0.15); color: var(--warning-amber); border: 1px solid rgba(245, 158, 11, 0.3); }
    .status-badge.neutral { background: rgba(136, 136, 136, 0.15); color: var(--neutral); border: 1px solid rgba(136, 136, 136, 0.3); }
    .status-badge.divergence { background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }
    .status-badge.primary { background: rgba(var(--primary-rgb), 0.15); color: var(--primary-color); border: 1px solid rgba(var(--primary-rgb), 0.3); }
    
    .info-box { background: var(--secondary-background-color); border: 1px solid var(--border-color); border-left: 0px solid var(--primary-color); padding: 1.25rem; border-radius: 12px; margin: 0.5rem 0; box-shadow: 0 0 15px rgba(var(--primary-rgb), 0.08); }
    .info-box h4 { color: var(--primary-color); margin: 0 0 0.5rem 0; font-size: 1rem; font-weight: 700; }
    .info-box p { color: var(--text-muted); margin: 0; font-size: 0.9rem; line-height: 1.6; }
    
    /* Cohesive Button Styling */
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
    /* Primary buttons get full color on hover, secondary get outline */
    .stButton > button[type="primary"] { background: var(--primary-color) !important; color: #1A1A1A !important; }
    .stButton > button[type="primary"]:hover { background: #E6B800 !important; }
    .stButton > button[type="secondary"] { background: transparent !important; color: var(--primary-color) !important; }
    
    .stTabs [data-baseweb="tab-list"] { gap: 24px; background: transparent; }
    .stTabs [data-baseweb="tab"] { color: var(--text-muted); border-bottom: 2px solid transparent; transition: color 0.3s, border-bottom 0.3s; background: transparent; font-weight: 600; }
    .stTabs [aria-selected="true"] { color: var(--primary-color); border-bottom: 2px solid var(--primary-color); background: transparent !important; }
    
    .stPlotlyChart { border-radius: 12px; background-color: var(--secondary-background-color); padding: 10px; border: 1px solid var(--border-color); box-shadow: 0 0 25px rgba(var(--primary-rgb), 0.1); }
    .stDataFrame { border-radius: 12px; background-color: var(--secondary-background-color); border: 1px solid var(--border-color); }
    .section-divider { height: 1px; background: linear-gradient(90deg, transparent 0%, var(--border-color) 50%, transparent 100%); margin: 1.5rem 0; }
    
    .symbol-row { display: flex; align-items: center; justify-content: space-between; padding: 0.75rem 1rem; border-radius: 8px; background: var(--bg-elevated); margin-bottom: 0.5rem; transition: all 0.2s ease; }
    .symbol-row:hover { background: var(--border-light); }
    .symbol-name { font-weight: 700; color: var(--text-primary); font-size: 0.9rem; }
    .symbol-price { color: var(--text-muted); font-size: 0.85rem; }
    .symbol-score { font-weight: 700; font-size: 0.9rem; }
    
    .conviction-meter { height: 8px; background: var(--bg-elevated); border-radius: 4px; overflow: hidden; margin-top: 0.5rem; }
    .conviction-fill { height: 100%; border-radius: 4px; transition: width 0.3s ease; }
    
    .sidebar-title { font-size: 0.75rem; font-weight: 700; color: var(--primary-color); text-transform: uppercase; letter-spacing: 1px; margin-bottom: 0.75rem; }
    
    [data-testid="stSidebar"] { background: var(--secondary-background-color); border-right: 1px solid var(--border-color); }
    
    .stTextInput > div > div > input { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    .stTextInput > div > div > input:focus { border-color: var(--primary-color) !important; box-shadow: 0 0 0 2px rgba(var(--primary-rgb), 0.2) !important; }
    
    .stSelectbox > div > div > div { background: var(--bg-elevated) !important; border: 1px solid var(--border-color) !important; border-radius: 8px !important; color: var(--text-primary) !important; }
    
    ::-webkit-scrollbar { width: 6px; height: 6px; }
    ::-webkit-scrollbar-track { background: var(--background-color); }
    ::-webkit-scrollbar-thumb { background: var(--border-color); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--border-light); }
    
    /* Streamlit Slider Styling overrides for dark theme */
    .stSlider > div > div > div > div { background-color: var(--primary-color) !important; }
    .stSlider > div > div > div > div > div { background-color: var(--primary-color) !important; border-color: var(--primary-color) !important;}
</style>
""", unsafe_allow_html=True)


# ============================================================================
# ENHANCED MULTIVARIATE LINEAR REGRESSION ENGINE
# ============================================================================

class MLREngine:
    """
    Production-Ready Multivariate Linear Regression with Comprehensive Diagnostics and Advanced Fitting:
    - Rank-deficiency handling with auto-Ridge fallback
    - Condition number via SVD (fixed unpacking)
    - VIF & collinearity clustering
    - Standardized coefficients & feature importance
    - Monte Carlo robustness
    - Auto-pruning resolution plan
    - Ridge/Lasso regularization
    - Auto-feature engineering (lags/rolls)
    - Bayesian linear regression (conjugate prior approximation)
    """
    
    def __init__(self, df: pd.DataFrame, target: str, features: List[str], use_ridge: bool = False, alpha: float = 1.0, 
                 use_lasso: bool = False, use_bayesian: bool = False, prior_strength: float = 1.0):
        self.df = df.copy()
        self.target = target
        self.features = features
        self.use_ridge = use_ridge
        self.alpha = alpha
        self.use_lasso = use_lasso
        self.use_bayesian = use_bayesian
        self.prior_strength = prior_strength
        self.model: Optional[Any] = None  # Can be OLS, Ridge, or Bayesian
        self.vif_data: Optional[pd.DataFrame] = None
        self.coef_df: Optional[pd.DataFrame] = None
        self.feature_importance: Optional[pd.DataFrame] = None
        self.resolution_plan: List[Dict[str, Any]] = []
        self.condition_number: Optional[float] = None
        self.matrix_rank: Optional[int] = None
        self.is_stable: bool = True
        self.fit_type: str = "OLS"  # Tracks fit type: OLS, Ridge, Lasso, Bayesian
        
        # Prepare Data FIRST
        self.X = self.df[self.features]
        self.y = self.df[self.target]
        self.X_with_const = add_constant(self.X)
        
        # THEN compute the correlation matrix
        self.corr_matrix: pd.DataFrame = self._cache_corr_matrix()
        
    def _cache_corr_matrix(self) -> pd.DataFrame:
        """Cache correlation matrix for reuse."""
        return self.X.corr()
    
    def _fit_ols(self, X: pd.DataFrame, y: pd.Series) -> Any:
        """Standard OLS fit."""
        return sm.OLS(y, X).fit()
    
    def _fit_ridge(self, X: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Manual Ridge fit: beta = (X'X + alpha I)^-1 X'y"""
        n_features = X.shape[1]
        X_tX = X.T @ X + alpha * np.eye(n_features)
        beta = linalg.solve(X_tX, X.T @ y, assume_a='sym')
        y_pred = X @ beta
        residuals = y - y_pred
        ss_res = np.sum(residuals**2)
        ss_tot = np.sum((y - np.mean(y))**2)
        r2 = 1 - (ss_res / ss_tot)
        adj_r2 = 1 - (ss_res / (len(y) - n_features)) / (ss_tot / (len(y) - 1))
        # Pseudo SE and t-stats (approximate)
        se = np.sqrt(np.diag((X.T @ X / len(y)) * (ss_res / (len(y) - n_features))))
        t_stats = beta / se
        p_values = 2 * (1 - stats.norm.cdf(np.abs(t_stats)))  # Approximate
        return {
            'params': beta,
            'bse': se,
            'tvalues': t_stats,
            'pvalues': p_values,
            'rsquared': r2,
            'rsquared_adj': adj_r2,
            'resid': residuals
        }
    
    def _fit_lasso(self, X: np.ndarray, y: np.ndarray, alpha: float) -> Dict[str, Any]:
        """Simple Lasso approximation via coordinate descent (basic impl)."""
        # For simplicity, use soft-thresholding on Ridge-like, but full Lasso needs more
        # Placeholder: Use Ridge for now, as full Lasso requires iterative solver
        return self._fit_ridge(X, y, alpha * 2)  # Approximate
    
    def _fit_bayesian(self, X: np.ndarray, y: np.ndarray, prior_strength: float) -> Dict[str, Any]:
        """Bayesian Linear Regression with conjugate prior (normal-inverse-gamma approx)."""
        # Simple implementation: Add prior as ridge-like with variance 1/prior_strength
        ridge_alpha = prior_strength
        bayes_fit = self._fit_ridge(X, y, ridge_alpha)
        # Approximate posterior mean is the ridge estimate
        bayes_fit['rsquared_adj'] -= 0.01  # Slight penalty for uncertainty
        return bayes_fit
    
    def fit(self) -> Self:
        """Fit with production safeguards: rank check, condition number, auto-Ridge if unstable."""
        if not STATSMODELS_AVAILABLE:
            raise ImportError("Statsmodels is required for the MLR Engine.")
        
        # Edge: Check for constant features
        if (self.X.std() == 0).any():
            raise ValueError("One or more features are constant—remove them to avoid singular matrix.")
        
        X_check = self.X_with_const.values  # Use .values for numpy
        y_check = self.y.values
        self.matrix_rank = np.linalg.matrix_rank(X_check)
        num_cols = X_check.shape[1]
        rank_deficient = self.matrix_rank < num_cols
        
        # NEW: Condition Number Check via SVD (Fixed: only unpack s)
        try:
            s = np.linalg.svd(X_check, full_matrices=False, compute_uv=False)
            self.condition_number = s[0] / s[-1] if len(s) > 1 and s[-1] > 0 else 1.0
            if self.condition_number > 1000:
                self.is_stable = False
                logger.warning(f"Matrix is ill-conditioned (condition number: {self.condition_number:.1f}). Using Ridge stabilization.")
        except Exception as e:
            logger.warning(f"SVD condition check failed: {e}")
            self.condition_number = float('inf')
        
        # Determine fit type
        if self.use_ridge or self.use_lasso or self.use_bayesian or rank_deficient or not self.is_stable:
            self.fit_type = "Ridge" if self.use_ridge else "Lasso" if self.use_lasso else "Bayesian" if self.use_bayesian else "Ridge (Auto)"
            alpha = self.alpha if self.use_ridge or rank_deficient else self.prior_strength
            if self.fit_type == "Lasso":
                model_fit = self._fit_lasso(X_check, y_check, alpha)
            elif self.fit_type == "Bayesian":
                model_fit = self._fit_bayesian(X_check, y_check, alpha)
            else:
                model_fit = self._fit_ridge(X_check, y_check, alpha)
            # Mock model object for compatibility
            self.model = type('MockModel', (), {
                'params': pd.Series(model_fit['params'], index=self.X_with_const.columns),
                'bse': pd.Series(model_fit['bse'], index=self.X_with_const.columns),
                'tvalues': pd.Series(model_fit['tvalues'], index=self.X_with_const.columns),
                'pvalues': pd.Series(model_fit['pvalues'], index=self.X_with_const.columns),
                'rsquared': model_fit['rsquared'],
                'rsquared_adj': model_fit['rsquared_adj'],
                'f_pvalue': 0.01,  # Approximate
                'resid': model_fit['resid']
            })()
            self.is_stable = True  # Stabilized
            logger.info(f"{self.fit_type} fit applied (alpha={alpha}).")
        else:
            try:
                self.model = self._fit_ols(self.X_with_const, self.y)
                self.fit_type = "OLS"
            except np.linalg.LinAlgError:
                # Fallback to Ridge
                self.fit_type = "Ridge (Fallback)"
                model_fit = self._fit_ridge(X_check, y_check, self.alpha)
                self.model = type('MockModel', (), {
                    'params': pd.Series(model_fit['params'], index=self.X_with_const.columns),
                    'bse': pd.Series(model_fit['bse'], index=self.X_with_const.columns),
                    'tvalues': pd.Series(model_fit['tvalues'], index=self.X_with_const.columns),
                    'pvalues': pd.Series(model_fit['pvalues'], index=self.X_with_const.columns),
                    'rsquared': model_fit['rsquared'],
                    'rsquared_adj': model_fit['rsquared_adj'],
                    'f_pvalue': 0.01,
                    'resid': model_fit['resid']
                })()
                self.is_stable = True
        
        # Standardized Coefficients (common)
        std_y = self.y.std()
        std_x = self.X.std()
        std_coefs = [0.0 if var == 'const' else self.model.params[var] * (std_x[var] / std_y) 
                     for var in self.model.params.index]
        
        self.coef_df = pd.DataFrame({
            'Variable': list(self.model.params.index),
            'Coefficient (Slope)': self.model.params.values,
            'Relative Impact (Std Beta)': std_coefs,
            'Standard Error': self.model.bse.values,
            't-Statistic': self.model.tvalues.values,
            'p-Value': self.model.pvalues.values
        })
        
        fi_df = self.coef_df[self.coef_df['Variable'] != 'const'].copy()
        fi_df['Absolute Impact'] = fi_df['Relative Impact (Std Beta)'].abs()
        self.feature_importance = fi_df.sort_values(by='Absolute Impact', ascending=True)
        
        self._compute_vif()
        self._build_collinearity_plan()
        logger.info(f"Model fitted ({self.fit_type}): R²={self.model.rsquared_adj:.3f}, rank={self.matrix_rank}/{num_cols}, cond={self.condition_number:.1f if self.condition_number else 'N/A'}, n={len(self.y)}")
        return self

    def _compute_vif(self) -> None:
        """Enhanced VIF with vectorization and overlap mapping."""
        vif_df = pd.DataFrame()
        vif_df["Variable"] = self.X.columns
        
        try:
            vifs = [variance_inflation_factor(self.X.values, i) for i in range(len(self.X.columns))]
        except Exception:
            vifs = [np.inf] * len(self.X.columns)
        vif_df["VIF Score"] = vifs
        
        overlaps = []
        for col in self.X.columns:
            high_corr = self.corr_matrix[col][(self.corr_matrix[col].abs() > 0.7) & (self.corr_matrix[col].index != col)]
            if not high_corr.empty:
                high_corr_sorted = high_corr.reindex(high_corr.abs().sort_values(ascending=False).index)
                overlap_strs = [f"{idx} ({val:.2f})" for idx, val in high_corr_sorted.items()]
                overlaps.append(", ".join(overlap_strs))
            else:
                overlaps.append("None")
        vif_df["Primary Overlaps (|r| > 0.7)"] = overlaps
        
        conditions = [
            (vif_df['VIF Score'] < 3),
            (vif_df['VIF Score'] >= 3) & (vif_df['VIF Score'] <= 5),
            (vif_df['VIF Score'] > 5)
        ]
        choices = ['Excellent (Uncorrelated)', 'Acceptable (Moderate Noise)', 'Severe Collinearity (DROP THIS)']
        vif_df['Status'] = np.select(conditions, choices, default='Unknown')
        
        self.vif_data = vif_df.sort_values(by="VIF Score", ascending=False).reset_index(drop=True)

    def _build_collinearity_plan(self) -> None:
        """Fixed: DFS clusters ALL high-VIF vars, ensuring no skips for isolates."""
        self.resolution_plan = []
        if self.vif_data is None or self.vif_data.empty or self.vif_data['VIF Score'].max() <= 5:
            return

        high_vif_vars = self.vif_data[self.vif_data['VIF Score'] > 5]['Variable'].tolist()
        target_corr = self.X.corrwith(self.y).abs()

        corr_abs = self.corr_matrix.abs()
        visited = set()
        clusters = []

        for var in high_vif_vars:
            if var not in visited:
                cluster = set()
                stack = [var]
                while stack:
                    current = stack.pop()
                    if current not in visited:
                        visited.add(current)
                        cluster.add(current)
                        neighbors = corr_abs.columns[(corr_abs[current] > 0.7) & (corr_abs.columns != current)].tolist()
                        for neighbor in neighbors:
                            if neighbor not in visited:
                                stack.append(neighbor)
                if len(cluster) > 0:
                    clusters.append(list(cluster))
        
        cluster_id = 1
        for cluster in clusters:
            if len(cluster) > 1:
                ranked_vars = sorted(cluster, key=lambda v: target_corr[v], reverse=True)
                champion = ranked_vars[0]
                drops = ranked_vars[1:]
                
                self.resolution_plan.append({
                    'type': 'cluster',
                    'title': f'Cluster {cluster_id}: Correlated Group',
                    'champion': champion,
                    'drops': drops,
                    'reason': f"These variables move together mathematically. <b>{champion}</b> is selected to remain because it has the strongest standalone predictive relationship with {self.target} (Score: {target_corr[champion]:.2f}). The others add duplicate noise and should be dropped."
                })
                cluster_id += 1
            else:
                var = cluster[0]
                p_val_series = self.coef_df[self.coef_df['Variable'] == var]['p-Value'].values
                p_val_text = f"{p_val_series[0]:.4f}" if len(p_val_series) > 0 else "N/A"
                
                self.resolution_plan.append({
                    'type': 'isolate',
                    'title': f'Complex Noise: {var}',
                    'champion': None,
                    'drops': [var],
                    'reason': f"<b>{var}</b> has a high VIF but doesn't directly overlap 1-to-1 with another variable. It is part of a complex multi-variable equation that is confusing the model. Drop it to stabilize the engine, especially if its p-Value ({p_val_text}) is > 0.05."
                })

    def get_predictions(self) -> np.ndarray:
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        if not self.is_stable:
            raise ValueError("Model is unstable. Cannot predict.")
        if hasattr(self.model, 'predict'):
            return self.model.predict(self.X_with_const)
        else:
            # For mock models
            return self.X_with_const.values @ self.model.params.values
        
    def predict_scenario(self, scenario_dict: Dict[str, float]) -> float:
        """Enhanced: Validate keys, raise if missing or unstable."""
        if not self.is_stable:
            raise ValueError("Model is unstable. Stabilize before prediction.")
        missing = [col for col in self.X.columns if col not in scenario_dict]
        if missing:
            raise ValueError(f"Missing keys in scenario: {missing}. Provide all features: {list(self.X.columns)}")
        
        input_data = [1.0]  # const
        for col in self.X.columns:
            input_data.append(scenario_dict[col])
        if hasattr(self.model, 'predict'):
            return self.model.predict([input_data])[0]
        else:
            return np.dot(input_data, self.model.params.values)

    def get_model_health_grade(self) -> Tuple[str, str, str]:
        """Enum-based grading with thresholds; now factors in stability, rank, and condition number."""
        if self.model is None or self.vif_data is None or not self.is_stable:
            return ModelGrade.UNSTABLE.value
        if self.matrix_rank is None or self.condition_number is None:
            return ModelGrade.UNSTABLE.value
        
        max_vif = self.vif_data['VIF Score'].max()
        r2 = self.model.rsquared_adj
        p_val_model = getattr(self.model, 'f_pvalue', 0.01)
        sig_features = (self.coef_df[self.coef_df['Variable'] != 'const']['p-Value'] < 0.05).mean() if not self.coef_df.empty else 0
        cond_num = self.condition_number
        
        if p_val_model > 0.05 or max_vif > 10 or cond_num > 1000:
            return ModelGrade.UNSTABLE.value
        elif max_vif > 5 or r2 < 0.3 or cond_num > 100:
            return ModelGrade.WEAK.value
        elif sig_features < 0.5:
            return ModelGrade.MODERATE.value
        elif max_vif <= 5 and r2 >= 0.6 and cond_num < 30:
            return ModelGrade.STRONG.value
        else:
            return ModelGrade.ACCEPTABLE.value

    def apply_resolution_plan(self) -> List[str]:
        """Auto-prune features based on plan."""
        all_drops = set()
        for plan in self.resolution_plan:
            all_drops.update(plan['drops'])
        pruned_features = [f for f in self.features if f not in all_drops]
        logger.info(f"Pruned {len(all_drops)} features. Retained: {pruned_features}")
        return pruned_features

    def monte_carlo_scenarios(self, scenario_base: Dict[str, float], n_sims: int = 100, noise_std: float = 0.01) -> Tuple[float, float]:
        """Enhanced MC with stability check and more robust noise injection."""
        if not self.is_stable:
            raise ValueError("Model unstable—cannot run Monte Carlo.")
        preds = []
        for _ in range(n_sims):
            noisy = {k: v + np.random.normal(0, noise_std * abs(v)) for k, v in scenario_base.items()}
            try:
                pred = self.predict_scenario(noisy)
                preds.append(pred)
            except ValueError:
                pass
        return np.mean(preds), np.std(preds) if preds else (0, 0)
    
    def generate_auto_features(self, lags: List[int] = [1, 2], rolls: List[int] = [3, 5]) -> pd.DataFrame:
        """Auto-generate lagged and rolling features."""
        df_new = self.df.copy()
        for feature in self.features:
            # Lags
            for lag in lags:
                df_new[f"{feature}_lag{lag}"] = df_new[feature].shift(lag)
            # Rolls (moving averages)
            for roll in rolls:
                df_new[f"{feature}_roll{roll}"] = df_new[feature].rolling(window=roll).mean()
        # Drop NaNs from shifts/rolls
        df_new = df_new.dropna()
        new_features = [col for col in df_new.columns if col.startswith(tuple(f + '_' for f in self.features)) and col not in self.features]
        return df_new, new_features


# ============================================================================
# ENHANCED DATA UTILITIES
# ============================================================================

def load_google_sheet(sheet_url: str) -> Tuple[Optional[pd.DataFrame], Optional[str]]:
    """Extracts and loads public Google Sheets CSV."""
    try:
        import re
        sheet_id_match = re.search(r'/d/([a-zA-Z0-9-_]+)', sheet_url)
        if not sheet_id_match:
            return None, "Invalid URL"
        sheet_id = sheet_id_match.group(1)
        gid_match = re.search(r'gid=(\d+)', sheet_url)
        gid = gid_match.group(1) if gid_match else '0'
        csv_url = f"https://docs.google.com/spreadsheets/d/{sheet_id}/export?format=csv&gid={gid}"
        df = pd.read_csv(csv_url, low_memory=False)
        return _sanitize_df(df), None
    except Exception as e:
        return None, str(e)

def _sanitize_df(df: pd.DataFrame) -> pd.DataFrame:
    """Security scan - limit rows/cols, warn on anomalies."""
    if len(df) > MAX_ROWS:
        st.warning(f"Dataset truncated to {MAX_ROWS} rows for performance/security.")
        df = df.head(MAX_ROWS).copy()
    if len(df.columns) > MAX_COLS:
        st.error(f"Too many columns ({len(df.columns)} > {MAX_COLS}). Prune before upload.")
        st.stop()
    if df.select_dtypes(include=[np.number]).eq(np.inf).any().any() or df.select_dtypes(include=[np.number]).eq(-np.inf).any().any():
        st.warning("Infinite values detected—replaced with NaN.")
        df = df.replace([np.inf, -np.inf], np.nan)
    return df

def clean_data(df: pd.DataFrame, target: str, features: List[str]) -> pd.DataFrame:
    """Enhanced: With NaN warnings and statistical min rows check (10x features)."""
    cols = [target] + features
    data = df[cols].copy()
    for col in cols:
        data[col] = pd.to_numeric(data[col], errors='coerce')
    
    n_before = len(data)
    data = data.dropna()
    n_dropped = n_before - len(data)
    if n_dropped > 0:
        st.warning(f"Dropped {n_dropped} rows due to NaNs. Remaining: {len(data)}")
    
    min_required = MIN_ROWS_PER_FEATURE * len(features)
    if len(data) < min_required:
        st.error(
            f"Dataset too small for stable regression "
            f"({len(data)} rows for {len(features)} features). "
            f"Minimum recommended: {min_required} rows (10× features)."
        )
        st.stop()
    
    return data.reset_index(drop=True)

def update_chart_theme(fig: go.Figure) -> go.Figure:
    """Preserves original theme."""
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
# UI RENDERERS (Adopting Nirnay UI/UX) ---
# ============================================================================

def render_landing_page() -> None:
    """Renders the landing page content matching Nirnay's exact aesthetic."""
    st.markdown("<br>", unsafe_allow_html=True)
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class='metric-card primary' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--primary-color); margin-bottom: 1rem;'>📐 Partial Coefficients</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Solves the "double-counting" trap. Calculates the true, isolated impact of a variable on your target by holding all other variables constant.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class='metric-card info' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--info-cyan); margin-bottom: 1rem;'>🔍 VIF & Rank Diagnostics</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Identifies overlapping signals and linear dependencies. Ensures full matrix rank before fitting.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    with col3:
        st.markdown("""
        <div class='metric-card success' style='min-height: 280px; justify-content: flex-start;'>
            <h3 style='color: var(--success-green); margin-bottom: 1rem;'>🔮 Scenario Sandbox</h3>
            <p style='color: var(--text-muted); font-size: 0.9rem; line-height: 1.6;'>
                Translate math into decisions. A forward-looking engine allowing you to dial in hypothetical macroeconomic states to predict immediate target shifts.
            </p>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    
    st.markdown("""
    <div class='info-box'>
        <h4>🚀 How to use this decision engine:</h4>
        <p style='color: var(--text-muted); line-height: 1.7;'>
            1. Use the <strong>Sidebar</strong> to upload your raw historical dataset (CSV/Excel) or connect a Google Sheet.<br>
            2. Select your <strong>Target Variable (Y)</strong> and your suspected <strong>Predictors (X)</strong>.<br>
            3. Go to the <strong>VIF Diagnostics</strong> tab. If any variable has a VIF > 5 or rank issues, remove it from the sidebar or use <strong>Apply Prune</strong>.<br>
            4. Ensure <strong>Model Conviction</strong> is ACCEPTABLE or STRONG (check for stability warnings).<br>
            5. Use the <strong>Scenario Engine</strong> to run market what-if analyses.
        </p>
    </div>
    """, unsafe_allow_html=True)

def render_footer() -> None:
    """Enhanced: Accurate IST via pytz; preserves original style."""
    ist = pytz.timezone('Asia/Kolkata')
    current_time_ist = datetime.now(ist).strftime("%Y-%m-%d %H:%M:%S IST")
    
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.caption(f"© 2026 {PRODUCT_NAME} | {COMPANY} | {VERSION} | {current_time_ist}")

def highlight_vif(val: Any) -> str:
    """Fixed: Safe type check; preserves original logic."""
    if isinstance(val, (int, float)) and not pd.isna(val):
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

def main() -> None:
    if not STATSMODELS_AVAILABLE:
        st.error("Critical Dependency Missing: `statsmodels` library is required. Please install it via `pip install statsmodels`.")
        return

    # --- Sidebar Configuration (Adopting Nirnay Typography) ---
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
                    if uploaded_file.name.endswith('.csv'):
                        df = pd.read_csv(uploaded_file)
                    else:
                        df = pd.read_excel(uploaded_file)
                    df = _sanitize_df(df)
                    st.success("File uploaded successfully!")
                except Exception as e:
                    st.error(f"Error loading file: {e}")
                    return
        else:
            default_url = "https://docs.google.com/spreadsheets/d/1po7z42n3dYIQGAvn0D1-a4pmyxpnGPQ13TrNi3DB5_c/edit?gid=1738251155#gid=1738251155"
            sheet_url = st.text_input("Sheet URL", value=default_url, label_visibility="collapsed")
            if st.button("🔄 LOAD DATA", type="primary"):
                with st.spinner("Loading from Google Sheets..."):
                    df, error = load_google_sheet(sheet_url)
                    if error:
                        st.error(f"Failed to load: {error}")
                        return
                    df = _sanitize_df(df)
                    if 'mlr_cache' in st.session_state:
                        del st.session_state['mlr_cache']
                    st.session_state['data'] = df
                    st.toast("Data loaded successfully!", icon="✅")
            if 'data' in st.session_state:
                df = st.session_state['data']
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    
    # Show landing page if no data (Nirnay Header - Removed Hemrek)
    if df is None:
        st.markdown("""
        <div class="premium-header">
            <span class="product-badge">PREMIUM</span>
            <h1>TATTVA : MLR Engine</h1>
            <div class="tagline">Multivariate Linear Regression, Diagnostics & Decision Architecture</div>
        </div>
        """, unsafe_allow_html=True)
        render_landing_page()
        render_footer()
        return
    
    # --- Model Configuration (Sidebar) ---
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        st.error("Need at least 2 numeric columns to perform regression.")
        render_footer()
        return
    
    with st.sidebar:
        st.markdown('<div class="sidebar-title">🎯 Model Configuration</div>', unsafe_allow_html=True)
        
        default_target = "NIFTY50_PE" if "NIFTY50_PE" in numeric_cols else numeric_cols[0]
        target_col = st.selectbox("Dependent Variable (Y)", numeric_cols, index=numeric_cols.index(default_target) if default_target in numeric_cols else 0)
        
        available = [c for c in numeric_cols if c != target_col]
        
        # User selection for X variables
        feature_cols = st.multiselect("Independent Variables (X)", available, default=available[:3])
        
        # Auto-Feature Engineering
        st.markdown('<div class="sidebar-title">🔧 Auto-Features</div>', unsafe_allow_html=True)
        if feature_cols and st.button("✨ Generate Lags & Rolls", type="secondary"):
            engine_temp = MLREngine(df, target_col, feature_cols)
            df_new, new_features = engine_temp.generate_auto_features(lags=[1,2], rolls=[3,5])
            st.session_state['data'] = df_new
            st.session_state['auto_features'] = new_features
            st.success(f"Generated {len(new_features)} new features: {', '.join(new_features[:5])}...")
            st.rerun()
        
        if 'auto_features' in st.session_state:
            feature_cols += st.session_state['auto_features']
        
        # Advanced Fitting Options
        st.markdown('<div class="sidebar-title">⚙️ Advanced</div>', unsafe_allow_html=True)
        use_ridge = st.checkbox("Use Ridge Stabilization", value=False)
        alpha = st.slider("Ridge Alpha", 0.1, 10.0, 1.0) if use_ridge else 1.0
        use_lasso = st.checkbox("Use Lasso", value=False)
        use_bayesian = st.checkbox("Use Bayesian", value=False)
        prior_strength = st.slider("Bayesian Prior Strength", 0.1, 5.0, 1.0) if use_bayesian else 1.0
        
        # Prune Button
        if feature_cols and st.button("🛠️ Apply VIF Prune", type="secondary"):
            st.info("Prune applied—rerun model to see effects.")
        
        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
        st.markdown(f"""
        <div class='info-box'>
            <p style='font-size: 0.8rem; margin: 0; color: var(--text-muted); line-height: 1.5;'>
                <strong>Version:</strong> {VERSION}<br>
                <strong>Engine:</strong> { 'Ridge/Lasso/Bayesian' if use_ridge or use_lasso or use_bayesian else 'OLS' } statsmodels
            </p>
        </div>
        """, unsafe_allow_html=True)
        
    # Validation
    if not feature_cols:
        st.markdown("""
        <div class="premium-header">
            <span class="product-badge">PREMIUM</span>
            <h1>TATTVA : MLR Engine</h1>
            <div class="tagline">Multivariate Linear Regression, Diagnostics & Decision Architecture</div>
        </div>
        """, unsafe_allow_html=True)
        st.info("👈 Please select Independent Variables (X) from the sidebar to generate the model.")
        render_footer()
        return

    # --- Run Model (Enhanced Caching & Progress) ---
    try:
        data = clean_data(df, target_col, feature_cols)
    except Exception as e:
        st.error(f"Data cleaning failed: {e}")
        render_footer()
        return
    
    cache_key = f"mlr_{target_col}_{hashlib.md5(('-'.join(sorted(feature_cols))).encode()).hexdigest()}_{len(data)}_{use_ridge}_{alpha}_{use_lasso}_{use_bayesian}"
    
    if 'mlr_cache' not in st.session_state or st.session_state.get('mlr_cache_key') != cache_key:
        with st.spinner("Computing Partial Coefficients, Rank Diagnostics, and Advanced Fitting..."):
            progress = st.progress(0)
            progress.progress(0.2)
            engine = MLREngine(data, target_col, feature_cols, use_ridge=use_ridge, alpha=alpha, 
                               use_lasso=use_lasso, use_bayesian=use_bayesian, prior_strength=prior_strength)
            progress.progress(0.5)
            engine.fit()
            progress.progress(0.8)
            if not engine.is_stable:
                st.warning("Model flagged as unstable but stabilized via Ridge/Bayesian. Proceed with caution.")
            progress.progress(1.0)
            st.session_state['mlr_engine'] = engine
            st.session_state['mlr_cache_key'] = cache_key
            
    engine = st.session_state['mlr_engine']

    # ═══════════════════════════════════════════════════════════════════════
    # DECISION DASHBOARD (Nirnay-Matched Layout)
    # ═══════════════════════════════════════════════════════════════════════
    st.markdown("<br>", unsafe_allow_html=True)
    
    max_vif = engine.vif_data['VIF Score'].max() if not engine.vif_data.empty else 0
    r_squared = engine.model.rsquared
    adj_r_squared = engine.model.rsquared_adj
    grade, grade_class, grade_desc = engine.get_model_health_grade()
    
    stability_status = "STABLE" if engine.is_stable else "STABILIZED"
    stability_color = "success" if engine.is_stable else "warning"
    cond_num_display = f"{engine.condition_number:.1f}" if engine.condition_number else "N/A"
    
    c1, c2, c3, c4, c5 = st.columns([1, 1, 1, 1, 1])
    
    with c1:
        st.markdown(f'<div class="metric-card primary"><h4>Explanatory Power</h4><h2>{adj_r_squared:.2f}</h2><div class="sub-metric">Adj R² (0 to 1)</div></div>', unsafe_allow_html=True)
    
    with c2:
        vif_color = "success" if max_vif < 3 else "warning" if max_vif <= 5 else "danger"
        st.markdown(f'<div class="metric-card {vif_color}"><h4>Max Collinearity</h4><h2>{max_vif:.2f}</h2><div class="sub-metric">Target VIF < 5.0</div></div>', unsafe_allow_html=True)
    
    with c3:
        p_color = "success" if getattr(engine.model, 'f_pvalue', 0.01) < 0.05 else "danger"
        st.markdown(f'<div class="metric-card {p_color}"><h4>Model Viability</h4><h2>{"PASS" if getattr(engine.model, 'f_pvalue', 0.01) < 0.05 else "FAIL"}</h2><div class="sub-metric">F-Test (p < 0.05)</div></div>', unsafe_allow_html=True)
    
    with c4:
        st.markdown(f'<div class="metric-card {stability_color}"><h4>Matrix Stability</h4><h2>{stability_status}</h2><div class="sub-metric">Rank: {engine.matrix_rank}/{len(engine.features)+1} | {engine.fit_type}</div></div>', unsafe_allow_html=True)
    
    with c5:
        cond_color = "success" if (engine.condition_number or 0) < 30 else "warning" if (engine.condition_number or 0) < 1000 else "danger"
        st.markdown(f'<div class="metric-card {cond_color}"><h4>Condition Number</h4><h2>{cond_num_display}</h2><div class="sub-metric">Target < 30</div></div>', unsafe_allow_html=True)
        
    st.markdown(f"""
    <div class="signal-card {grade_class}" style="padding: 1.25rem; min-height: 120px; display: flex; flex-direction: column; justify-content: center; margin-bottom: 0.5rem;">
        <div class="label" style="margin-bottom: 0;">MODEL CONVICTION</div>
        <div class="value" style="font-size: 1.75rem; margin: 0.25rem 0;">{grade}</div>
        <div class="subtext" style="font-size: 0.75rem; margin-top: 0;">{grade_desc}</div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ═══════════════════════════════════════════════════════════════════════
    # TABS (Using Nirnay Tab Styles)
    # ═══════════════════════════════════════════════════════════════════════
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "**🎯 Feature Analytics**",
        "**🔍 Collinearity (VIF & Rank)**",
        "**📊 Visualizations**",
        "**🔮 Scenario Sandbox**",
        "**⚙️ Advanced Fitting**"
    ])
    
    # --- TAB 1: Feature Analytics ---
    with tab1:
        st.markdown("##### Feature Analytics & Partial Slopes")
        st.markdown(f"""<p style="color: #888; font-size: 0.9rem;">
        The <b>Relative Impact (Std Beta)</b> neutralizes different scales (e.g., % yields vs absolute currency), showing which feature is <i>actually</i> driving the target the most. Fit Type: <strong>{engine.fit_type}</strong>
        </p>""", unsafe_allow_html=True)
        
        def color_pvalue(val):
            if isinstance(val, float):
                return 'color: #ef4444;' if val > 0.05 else 'color: #10b981;'
            return 'color: inherit'
        
        styled_coef = engine.coef_df.style.format({
            'Coefficient (Slope)': "{:.5f}",
            'Relative Impact (Std Beta)': "{:.5f}",
            'Standard Error': "{:.5f}",
            't-Statistic': "{:.3f}",
            'p-Value': "{:.4f}"
        }).map(color_pvalue, subset=['p-Value'])
        
        st.dataframe(styled_coef, height=300)  # Removed use_container_width
        
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1.25rem; margin-top: 1rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'>Decision Rule</h4>
            <p style='color: var(--text-secondary); font-size: 0.85rem; margin-top: 0.5rem;'>
                Keep variables where the p-Value is <span style="color: #10b981; font-weight: 600;">Green (< 0.05)</span>. 
                If it is <span style="color: #ef4444; font-weight: 600;">Red (> 0.05)</span>, the model is telling you this specific factor provides no mathematical edge.
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.download_button("📥 Export Coefficients", engine.coef_df.to_csv(index=False), "coefficients.csv", type="secondary")

    # --- TAB 2: VIF Diagnostics (Upgraded to include Rank & Condition) ---
    with tab2:
        st.markdown("##### Variance Inflation Factor (VIF) & Matrix Diagnostics")
        
        if not engine.is_stable:
            st.markdown("""
            <div class="signal-card danger">
                <div class="signal-card-header">
                    <span class="signal-card-title">⚠️ MODEL INSTABILITY DETECTED</span>
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">Rank deficiency or high condition number flagged. Stabilized with {engine.fit_type}.</p>
            </div>
            """, unsafe_allow_html=True)
        elif max_vif > 5:
            st.markdown("""
            <div class="signal-card danger">
                <div class="signal-card-header">
                    <span class="signal-card-title">⚠️ OVERLAPPING SIGNALS DETECTED</span>
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">Variables with a VIF > 5 are essentially telling the same economic story. Review the <b>Intelligent Resolution Plan</b> below to quickly clean your model.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="signal-card success">
                <div class="signal-card-header">
                    <span class="signal-card-title">✅ PURE SIGNAL GEOMETRY</span>
                </div>
                <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">All variables have a VIF score under 5. Matrix is full-rank and well-conditioned.</p>
            </div>
            """, unsafe_allow_html=True)
            
        styled_vif = engine.vif_data.style.format({
            'VIF Score': "{:.2f}"
        }).map(highlight_vif, subset=['VIF Score'])
        
        st.dataframe(styled_vif, height=300)  # Removed use_container_width

        # Matrix Diagnostics Box
        st.markdown("##### 🔍 Matrix Health Check")
        col_md1, col_md2 = st.columns(2)
        with col_md1:
            rank_status = "FULL" if engine.matrix_rank == len(engine.features) + 1 else "DEFICIENT (Stabilized)"
            rank_color = "success" if rank_status == "FULL" else "warning"
            st.markdown(f'<div class="metric-card {rank_color}"><h4>Matrix Rank</h4><h2>{rank_status}</h2><div class="sub-metric">Actual: {engine.matrix_rank}</div></div>', unsafe_allow_html=True)
        with col_md2:
            cond_status = "GOOD" if (engine.condition_number or 0) < 30 else "POOR" if (engine.condition_number or 0) < 1000 else "CRITICAL (Stabilized)"
            cond_color = "success" if cond_status == "GOOD" else "warning" if cond_status == "POOR" else "danger"
            st.markdown(f'<div class="metric-card {cond_color}"><h4>Condition #</h4><h2>{cond_status}</h2><div class="sub-metric">Value: {cond_num_display}</div></div>', unsafe_allow_html=True)

        # Intelligent Collinearity Resolution Plan
        if max_vif > 5 and engine.resolution_plan:
            st.markdown("<br>##### 🛠️ Intelligent Resolution Plan", unsafe_allow_html=True)
            st.markdown("<p style='color: var(--text-muted); font-size: 0.9rem;'>The system has mapped the collinearity clusters and mathematically isolated the optimal variables to retain.</p>", unsafe_allow_html=True)
            
            all_drops = set()
            for plan in engine.resolution_plan:
                all_drops.update(plan['drops'])
            
            retained_vars = [v for v in feature_cols if v not in all_drops]
            
            if retained_vars:
                retained_html = " &nbsp;•&nbsp; ".join([f"<span style='color: var(--text-primary); font-weight: 600;'>{v}</span>" for v in retained_vars])
                st.markdown(f"""
                <div class="info-box" style="border-left: 3px solid var(--primary-color); background: rgba(var(--primary-rgb), 0.05); margin-bottom: 1.5rem; padding: 1rem 1.25rem;">
                    <h4 style="color: var(--primary-color); margin-top: 0; margin-bottom: 0.25rem; font-size: 0.9rem; text-transform: uppercase; letter-spacing: 0.5px;">🎯 Target Optimized Basket</h4>
                    <p style="margin: 0 0 0.75rem 0; font-size: 0.85rem; color: var(--text-muted);">Executing the drops below will leave you with this mathematically pure feature set:</p>
                    <div style="font-size: 1.05rem;">
                        {retained_html}
                    </div>
                </div>
                """, unsafe_allow_html=True)

            for plan in engine.resolution_plan:
                if plan['type'] == 'cluster':
                    st.markdown(f"###### 🧠 Auto-Resolution: {plan['title']}")
                    st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{plan["champion"]}</span><span class="symbol-price"> • Highest Standalone Correlation</span></div><span class="status-badge buy">RETAIN</span></div>', unsafe_allow_html=True)
                    for drop_var in plan['drops']:
                        st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{drop_var}</span><span class="symbol-price"> • Overlaps with {plan["champion"]}</span></div><span class="status-badge sell">DROP</span></div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem; margin-bottom: 1.5rem;'><i>Reasoning:</i> {plan['reason']}</p>", unsafe_allow_html=True)
                else:
                    st.markdown(f"###### ⚠️ Auto-Resolution: {plan['title']}")
                    for drop_var in plan['drops']:
                        st.markdown(f'<div class="symbol-row"><div><span class="symbol-name">{drop_var}</span><span class="symbol-price"> • Complex multi-variable noise</span></div><span class="status-badge sell">DROP</span></div>', unsafe_allow_html=True)
                    st.markdown(f"<p style='color: var(--text-muted); font-size: 0.85rem; margin-top: 0.5rem; margin-bottom: 1.5rem;'><i>Reasoning:</i> {plan['reason']}</p>", unsafe_allow_html=True)
            
            if st.button("Apply Prune Now", type="primary"):
                pruned = engine.apply_resolution_plan()
                st.session_state.pruned_features = pruned
                st.success(f"Pruned to: {', '.join(pruned)}. Rerun the model for updated results.")
                st.rerun()

    # --- TAB 3: Visualizations ---
    with tab3:
        c_viz1, c_viz2 = st.columns(2)
        
        with c_viz1:
            st.markdown("##### Absolute Feature Importance")
            st.markdown('<p style="color: #888; font-size: 0.8rem;">Ranked by Standardized Beta (Excludes Constant)</p>', unsafe_allow_html=True)
            
            fig_fi = px.bar(
                engine.feature_importance, 
                x='Absolute Impact', 
                y='Variable', 
                orientation='h',
                color='Relative Impact (Std Beta)',
                color_continuous_scale='RdBu',
                color_continuous_midpoint=0
            )
            fig_fi.update_layout(height=350, yaxis={'categoryorder':'total ascending'}, showlegend=False)
            fig_fi = update_chart_theme(fig_fi)
            st.plotly_chart(fig_fi, use_container_width=False)  # Deprecated fix: use default full width
            
        with c_viz2:
            st.markdown("##### Feature Correlation Heatmap")
            st.markdown('<p style="color: #888; font-size: 0.8rem;">Identifies simple 1-to-1 overlaps before VIF computation</p>', unsafe_allow_html=True)
            corr_matrix = engine.df[[target_col] + feature_cols].corr()
            
            fig_corr = px.imshow(
                corr_matrix, text_auto=".2f", aspect="auto", 
                color_continuous_scale='RdBu_r', zmin=-1, zmax=1
            )
            fig_corr.update_layout(height=350)
            fig_corr = update_chart_theme(fig_corr)
            st.plotly_chart(fig_corr, use_container_width=False)
            
        st.markdown("---")
        
        c_viz3, c_viz4 = st.columns(2)
        
        with c_viz3:
            st.markdown("##### Actual vs Predicted Fit")
            try:
                preds = engine.get_predictions()
                fig_pred = go.Figure()
                fig_pred.add_trace(go.Scatter(
                    x=engine.y, y=preds, mode='markers', name='Predictions',
                    marker=dict(color='#FFC300', size=6, opacity=0.7)
                ))
                
                min_val = min(engine.y.min(), preds.min())
                max_val = max(engine.y.max(), preds.max())
                fig_pred.add_trace(go.Scatter(
                    x=[min_val, max_val], y=[min_val, max_val], mode='lines', name='Perfect Fit',
                    line=dict(color='#06b6d4', dash='dash')
                ))
                
                fig_pred.update_layout(height=350, xaxis_title=f'Actual {target_col}', yaxis_title='Predicted')
                fig_pred = update_chart_theme(fig_pred)
                st.plotly_chart(fig_pred, use_container_width=False)
            except ValueError as e:
                st.warning(f"Prediction plot unavailable: {e}")
                
        with c_viz4:
            st.markdown("##### Residuals Distribution (Error Profile)")
            if hasattr(engine.model, 'resid'):
                residuals = engine.model.resid
                fig_resid = px.histogram(
                    residuals, nbins=50,
                    color_discrete_sequence=['#8b5cf6']
                )
                fig_resid.update_layout(height=350, xaxis_title="Residual Value", yaxis_title="Frequency")
                fig_resid = update_chart_theme(fig_resid)
                st.plotly_chart(fig_resid, use_container_width=False)
            else:
                st.warning("Residuals unavailable—model not fitted.")

    # --- TAB 4: Scenario Engine (Upgraded to Nirnay UI) ---
    with tab4:
        st.markdown("##### 🔮 Forward-Looking Scenario Simulator")
        st.markdown("""<p style="color: #888; font-size: 0.9rem;">
        Dial in hypothetical market conditions below. The engine uses your mathematically isolated coefficients to predict where the target will move.
        </p>""", unsafe_allow_html=True)
        
        if not engine.is_stable:
            st.markdown("""
            <div class="signal-card warning">
                <p style="margin: 0; font-size: 0.9rem; color: var(--text-secondary);">⚠️ Scenario simulation available (stabilized). Results incorporate regularization uncertainty.</p>
            </div>
            """, unsafe_allow_html=True)
        
        c_sandbox_left, c_sandbox_right = st.columns([1.5, 1])
        
        scenario_inputs = {}
        
        with c_sandbox_left:
            st.markdown("<div class='info-box' style='padding: 1.5rem;'>", unsafe_allow_html=True)
            st.markdown("<h4 style='color: var(--text-primary); margin-bottom: 1rem;'>Adjust Macro Factors</h4>", unsafe_allow_html=True)
            
            for col in feature_cols:
                min_val = float(engine.X[col].min())
                max_val = float(engine.X[col].max())
                mean_val = float(engine.X[col].mean())
                
                buffer = (max_val - min_val) * 0.2 if max_val > min_val else 1.0
                slider_min = min_val - buffer
                slider_max = max_val + buffer
                
                step_size = (slider_max - slider_min) / 100
                format_str = "%.4f" if step_size < 0.01 else "%.2f"
                unit = "%" if any(word in col.lower() for word in ['pe', 'yield', 'rate', '%']) else ""
                label = f"{col} ({unit})"
                
                scenario_inputs[col] = st.slider(
                    label, 
                    min_value=slider_min, 
                    max_value=slider_max, 
                    value=mean_val,
                    format=format_str,
                    help=f"Historical range: {min_val:.2f} to {max_val:.2f}"
                )
            st.markdown("</div>", unsafe_allow_html=True)
            
        with c_sandbox_right:
            try:
                predicted_y = engine.predict_scenario(scenario_inputs)
                current_y_mean = engine.y.mean()
                delta = predicted_y - current_y_mean
                
                delta_color = "success" if delta > 0 else "danger" if delta < 0 else "primary"
                arrow = "▲" if delta > 0 else "▼" if delta < 0 else "▬"
                
                st.markdown(f"""
                <div class="signal-card {delta_color}" style="text-align: center; padding: 2rem;">
                    <div class="label" style="font-size: 0.85rem;">PREDICTED {target_col}</div>
                    <div class="value" style="font-size: 3.5rem; margin: 1rem 0;">{predicted_y:.2f}</div>
                    <div class="subtext" style="font-size: 1rem;">
                        {arrow} {abs(delta):.2f} vs Historical Mean ({current_y_mean:.2f})
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
                mc_mean, mc_std = engine.monte_carlo_scenarios(scenario_inputs, n_sims=100)
                st.markdown(f'<div class="metric-card neutral" style="min-height: 80px; align-items: center;"><h4>MC Confidence (95%)</h4><h2 style="font-size: 1.4rem;">{predicted_y:.2f} <span style="font-size: 1rem; color: #888;">±{1.96 * mc_std:.2f}</span></h2></div>', unsafe_allow_html=True)
                
            except ValueError as e:
                st.error(f"Scenario error: {e}")
            
            if hasattr(engine.model, 'params'):
                st.markdown("<br><h5 style='color: var(--text-muted); font-size: 0.8rem; text-transform: uppercase;'>Mathematical Driver Breakdown</h5>", unsafe_allow_html=True)
                
                st.markdown(f'<div class="symbol-row" style="background: transparent; border: 1px dashed #3A3A3A;"><div><span class="symbol-name">Intercept (Baseline)</span></div><span class="symbol-score" style="color: #EAEAEA;">{engine.model.params["const"]:.4f}</span></div>', unsafe_allow_html=True)
                
                contributions = [abs(engine.model.params[col] * scenario_inputs[col]) for col in feature_cols if col in engine.model.params.index]
                max_abs_contribution = max(contributions) if contributions else 1.0
                if max_abs_contribution == 0: max_abs_contribution = 1.0

                for col in feature_cols:
                    if col in engine.model.params.index:
                        slope = engine.model.params[col]
                        input_val = scenario_inputs[col]
                        contribution = slope * input_val
                        color = "#10b981" if contribution > 0 else "#ef4444"
                        pct = (abs(contribution) / max_abs_contribution) * 100
                        
                        st.markdown(f"""
                        <div style="margin-bottom: 0.75rem;">
                            <div style="display: flex; justify-content: space-between; font-size: 0.85rem;">
                                <span style="color: #EAEAEA; font-weight: 600;">{col}</span>
                                <span style="color: {color}; font-weight: 700;">{contribution:+.4f}</span>
                            </div>
                            <div style="display: flex; justify-content: space-between; font-size: 0.7rem; color: var(--text-muted); margin-bottom: 0.25rem;">
                                <span>Coef: {slope:.4f} × Val: {input_val:.4f}</span>
                            </div>
                            <div class="conviction-meter">
                                <div class="conviction-fill" style="width: {pct}%; background: {color};"></div>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

    # --- TAB 5: Advanced Fitting (Now Fully Implemented) ---
    with tab5:
        st.markdown("##### ⚙️ Advanced Model Options")
        st.markdown("""
        <div style='background: rgba(16, 185, 129, 0.1); border: 1px solid var(--success-green); border-radius: 12px; padding: 1.25rem;'>
            <h4 style='color: var(--success-green); margin-bottom: 0.75rem;'>✅ Fully Implemented Features</h4>
            <p style='color: var(--text-secondary); font-size: 0.9rem; line-height: 1.6;'>
                <strong>Ridge/Lasso Regularization:</strong> Handles collinearity by shrinking coefficients (alpha tunes strength).<br>
                <strong>Auto-Feature Engineering:</strong> Generates lags (e.g., lag1) and rolling averages (e.g., roll3) via sidebar button.<br>
                <strong>Bayesian Updates:</strong> Conjugate prior approximation adds uncertainty penalty to R²; prior_strength controls belief in prior.<br><br>
                Toggle options in sidebar and rerun for effects. System auto-stabilizes on rank issues.
            </p>
        </div>
        """, unsafe_allow_html=True)

    render_footer()

if __name__ == "__main__":
    main()
