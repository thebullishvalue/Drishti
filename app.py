"""
Tattva — Main Streamlit entrypoint.
तत्त्व (Tattva) — "Principle / Essence"

TATTVA — Two systems. One conclusion. A top-down macro forecast and a bottom-up
basket regime read — across commodities, FX, and equity indices — unified by
adaptive convergence.

Usage:
    streamlit run app.py
"""

from __future__ import annotations

import os

# ── BLAS thread pinning (MUST run before numpy/sklearn import) ────────────────
# The walk-forward fits hundreds of small models sequentially. On Streamlit
# Community Cloud the container is throttled to ~1 shared vCPU but the host
# reports many logical CPUs, so OpenBLAS/MKL spawn one thread per reported core
# and thrash — turning each tiny PCA/Ridge solve into a thread-contention storm
# (the #1 reason the walk-forward is far slower on cloud than locally). One
# thread per process is strictly faster for many-small-matrix workloads here.
# os.environ.setdefault → respects any explicit override from the environment.
for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

# ── Numba cache OUTSIDE the app tree (MUST run before numba is imported) ──────
# @njit(cache=True) kernels write .nbc/.nbi artifacts. If those land in the app
# directory (default: <module>/__pycache__), Streamlit's file watcher treats each
# write as a source change and reruns the script — restarting the whole pipeline
# mid-compile. Point Numba's cache at the home cache dir (writable, NOT watched).
os.environ.setdefault(
    "NUMBA_CACHE_DIR",
    os.path.join(os.path.expanduser("~"), ".cache", "tattva", "numba"),
)

import sys
import time
import warnings
from datetime import datetime, timezone, timedelta
from pathlib import Path

import html
import numpy as np
import pandas as pd
import streamlit as st

# ── Warning suppression ──────────────────────────────────────────────────────
# A blanket `category=RuntimeWarning` filter used to sit here — it silenced
# every RuntimeWarning process-wide, including any GENUINE numeric issue
# (overflow, invalid divide, degenerate log/sqrt) anywhere in the math stack,
# not just the known-noisy sources it was meant to cover (audit finding C6).
# The one legitimate source found by auditing (nanmean's "Mean of empty
# slice" on the engine's own warm-up rows) is now scoped locally at its call
# site (engines/fvo.py's _compute_breadth_metrics) instead. FutureWarning
# stays broadly suppressed — it's pandas/numpy API-deprecation noise, not a
# correctness signal, so it doesn't carry the same risk of masking a real bug.
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*YF.download.*")
warnings.filterwarnings("ignore", message=".*auto_adjust.*")
warnings.filterwarnings("ignore", category=UserWarning, module="yfinance")
pd.options.mode.chained_assignment = None

# ── Path setup ───────────────────────────────────────────────────────────────
# Force PROJECT_ROOT to the FRONT of sys.path (ahead of site-packages) so the
# project's own packages (analytics, core, data, …) always win over any
# same-named package that happens to be installed in the environment. The
# project dirs carry __init__.py so they resolve as regular packages.
PROJECT_ROOT = Path(__file__).resolve().parent
_pr = str(PROJECT_ROOT)
if _pr in sys.path:
    sys.path.remove(_pr)
sys.path.insert(0, _pr)

# ── UI ───────────────────────────────────────────────────────────────────────
from ui.theme import inject_css, VERSION, PRODUCT_NAME, COMPANY, progress_bar
from ui.tabs.tab_convergence import render_convergence_tab
from ui.components import (
    render_header,
    render_info_box,
    build_hero_verdict,
    render_hero_card,
    render_warning_box,
    render_control_hint,
    render_ticker,
    section_gap,
)
from ui.tabs.tab_fvo import render_fvo_tab
from ui.tabs.tab_swayam import render_swayam_tab
from ui.tabs.tab_diagnostics import render_diagnostics_tab
from ui.tabs.tab_data import render_data_tab
from ui.tabs.tab_precedent import render_precedent_tab

# ── Data ─────────────────────────────────────────────────────────────────────
from data.fetcher import fetch_constituent_ohlcv, fetch_macro_live, fetch_commodity_dataset, fetch_stock_target_series
from data.calendars import trading_days_behind, is_session, session_mask, resolve_exchange
from data.universe import resolve_stock_symbol

# ── Engines ──────────────────────────────────────────────────────────────────
from engines.fvo import FairValueEngine
from engines.fvo.blocks import block_membership
from engines.swayam import aggregate_views
from engines.swayam.kernel import view_skill_weights

# ── Convergence ──────────────────────────────────────────────────────────────
from convergence.cross_validator import CrossValidator
from convergence.conviction_model import UnifiedConvictionModel
from convergence.divergence_detector import CrossSystemDivergenceDetector

# ── Logger & Config ──────────────────────────────────────────────────────────
from core.logger_config import console, generate_run_id, Colors
from core.config import LOOKBACK_WINDOWS, MIN_DATA_POINTS, STALENESS_DAYS, SESSION_FRESH_FLOOR, TARGET_EXCLUDED_PREDICTORS, ALL_TARGETS, TARGET_CATEGORIES, is_stock_target, FORECAST_HORIZON, RAW_YIELD_PREDICTORS, DIV_LOOKBACK, TIMEFRAME_TRADING_DAYS, swayam_macro_columns, FREEFORM_STOCK_CATEGORIES, register_stock_target, get_instrument_config
from engines.swayam import build_swayam_frames, effective_member_count, default_swayam_members
from core.config import GLOBAL_MACRO_MAP, MACRO_SYMBOLS_YF, INDEX_TARGETS_MAP

# Friendly column name → ticker, for resolving each predictor/target column to its
# exchange (holiday-aware data freshness + target-session spine filtering). Targets
# (incl. sheet/NCDEX sentinels) are merged last so they win any name collision.
_COLUMN_TICKERS = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **INDEX_TARGETS_MAP, **ALL_TARGETS}


# ─── Per-config result cache ─────────────────────────────────────────────────
# The full result of an analysis is the set of session-state keys below. We
# snapshot them per cache_key so revisiting a previously-computed config (e.g.
# the user switches Gold → Silver → Gold) restores instantly instead of
# recomputing the whole 5-phase pipeline. Bounded (LRU) to cap memory.
_BUNDLE_KEYS = (
    "engine", "fvo_ts", "swayam_daily", "swayam_view_dfs",
    "convergence_df", "divergence_events", "nishkarsh_result", "last_agreement",
    "nishkarsh_conv_normalized", "wf_results",
    "intelligence_active_weights", "intelligence_active_thresholds",
    "intelligence_active_profile",
    # The consensus's full history (Convergence-tab marker tiers) and the
    # weighted composite (hero WEIGHTED row / amber overlay) — must travel
    # with the bundle so a cached target switch-back doesn't leave the
    # PREVIOUS target's state in session.
    "hero_series",
    "nishkarsh_calibrated_score", "nishkarsh_calibrated_signal",
    "calibrated_conv_series",
    # Per-target UI metadata that must travel with the result bundle —
    # otherwise a cached target switch-back leaves the PREVIOUS target's
    # value in session state (e.g. the Swayam tab showing a stale
    # "basket source: snapshot" hint for a target resolved live, or the
    # Convergence tab's "breadth carried forward" notice firing/missing
    # based on the WRONG target's basket-freshness timestamp).
    "swayam_native_last", "swayam_n_eff",
)
# Keep the last N configs. The comment here previously said "the 3
# commodities" — stale since the universe grew to 30+ targets (commodities,
# FX, India/US indices, sector ETFs; audit finding E5). Each entry is a full
# 5-phase pipeline result, so this stays modest rather than trying to cover
# the whole universe; 6 covers a session that browses a handful of targets
# (e.g. all commodities, or an index + its close comparators) without
# recomputing.
_RESULTS_CACHE_MAX = 6

# Baskets at/above this size get their per-constituent frames trimmed before
# entering the _RESULTS_CACHE_MAX-deep results_cache LRU (audit finding F19).
# swayam_view_dfs carries ~200 columns per constituent (the full
# kernel output); only the ~9 the Swayam tab's drill-down actually
# displays (_SWAYAM_DRILLDOWN_COLS) are needed once the result is just sitting
# in the switch-back cache. A small commodity basket (~15-20 names) is cheap
# either way and kept at full width so nothing else that might read the wider
# frame in-session breaks; an uncapped large index (S&P 500 ~500 names) is
# where the ~200-column full width, multiplied across up to 6 LRU entries,
# actually matters.
_CONSTITUENT_TRIM_THRESHOLD = 60
_SWAYAM_DRILLDOWN_COLS = (
    "Close", "MSF_Osc", "MMR_Osc", "Unified_Osc", "Condition",
    "Regime", "Vol_Regime", "Change_Point", "Confidence",
)


def _bundle_swayam_view_dfs(dfs: dict) -> dict:
    """Trim swayam_view_dfs to the Swayam tab's drill-down columns
    before it enters the per-config results_cache LRU, for baskets at/above
    _CONSTITUENT_TRIM_THRESHOLD names. Only affects the SNAPSHOT stored in
    results_cache — the live session_state copy the active render reads
    (and engines.swayam.aggregate_views, which needs the
    full width and runs before this snapshot is taken) is never touched.
    """
    if not dfs or len(dfs) < _CONSTITUENT_TRIM_THRESHOLD:
        return dfs
    trimmed = {}
    for sym, df in dfs.items():
        cols = [c for c in _SWAYAM_DRILLDOWN_COLS if c in df.columns]
        trimmed[sym] = df[cols] if cols else df.iloc[:, :0]
    return trimmed


def _ensure_stock_target_column(df: pd.DataFrame, active_target: str) -> pd.DataFrame:
    """Inject a free-form stock target's Close into the model matrix.

    Individual-stock targets (``is_stock_target``) are deliberately
    NOT part of the macro batch universe fetch_commodity_dataset pulls (cache
    coherence — see fetch_stock_target_series's docstring); their price
    column is injected per-target here, the same pattern
    data.fetcher._fetch_exogenous_targets uses for sheet targets: aligned to
    the matrix's DATE spine, ffilled, leading NaNs left for the per-target
    dropna downstream. No-op when the column already exists or the target
    isn't a stock. Mutates st.session_state['data'] too, so a target switch
    or a cached rerun sees the column without re-fetching.
    """
    if active_target in df.columns or not is_stock_target(active_target):
        return df
    ticker = ALL_TARGETS.get(active_target)
    if not ticker:
        return df
    end = pd.Timestamp.today()
    s = fetch_stock_target_series(ticker, end - pd.Timedelta(days=365 * 9), end)
    if s is None:
        return df                      # the guard right after this call fires cleanly
    spine = pd.to_datetime(df["DATE"], errors="coerce").dt.normalize()
    s.index = pd.DatetimeIndex(s.index).normalize()
    s = s[~s.index.duplicated(keep="last")]
    df = df.copy()
    df[active_target] = s.reindex(spine).ffill().to_numpy()
    st.session_state["data"] = df
    return df


# ─── UI Rendering helpers ────────────────────────────────────────────────────

def _render_header(frame=None) -> None:
    """Masthead, then the tape.

    The tape sits directly under the masthead and above everything else: it is
    ambient context (where is the world today) that every reading below is
    relative to, and it belongs where the eye lands before it starts working.
    It draws from the run's OWN macro panel, so it cannot disagree with the
    valuation underneath it.
    """
    render_header(
        title=f"{PRODUCT_NAME}",
        tagline="Cross-Asset Fair Value · Self-Referential Breadth · Unified Convergence",
    )
    if frame is not None:
        render_ticker(frame)


def _render_landing_page() -> None:
    """Render the landing page with three system cards."""
    section_gap()
    col1, col2, col3 = st.columns(3, gap="small")
    with col1:
        st.markdown("""
        <div class='system-card fvo'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>
                FVO
            </h3>
            <p>Walk-forward ensemble regression on the selected target (commodities, FX, indices & ETFs) vs the macro/FX universe, with robust quantile z-scores and DDM filtering.</p>
            <div class='spec'>
                <span>Ensemble:</span> PCA-OLS + Huber<br>
                <span>Validation:</span> Walk-forward OOS<br>
                <span>Bounds:</span> Rolling robust quantiles
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='system-card swayam'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>
                SWAYAM
            </h3>
            <p>Per-instrument MSF + MMR analysis across a basket of related ETFs & miners, with HMM/GARCH/CUSUM regime intelligence aggregation.</p>
            <div class='spec'>
                <span>Signal:</span> MSF + MMR oscillator<br>
                <span>Breadth:</span> Oversold / Overbought %<br>
                <span>Regime:</span> HMM · GARCH · CUSUM
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class='system-card convergence'>
            <h3>
                <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="12" y1="8" x2="12" y2="16"/><line x1="8" y1="12" x2="16" y2="12"/></svg>
                CONVERGENCE
            </h3>
            <p>Adaptive-weighted composite of 4 dimensions: Direction, Breadth, Magnitude, Regime — with DDM.</p>
            <div class='spec'>
                <span>Fusion:</span> FVO + Swayam<br>
                <span>Smoothing:</span> Leaky DDM<br>
                <span>Range:</span> Soft \u00b1100 limit
            </div>
        </div>
        """, unsafe_allow_html=True)
    section_gap()
    st.markdown("""
    <div class='landing-prompt'>
        <h4>
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>
            AWAITING DATA
        </h4>
        <p>Pick an <strong>Asset Class → Target</strong> (Commodities · FX · India &amp; US Indices · Sector ETFs) in the <strong>Sidebar</strong>,<br>
           then execute <strong>Run Analysis</strong> to fetch the live yfinance data and initialize both engines.</p>
    </div>
    """, unsafe_allow_html=True)


def _render_primary_signal(nishkarsh_norm, agreement, fvo_signal) -> None:
    """Render the hero card.

    All interpretation lives in ``ui.components.build_hero_verdict`` (a pure,
    unit-testable function — see research/test_hero_verdict.py); this wrapper
    only gathers session-state inputs and hands the verdict to
    ``render_hero_card``.

    The card is a CONVICTION CHAIN: direction from FVO, then six gates whose
    product is the conviction, with the smallest gate named as the binding
    constraint. Every engine contributes exactly one gate — FVO (mispricing
    and reversion), Swayam (breadth), Convergence (agreement + normalized
    consensus), the walk-forward read (edge), and Precedent (base rate).
    """
    wf          = st.session_state.get("wf_results")                   # list[dict] | None
    div_events  = st.session_state.get("divergence_events")            # DataFrame | None
    prec        = st.session_state.get("precedent_summary")            # dict | None

    # DEGENERATE-CONVERGENCE GATE: `nishkarsh_norm` is None exactly when the
    # FVO/Swayam alignment found no overlap. The verdict handles that case
    # directly now — `swayam_breadth=None` leaves the corroboration gate at a
    # neutral 0.5 and says "no breadth read" on the card, rather than the
    # signal quietly becoming half-weight FVO wearing a convergence label.
    # Divergences are silenced for the same reason: the detector would be
    # comparing FVO against breadth that does not exist.
    if nishkarsh_norm is None:
        div_events = None

    # Active instrument's forecast horizon - for interpretation copy only.
    try:
        FWD_HORIZON = get_instrument_config(st.session_state.get("active_target", "")).forecast_horizon
    except KeyError:
        FWD_HORIZON = FORECAST_HORIZON

    wf_ics = [r["ic"] for r in wf if isinstance(r, dict) and r.get("ic") == r.get("ic")] if wf else []
    # The edge number IS the walk-forward mean now. There is no calibration to
    # hold out from: weights are learned forward, so every window's IC is
    # already out-of-sample with respect to everything after it, and a separate
    # "Val IC" would just be one more window of the same thing.
    oos_ic = float(np.mean(wf_ics)) if wf_ics else None
    wf_pos = (sum(1 for v in wf_ics if v > 0) / len(wf_ics)) if wf_ics else None
    wf_n = len(wf_ics) if wf_ics else None
    # RECENT divergence count only (audit finding F7) — div_events spans the
    # WHOLE history (6+ years), so a bare len() reads in the hundreds and is a
    # permanent, meaningless alarm ("N divergence events flagged"). Count only
    # events within the last DIV_LOOKBACK trading days of the series (the same
    # window CrossSystemDivergenceDetector uses for its own persistence flag),
    # anchored on the LATEST event date in the table (a proxy for "today" —
    # div_events carries no direct handle on the engine's current as-of date).
    n_div = 0
    if (div_events is not None
            and hasattr(div_events, "__len__") and len(div_events)):
        try:
            _div_dates = pd.to_datetime(div_events.index, errors="coerce")
            _valid_dates = _div_dates.dropna()
            if len(_valid_dates):
                _cutoff = _valid_dates.max() - pd.Timedelta(days=int(DIV_LOOKBACK * 1.5))
                n_div = int((_div_dates >= _cutoff).sum())
            else:
                n_div = int(len(div_events))
        except Exception:
            n_div = int(len(div_events))

    # Swayam's current breadth — the corroboration gate. Read from the daily
    # frame's last row rather than a session scalar so it is the same object
    # the Swayam tab shows.
    _sw = st.session_state.get("swayam_daily")
    swayam_breadth = None
    if _sw is not None and not _sw.empty:
        _last = _sw.iloc[-1]
        swayam_breadth = {
            "oversold_pct": float(_last.get("Oversold_Pct", 50.0)),
            "overbought_pct": float(_last.get("Overbought_Pct", 50.0)),
        }

    # The convergence layer's own output: how far the two engines concur
    # (dimension-weighted by learned skill) and where the normalized consensus
    # — the series the Convergence tab plots — currently points.
    _cdf = st.session_state.get("convergence_df")
    convergence_read = None
    if _cdf is not None and not _cdf.empty:
        convergence_read = {
            "agreement_ratio": float(_cdf["agreement_ratio"].iloc[-1])
            if "agreement_ratio" in _cdf.columns else 0.5,
            "consensus": (float(nishkarsh_norm["value"])
                          if nishkarsh_norm and nishkarsh_norm.get("value") is not None
                          else None),
        }

    verdict = build_hero_verdict(
        fvo_signal=fvo_signal,
        swayam_breadth=swayam_breadth,
        convergence=convergence_read,
        wf_ic=oos_ic,
        wf_pos=wf_pos,
        wf_n=wf_n,
        precedent=prec,
        n_divergences=n_div,
        horizon_days=FWD_HORIZON,
        div_window=DIV_LOOKBACK,
    )
    render_hero_card(verdict)
    section_gap()


def _render_model_passport_sidebar(current_universe: str, current_index: str | None = None) -> None:
    """Sidebar Passport — what the model is doing right now.

    This used to be a PROFILE manager: it showed which calibrated profile was
    loaded, its train/val IC and timestamp, warned when the profile had been fit
    on a different universe, and offered Import / Export / Reset. All of that
    existed because calibration produced a persisted artefact that could be
    stale, mismatched, or shared — and because the output depended on which
    artefact happened to be loaded.

    Nothing is persisted now. Weights are learned forward from the data in
    front of the model, every run, so there is no profile to import, no
    mismatch to warn about, and nothing to reset to. What remains worth showing
    is the state the run actually reached.
    """
    st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">Model Passport</div>', unsafe_allow_html=True)

    w = st.session_state.get("intelligence_active_weights") or {}
    wf = st.session_state.get("wf_results") or []

    if not w:
        render_control_hint("Run an analysis to populate.")
        return

    _top = sorted(w.items(), key=lambda kv: -kv[1])
    st.markdown(
        '<div style="font-family:var(--data);font-size:0.72rem;color:var(--ink-secondary);'
        'line-height:1.7;padding:0.2rem 0 0.4rem 0;">'
        + "".join(
            f'<div style="display:flex;justify-content:space-between;">'
            f'<span>{k}</span><span style="color:var(--amber);font-weight:700;">{v:.3f}</span></div>'
            for k, v in _top)
        + '</div>',
        unsafe_allow_html=True,
    )
    render_control_hint("Dimension weights · learned forward from resolved outcomes")

    if wf:
        _ics = [r["ic"] for r in wf if np.isfinite(r.get("ic", float("nan")))]
        if _ics:
            _mean = float(np.mean(_ics))
            _pos = sum(1 for v in _ics if v > 0)
            render_control_hint(
                f"Walk-forward IC {_mean:+.3f} · {_pos}/{len(_ics)} windows positive")


def _render_footer() -> None:
    utc_now = datetime.now(timezone.utc)
    ist_now = utc_now + timedelta(hours=5, minutes=30)
    st.markdown(
        f'<div class="app-footer">'
        f'<div class="content">'
        f'\u00a9 {ist_now.year} <strong>{PRODUCT_NAME}</strong> &nbsp;\u00b7&nbsp; {COMPANY} &nbsp;\u00b7&nbsp; v{VERSION} &nbsp;\u00b7&nbsp; {ist_now.strftime("%Y-%m-%d %H:%M:%S IST")}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# ══════════════════════════════════════════════════════════════════════════════
# MAIN APPLICATION
# ══════════════════════════════════════════════════════════════════════════════

def main():
    st.set_page_config(
        page_title="TATTVA | Unified Convergence",
        page_icon="data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI+PGNpcmNsZSBjeD0iMTIiIGN5PSIxMiIgcj0iMTAiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIi8+PHBhdGggZD0iTTggMTRsMy01IDIgMyAzLTQiIGZpbGw9Im5vbmUiIHN0cm9rZT0iI0Q0QTg1MyIgc3Ryb2tlLXdpZHRoPSIyIiBzdHJva2UtbGluZWNhcD0icm91bmQiIHN0cm9rZS1saW5lam9pbj0icm91bmQiLz48L3N2Zz4=",
        layout="wide", initial_sidebar_state="expanded",
    )
    inject_css()

    # Replay dynamic stock-target registration on every rerun. register_stock_target
    # mutates module-level core.config dicts (ALL_TARGETS etc.) which survive
    # Streamlit reruns WITHIN a process but are never persisted — only
    # st.session_state survives a rerun as the durable record, so a freeform
    # symbol resolved earlier this session (e.g. "RELIANCE (NSE)") must be
    # re-registered before anything below resolves active_target against
    # ALL_TARGETS/STOCK_TARGET_MARKETS. Idempotent — safe every rerun.
    for _dname, _dmeta in st.session_state.get("dynamic_stock_targets", {}).items():
        register_stock_target(_dname, _dmeta["ticker"], _dmeta["market"])

    # Single main-area progress slot, created up front (outside the sidebar) so the
    # SAME themed progress bar drives everything from the moment "Run Analysis" is
    # clicked — the fetch, the data-prep spine, the engines, convergence — instead of
    # a sidebar spinner that then hands off to a separate bar with a gap between them.
    # Empty (invisible) until the first progress_bar() call; cleared when a run ends.
    progress_container = st.empty()

    # ─── Sidebar ──────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown(
            """
        <div style="text-align:center;padding:0.5rem 0 0.75rem 0;">
            <div style="font-family:var(--display);font-size:1.35rem;font-weight:700;color:var(--amber);letter-spacing:0.04em;">TATTVA</div>
            <div style="font-family:var(--data);color:var(--ink-tertiary);font-size:0.6rem;margin-top:0.1rem;letter-spacing:0.06em;text-transform:uppercase;">तत्त्व | Unified Convergence</div>
        </div>
        <hr style="margin: 0.5rem 0; opacity: 0.1;">
        """,
            unsafe_allow_html=True,
        )

        # Two-level selection: Asset Class → Target. Keeps the growing target
        # roster (commodities, FX, India & US indices, sector-ETF universe)
        # logically grouped instead of one long flat list.
        all_names = list(ALL_TARGETS.keys())
        prev_commodity = st.session_state.get("selected_commodity", all_names[0])
        if prev_commodity not in all_names:
            prev_commodity = all_names[0]

        _categories = list(TARGET_CATEGORIES.keys())
        # Freeform categories (India/US Stocks) stay EMPTY in TARGET_CATEGORIES
        # (a dynamic name renders a text input, not a list entry — see
        # core.config.register_stock_target), so plain membership can never
        # find a previously-resolved stock target there. Check
        # dynamic_stock_targets first so re-selecting a stock category across
        # reruns doesn't silently snap back to the first category.
        _dyn_meta = st.session_state.get("dynamic_stock_targets", {}).get(prev_commodity)
        if _dyn_meta is not None:
            prev_cat = next(
                (cat for cat, mkt in FREEFORM_STOCK_CATEGORIES.items() if mkt == _dyn_meta["market"]),
                _categories[0],
            )
        else:
            prev_cat = next(
                (c for c, names in TARGET_CATEGORIES.items() if prev_commodity in names),
                _categories[0],
            )
        # Seed widget state BEFORE instantiation so the (options-changing) target
        # selectbox never holds a value outside its current category — the classic
        # Streamlit "key + dynamic options" pitfall. We drive both via session_state
        # keys, not `index=`.
        st.session_state.setdefault("target_category", prev_cat)
        if st.session_state["target_category"] not in _categories:
            st.session_state["target_category"] = prev_cat

        st.markdown('<div class="sidebar-title">Asset Class</div>', unsafe_allow_html=True)
        sel_cat = st.selectbox(
            "Asset Class", _categories,
            label_visibility="collapsed", key="target_category",
            help="Choose an asset class, then a target within it.",
        )

        if sel_cat in FREEFORM_STOCK_CATEGORIES:
            # India Stocks / US Stocks: no constituent basket to browse — enter
            # a symbol directly. The asset class supplies the suffix policy
            # (data.universe.resolve_stock_symbol): India tries SYMBOL.NS
            # first, then SYMBOL.BO; US uses the bare symbol.
            _market = FREEFORM_STOCK_CATEGORIES[sel_cat]
            st.markdown('<div class="sidebar-title" style="margin-top:0.5rem;">Symbol</div>', unsafe_allow_html=True)
            _raw_symbol = st.text_input(
                "Symbol", key=f"stock_symbol_{_market}",
                label_visibility="collapsed",
                placeholder="e.g. RELIANCE, TATASTEEL" if _market == "india" else "e.g. AAPL, BRK.B",
                help="Swayam runs in Swayam self-mode on this instrument's own OHLCV "
                     "(no constituent basket exists for a single stock).",
            )
            selected_commodity = None
            if _raw_symbol and _raw_symbol.strip():
                with st.spinner("Resolving symbol…"):
                    _ticker, _exch_or_err = resolve_stock_symbol(_raw_symbol, _market)
                if _ticker is None:
                    st.error(_exch_or_err)
                else:
                    _base = _ticker.rsplit(".", 1)[0] if _market == "india" else _raw_symbol.strip().upper()
                    selected_commodity = f"{_base} ({_exch_or_err})"
                    register_stock_target(selected_commodity, _ticker, _market)
                    _dyn = st.session_state.setdefault("dynamic_stock_targets", {})
                    _dyn[selected_commodity] = {"ticker": _ticker, "market": _market}
                    render_control_hint(f"{_raw_symbol.strip().upper()} → {_ticker} · {_exch_or_err}")
            else:
                render_control_hint(
                    "NSE (.NS) checked first, then BSE (.BO)" if _market == "india"
                    else "US listing · symbol as typed"
                )
        else:
            cat_targets = TARGET_CATEGORIES.get(sel_cat, all_names)

            # Keep the target selection valid for the chosen category.
            if st.session_state.get("target_select") not in cat_targets:
                st.session_state["target_select"] = (
                    prev_commodity if prev_commodity in cat_targets else cat_targets[0]
                )
            st.markdown('<div class="sidebar-title" style="margin-top:0.5rem;">Target</div>', unsafe_allow_html=True)
            selected_commodity = st.selectbox(
                "Target", cat_targets,
                label_visibility="collapsed", key="target_select",
                help="FVO forecasts this target's forward return; Swayam reads "
                     "bottom-up breadth — across its constituent basket (index members, "
                     "producers, sector ETFs), or as a Swayam self-ensemble on the "
                     "instrument's own price (commodities & stocks).",
            )
        # Breadth source hint. Every target now reads the same way, so this is
        # a statement of method rather than a routing label. Suppressed for a
        # FREEFORM stock, where the resolution hint just above already states
        # the ticker/exchange and the Symbol help text explains Swayam.
        if selected_commodity and sel_cat not in FREEFORM_STOCK_CATEGORIES:
            render_control_hint("Swayam · self-referential view bank (own OHLCV)")

        df = None
        has_data = "data" in st.session_state and "run_analysis" in st.session_state

        if selected_commodity is None:
            # Freeform stock category with no symbol resolved yet (empty input
            # or a resolution error already shown above) — nothing to run/switch
            # to, so don't render either button.
            render_control_hint("Enter a symbol above to continue.")
            if has_data:
                df = st.session_state["data"]
        elif not has_data:
            # Initial load. The fetch pulls the entire macro universe once and
            # is target-agnostic — the chosen commodity only selects FVO's
            # target column and Swayam's basket.
            if st.button("Run Analysis", type="primary"):
                # No spinner — drive the main-area progress bar from the very first
                # click. The fetch is one blocking call, so we show the stage before it
                # (3%) and after it (15%); the analysis picks the bar up from there on
                # the rerun, so the experience reads as one continuous progress bar.
                progress_bar(progress_container, 3, "Fetching Market Data",
                             "yfinance · global macro universe · ~9y daily history")
                _end = pd.Timestamp.today()
                # Walk-forward needs MIN_DATA_POINTS (1500) daily observations.
                # ~9 years of calendar history clears that with headroom.
                _start = _end - pd.Timedelta(days=365 * 9)
                df, error = fetch_commodity_dataset(_start, _end)
                if error or df is None:
                    progress_container.empty()
                    st.error(f"Failed: {error}")
                    return
                progress_bar(progress_container, 15, "Market Data Loaded",
                             f"{df.shape[1]} series × {df.shape[0]} rows · preparing analysis…")
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state["data"] = df
                st.session_state["selected_commodity"] = selected_commodity
                st.session_state["active_target"] = selected_commodity
                st.session_state["nishkarsh_index"] = selected_commodity
                st.session_state["run_analysis"] = True
                st.rerun()
        else:
            df = st.session_state["data"]
            # Post-load target switch — re-runs the engines on the already
            # fetched universe (no re-fetch; only the Swayam basket re-pulls).
            if selected_commodity != st.session_state.get("active_target"):
                if st.button(f"Switch target → {selected_commodity}", type="primary"):
                    st.session_state["selected_commodity"] = selected_commodity
                    st.session_state["active_target"] = selected_commodity
                    st.session_state["nishkarsh_index"] = selected_commodity
                    st.session_state.pop("active_features", None)  # re-default predictors for new target
                    st.session_state.pop("engine", None)
                    st.session_state.pop("engine_cache", None)
                    st.rerun()

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

    # ─── Landing page if no data loaded ──────────────────────────────────
    if df is None:
        _render_header()
        _render_landing_page()
        _render_footer()
        return

    # A stock target's price column is injected here — BEFORE numeric_cols/
    # commodity_options are computed below — so target_col never falls back
    # to some other target for a stock on its first render (Model
    # Configuration's "Apply Configuration" button, further down, writes
    # that fallback straight into st.session_state["active_target"] if it's
    # ever wrong). Cheap no-op once the column already exists (cached fetch).
    df = _ensure_stock_target_column(df, st.session_state.get("active_target", ""))

    # ─── Sidebar: Model Configuration ──────────────────────────────────────
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    all_cols = df.columns.tolist()
    if len(numeric_cols) < 2:
        st.error("Need 2+ numeric columns.")
        return

    with st.sidebar:
        st.markdown('<div class="sidebar-title">Model Configuration</div>', unsafe_allow_html=True)

        # Target is chosen once in the sidebar "Target Commodity" selector;
        # resolve it here for predictor configuration.
        commodity_options = [c for c in ALL_TARGETS if c in numeric_cols] or numeric_cols
        target_col = st.session_state.get("active_target", commodity_options[0])
        if target_col not in numeric_cols:
            target_col = commodity_options[0]

        # Date column is always the dataset's DATE column — auto-detected.
        date_candidates = [c for c in all_cols if "date" in c.lower()]
        date_col = date_candidates[0] if date_candidates else "None"

        # Read-only target chip (set via the Target Commodity selector above).
        st.markdown(
            '<div style="display:flex;align-items:baseline;gap:0.5rem;'
            'padding:0.35rem 0 0.55rem 0;font-family:var(--data);">'
            '<span style="color:var(--ink-tertiary);text-transform:uppercase;'
            'letter-spacing:0.1em;font-size:0.58rem;">Target</span>'
            f'<span style="color:var(--amber);font-weight:700;font-size:0.92rem;">{target_col}</span>'
            '</div>',
            unsafe_allow_html=True,
        )

        # The valuation panel is the WHOLE macro cross-section, minus this
        # target's self-replicating near-duplicates (e.g. GLTR for a precious
        # metal, which would let the regression explain gold with gold). There
        # is no predictor picker any more: the FVO engine prices the target
        # against the traded opportunity set, and hand-deselecting instruments
        # from that set does not make it a better opportunity set — it makes it
        # a smaller one with an undocumented reason. The engine already handles
        # a wide panel on its own terms (Marchenko-Pastur decides how many
        # factors are real; an instrument that never prints is never admitted).
        _excluded = set(TARGET_EXCLUDED_PREDICTORS.get(target_col, []))
        available = [c for c in numeric_cols if c != target_col and c not in _excluded]
        st.session_state["active_features"] = tuple(available)
        st.session_state["active_date_col"] = date_col
        render_control_hint(
            f"{len(available)} macro instruments · full cross-section"
            + (f" ({len(_excluded)} excluded as self-replicating)" if _excluded else ""))

        st.markdown('<div class="section-divider"></div>', unsafe_allow_html=True)

        if "run_analysis" in st.session_state and st.session_state.get("run_analysis"):
            if st.button("Reset Analysis", type="secondary", use_container_width=True):
                st.session_state.pop("data", None)
                st.session_state.pop("engine", None)
                st.session_state.pop("engine_cache", None)
                st.session_state.pop("fvo_engine", None)
                st.session_state.pop("fvo_fit_key", None)
                st.session_state.pop("wf_results", None)
                st.session_state.pop("results_cache", None)  # drop all cached configs
                st.session_state.pop("run_analysis", None)
                st.session_state.pop("nishkarsh_result", None)
                st.rerun()

            # Force a live re-pull of the whole universe, then recompute — for when
            # the data is stale/partial (the freshness notices point here). Reset =
            # re-run on cached data (fast); Refresh = re-fetch live + re-run (slower).
            # Snapshot-preserving: if the live pull fails (rate-limit / circuit open),
            # the cache's stale fallback keeps the app working on last-good data.
            if st.button("Refresh Data", type="secondary", use_container_width=True):
                from data.cache import begin_force_refresh
                begin_force_refresh()   # next fetches bypass TTL; disk snapshot kept
                # Same main-area progress bar as Run Analysis (no spinner) — the recompute
                # on rerun picks it up from ~15%, so refresh reads as one continuous bar.
                progress_bar(progress_container, 3, "Re-fetching Live Market Data",
                             "yfinance · full universe · bypassing cache · ~30–60s")
                _rend = pd.Timestamp.today()
                _rdf, _rerr = fetch_commodity_dataset(_rend - pd.Timedelta(days=365 * 9), _rend)
                if _rdf is not None:
                    progress_bar(progress_container, 15, "Live Data Refreshed",
                                 f"{_rdf.shape[1]} series × {_rdf.shape[0]} rows · recomputing…")
                    st.session_state["data"] = _rdf   # keep run_analysis → stay in results
                else:
                    progress_container.empty()
                for _k in ("engine", "engine_cache", "fvo_engine", "fvo_fit_key",
                           "wf_results", "results_cache", "nishkarsh_result",
                           "precedent_summary", "_prec_key", "_precedent_analogs_cache", "conv_norm_params",
                           # Horizon-independent Swayam cache (audit finding F17) —
                           # must be dropped on a live re-fetch too, else Refresh
                           # Data re-pulls the FVO macro universe live but
                           # silently keeps serving the PRE-refresh Swayam
                           # basket/constituent analysis.
                           "_swayam_fetch_cache", "_swayam_analysis_cache"):
                    st.session_state.pop(_k, None)
                # The convergence tab's actual per-config normalization cache key is
                # "conv_norm_causal::<engine_cache>" (ui/tabs/tab_convergence.py) — the
                # legacy "conv_norm_params" prefix below predates that rename and no
                # longer matches anything, so those z-score caches survived every
                # "Refresh Data" click unpruned (audit finding C1). Sweep both
                # prefixes so a future rename doesn't reintroduce the same gap.
                for _prefix in ("conv_norm_params", "conv_norm_causal::"):
                    for _k in [k for k in list(st.session_state) if str(k).startswith(_prefix)]:
                        st.session_state.pop(_k, None)
                st.rerun()
            render_control_hint("Force-fetch live data · recompute · slower than Reset")

        # ── Model Passport (Sanket-style) ──────────────────────────────
        # Surfaces the learned dimension weights + walk-forward read. (Each
        # target used to key its own persisted profile here — see
        # _intel_index below).
        _current_universe = st.session_state.get("active_target") or st.session_state.get("selected_commodity", "Gold")
        _current_index = st.session_state.get("nishkarsh_index", _current_universe)
        _render_model_passport_sidebar(_current_universe, _current_index)

        st.markdown('<hr style="margin: 1rem 0 0.75rem 0; opacity: 0.05;">', unsafe_allow_html=True)
        st.markdown(
            '<div class="system-spec">'
            f'<div class="spec-row"><span class="spec-label">Version</span><span class="spec-value">{VERSION}</span></div>'
            '<div class="spec-row"><span class="spec-label">Engine</span><span class="spec-value">Convergence</span></div>'
            '<div class="spec-row"><span class="spec-label">Data</span><span class="spec-value">yfinance</span></div>'
            '</div>',
            unsafe_allow_html=True,
        )

    # ─── Resolve active configuration ──────────────────────────────────────
    active_target = st.session_state.get("active_target", target_col)
    # Per-instrument config — every engine knob (Swayam/Swayam, FVO forecast,
    # DDM, convergence weights, precedent) is read from THIS target's own config
    # (core.config.INSTRUMENT_CONFIGS), so an instrument can be retuned in
    # isolation. Falls back to the base defaults for any target that somehow
    # isn't registered (shouldn't happen — catalogue targets register at import,
    # stocks via register_stock_target before analysis).
    try:
        _icfg = get_instrument_config(active_target)
    except KeyError:
        from core.config import InstrumentConfig as _IC
        _icfg = _IC()
    active_features = list(st.session_state.get("active_features", [c for c in numeric_cols if c != active_target]))
    # Never let the target — or a self-replicating predictor (e.g. GLTR for a
    # precious metal) — leak into its own predictor set.
    _excluded_feats = {active_target, *TARGET_EXCLUDED_PREDICTORS.get(active_target, [])}
    active_features = [f for f in active_features if f not in _excluded_feats]
    active_date = st.session_state.get("active_date_col", date_col)

    # ─── Data freshness notice ──────────────────────────────────────────────
    # Measured in TRADING days behind (weekends ignored) so Friday data read on a
    # Sunday is "current", not stale. Tiered, design-consistent: a calm info note
    # when 1–2 trading days behind (today's bar often isn't published yet), and a
    # prominent warning once genuinely stale (source hasn't updated). The signal
    # always reflects the as-of date shown, never "today".
    # "Trading days behind" is counted on the TARGET's own exchange calendar via
    # data.calendars.trading_days_behind — holiday-aware when exchange_calendars is
    # installed (so Diwali/Thanksgiving no longer over-count by ~1), else it degrades
    # to the exact legacy Mon–Fri busday count. The partial-session check below remains
    # the calendar-agnostic primary freshness signal (native coverage).
    if active_date != "None" and active_date in df.columns:
        try:
            dates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True).dropna()
            if len(dates) > 0:
                latest_date = dates.max().to_pydatetime()
                if latest_date.tzinfo is not None:
                    latest_date = latest_date.replace(tzinfo=None)
                # `latest_date` is a tz-naive EXCHANGE-LOCAL date, but "today" has no
                # single frame: a UTC-hosted deploy (Streamlit Cloud) rolls past
                # midnight ahead of an IST/EST exchange, over-counting a current bar
                # as "1 day behind". Anchor to the EARLIER of UTC and machine-local —
                # that brackets the realistic tz band, so a tz skew never *overstates*
                # staleness. Genuine staleness (≥ STALENESS_DAYS) and the exact
                # partial-session gate below still fire normally.
                today = min(datetime.now(timezone.utc).date(), datetime.now().date())
                # trading days strictly after the data date, up to & including today,
                # on the target exchange's calendar (holiday-aware when available).
                _tgt_ticker = ALL_TARGETS.get(active_target)
                behind = trading_days_behind(_tgt_ticker, latest_date.date(), today)
                ds = latest_date.strftime("%d %b %Y")
                if behind >= STALENESS_DAYS:
                    render_warning_box(
                        title="Latest data unavailable",
                        content=(f"Newest data is {ds} — {behind} trading days behind. The price source "
                                 f"(yfinance) hasn't published more recent data, so every signal below "
                                 f"reflects {ds}, not today. Use Refresh Data in the sidebar to pull the "
                                 f"latest once the source updates."),
                    )
                elif behind >= 1:
                    render_info_box(
                        "Data freshness",
                        f"Signals are as of {ds} ({behind} trading day"
                        f"{'s' if behind > 1 else ''} behind — today's bar may not be published yet).",
                    )

                # Predictors carried from a snapshot backfill (data.fetcher's
                # rate-limit recovery, audit finding B1): a rate-limited ticker
                # this run was refilled from the most recent prior snapshot
                # that HAD it, which may itself be stale. Surface which
                # columns and how old, rather than a silent log.warning no
                # one watching a Streamlit deploy will ever see.
                from data.fetcher import _current_stale_backfills
                _stale_backfills = _current_stale_backfills()
                if _stale_backfills:
                    _sb_items = sorted(_stale_backfills.items(), key=lambda kv: kv[1])
                    _sb_preview = ", ".join(f"{k} (as of {v})" for k, v in _sb_items[:5])
                    _sb_more = f" +{len(_sb_items) - 5} more" if len(_sb_items) > 5 else ""
                    render_info_box(
                        "Predictors carried from snapshot",
                        f"{len(_sb_items)} predictor(s) were rate-limited this fetch and refilled "
                        f"from a prior cached snapshot: {_sb_preview}{_sb_more}. Their momentum is "
                        f"flat until the next successful live fetch.",
                    )

                # Session completeness (Phase 2 — exchange-aware): of the inputs whose
                # market was OPEN on the latest date, how many actually posted a fresh
                # value vs are still forward-filled? Columns whose exchange was CLOSED
                # that day (e.g. US on Thanksgiving) are legitimately carried forward
                # and EXCLUDED — only genuinely-lagging open markets count, so a global
                # holiday no longer trips a false "partial session". Native freshness =
                # changed vs the prior row (continuous prices move every session). With
                # the calendar lib absent, is_session is "is a weekday" → every column
                # counts → identical to the prior calendar-agnostic gate.
                num = df.select_dtypes(include=[np.number])
                if len(num) >= 2:
                    cols = list(num.columns)
                    last_r = num.iloc[-1].to_numpy(dtype=np.float64)
                    prev_r = num.iloc[-2].to_numpy(dtype=np.float64)
                    finite = np.isfinite(last_r) & np.isfinite(prev_r)
                    should_post = np.array(
                        [is_session(_COLUMN_TICKERS.get(c), latest_date.date()) for c in cols]
                    )
                    judged = finite & should_post
                    denom = int(judged.sum())
                    if denom >= 3:   # need a few open markets before judging completeness
                        fresh_frac = float(((last_r != prev_r) & judged).sum() / denom)
                        # Skip the warning when latest_date is today: the session is
                        # still in progress, so most prices are forward-filled by
                        # design — that is expected, not a data problem.
                        _session_is_live = (latest_date.date() >= today)
                        if fresh_frac < SESSION_FRESH_FLOOR and not _session_is_live:
                            render_warning_box(
                                title="Partial latest session",
                                content=(f"Only {fresh_frac:.0%} of the markets open on {ds} have posted — the "
                                         f"rest are forward-filled from the prior session, so the macro predictors "
                                         f"and bottom-up breadth behind the latest signal are stale. Treat it as "
                                         f"provisional; use Refresh Data in the sidebar once those markets post."),
                            )

                # Per-source freshness for the ACTIVE target specifically — it can
                # lag the macro universe (sheet behind, or its market shut on a
                # holiday) with the gap forward-filled. Find its true last update:
                #   • sheet target  → exact, from the source series.
                #   • yfinance/other → detect the ff-filled tail (continuous prices
                #     don't repeat, so a run of identical closes = forward-filled days).
                try:
                    from data.sheets import SHEET_SOURCES, fetch_sheet_series
                    t_last = None
                    if active_target in SHEET_SOURCES:
                        s = fetch_sheet_series(active_target)
                        if s is not None and len(s):
                            t_last = pd.Timestamp(s.index.max()).to_pydatetime()
                    elif active_target in df.columns and active_date in df.columns:
                        tv = pd.to_numeric(df[active_target], errors="coerce").to_numpy()
                        tdates = pd.to_datetime(df[active_date], errors="coerce", dayfirst=True)
                        j = len(tv) - 1
                        while j > 0 and np.isfinite(tv[j]) and tv[j] == tv[j - 1]:
                            j -= 1
                        if j < len(tv) - 1 and pd.notna(tdates.iloc[j]):
                            t_last = pd.Timestamp(tdates.iloc[j]).to_pydatetime()
                    if t_last is not None:
                        t_behind = trading_days_behind(_tgt_ticker, t_last.date(), today)
                        if t_behind >= 1:
                            _today_is_session = is_session(_tgt_ticker, today)
                            if _today_is_session and t_behind == 1:
                                # Market is open but today's bar is forward-filled —
                                # most likely yfinance rate-limited this ticker during
                                # the last fetch and the backfill used a prior snapshot.
                                # Prompt a manual refresh rather than crying "stale".
                                render_info_box(
                                    f"{active_target} price not yet updated",
                                    (f"Today's bar is carried forward from {t_last.strftime('%d %b %Y')} — "
                                     f"the {active_target} market is open but yfinance may have rate-limited "
                                     f"this ticker during the last fetch. Use Refresh Data in the sidebar "
                                     f"to pull the latest price."),
                                )
                            else:
                                render_warning_box(
                                    title=f"{active_target} data is lagging",
                                    content=(f"This target last updated {t_last.strftime('%d %b %Y')} "
                                             f"({t_behind} trading day{'s' if t_behind > 1 else ''} behind the macro "
                                             f"universe) — more recent rows are forward-filled from that value, so "
                                             f"its latest signal may be stale."),
                                )
                except Exception:
                    pass
        except Exception:
            pass

    # ─── Clean & Fit Engine ────────────────────────────────────────────────
    # Guard: a selected target whose column failed to fetch (e.g. a sheet/source
    # outage on a later run, while it stays selected) is silently dropped by the
    # column filter below and would KeyError at the per-column coercion. Fail clean.
    # Stock targets (archetype 'self') are never IN the macro batch fetch to begin
    # with — inject their price column here (single-ticker fetch, cached) before
    # the guard checks for it.
    df = _ensure_stock_target_column(df, active_target)
    if active_target not in df.columns:
        _tgt_ticker_guard = ALL_TARGETS.get(active_target, "?")
        if is_stock_target(active_target):
            console.failure("Stock target fetch failed", f"'{active_target}' (ticker {_tgt_ticker_guard}) — yfinance returned no usable data.")
            st.error(f"'{active_target}' price fetch failed (ticker {_tgt_ticker_guard}) — yfinance returned no data. "
                     f"Check the symbol, or try again once yfinance recovers.")
        else:
            console.failure("Target column missing", f"'{active_target}' not in fetched dataset — its source fetch failed.")
            st.error(f"'{active_target}' data is currently unavailable (its source fetch failed). "
                     f"Pick another target, or re-run once the source is back online.")
        return
    # Data-preparation diagnostics — collected at each stage so the terminal can
    # show exactly how the row count evolves (no "dark spots"), and so a failure
    # explains itself instead of a bare "Need 1500+". Emitted via _log_prep() below.
    _prep = {"target": active_target, "min_required": MIN_DATA_POINTS}
    _tgt_ticker_prep = ALL_TARGETS.get(active_target)
    _tgt_exch_prep = resolve_exchange(_tgt_ticker_prep) or "weekday"

    def _log_prep(stage: str = "complete") -> None:
        """Print the data-prep pipeline to the terminal (visible on success & failure)."""
        console.section("DATA PREPARATION")
        console.item("Target", f"{active_target}  (ticker={_tgt_ticker_prep or 'n/a'}, exch={_tgt_exch_prep})")
        console.item("Scoring Horizon", f"{_prep.get('fwd_h','?')}d "
                     "(convergence / analogs / display — the engine is fit to no label)")
        console.item("Rows · fetched", _prep.get("rows_initial", "?"))
        console.item("Rows · after session spine", f"{_prep.get('rows_session','?')}  ({_prep.get('sessions_dropped','?')} non-session rows removed)")
        console.item("Instruments · requested", _prep.get("feats_requested", "?"))
        console.item("Instruments · dropped (short history)", f"{len(_prep.get('feats_dropped', []))}"
                     + (f" → {', '.join(_prep['feats_dropped'][:6])}{'…' if len(_prep.get('feats_dropped', [])) > 6 else ''}" if _prep.get("feats_dropped") else ""))
        if _prep.get("yield_cols_dropped"):
            # Raw yield LEVELS are rate series, not prices: they print at/near/
            # below zero and the engine's log transform is undefined there. The
            # tradeable expression of the curve is already in the panel as the
            # Treasury ETF complex, so they are excluded rather than transformed.
            console.item("Instruments · dropped (raw yield levels, not prices)",
                         _prep["yield_cols_dropped"])
        console.item("Instruments · in valuation panel", _prep.get("feats_kept", "?"))
        console.item("Rows · after dropna (final spine)", _prep.get("rows_final", "?"))
        if "burn_in" in _prep and isinstance(_prep.get("rows_final"), int):
            # Unlike a rolling momentum window there is no warm-up head to trim
            # here — every row carries a finite level for the target and every
            # surviving instrument. The engine applies its own burn-in
            # internally and flags those rows `Valid = False`.
            _pub = max(0, _prep["rows_final"] - _prep["burn_in"])
            console.item("Rows · valued (after engine burn-in)",
                         f"{_pub}  ({_prep['burn_in']} burn-in rows published as Valid=False)")
        if stage == "complete":
            console.checkpoint(f"Data spine ≥ {MIN_DATA_POINTS}", "OK")

    cols = [active_target] + active_features + ([active_date] if active_date != "None" and active_date in df.columns else [])
    data = df[[c for c in cols if c in df.columns]].copy()
    _prep["rows_initial"] = len(data)
    _prep["feats_requested"] = len(active_features)
    if active_date != "None" and active_date in data.columns:
        data[active_date] = pd.to_datetime(data[active_date], errors="coerce", dayfirst=True)
        data = data.dropna(subset=[active_date]).sort_values(active_date)
    for col in [active_target] + active_features:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    _rows_pre_session = len(data)
    # Phase 3 — target-exchange session spine, applied FIRST so every filter below
    # operates in the target's real trading-session space (not the US-weekday spine).
    # The fetched matrix is a Mon–Fri spine (FX trades every weekday), so a row on the
    # TARGET's own market holiday carries its last close forward: a fake no-change bar
    # with stale predictors. Restricting to genuine sessions up front matters because
    # the feature-history guard and dropna below count rows — measuring them on the US
    # spine while the walk-forward actually runs on (fewer) India/exchange sessions can
    # leave a target just under MIN_DATA_POINTS (India indices have more holidays, so
    # ~1582 US weekdays = ~1496 NSE sessions). No-op for 24×5 FX and under the weekday
    # fallback (lib absent); the `.any()` guard refuses to blank the frame on misfire.
    if active_date != "None" and active_date in data.columns and len(data):
        _smask = session_mask(ALL_TARGETS.get(active_target), data[active_date])
        if _smask.any():
            data = data[_smask].reset_index(drop=True)
    _prep["rows_session"] = len(data)
    _prep["sessions_dropped"] = max(0, _rows_pre_session - len(data))
    # NOTE on the FVO print mask. The engine admits an instrument to the
    # cross-section only on days it genuinely traded — a carried-forward quote
    # otherwise enters as a fabricated zero return and drags whatever factor it
    # loads on toward zero on every foreign holiday. The exact answer is the
    # vendor's own NaN mask, but it no longer exists by the time this runs:
    # data/fetcher.py forward-fills the combined frame at source (see its
    # `combined.ffill()`), so a mask taken here would be all-True and would
    # silently assert that a Nikkei quote on a Tokyo holiday is a real print.
    # Rather than pass a mask that looks exact and is not, the panel is handed
    # over without one and the engine infers prints from where values actually
    # change (FairValueEngine._infer_printed) — conservative in the same
    # direction as the gate itself. Threading the true mask would mean
    # returning it from the fetcher alongside the prices, through the cache
    # layer; the `printed=` parameter on fit() is there for when that happens.
    data[[active_target] + active_features] = data[[active_target] + active_features].ffill()
    # Drop features with insufficient real history. We ffill (causal: carry last known
    # value forward) but deliberately do NOT bfill — backfilling leading NaNs would inject
    # future values into the past (look-ahead bias). The consequence is that a young or
    # near-empty series (e.g. a just-listed ETF, or a ticker yfinance returned ~nothing for)
    # keeps its leading NaNs, and dropna(subset=all features) would then collapse the whole
    # window to the intersection — as little as 1 row. So drop any feature still carrying a
    # NaN within the most recent MIN_DATA_POINTS *target-session* rows: those can't support
    # the walk-forward window without backfilled fakery. Measuring the tail in session space
    # (after the restriction above) means a feature too young for THIS target's calendar
    # (e.g. SGOV, listed 2020, vs an NSE target) is dropped, extending the usable window back
    # rather than capping it. Survivors are non-null over the tail, so the dropna below
    # retains >= MIN_DATA_POINTS rows whenever the target itself has the history.
    _win = min(MIN_DATA_POINTS, len(data)) if len(data) else 0
    _feats_before_guard = list(active_features)
    active_features = [
        f for f in active_features
        if f in data.columns and _win and data[f].tail(_win).notna().all()
    ]
    _prep["feats_dropped"] = [f for f in _feats_before_guard if f not in active_features]
    _prep["feats_kept"] = len(active_features)
    data = data.dropna(subset=[active_target] + active_features).reset_index(drop=True)
    _prep["rows_final"] = len(data)
    if len(data) < MIN_DATA_POINTS:
        # Explain the shortfall on the terminal — which stage cost the rows.
        _log_prep(stage="fail")
        console.failure(
            "Insufficient data spine for walk-forward",
            f"{active_target}: {len(data)} usable {_tgt_exch_prep} sessions after cleaning, "
            f"need ≥{MIN_DATA_POINTS}. Fetched {_prep['rows_initial']} rows → "
            f"{_prep['rows_session']} after session spine → {len(data)} after dropna "
            f"({_prep['feats_kept']} features kept, {len(_prep['feats_dropped'])} dropped for short history).",
        )
        st.error(
            f"Need {MIN_DATA_POINTS}+ data points for walk-forward analysis — "
            f"'{active_target}' yielded only {len(data)} usable {_tgt_exch_prep} trading sessions "
            f"after cleaning. Try a longer history, a target with more data, or fewer young predictors."
        )
        return
    active_features = [f for f in active_features if f in data.columns]
    if not active_features:
        _log_prep(stage="fail")
        console.failure("No valid features", f"{active_target}: every predictor was dropped for short history.")
        st.error("No valid features found after data cleaning.")
        return
    # The valuation regression models LOG price → the target must be strictly
    # positive. Every shipped target is a price/level/ratio, but a future target
    # (a spread, a net position, a yield differential) could go ≤0; fail clean
    # rather than silently producing all-NaN valuations.
    if (pd.to_numeric(data[active_target], errors="coerce") <= 0).any():
        _log_prep(stage="fail")
        console.failure("Non-positive target", f"{active_target}: contains values ≤ 0; the log-level valuation engine needs a strictly positive series.")
        st.error(f"'{active_target}' has non-positive values — the valuation engine models "
                 f"log price and needs a strictly positive series.")
        return

    # ── Valuation representation: price the target against the traded
    # opportunity set, rather than forecasting its next move.
    #   • Panel   = the LEVEL of every macro predictor. The FVO engine takes
    #     its own logs and differences internally, integrates the resulting
    #     factors back into levels, and regresses log price on them with
    #     time-varying coefficients. It therefore wants prices, not the
    #     pre-engineered momentum features the previous engine consumed.
    #   • Target  = the target's own price level. There is no forward label
    #     and so no forward-label overlap, no purge gap, and no zero-filled
    #     tail: the newest row is valued the same way every other row is.
    # FWD_HORIZON survives as a SCORING horizon only — the convergence layer,
    # the precedent analogs and the Intelligence calibrator all score against
    # the h-day forward return, and the UI projects the current mispricing
    # over it. Daily bars throughout.
    FWD_HORIZON = _icfg.forecast_horizon   # scoring / display horizon (trading days)
    _prep["fwd_h"] = FWD_HORIZON
    # RAW_YIELD_PREDICTORS (^IRX/^FVX/^TNX/^TYX) are percent-point RATE series,
    # not prices: they print at/near/below zero (2020-21 zero-rate era), and
    # the engine's log transform is undefined there. A yield LEVEL is also not
    # an instrument the target can be valued against — the tradeable expression
    # of the curve is already in the panel as the Treasury ETF complex
    # (SHY/IEF/TLT/…), which the block map classifies as "Rates". So they are
    # dropped from the valuation panel rather than transformed.
    _yield_feats = [f for f in active_features if f in RAW_YIELD_PREDICTORS]
    if _yield_feats:
        active_features = [f for f in active_features if f not in RAW_YIELD_PREDICTORS]
        _prep["yield_cols_dropped"] = len(_yield_feats)
        # Re-stamp the kept count: it was recorded after the short-history guard
        # but before this exclusion, so the prep log would otherwise report a
        # panel four instruments wider than the one the engine is handed.
        _prep["feats_kept"] = len(active_features)
    if not active_features:
        _log_prep(stage="fail")
        console.failure("Empty valuation panel", f"{active_target}: no price-level predictors survived.")
        st.error("No price-level predictors remain — the valuation engine needs a cross-section of prices.")
        return
    # No warm-up head to trim, and no row-validity mask: every surviving row
    # already carries a finite level for the target and every instrument (the
    # dropna above). The predecessor needed both — a rolling momentum window
    # left a NaN head, and its forward labels left a zero-filled tail — so the
    # prep log reported "usable rows" and "real labels" as separate counts.
    # Here the only exclusion is the engine's own burn-in, applied internally
    # and reported as `Valid`.
    _prep["burn_in"] = int(_icfg.fvo_burn_in)
    # Date-range fingerprint for the cache key. `data` carries a RangeIndex (reset at
    # load), so the real dates live in the active_date column, not the index — using
    # the index here would be integers (AttributeError on .date()). Fall back to a
    # row-count surrogate when there's no date column.
    if active_date != "None" and active_date in data.columns:
        _vd = pd.to_datetime(data[active_date], errors="coerce").dropna()
        _date_range = f"{_vd.iloc[0].date()}_{_vd.iloc[-1].date()}" if len(_vd) else f"n{len(data)}"
    else:
        _date_range = f"n{len(data)}"
    cache_key = f"fvo{FWD_HORIZON}|{active_target}|{'|'.join(sorted(active_features))}|{_date_range}"
    if st.session_state.get("engine_cache") != cache_key:
        # ── Restore from the per-config result cache if this exact config was
        # already computed this session (e.g. the user switched commodities and
        # came back) — full reuse, no recompute. ─────────────────────────────
        _rcache = st.session_state.setdefault("results_cache", {})
        if cache_key in _rcache:
            for _bk, _bv in _rcache[cache_key].items():
                st.session_state[_bk] = _bv
            _rcache[cache_key] = _rcache.pop(cache_key)  # mark most-recently-used
            st.session_state["engine_cache"] = cache_key
            console.header("TATTVA — Cached Result Restored", f"v{VERSION}")
            console.success(f"Restored {active_target} from session cache — no recompute")
            st.rerun()
        if "engine" in st.session_state:
            del st.session_state["engine"]

        # ════ RUN HEADER ════
        console.header("TATTVA — Unified Convergence Analysis", f"v{VERSION}")
        console.main_header("ANALYSIS CONFIGURATION", {
            "Run ID": generate_run_id(),
            "Target": active_target,
            "Predictors": f"{len(active_features)} columns",
            "Date Range": f"{data.shape[0]} observations",
        })
        # Full data-preparation trace (row evolution, session spine, dropped features)
        # so the pipeline has no dark spots — printed once per new computation.
        _log_prep(stage="complete")

        # Reuse the hoisted main-area progress slot (created at the top of main())
        # so the bar continues from where the fetch left it (~15%) with no gap.

        # ── Phase 1: Data Loading ─────────────────────────────────────────
        # HORIZON-INDEPENDENT: the macro fetch and the target's own OHLCV depend
        # only on active_target, never on the scoring horizon, so they are cached
        # separately (audit finding F17) and survive a re-run.
        #
        # There is no mode resolution left to do. Every target reads breadth off
        # its own price through Swayam; the branch that used to choose between
        # that and a constituent basket — and the basket resolution, the
        # snapshot fallbacks, and the up-to-503-symbol OHLCV fetch behind it —
        # went with the basket engine. On a large index that fetch was ~13 of
        # the run's ~14 minutes.
        _swayam_fetch_key = f"swayam_fetch::{active_target}"
        _nf_cache = st.session_state.get("_swayam_fetch_cache")
        if _nf_cache is not None and _nf_cache.get("key") == _swayam_fetch_key:
            console.start_phase("DATA ACQUISITION", 1, 5)
            target_ohlcv = _nf_cache["target_ohlcv"]
            swayam_macro_df = _nf_cache["swayam_macro_df"]
            macro_cols_list = _nf_cache["macro_cols_list"]
            console.item("Macro/OHLCV", "reused cached fetch (horizon-independent)")
            progress_bar(progress_container, 20, "Data Acquisition Reused", f"{len(macro_cols_list)} Macros (cached)")
            console.end_phase("DATA ACQUISITION")
        else:
            console.start_phase("DATA ACQUISITION", 1, 5)
            progress_bar(progress_container, 16, "Resolving Swayam Source",
                         f"{active_target} · own OHLCV (self-referential ensemble)")

            console.section("Macro Data")
            end_date = pd.Timestamp.today()
            # Match the FVO model-dataset window (~9y) so the Swayam views and
            # macro drivers overlap the FULL series — convergence then runs on
            # real data, not neutral placeholders.
            start_date = end_date - pd.Timedelta(days=365 * 9)
            macro_df = fetch_macro_live(start_date, end_date)
            console.item("Date Range", f"{start_date.date()} to {end_date.date()}")
            if not macro_df.empty:
                console.item("YF Columns", f"{len(macro_df.columns)} symbols")
                console.item("Rows", len(macro_df))
                console.success(f"Macro data: {len(macro_df.columns)} symbols × {len(macro_df)} rows")
            else:
                console.warning("No macro data available")

            console.section("Target OHLCV")
            # Swayam needs the target's own OHLC (and volume where it exists —
            # the volume-dependent views abstain when it does not; see
            # ensemble._is_volume_dependent).
            _tgt_ticker = ALL_TARGETS[active_target]
            progress_bar(progress_container, 18, "Fetching Target OHLCV", f"yfinance · {_tgt_ticker}")
            _ohlcv = fetch_constituent_ohlcv([_tgt_ticker], start_date, end_date)
            target_ohlcv = _ohlcv.get(_tgt_ticker)
            if target_ohlcv is not None and not target_ohlcv.empty:
                console.item("Symbol", _tgt_ticker)
                console.item("Rows", len(target_ohlcv))
                console.item("Has Volume", bool("Volume" in target_ohlcv.columns
                                                and target_ohlcv["Volume"].fillna(0).abs().sum() > 0))
                console.success(f"Target OHLCV: {len(target_ohlcv)} rows")
            else:
                console.warning(f"No OHLCV for {_tgt_ticker} — Swayam breadth will be unavailable")

            console.section("Swayam Macro Assembly")
            swayam_macro_df = macro_df.copy() if macro_df is not None and not macro_df.empty else pd.DataFrame()
            if not swayam_macro_df.empty:
                console.item("YF Symbols", len(swayam_macro_df.columns))
                console.success(f"Macro indicators: {len(swayam_macro_df.columns)} × {len(swayam_macro_df)} rows")
            macro_cols_list = list(swayam_macro_df.columns) if not swayam_macro_df.empty else []
            console.end_phase("DATA ACQUISITION")
            progress_bar(progress_container, 20, "Data Acquisition Complete", f"{len(swayam_macro_df.columns)} Macros")

            st.session_state["_swayam_fetch_cache"] = {
                "key": _swayam_fetch_key,
                "target_ohlcv": target_ohlcv,
                "swayam_macro_df": swayam_macro_df,
                "macro_cols_list": macro_cols_list,
            }

        # ── Phase 2: FVO FairValueEngine ─────────────────────────────────
        console.start_phase("FVO ENGINE", 2, 5)
        progress_bar(progress_container, 20, "Running FVO Engine", f"Valuation · {len(active_features)} Instruments · {len(data)} Rows")

        _price_level = data[active_target].to_numpy(dtype=np.float64)
        _cal = (pd.to_datetime(data[active_date].values)
                if active_date != "None" and active_date in data.columns
                else pd.RangeIndex(len(data)))
        _tgt_px = pd.Series(_price_level, index=_cal, name=active_target)
        _expl_px = data[active_features].astype(float)
        _expl_px.index = _cal

        _blk_names, _blk_map = block_membership(active_features)

        console.section("Engine Configuration")
        console.item("Mode", "Valuation · dynamic cointegrating regression on log price")
        console.item("Target", active_target)
        console.item("Cross-section", f"{len(active_features)} macro instruments → {len(_blk_names)} asset-class blocks")
        console.item("Blocks", ", ".join(f"{b}({sum(1 for v in _blk_map.values() if v == b)})" for b in _blk_names))
        console.item("Observations", f"{len(data)} rows")
        console.item("Burn-in", f"{_icfg.fvo_burn_in} rows before first publication")
        console.item("Print Floor", f"{_icfg.fvo_min_prints} prints before an instrument may join")
        console.item("Coefficient Memory", f"deltas {_icfg.fvo_valuation_deltas}")
        console.item("Min Data Points", MIN_DATA_POINTS)
        console.item("Lookback Windows", f"{LOOKBACK_WINDOWS}")

        console.section("Recursive Valuation")
        # Reuse an already-fit FVO engine for this exact config if a prior
        # (possibly interrupted) execution in THIS session already produced one.
        # `engine_cache` is only set at the end of Phase 5, so a Streamlit rerun
        # mid-pipeline (yfinance retry, cloud reconnect, stray interaction) would
        # otherwise re-enter this block and re-run the expensive valuation pass.
        # Keyed by cache_key → identical inputs → identical fit, so reuse is safe.
        if (st.session_state.get("fvo_fit_key") == cache_key
                and isinstance(st.session_state.get("fvo_engine"), FairValueEngine)):
            engine = st.session_state["fvo_engine"]
            console.item("Valuation", "reused cached fit (resumed run)")
            progress_bar(progress_container, 40, "FVO Engine Reused", "Cached valuation pass")
        else:
            engine = FairValueEngine()
            # `config=_icfg` threads this instrument's per-instrument FVO knobs
            # (burn-in, print floor, discount grid, lookback windows) into the
            # valuation, so FVO is tuned per instrument / asset class exactly
            # like Swayam and Swayam.
            engine.fit(
                _tgt_px, _expl_px,
                feature_names=active_features, config=_icfg,
                progress_callback=lambda pct, msg: progress_bar(
                    progress_container, int(20 + pct * 20), "Running FVO Engine", msg),
            )
            # Carry the raw price LEVEL on the engine output too. `Actual` is
            # already the price here, but the analog matcher and Intelligence
            # tuner both key off a column literally named "Price".
            engine.ts_data["Price"] = _price_level
            st.session_state["fvo_engine"] = engine
            st.session_state["fvo_fit_key"] = cache_key

        sig = engine.get_current_signal()
        stats = engine.get_model_stats()
        console.section("Engine Results")
        console.item("Signal", f"{sig['signal']} ({sig['strength']})")
        console.item("Conviction", f"{sig['conviction_score']:+.0f}")
        console.item("FVO", f"{sig['fvo']:+.2f}σ ({sig['pct_mispricing'] * 100:+.2f}% vs fair value)")
        console.item("Fair Value", f"{sig['fair_value']:,.2f} vs price {sig['actual']:,.2f}")
        console.item("OOS R²", f"{stats['r2_oos']:.3f} (log price vs fitted level)")
        console.item("R² vs Trailing Mean", f"{stats['r2_vs_anchor']:+.3f} (edge over a 252d anchor)")
        console.item("Model Spread", f"{sig['model_spread'] * 10000:.1f} bps (predictive SD of fair value)")
        console.item("Valuation Confidence", f"{sig['valuation_confidence']:.2f} "
                     f"(mean-reversion {sig['mr_prob']:.2f} × cross-sectional agreement {sig['xs_consistency']:.2f})")
        # DFA Hurst is deliberately not printed: its single log-log slope is
        # biased sharply upward for a short-memory series and would read
        # "trending" for a gap ADF calls strongly stationary. See
        # FairValueEngine._compute_hurst.
        console.item("Gap Half-Life", f"{sig['gap_half_life']:.0f}d (online AR1) · OU {sig['ou_half_life']:.0f}d")
        console.item("Gap Stationarity", f"ADF p={sig['adf_pvalue']:.4f} "
                     f"({'stationary — reversion licensed' if sig['adf_pvalue'] < 0.05 else 'unit root not rejected'})")
        console.item("Factors", f"k={sig['k_factors']} above the MP edge · {sig['n_available']} instruments admitted today")
        console.item("Market Regime", f"{sig['market_regime']} (stress pct {sig['stress']:.2f})")
        console.success(f"FVO engine complete | {len(engine.ts_data)} output rows "
                        f"({int(engine.ts_data['Valid'].sum())} valued, {engine.min_train_size} burn-in)")
        console.end_phase("FVO ENGINE")
        progress_bar(progress_container, 40, "FVO Engine Complete", f"Signal: {sig['signal']} ({sig['strength']}) · Conviction: {sig['conviction_score']:+.0f}")

        # ── Phase 3: Swayam Breadth ───────────────────────────────────────
        # HORIZON-INDEPENDENT (audit finding F17): the view bank and its
        # aggregation depend only on the target's own OHLCV + the macro driver
        # window, never on the scoring horizon. Cached under the SAME
        # _swayam_fetch_key as Phase 1; only the target-calendar reindex below
        # (cheap — no yfinance calls) re-runs.
        console.start_phase("SWAYAM ENGINE", 3, 5)
        progress_bar(progress_container, 42, "Running Swayam Engine",
                     "MSF+MMR+Regime · self-referential view bank")

        _na_cache = st.session_state.get("_swayam_analysis_cache")
        if _na_cache is not None and _na_cache.get("key") == _swayam_fetch_key:
            swayam_view_dfs = _na_cache["swayam_view_dfs"]
            swayam_daily_pre_reindex = _na_cache["swayam_daily_pre_reindex"]
            if "n_eff" in _na_cache:
                st.session_state["swayam_n_eff"] = _na_cache["n_eff"]
            console.item("View Bank", "reused cached fit (horizon-independent)")
            progress_bar(progress_container, 74, "Swayam Engine Reused", f"{len(swayam_view_dfs)} Views (cached)")
        else:
            swayam_daily_pre_reindex = pd.DataFrame()
            swayam_view_dfs = {}

            if target_ohlcv is not None and not target_ohlcv.empty:
                console.section("Self-Referential View Bank")
                # Leakage guard: drop the target's own macro column + its
                # excluded-predictor near-replicas from the MMR driver pool — a
                # view's Close correlates ~1.0 with the target's own macro
                # column, which would let MMR "explain" the target with itself
                # and silently zero the deviation oscillator.
                swayam_cols = swayam_macro_columns(active_target, macro_cols_list)
                _swayam_members = default_swayam_members(_icfg.swayam_lengths, _icfg.swayam_roc_frac)
                console.item("Views (bank)", f"{len(_swayam_members)} · timescale × information-set × mechanism")
                console.item("Timescale Span", str(_icfg.swayam_lengths))
                console.item("Regime Sensitivity", _icfg.swayam_regime_sensitivity)
                console.item("Base Weight", _icfg.swayam_base_weight)
                console.item("Macro Columns (post-leakage-guard)", len(swayam_cols))

                def _swayam_progress(done, total, name):
                    pct_val = int(45 + done / max(total, 1) * 30)
                    progress_bar(progress_container, pct_val, f"View {name}", f"{done}/{total} views")

                swayam_view_dfs = build_swayam_frames(
                    target_ohlcv, swayam_macro_df, swayam_cols,
                    members=_swayam_members,
                    regime_sensitivity=_icfg.swayam_regime_sensitivity,
                    base_weight=_icfg.swayam_base_weight,
                    num_vars=_icfg.swayam_mmr_num_vars,
                    oversold=_icfg.swayam_oversold, overbought=_icfg.swayam_overbought,
                    progress_cb=_swayam_progress,
                )
                n_eff = effective_member_count(swayam_view_dfs)
                st.session_state["swayam_n_eff"] = n_eff
                console.success(f"View bank: {len(swayam_view_dfs)} views · ~{n_eff:.1f} effective")

            if swayam_view_dfs:
                console.section("Aggregation")
                # Views are weighted by their own realised skill, estimated
                # causally (analytics.adaptive) — a timescale that has stopped
                # predicting this instrument contributes less to breadth, and
                # no grid had to be chosen for that to happen.
                _view_w = view_skill_weights(swayam_view_dfs, horizon=FWD_HORIZON)
                swayam_daily_pre_reindex = aggregate_views(swayam_view_dfs, weights=_view_w)
                if not _view_w.empty:
                    _last = _view_w.iloc[-1].sort_values(ascending=False)
                    console.item("View Weighting", f"skill-weighted · {len(_last)} views")
                    console.item("Top Views", " · ".join(f"{k} {v:.2f}" for k, v in _last.head(4).items()))
                    console.item("Weakest View", f"{_last.index[-1]} {_last.iloc[-1]:.2f}")

            st.session_state["_swayam_analysis_cache"] = {
                "key": _swayam_fetch_key,
                "swayam_view_dfs": swayam_view_dfs,
                "swayam_daily_pre_reindex": swayam_daily_pre_reindex,
                "n_eff": st.session_state.get("swayam_n_eff"),
            }

        # ── HORIZON-DEPENDENT tail: reindex onto the target's calendar ──────
        # Cheap (pure pandas, no yfinance) — re-runs since the horizon's
        # warm-up trim can shift the target's date spine.
        swayam_daily = swayam_daily_pre_reindex
        if not swayam_daily.empty:
            # Carry the basket forward onto the TARGET's trading calendar. The
            # views share the target's own calendar by construction, but the macro
            # driver pool behind MMR does not
            # than the target — on a Monday-morning IST run, or when the target's
            # market is open but the basket's is on holiday, the basket's last close
            # IS its current value. Reindexing it onto the target's dates (ff-fill)
            # lets the SIGNAL, cards and plots all reach the target's latest session
            # instead of truncating to the slowest constituent. We record the
            # basket's true last-native date so the UI can flag how much is carried
            # over (the partial-session notice covers the row-level staleness).
            st.session_state["swayam_native_last"] = pd.Timestamp(swayam_daily.index.max())
            if active_date in data.columns:
                _cal = pd.DatetimeIndex(sorted(pd.to_datetime(
                    data[active_date], errors="coerce").dropna().dt.normalize().unique()))
                _nd = swayam_daily.copy()
                _nd.index = pd.to_datetime(_nd.index).normalize()
                _nd = _nd[~_nd.index.duplicated(keep="last")].sort_index()
                # _Native marks rows that are a genuine basket observation
                # (present in _nd BEFORE the reindex) vs carried forward by
                # the ffill below (the basket's market was closed/hadn't
                # posted that day). Carried through so the calibration
                # overlap gate can require NATIVE overlap, not ffilled
                # rows masquerading as fresh Swayam signal (audit finding
                # F21) — the UI's own "breadth carried forward" notice
                # already discloses this to the user; the gate didn't.
                _native_dates = set(_nd.index)
                swayam_daily = _nd.reindex(_cal, method="ffill").dropna(how="all")
                swayam_daily["_Native"] = swayam_daily.index.isin(_native_dates)
            console.item("Trading Days", len(swayam_daily))
            if len(swayam_daily) > 0:
                last = swayam_daily.iloc[-1]
                console.item("Avg Signal", f"{last.get('Avg_Signal', 0):+.2f}")
                console.item("Oversold %", f"{last.get('Oversold_Pct', 0):.0f}%")
                console.item("Overbought %", f"{last.get('Overbought_Pct', 0):.0f}%")
                console.item("Buy Signals", int(last.get('Buy_Signals', 0)))
                console.item("Sell Signals", int(last.get('Sell_Signals', 0)))
            console.success(f"Swayam aggregation: {len(swayam_daily)} trading days")

        console.end_phase("SWAYAM ENGINE")
        progress_bar(progress_container, 75, "Swayam Engine Complete", f"{len(swayam_view_dfs)} Views · {len(swayam_daily)} Trading Days")

        # ── Phase 4: Convergence ──────────────────────────────────────────
        console.start_phase("CONVERGENCE", 4, 5)
        progress_bar(progress_container, 78, "Computing Convergence", "Cross-Validation · DDM Filtering")

        console.section("Cross-Validation Setup")
        # ── Convergence weights: prior, then learned forward ─────────────
        # The first pass builds the dim_* matrix with the PRIOR weights. The
        # composite is then recomputed (Phase 4b) with weights learned online
        # from resolved outcomes only, so no score depends on data that had
        # not happened when it was published.
        #
        # There is no profile to resolve and no mode to be in. The Optuna
        # search that used to run here fit the whole history at once and wrote
        # its winner to disk, which meant the output depended both on future
        # data and on when you last calibrated; both are gone.
        from convergence import intelligence as _intel_mod
        _prior_w = _icfg.weights_seed()
        console.item("Dimension weights", "prior → learned online (causal, no calibration step)")

        # First pass builds the dim_* sub-scores; the composite it also writes
        # is superseded below by the online-weighted recomputation.
        _validator_weights = _prior_w
        # The vote count is the Swayam bank's member count. Every view always
        # reports, so coverage reads 1.0 — unlike a basket, where a constituent
        # could simply be missing that day and breadth was read off a partial
        # cross-section without saying so.
        _expected_constituents = len(swayam_view_dfs) or None
        validator = CrossValidator(
            active_weights=_validator_weights,
            expected_constituents=_expected_constituents,
        )
        divergence_detector = CrossSystemDivergenceDetector()

        fvo_ts = engine.ts_data.copy()  # carries "Price" (set after fit)
        if active_date != "None" and active_date in data.columns:
            fvo_ts["Date"] = pd.to_datetime(data[active_date].values)
            fvo_ts = fvo_ts.set_index("Date")
        else:
            fvo_ts["Date"] = np.arange(len(fvo_ts))
        fvo_ts = fvo_ts[~fvo_ts.index.duplicated(keep="last")]
        console.item("FVO Dates", len(fvo_ts))

        swayam_by_date = {}
        if not swayam_daily.empty:
            swayam_unique = swayam_daily[~swayam_daily.index.duplicated(keep="last")]
            for idx in swayam_unique.index:
                key = str(idx.date()) if hasattr(idx, "date") else str(pd.Timestamp(idx).date())
                swayam_by_date[key] = swayam_unique.loc[idx]
            console.item("Swayam Dates", len(swayam_by_date))

        console.section("Daily Convergence Scoring")
        overlap_count = 0
        native_overlap_count = 0
        skipped_warmup = 0
        total_dates = len(fvo_ts.index)
        for i, ts_idx in enumerate(fvo_ts.index):
            ts_date = ts_idx.date() if hasattr(ts_idx, "date") else pd.Timestamp(ts_idx).date()
            date_str = str(ts_date)
            row_a = fvo_ts.loc[ts_idx]
            if isinstance(row_a, pd.DataFrame):
                row_a = row_a.iloc[-1]
            # Skip the engine's own [0, MIN_TRAIN_SIZE) warm-up rows — the
            # `Valid` column (engines/fvo.py) is False there because no
            # genuine walk-forward forecast covers them (see A3 in the audit).
            # Scoring them would feed the Intelligence calibration frame and
            # the walk-forward IC a fabricated "neutral" convergence reading
            # instead of genuinely excluding the unfit region.
            if not bool(row_a.get("Valid", True)):
                skipped_warmup += 1
                continue
            fvo_sig = {
                "conviction_score": float(row_a.get("ConvictionBounded", 0)),
                "oversold_breadth": float(row_a.get("OversoldBreadth", 50)),
                "regime": str(row_a.get("Regime", "NEUTRAL")),
            }
            if date_str in swayam_by_date:
                row_n = swayam_by_date[date_str]
                swayam_stats = {
                    "oversold_pct": float(row_n.get("Oversold_Pct", 50)),
                    "overbought_pct": float(row_n.get("Overbought_Pct", 50)),
                    "avg_unified_osc": float(row_n.get("Avg_Signal", 0)),
                    "regime_bull_pct": float(row_n.get("Regime_Bull_Pct", 33)),
                    "regime_bear_pct": float(row_n.get("Regime_Bear_Pct", 33)),
                    "regime_neutral": float(row_n.get("Regime_Neutral", 34)),
                    "num_constituents": int(row_n.get("Total_Analyzed", 0)),
                }
                overlap_count += 1
                if bool(row_n.get("_Native", True)):
                    native_overlap_count += 1
            else:
                swayam_stats = {
                    "oversold_pct": 50, "overbought_pct": 50, "avg_unified_osc": 0,
                    "regime_bull_pct": 33, "regime_bear_pct": 33,
                    "regime_neutral": 34, "num_constituents": 0,
                }
            validator.compute_convergence(fvo_sig, swayam_stats, date_str)
            divergence_detector.detect(fvo_sig, swayam_stats, date_str)

            if (i + 1) % 10 == 0 or i == total_dates - 1:
                pct_val = int(78 + (i + 1) / total_dates * 7)
                progress_bar(progress_container, pct_val, "Computing Convergence", f"{i + 1}/{total_dates} Dates Scored")

        console.item("Total FVO Dates", len(fvo_ts))
        console.item("Skipped (warm-up, no genuine forecast)", skipped_warmup)
        console.item("Overlap Dates", f"{overlap_count} ({native_overlap_count} native, "
                     f"{overlap_count - native_overlap_count} carried-forward)")
        console.success("Convergence scoring complete")

        # ── Online-weighting overlap gate ────────────────────────────────
        # Skip the online re-weighting when the FVO/Swayam overlap is too thin.
        # With no genuine overlap every date takes the same neutral swayam_stats
        # default, so the continuous consensus direction degenerates to
        # fvo_bull/2 and every Swayam-driven dim score is constant — the learner
        # would then be scoring what is really a half-weight FVO-only signal and
        # attributing its skill to dimensions that never varied.
        #
        # Gated on NATIVE overlap, not raw overlap (audit finding F21):
        # swayam_daily is forward-filled onto the target's calendar before the
        # scoring loop, so `overlap_count` alone would count a long run of
        # carried-forward, non-fresh breadth rows as new information each day.
        # 60 dates (~3 trading months) excludes the genuinely-degenerate case
        # without second-guessing a real but short history.
        _MIN_OVERLAP_FOR_LEARNING = 60
        _learn_weights = native_overlap_count >= _MIN_OVERLAP_FOR_LEARNING
        if not _learn_weights:
            console.warning(
                f"Online dimension weighting skipped: only {native_overlap_count} NATIVE "
                f"FVO/Swayam overlap dates (< {_MIN_OVERLAP_FOR_LEARNING}; "
                f"{overlap_count} total incl. carried-forward) — the dimensions would be "
                f"constant and their measured skill meaningless. Prior weights stand."
            )

        # ── 4a. First-pass conviction model ─────────────────────────────
        # First-pass DDM filter on the convergence_score from the first
        # validator pass. Labeled "first-pass" only when Intelligence Mode
        # is ON (a second pass will follow); just "Conviction Model" otherwise.
        _first_pass_label = "First-Pass Conviction Model" if _learn_weights else "Conviction Model"
        console.section("Conviction Model (initial pass)" if _learn_weights else "Conviction Model")
        progress_bar(progress_container, 83, _first_pass_label, "DDM Filter · Prior Weights")
        convergence_df = validator.get_convergence_series()
        # DDM smoothing = the shared consensus-filter tuning (CONV_DDM_*).
        conviction_model = UnifiedConvictionModel(
            leak_rate=_icfg.ddm_leak,
            drift_scale=_icfg.ddm_drift,
            long_run_var=_icfg.ddm_lrv,
        )
        results = conviction_model.fit(
            convergence_df["convergence_score"].tolist(),
            convergence_df.index.tolist(),
        )
        if results:
            latest = results[-1]
            _pre_label = "DDM Conviction (pre-weighting)" if _learn_weights else "DDM Conviction"
            _sig_label = "DDM Signal (pre-weighting)" if _learn_weights else "DDM Signal"
            console.item(_pre_label, f"{latest.nishkarsh_conviction:+.0f}")
            console.item(_sig_label, latest.nishkarsh_signal)
        console.success(f"Initial conviction: {len(results)} scores computed")

        # ── 4b. Learn the dimension weights, forward only ───────────────
        # Replaces the Optuna calibration that used to sit here. That search
        # fit the whole history and applied its winner back across the same
        # history, so every published score changed whenever a session was
        # added; it also persisted the winner to disk, making the output depend
        # on when it was last run. This learns the same thing — which
        # dimensions actually predict this instrument — from resolved outcomes
        # only, at a fraction of the cost and with no state carried between
        # runs.
        _weight_hist = pd.DataFrame()
        if _learn_weights:
            console.section("Online Dimension Weighting")
            progress_bar(progress_container, 86, "Learning Dimension Weights",
                         "Causal · discounted directional skill per dimension")
            try:
                convergence_df, _weight_hist = _intel_mod.apply_online_weights(
                    convergence_df, fvo_ts, horizon=FWD_HORIZON, target_col="Price",
                )
                if not _weight_hist.empty:
                    _wnow = _weight_hist.iloc[-1]
                    _wprior = _intel_mod.PRIOR_WEIGHTS
                    console.item("Horizon", f"{FWD_HORIZON}d (outcome resolves h days after the call)")
                    console.item("Learned Weights",
                                 " · ".join(f"{k} {v:.3f}" for k, v in _wnow.sort_values(ascending=False).items()))
                    console.item("Prior Weights",
                                 " · ".join(f"{k} {_wprior[k]:.2f}" for k in _wnow.index))
                    _moved = max(abs(_wnow[k] - _wprior[k]) for k in _wnow.index)
                    console.item("Max Shift From Prior", f"{_moved:+.3f}")
                    console.success(f"Dimension weights learned over {len(_weight_hist)} dates (no repaint)")
                else:
                    console.warning("Online weighting produced no coverage — prior weights stand")
            except Exception as _we:
                console.warning(f"Online weighting failed: {_we} — prior weights stand")

            # Re-fit the conviction model on the re-weighted composite.
            progress_bar(progress_container, 92, "Re-Fitting Conviction Model",
                         "DDM pass on the online-weighted composite")
            conviction_model = UnifiedConvictionModel(
                leak_rate=_icfg.ddm_leak, drift_scale=_icfg.ddm_drift, long_run_var=_icfg.ddm_lrv,
            )
            results = conviction_model.fit(
                convergence_df["convergence_score"].tolist(),
                convergence_df.index.tolist(),
            )
            console.section("Conviction Model (re-weighted)")
            if results:
                latest = results[-1]
                console.item("DDM Conviction", f"{latest.nishkarsh_conviction:+.0f}")
                console.item("DDM Signal", latest.nishkarsh_signal)
                console.item("DDM Band", f"[{latest.confidence_lower:.0f}, {latest.confidence_upper:.0f}]")
            console.success("Re-fit complete on online-weighted convergence")

        _active_w = (dict(_weight_hist.iloc[-1]) if not _weight_hist.empty
                     else _icfg.weights_seed())
        _active_t = _icfg.composite_thresholds()

        # Publish to session state so the Passport sidebar + Convergence cards
        # see the calibrated state immediately on the next rerun.
        st.session_state["intelligence_active_weights"] = _active_w
        st.session_state["intelligence_active_thresholds"] = _active_t
        st.session_state["intelligence_active_profile"] = (
            {"weights": _active_w, "learned": bool(_learn_weights)}
        )

        # ── 4d. NORMALIZED CONSENSUS ─────────────────────────────────────────
        # The causal expanding-z average of FVO's ConvictionRaw and Swayam's
        # Avg_Signal. It headlined the hero card until the card became a
        # conviction chain; it is now an INPUT to that chain's convergence
        # gate (a consensus pointing against the FVO call shuts the gate), and
        # the top row of the Unified Signal plot. `consensus_series` is the
        # single source for the full history; the dict is its last point.
        from convergence.normalization import (
            compute_normalized_convergence, consensus_series, classify_convergence_score,
        )
        _consensus_full = consensus_series(fvo_ts, swayam_daily)
        _nishkarsh_norm = compute_normalized_convergence(fvo_ts, swayam_daily)
        if _nishkarsh_norm:
            console.section("Normalized Consensus (headline)")
            console.item("Conviction", f"{_nishkarsh_norm['value']:+.2f}")
            console.item("Signal", _nishkarsh_norm['signal'])
            console.item("  FVO contribution", f"{_nishkarsh_norm['fvo_norm']:+.2f}")
            console.item("  Swayam contribution",  f"{_nishkarsh_norm['swayam_norm']:+.2f}")

        # ── 4e. WEIGHTED COMPOSITE — the second construction ────────────────
        # convergence_score (post online re-weighting, ±100 scale) IS the exact
        # quantity the learner scores, so this is the one place the learned
        # weights are semantically valid (audit findings F1/F2). It feeds the
        # hero's WEIGHTED evidence row — a genuine second opinion, being a
        # dimension-weighted construction of the same two engines rather than
        # their plain 50/50 mean. The
        # RAW factory-weight composite is no longer surfaced in the UI at all
        # — it remains the research baseline in
        # research/calibration_lift_study.py (raw-vs-calibrated ablation).
        _calibrated_score = float(convergence_df["convergence_score"].iloc[-1]) if not convergence_df.empty else 0.0
        _calibrated_signal = classify_convergence_score(_calibrated_score, _active_t)
        console.section("Calibrated Signal (evidence)")
        console.item("Score", f"{_calibrated_score:+.1f}")
        console.item("Signal", _calibrated_signal)

        console.section("Divergence Detection")
        progress_bar(progress_container, 93, "Detecting Divergences", "Cross-System Disagreement Analysis")
        events = divergence_detector.get_events()
        console.item("Total Events", len(events))
        if not events.empty:
            event_types = events['divergence_type'].value_counts()
            for etype, count in event_types.items():
                console.item(f"  {etype}", count)
        console.success("Divergence analysis complete")

        # ── Walk-Forward Validation (durability check, runs every analysis) ──
        # Re-calibrates on each expanding window and scores IC on the next
        # unseen block. Many genuine OOS grades → distinguishes a durable edge
        # from a lucky recent regime. Results power the Diagnostics tab.
        console.section("Walk-Forward Validation")
        progress_bar(progress_container, 94, "Walk-Forward Validation", "Rolling OOS IC · Re-Calibration")
        try:
            _hold_grid = _icfg.hold_horizons  # IC durability at this instrument's forecast horizons
            _wf_frame = _intel_mod._build_calibration_frame(
                convergence_df, fvo_ts, target_col="Price", horizons=_hold_grid,
            )
            _wf_results = _intel_mod.walk_forward_ic(_wf_frame, horizons=_hold_grid)
            st.session_state["wf_results"] = _wf_results
            _wf_ics = [r["ic"] for r in _wf_results if r["ic"] == r["ic"]]  # drop NaN
            if _wf_ics:
                _wf_mean = sum(_wf_ics) / len(_wf_ics)
                _wf_pos = sum(1 for v in _wf_ics if v > 0)
                console.item("Windows", len(_wf_ics))
                console.item("Mean OOS IC", f"{_wf_mean:+.3f}")
                console.item("Positive", f"{_wf_pos}/{len(_wf_ics)}")
                console.success(f"Walk-forward: mean OOS IC {_wf_mean:+.3f} ({_wf_pos}/{len(_wf_ics)} +ve)")
            else:
                console.warning("Walk-forward produced no scorable windows")
        except Exception as _wf_e:
            st.session_state["wf_results"] = None
            console.warning(f"Walk-forward validation skipped: {_wf_e}")

        console.end_phase("CONVERGENCE")
        _conv_complete_sub = (
            f"{overlap_count} Overlap Dates · {len(events)} Divergence Events · "
            f"{'Online-Weighted' if _learn_weights else 'Prior Weights'}"
        )
        progress_bar(progress_container, 95, "Convergence Phase Complete", _conv_complete_sub)

        # ── Phase 5: Final Assembly ───────────────────────────────────────
        console.start_phase("FINAL ASSEMBLY", 5, 5)
        progress_bar(progress_container, 96, "Storing Results", "Session State · Cache")
        console.section("Session State")

        st.session_state["engine"] = engine
        st.session_state["engine_cache"] = cache_key
        st.session_state["fvo_ts"] = fvo_ts
        st.session_state["swayam_daily"] = swayam_daily
        st.session_state["swayam_view_dfs"] = swayam_view_dfs
        st.session_state["convergence_df"] = convergence_df
        st.session_state["divergence_events"] = events
        st.session_state["nishkarsh_result"] = results[-1] if results else None
        st.session_state["last_agreement"] = convergence_df["agreement_ratio"].iloc[-1] if not convergence_df.empty else 0
        # The weighted composite — the hero's WEIGHTED evidence row and the
        # Unified Signal plot's amber overlay.
        st.session_state["nishkarsh_calibrated_score"] = _calibrated_score
        st.session_state["nishkarsh_calibrated_signal"] = _calibrated_signal

        # `hero_series` — the normalized consensus's full history (from
        # consensus_series, the single source shared with the Unified Signal
        # plot's top row). It is no longer a "hero" series in the sense of
        # being what the card plots: the card has no plot. It survives because
        # the Convergence tab's marker tiers are causal quantiles of THIS
        # distribution (see _series_tier there), which needs the history.
        #
        # `hero_smoothed` (a DDM of the same consensus) went with the TREND
        # evidence row it existed to feed.
        st.session_state["hero_series"] = (
            _consensus_full["Consensus"].rename("HeroConsensus")
            if not _consensus_full.empty else None)

        if results:
            _ccs_index = pd.to_datetime(convergence_df.index, errors="coerce")
            st.session_state["calibrated_conv_series"] = pd.Series(
                [r.nishkarsh_conviction / 100.0 for r in results],
                index=_ccs_index, name="CalibratedConvergence",
            )
        else:
            st.session_state["calibrated_conv_series"] = None

        # THE HEADLINE: the normalized consensus dict (value + signal +
        # per-engine contributions) — single source of truth from
        # convergence/normalization.py, shared verbatim with the TATTVA
        # CONVICTION card and the Unified Signal plot's top row.
        st.session_state["nishkarsh_conv_normalized"] = _nishkarsh_norm

        # Display signal = what the UI cards show: the consensus headline,
        # then the DDM signal as a last resort.
        display_signal = (
            _nishkarsh_norm["signal"] if _nishkarsh_norm
            else (results[-1].nishkarsh_signal if results else "N/A")
        )

        console.item("FVO Engine", "✅ Cached")
        console.item("Swayam Daily", f"✅ {len(swayam_daily)} rows")
        console.item("View Bank", f"✅ {len(swayam_view_dfs)} views")
        console.item("Convergence DF", f"✅ {len(convergence_df)} rows")
        console.item("Convergence Result", f"✅ {display_signal}")

        console.end_phase("FINAL ASSEMBLY")

        console.summary("RUN SUMMARY", {
            "Total Phases": "5/5 complete",
            "FVO Rows": len(engine.ts_data),
            "Swayam Views": len(swayam_view_dfs),
            "Swayam Trading Days": len(swayam_daily),
            "Convergence Scores": len(convergence_df),
            "Overlap Dates": overlap_count,
            "Divergence Events": len(events),
            "Status": "SUCCESS",
        })

        console.line('═', 70)
        console._write(f"  {Colors.BOLD}{Colors.GREEN}Analysis Complete{Colors.RESET}")
        console.line('═', 70)
        console._write()

        # Snapshot this config's full result into the bounded per-config cache
        # so revisiting it later (commodity switch-back, predictor toggle-back)
        # restores instantly. LRU-evict to keep memory bounded.
        _rcache = st.session_state.setdefault("results_cache", {})
        _rcache.pop(cache_key, None)
        _bundle_snapshot = {bk: st.session_state.get(bk) for bk in _BUNDLE_KEYS}
        # Trim large baskets' per-constituent frames before they enter the LRU
        # (audit finding F19) — see _bundle_swayam_view_dfs's docstring.
        _bundle_snapshot["swayam_view_dfs"] = _bundle_swayam_view_dfs(
            _bundle_snapshot.get("swayam_view_dfs") or {}
        )
        _rcache[cache_key] = _bundle_snapshot
        while len(_rcache) > _RESULTS_CACHE_MAX:
            _rcache.pop(next(iter(_rcache)))

        progress_bar(progress_container, 100, "Analysis Complete", f"Convergence: {display_signal}")
        time.sleep(0.25)
        progress_container.empty()
        st.session_state["run_requested"] = True
        st.rerun()

    engine: FairValueEngine = st.session_state["engine"]
    signal = engine.get_current_signal()
    model_stats = engine.get_model_stats()
    regime_stats = engine.get_regime_stats()
    ts = engine.ts_data.copy()
    if active_date != "None" and active_date in data.columns:
        ts["Date"] = pd.to_datetime(data[active_date].values)
    else:
        ts["Date"] = np.arange(len(ts))
    if "fvo_ts" not in st.session_state:
        st.session_state["fvo_ts"] = ts.copy()

    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    agreement = st.session_state.get("last_agreement", 0)

    # ─── Precedent base rate for the hero (co-equal second opinion) ────────
    # A 33-target non-overlapping study (hero_study.py) found the analog precedent
    # is a STRONGER directional read than the convergence signal, and adds genuine,
    # independent value — while the plot markers add nothing (they ARE the
    # convergence's own inputs). So the hero reads the precedent alongside its
    # signal: agreement raises confidence, disagreement is flagged as a divergence.
    # NOTE: the specific quoted numbers in the original study (IC +0.226 vs +0.158)
    # were measured under a since-changed analog config (the old .55/.35/.10
    # Mahalanobis/trajectory/recency blend, not the current pure-Mahalanobis 1/0/0 —
    # see analytics.analogs.ANALOG_W_*) and with a look-ahead full-sample
    # normalization the live tab avoids (causal expanding z-scores). Re-run
    # hero_study.py with the shipped config before quoting a specific number again;
    # the qualitative conclusion (precedent >= convergence, markers add nothing) is
    # the part that's load-bearing here, not the exact ICs.
    # Content-aware key (not just row count): include the latest Price so an intraday
    # refresh that updates the last bar without adding a row still recomputes.
    #
    # Computed ONCE here over the FIXED precedent term structure
    # (core.config.PRECEDENT_HORIZONS = 1/3/5/10/20/60d) and cached as the raw
    # analog list — the tab (ui/tabs/tab_precedent.py) previously called
    # find_similar_periods AGAIN with its own hold_horizons on every render,
    # re-running the expensive part (feature-frame build incl. rolling Hurst,
    # Mahalanobis distance, Theiler selection) a second time for the same
    # ts/target/mom_window (audit finding F18). summarize_forward is cheap
    # (pure aggregation), so both the hero's single-horizon read and the tab's
    # per-horizon cards derive from this one cached analog list.
    #
    # The analog STATE features use this instrument's forecast-momentum window;
    # the term structure is this instrument's precedent_horizons span.
    _plast = float(ts["Price"].iloc[-1]) if "Price" in ts.columns and len(ts) else 0.0
    _pkey = f"{active_target}|{len(ts)}|{_plast:.6g}"
    if st.session_state.get("_prec_key") != _pkey:   # recompute only when inputs change
        _prec_summary = None
        _cached_analogs: list = []
        _cached_display_hold: tuple = ()
        try:
            from analytics.analogs import find_similar_periods as _fsp, summarize_forward as _sf
            _display_hold = _icfg.precedent_horizons
            _analogs = _fsp(ts, active_target, hold_horizons=_display_hold,
                            mom_window=_icfg.analog_mom_window)
            _cached_analogs = _analogs
            _cached_display_hold = _display_hold
            _ps = _sf(_analogs, _display_hold) if _analogs else {}
            # Hero precedent second-opinion reads at this instrument's forecast
            # horizon (a member of its precedent_horizons); fall back to the
            # nearest available shorter horizon if it is ever absent.
            _hp = int(_icfg.forecast_horizon)
            if _hp not in _ps:
                _cands = [h for h in _display_hold if h <= _hp]
                _hp = max(_cands) if _cands else min(_display_hold)
            _row = _ps.get(int(_hp))
            if _row:
                _med = _row["median"]
                _prec_summary = {
                    "horizon": int(_hp), "median": float(_med),
                    "positive_pct": float(_row["positive_pct"]),
                    # n_eff, not n, is what the hero's precedent gate reads:
                    # ten analogs carried by one episode is one observation.
                    "n": int(round(_row.get("n_eff", _row["n"]))),
                    "n_raw": int(_row["n"]),
                    "n_eff": float(_row.get("n_eff", _row["n"])),
                    "p25": float(_row.get("p25", float("nan"))),
                    "p75": float(_row.get("p75", float("nan"))),
                    "usable": bool(_row.get("usable", True)),
                    "note": str(_row.get("note", "")),
                    "dir": 1 if _med > 0 else -1 if _med < 0 else 0,
                }
        except Exception:
            _prec_summary = None
        st.session_state["precedent_summary"] = _prec_summary
        st.session_state["_precedent_analogs_cache"] = {
            "pkey": _pkey, "periods": _cached_analogs, "display_hold": _cached_display_hold,
        }
        st.session_state["_prec_key"] = _pkey

    # ─── Masthead + tape, then the verdict (above tabs, always visible) ────
    # The tape reads from `data` — the same panel the valuation engine was fit
    # on this run — so the ambient context above the card cannot disagree with
    # the card underneath it.
    _render_header(frame=data)
    _render_primary_signal(nishkarsh_norm, agreement, signal)

    # ─── Sidebar Discovery Hint (passive — the sidebar collapse control lives
    # in Streamlit's own chrome; this is a directional pointer, not a button) ──
    st.markdown(
        """
        <div class="sidebar-hint">
            <svg class="sidebar-hint-arrow" width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">
                <polyline points="15 18 9 12 15 6"></polyline>
            </svg>
            <span class="sidebar-hint-label">CONFIGURE</span>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # ─── Timeframe Filter — with robust persistence ───────────────────────
    if 'tf_selected' not in st.session_state:
        st.session_state.tf_selected = '6M'
    # Derived from TIMEFRAME_TRADING_DAYS (core/config.py) rather than a
    # second hard-coded {3M:63, 6M:126, ...} literal — the two used to drift
    # independently with no shared source (audit finding F15).
    TIMEFRAMES = {**TIMEFRAME_TRADING_DAYS, 'ALL': None}

    tf_cols = st.columns(len(TIMEFRAMES), gap="small")
    for i, tf in enumerate(TIMEFRAMES.keys()):
        with tf_cols[i]:
            btn_type = "primary" if st.session_state.tf_selected == tf else "secondary"
            if st.button(tf, key=f"tf_{tf}", type=btn_type, width='stretch'):
                st.session_state.tf_selected = tf
                st.rerun()
    selected_tf = st.session_state.tf_selected

    # Ensure timeframe survives config changes by always applying it
    ts_filtered = ts.copy()
    if selected_tf != "ALL":
        if active_date != "None" and pd.api.types.is_datetime64_any_dtype(ts["Date"]):
            from pandas import DateOffset
            max_date = ts["Date"].max()
            offsets = {"3M": DateOffset(months=3), "6M": DateOffset(months=6), "1Y": DateOffset(years=1), "2Y": DateOffset(years=2)}
            cutoff = max_date - offsets.get(selected_tf, DateOffset(years=1))
            ts_filtered = ts[ts["Date"] >= cutoff]
        else:
            n_days = TIMEFRAME_TRADING_DAYS.get(selected_tf, 252)
            ts_filtered = ts.iloc[max(0, len(ts) - n_days):]
    x_axis = ts_filtered["Date"]
    x_title = "Date" if active_date != "None" else "Index"

    # ─── Tabs with Error Boundaries ─────────────────────────────────────────
    # Streamlit renders every tab's content on each script run (there is no
    # built-in lazy-loading of inactive tabs) — the CSS just hides the
    # inactive panels. A `rendered_tabs` session-state set was previously
    # written here on every render but never read anywhere, under a comment
    # claiming lazy loading that isn't actually happening (audit finding C5).
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "CONVERGENCE", "FVO", "SWAYAM", "PRECEDENT", "DIAGNOSTICS", "DATA",
    ])

    # Error boundary wrapper
    def _safe_render(name, render_fn):
        """Render a tab with graceful error handling."""
        try:
            render_fn()
        except Exception as e:
            st.markdown(
                f'<div class="warning-box">'
                f'<div class="interp-title">Error in {html.escape(name)}</div>'
                f'<div class="interp-body">{html.escape(str(e))}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    with tab1:
        _safe_render("Convergence", lambda: render_convergence_tab(ts_filtered))
    with tab2:
        _safe_render("FVO", lambda: render_fvo_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats, regime_stats, ts, active_target))
    with tab3:
        _safe_render("Swayam", lambda: render_swayam_tab(selected_tf=selected_tf))
    # Reuse the analog list already computed above (Precedent base-rate for the
    # hero) instead of having the tab call find_similar_periods a second time
    # for the same (ts, target, mom_window) — audit finding F18. Guarded on
    # the pkey matching THIS render's ts/target/horizon; a mismatch (shouldn't
    # happen since the precompute above always runs first) falls back to None,
    # and the tab recomputes itself exactly as before.
    _prec_cache = st.session_state.get("_precedent_analogs_cache")
    _cached_periods = (
        _prec_cache["periods"] if _prec_cache and _prec_cache.get("pkey") == _pkey else None
    )
    with tab4:
        # Precedent term structure + momentum/horizon come from this instrument's
        # own config (precedent_horizons / analog_mom_window / forecast_horizon).
        _safe_render("Precedent", lambda: render_precedent_tab(
            ts, active_target, _icfg.precedent_horizons, _icfg.analog_mom_window, _icfg.forecast_horizon,
            precomputed_periods=_cached_periods))
    with tab5:
        _safe_render("Diagnostics", lambda: render_diagnostics_tab(engine, ts_filtered, x_axis, x_title, signal, model_stats))
    with tab6:
        _safe_render("Data", lambda: render_data_tab(ts_filtered, ts, active_target))

    _render_footer()


if __name__ == "__main__":
    main()
