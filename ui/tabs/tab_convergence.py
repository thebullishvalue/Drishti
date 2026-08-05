"""
Tattva — Convergence tab: where the two engines agree, and where they do not.

The headline object of the whole app: the normalized consensus of FVO's
valuation conviction and Swayam's breadth, plus the divergences worth knowing
about when they disagree.

Reading order — the house convention every analysis tab follows:

  1 TRUST     can this reading be believed?      Agreement ratio + coverage
  2 ANCHOR    what is the underlying claim?      Convergence analysis cards
  3 SIGNAL    what does it say to do?            Hero signal history
  4 STATE     how does that sit historically?    Unified Signal plot
  5 DETAIL    the evidence behind it             Recent divergences
"""

from __future__ import annotations

import streamlit as st

from analytics.adaptive import tier_now
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from ui.theme import (chart_layout, style_axes,
                      chart_color, chart_rgba, grid_rgba)
from ui.components import (render_metric_card, render_section_header, section_gap,
                           render_info_box, render_empty_state,
                           render_chart_panel, render_note)
from convergence.normalization import (
    align_fvo_swayam,
    causal_normalize,
)
from core.config import (
    get_instrument_config, InstrumentConfig,  # per-instrument marker/tier anchors
    # Marker/tier constants are NOT imported here — they are resolved per-instrument
    # off get_instrument_config(active_target) at render time (see below).
    UI_CHART_HEIGHT_STACKED,
)

# (Tab-local colour aliases stood here as module-level constants. They were
# evaluated ONCE at import, when there is no session to read a theme from,
# so every chart drawn through them was frozen to whichever theme happened
# to be active at first import — the same import-time binding that made the
# original COLOR_* constants unable to follow Paper mode. Colours are
# resolved at the call site now, per render.

# ── Tooltip definitions ────────────────────────────────────────────────────
TOOLTIPS = {
    "nishkarsh_conviction": (
        "Composite score combining FVO (top-down) and Swayam (bottom-up) into a single "
        "signal. Near 0 = both systems uncertain — avoid new positions. Large absolute values "
        "= high-conviction opportunities."
    ),
    "fvo_conviction": (
        "FVO's fair-value breadth: how many lookback windows see the market as overbought "
        "vs. oversold. Below -20 = most stocks cheap (bullish); above +20 = most expensive (bearish)."
    ),
    "swayam_avg": (
        "Average technical signal across the Swayam bottom-up units — basket constituents, or "
        "self-ensemble views of the instrument's own price (Swayam self mode). Negative = net "
        "bullish; positive = net bearish. Moves slowly and confirms (or contradicts) FVO's "
        "top-down view."
    ),
    "agreement": (
        "How often FVO and Swayam point in the same direction. Above 70% = both systems "
        "agree — trust the signal. Below 50% = they disagree — stay flat until alignment improves."
    ),
}


def _dynamic_range(vals, padding=0.15):
    """Compute a padded y-axis range from a list of values."""
    valid = [v for v in vals if v is not None and not np.isnan(v)]
    if not valid:
        return (-1, 1)
    mn, mx = min(valid), max(valid)
    span = mx - mn if mx != mn else 1.0
    pad = span * padding
    return (round(mn - pad, 2), round(mx + pad, 2))


def render_convergence_tab(ts_filtered=None):
    """Render the convergence dashboard tab with amber-gold system identity."""
    # System identity background
    st.markdown(
        '<div class="tab-bg convergence"></div>',
        unsafe_allow_html=True,
    )

    convergence_df = st.session_state.get("convergence_df")
    nishkarsh_norm = st.session_state.get("nishkarsh_conv_normalized")
    fvo_ts = st.session_state.get("fvo_ts")
    swayam_daily = st.session_state.get("swayam_daily")
    # Swayam's bottom-up units are basket CONSTITUENTS in basket mode and
    # self-ensemble VIEWS in Swayam self mode — keep copy accurate for both.
    _self_mode = st.session_state.get("swayam_mode") == "self"
    _units = "views" if _self_mode else "constituents"

    # ── Per-instrument marker / tier anchors ────────────────────────────────
    # Marker tiers for THIS render. The config values are warm-up priors; once
    # a series has a year of its own history the tier is its causal p90/p75 —
    # so the "extreme" line on each row means extreme *for this instrument*
    # rather than extreme for the pooled universe the constants were anchored
    # to. Shadow the module-global names for the rest of this render.
    try:
        _icfg = get_instrument_config(st.session_state.get("active_target", ""))
    except KeyError:
        _icfg = InstrumentConfig()

    def _tier(series, prior, q):
        """Causal tier for one marker row, or the prior while history is short."""
        if series is None or not len(series):
            return float(prior)
        return tier_now(np.asarray(series, dtype=float), float(prior), q=q)

    _hero = st.session_state.get("hero_series")
    _cons_hist = _hero.to_numpy(dtype=float) if _hero is not None and len(_hero) else None
    _fvo_ts = st.session_state.get("fvo_ts")
    _conv_raw = (_fvo_ts["ConvictionRaw"].to_numpy(dtype=float)
                 if _fvo_ts is not None and "ConvictionRaw" in _fvo_ts.columns else None)
    _swayam_d = st.session_state.get("swayam_daily")
    _swayam_avg = (_swayam_d["Avg_Signal"].to_numpy(dtype=float)
                   if _swayam_d is not None and "Avg_Signal" in getattr(_swayam_d, "columns", ())
                   else None)
    _agree = st.session_state.get("convergence_df")
    _agree = (_agree["agreement_ratio"].to_numpy(dtype=float)
              if _agree is not None and "agreement_ratio" in getattr(_agree, "columns", ()) else None)

    UI_CONSENSUS_STRONG = _tier(_cons_hist, _icfg.ui_consensus_strong, 0.90)
    UI_CONSENSUS_MODERATE = _tier(_cons_hist, _icfg.ui_consensus_moderate, 0.75)
    UI_CONVRAW_STRONG = _tier(_conv_raw, _icfg.ui_convraw_strong, 0.90)
    UI_CONVRAW_MODERATE = _tier(_conv_raw, _icfg.ui_convraw_moderate, 0.75)
    UI_SWAYAM_AVG_THRESHOLD = _tier(_swayam_avg, _icfg.ui_swayam_avg_threshold, 0.75)
    # Agreement is a [0,1] ratio, not a signed oscillator — its tiers are
    # quantiles of the ratio itself, so |x| would be a no-op and the p-levels
    # read directly.
    UI_AGREEMENT_STRONG = _tier(_agree, _icfg.ui_agreement_strong, 0.90)
    UI_AGREEMENT_MODERATE = _tier(_agree, _icfg.ui_agreement_moderate, 0.75)
    CONVICTION_MODERATE = _tier(_conv_raw, _icfg.conviction_moderate, 0.75)
    UI_SWAYAM_BULLISH = -UI_SWAYAM_AVG_THRESHOLD
    UI_SWAYAM_BEARISH = UI_SWAYAM_AVG_THRESHOLD

    if convergence_df is None or convergence_df.empty:
        render_empty_state(
            "No convergence data available",
            "Convergence fuses FVO's valuation read with Swayam's breadth — run an "
            "analysis first so both engines have something to agree or disagree over.",
            eyebrow="Convergence",
            action_label="Run analysis in the sidebar, then return to this page.",
        )
        return

    # ── SINGLE SOURCE OF TRUTH ───────────────────────────────────────────────
    # Align FVO + Swayam ONCE, here, before anything renders. The metric cards
    # AND the 3-row plot below both read these exact arrays, so a card can never
    # disagree with the plot point it mirrors (the old bug: card read the raw last
    # ts row, the plot read the Swayam-aligned last row → drift on calendar gaps).
    if ts_filtered is not None and not ts_filtered.empty:
        if "Date" in ts_filtered.columns:
            filtered_dates = set(pd.to_datetime(ts_filtered["Date"]).dt.date.astype(str))
        else:
            filtered_dates = set(ts_filtered.index.astype(str))
    else:
        filtered_dates = None

    aligned_dates, aligned_fvo_raw, aligned_swayam_raw = align_fvo_swayam(
        fvo_ts, swayam_daily, filter_dates=filtered_dates,
    )
    has_overlap = bool(aligned_dates)

    norm_a = norm_n = norm_avg = np.array([], dtype=np.float64)
    aligned_conv_raw: list = []
    if has_overlap:
        # Key by the full engine config (target + features + horizon + date range) so
        # switching predictor sets with the same target never reuses stale z-scores.
        # Also fold in content (row count + latest raw FVO/Swayam reading): a
        # "Refresh Data" that updates the LAST bar's value without changing the
        # date-range fingerprint that engine_cache is built from would otherwise
        # keep this key unchanged and silently reuse pre-refresh z-scores against
        # the post-refresh raw series (audit finding C1).
        _last_a = aligned_fvo_raw[-1] if aligned_fvo_raw else 0.0
        _last_n = aligned_swayam_raw[-1] if aligned_swayam_raw else 0.0
        _np_key = (
            f"conv_norm_causal::{st.session_state.get('engine_cache', st.session_state.get('active_target', ''))}"
            f"|{len(aligned_dates)}|{_last_a:.6g}|{_last_n:.6g}"
        )
        if _np_key not in st.session_state:
            # Compute per-date CAUSAL expanding-window z-scores over the FULL aligned
            # series.  Applying terminal-point μ/σ to a historical slice is look-ahead
            # bias: earlier bars appear less extreme than they were at the time because
            # σ is estimated from data that didn't yet exist.
            # causal_normalize is the SAME transform convergence.normalization's
            # compute_normalized_convergence uses (audit finding F16) — a
            # hand-duplicated copy here previously had to be kept in sync by
            # inspection for this plot to match the Convergence-tab cards.
            _full_dates, full_a, full_n = align_fvo_swayam(fvo_ts, swayam_daily)
            fa = np.array(full_a, dtype=np.float64)
            fn = np.array(full_n, dtype=np.float64)
            na_full = causal_normalize(fa)
            nn_full = causal_normalize(fn)
            def _dk(d):
                return str(d.date()) if hasattr(d, "date") else str(d)
            _p = {
                "a": {_dk(d): v for d, v in zip(_full_dates, na_full)},
                "n": {_dk(d): v for d, v in zip(_full_dates, nn_full)},
                "_n": len(full_a),
            }
            st.session_state[_np_key] = _p
        params = st.session_state[_np_key]
        def _dk(d):
            return str(d.date()) if hasattr(d, "date") else str(d)
        norm_a = np.array([params["a"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_n = np.array([params["n"].get(_dk(d), 0.0) for d in aligned_dates])
        norm_avg = (norm_a + norm_n) / 2.0
        at_dedup = fvo_ts[~fvo_ts.index.duplicated(keep="last")] if fvo_ts is not None else None
        for d in aligned_dates:
            d_str = str(d.date()) if hasattr(d, "date") else str(d)
            val = None
            if at_dedup is not None:
                if d in at_dedup.index:
                    val = float(at_dedup.loc[d]["ConvictionRaw"])
                elif "Date" in at_dedup.columns:
                    mask = at_dedup["Date"].astype(str).str.contains(d_str)
                    if mask.any():
                        val = float(at_dedup.loc[mask, "ConvictionRaw"].iloc[0])
            aligned_conv_raw.append(val)

    # System identity background

    # ═══════════════════════════════════════════════════════════════════════
    # HEADER + METRIC CARDS
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Convergence Analysis",
        "FVO top-down vs Swayam bottom-up. Agreement = reliable signal. Divergence = stand aside.",
        icon="target",
    )

    col1, col2, col3, col4 = st.columns(4, gap="small")

    with col1:
        # Mirrors Row 1 of the Unified Signal plot: average of normalized FVO
        # + Swayam z-scores, in [-1, +1].
        if nishkarsh_norm:
            score = nishkarsh_norm["value"]
            sig = nishkarsh_norm["signal"]
            color = "success" if "BUY" in sig else "danger" if "SELL" in sig else "neutral"
            render_metric_card("TATTVA CONVICTION", f"{score:+.2f}", sig, color, tooltip=TOOLTIPS["nishkarsh_conviction"])
        else:
            render_metric_card("TATTVA CONVICTION", "N/A", "Not computed", "neutral")

    with col2:
        # Mirrors Row 2 of the plot — reads the SAME aligned ConvictionRaw last point
        # (falls back to the raw last ts row only when there is no Swayam overlap).
        a_conv = None
        if has_overlap and aligned_conv_raw and aligned_conv_raw[-1] is not None:
            a_conv = aligned_conv_raw[-1]
        elif fvo_ts is not None and "ConvictionRaw" in fvo_ts.columns:
            a_conv = float(fvo_ts["ConvictionRaw"].iloc[-1])
        if a_conv is not None:
            render_metric_card("FVO CONVICTION", f"{a_conv:+.2f}", "Market breadth: oversold vs overbought",
                               "success" if a_conv < -CONVICTION_MODERATE else "danger" if a_conv > CONVICTION_MODERATE else "neutral",
                               tooltip=TOOLTIPS["fvo_conviction"])
        else:
            render_metric_card("FVO CONVICTION", "N/A", "", "neutral")

    with col3:
        # Mirrors Row 3 of the plot — reads the SAME aligned Swayam Avg Signal point.
        if has_overlap and len(aligned_swayam_raw):
            n_avg = float(aligned_swayam_raw[-1])
            render_metric_card("SWAYAM AVG SIGNAL", f"{n_avg:.2f}", f"Bottom-up {_units[:-1]} momentum",
                               "success" if n_avg < UI_SWAYAM_BULLISH else "danger" if n_avg > UI_SWAYAM_BEARISH else "neutral",
                               tooltip=TOOLTIPS["swayam_avg"])
        else:
            render_metric_card("SWAYAM AVG SIGNAL", "N/A", f"No {_units[:-1]} data", "neutral")

    with col4:
        agreement = convergence_df["agreement_ratio"].iloc[-1]
        render_metric_card("AGREEMENT", f"{agreement:.0%}", "FVO and Swayam alignment",
                           "success" if agreement > UI_AGREEMENT_STRONG else "warning" if agreement > UI_AGREEMENT_MODERATE else "neutral",
                           tooltip=TOOLTIPS["agreement"])

    section_gap()

    # ═══════════════════════════════════════════════════════════════════════
    # UNIFIED NORMALIZED SIGNAL — 3-row stacked chart
    # ═══════════════════════════════════════════════════════════════════════
    render_section_header(
        "Unified Signal — Normalized Convergence",
        "Z-scored to [−1, 1]. Combined signal (top) decomposed into constituent inputs (below).",
        icon="layers",
        accent="cyan",
    )

    # Aligned series already computed once at the top (single source of truth with
    # the metric cards). FVO-only targets (no Swayam basket) have no overlap →
    # the cards above still rendered; the plot just can't be drawn.
    if not has_overlap:
        render_empty_state(
            "No engine overlap",
            "FVO and Swayam share no dates for this target, so the consensus overlay "
            "cannot be drawn. The metric cards above are FVO-only reads and remain valid.",
            eyebrow="Convergence",
            action_label="Pick a target with a resolvable Swayam view bank.",
        )
        return

    # Short-history guard: z-scoring needs a stable σ. When the FULL FVO∩Swayam
    # overlap is tiny (brand-new sheet target, freshly-listed basket constituents),
    # σ collapses to its 1e-10 floor and the whole normalized plot flat-lines at 0 —
    # which misreads as a confident "neutral". The cards above already show the raw
    # latest reads honestly; here we suppress the misleading plot and say why.
    MIN_CONV_NORM_POINTS = 10
    _n_full = int(params.get("_n", len(aligned_dates)))
    if _n_full < MIN_CONV_NORM_POINTS:
        render_info_box(
            "Building convergence history",
            f"Only {_n_full} overlapping session{'s' if _n_full != 1 else ''} between FVO and Swayam "
            f"so far — too few to z-score into a stable convergence view (the plot would flat-line at zero "
            f"and misread as neutral). The cards above reflect the latest raw reads; this view populates "
            f"once {MIN_CONV_NORM_POINTS}+ shared sessions accrue.",
            color="cyan",
        )
        return

    # Honesty for the carry-forward: when the bottom-up source's native data ends
    # before the latest plotted session (its market(s) closed / haven't posted),
    # say so — those trailing breadth points are carried forward, provisional.
    # In self mode the "source" is the instrument's own OHLCV (self-ensemble
    # views), not a constituent basket — keep the copy accurate.
    _nn_last = st.session_state.get("swayam_native_last")
    _plot_last = aligned_dates[-1] if aligned_dates else None
    try:
        if _nn_last is not None and _plot_last is not None \
                and pd.Timestamp(_nn_last).normalize() < pd.Timestamp(_plot_last).normalize():
            _src = ("The instrument's own price data" if _self_mode
                    else "The constituent basket's data")
            _why = ("the instrument's market is closed or hasn't posted yet" if _self_mode
                    else "the constituents' markets are closed or haven't posted yet")
            render_info_box(
                "Breadth carried forward",
                f"{_src} ends {pd.Timestamp(_nn_last):%d %b %Y}; later sessions "
                f"(through {pd.Timestamp(_plot_last):%d %b %Y}) carry its last reads forward — {_why}, "
                f"so bottom-up breadth on those bars is provisional.",
                color="amber",
            )
    except Exception:
        pass

    # Compute ranges
    unified_y = _dynamic_range(norm_avg)
    conv_y = _dynamic_range(aligned_conv_raw)
    swayam_y = _dynamic_range(aligned_swayam_raw)

    # ── Build 3-row chart ───────────────────────────────────────────────
    fig = make_subplots(
        rows=3, cols=1, shared_xaxes=True,
        row_heights=[0.50, 0.25, 0.25],
        vertical_spacing=0.05,
    )

    # Convergence color mapping
    avg_colors, avg_sizes = [], []
    for v in norm_avg:
        if v < -UI_CONSENSUS_STRONG:
            avg_colors.append(chart_color("emerald")); avg_sizes.append(7)
        elif v <= -UI_CONSENSUS_MODERATE:
            avg_colors.append(chart_rgba("emerald", 0.55)); avg_sizes.append(5)
        elif v > UI_CONSENSUS_STRONG:
            avg_colors.append(chart_color("rose")); avg_sizes.append(7)
        elif v >= UI_CONSENSUS_MODERATE:
            avg_colors.append(chart_rgba("rose", 0.55)); avg_sizes.append(5)
        else:
            avg_colors.append(chart_rgba("slate", 0.45)); avg_sizes.append(4)

    # ── Row 1: Unified normalized ─────────────────────────────────────
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(norm_avg, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.06),
        line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_a, mode="lines", name="FVO",
        line=dict(color=chart_rgba("slate", 0.25), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_n, mode="lines", name="Swayam",
        line=dict(color=chart_rgba("cyan", 0.2), width=1, dash="dot"),
    ), row=1, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=norm_avg, mode="lines+markers", name="Consensus (50/50)",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=dict(size=avg_sizes, color=avg_colors),
    ), row=1, col=1)
    # (A dashed accent 'Calibrated Model' overlay was drawn here. It plotted
    # a second smoothed read of the same signal on top of the consensus
    # line, in the ONE colour this system reserves for interaction —
    # two lines making the same claim, the louder of which meant nothing.)
    fig.add_hline(y=UI_CONSENSUS_STRONG, line_dash="dot", line_color=chart_rgba("rose", 0.15), line_width=0.5, row=1, col=1)
    fig.add_hline(y=-UI_CONSENSUS_STRONG, line_dash="dot", line_color=chart_rgba("emerald", 0.15), line_width=0.5, row=1, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=1, col=1)

    # ── Row 2: Base Conviction ────────────────────────────────────────
    conv_vals = [float(v) if v is not None else np.nan for v in aligned_conv_raw]
    conv_colors, conv_sizes = [], []
    for v in aligned_conv_raw:
        if v is None:
            conv_colors.append(chart_rgba("slate", 0.45)); conv_sizes.append(4)
        elif v > UI_CONVRAW_STRONG:
            conv_colors.append(chart_color("rose")); conv_sizes.append(7)
        elif v >= UI_CONVRAW_MODERATE:
            conv_colors.append(chart_rgba("rose", 0.55)); conv_sizes.append(5)
        elif v < -UI_CONVRAW_STRONG:
            conv_colors.append(chart_color("emerald")); conv_sizes.append(7)
        elif v <= -UI_CONVRAW_MODERATE:
            conv_colors.append(chart_rgba("emerald", 0.55)); conv_sizes.append(5)
        else:
            conv_colors.append(chart_rgba("slate", 0.45)); conv_sizes.append(4)

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(conv_vals, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=2, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=conv_vals, mode="lines+markers", name="Base Conviction",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=dict(size=conv_sizes, color=conv_colors),
    ), row=2, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=2, col=1)
    fig.add_hline(y=UI_CONVRAW_STRONG, line_dash="dot", line_color=chart_rgba("rose", 0.12), line_width=0.5, row=2, col=1)
    fig.add_hline(y=-UI_CONVRAW_STRONG, line_dash="dot", line_color=chart_rgba("emerald", 0.12), line_width=0.5, row=2, col=1)

    # ── Row 3: Swayam Avg Signal ──────────────────────────────────────
    # Same three tiers, same colours, same sizes as the two rows above.
    swayam_colors = [chart_color("emerald") if v < -UI_SWAYAM_AVG_THRESHOLD
                     else chart_color("rose") if v > UI_SWAYAM_AVG_THRESHOLD
                     else chart_rgba("slate", 0.45) for v in aligned_swayam_raw]
    swayam_sizes = [7 if abs(v) > UI_SWAYAM_AVG_THRESHOLD else 4 for v in aligned_swayam_raw]

    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_swayam_raw, 0, None),
        fill="tozeroy", fillcolor=chart_rgba("rose", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=np.clip(aligned_swayam_raw, None, 0),
        fill="tozeroy", fillcolor=chart_rgba("emerald", 0.05), line=dict(width=0), showlegend=False, hoverinfo="skip",
    ), row=3, col=1)
    fig.add_trace(go.Scatter(
        x=aligned_dates, y=aligned_swayam_raw, mode="lines+markers", name="Avg Signal",
        line=dict(color=chart_rgba("slate", 0.55), width=1.2),
        marker=dict(size=swayam_sizes, color=swayam_colors),
    ), row=3, col=1)
    fig.add_hline(y=UI_SWAYAM_AVG_THRESHOLD, line_dash="dot", line_color=chart_rgba("rose", 0.15), line_width=0.5, row=3, col=1)
    fig.add_hline(y=-UI_SWAYAM_AVG_THRESHOLD, line_dash="dot", line_color=chart_rgba("emerald", 0.15), line_width=0.5, row=3, col=1)
    fig.add_hline(y=0, line_color=grid_rgba(0.06), line_width=0.5, row=3, col=1)

    # ── Layout ────────────────────────────────────────────────────────
    fig.update_layout(**chart_layout(height=UI_CHART_HEIGHT_STACKED, show_legend=False))
    style_axes(fig, y_title="Normalized", y_range=unified_y, row=1, col=1)
    style_axes(fig, y_title="Conviction", y_range=conv_y, row=2, col=1)
    style_axes(fig, y_title="Avg Signal", y_range=swayam_y, row=3, col=1)

    render_chart_panel(fig, "convergence_overlay", units="normalized · conviction · signal", window=True)
    render_note(f"{len(aligned_dates)} overlapping trading days")
