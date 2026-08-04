"""
Tattva — Reusable UI components: metric cards, signal badges, headers, section headers.
तत्त्व (Tattva) — "Principle / Essence"

UI — Obsidian Quant Terminal design language.
"""

from __future__ import annotations

import datetime as _dt
import html as html_mod

import pandas as pd
import numpy as np
import streamlit as st
from streamlit.components.v1 import html as _components_html


# ── SVG Icons (inline, no external deps) — with ARIA labels for accessibility

ICONS = {
    "chart":      '<svg aria-label="Chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "cube":       '<svg aria-label="Cube icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 16V8a2 2 0 0 0-1-1.73l-7-4a2 2 0 0 0-2 0l-7 4A2 2 0 0 0 3 8v8a2 2 0 0 0 1 1.73l7 4a2 2 0 0 0 2 0l7-4A2 2 0 0 0 21 16z"/><polyline points="3.27 6.96 12 12.01 20.73 6.96"/><line x1="12" y1="22.08" x2="12" y2="12"/></svg>',
    "target":     '<svg aria-label="Target icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><circle cx="12" cy="12" r="6"/><circle cx="12" cy="12" r="2"/></svg>',
    "layers":     '<svg aria-label="Layers icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="12 2 2 7 12 12 22 7 12 2"/><polyline points="2 17 12 22 22 17"/><polyline points="2 12 12 17 22 12"/></svg>',
    "bar-chart":  '<svg aria-label="Bar chart icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><line x1="18" y1="20" x2="18" y2="10"/><line x1="12" y1="20" x2="12" y2="4"/><line x1="6" y1="20" x2="6" y2="14"/></svg>',
    "activity":   '<svg aria-label="Activity icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="22 12 18 12 15 21 9 3 6 12 2 12"/></svg>',
    "crosshair":  '<svg aria-label="Crosshair icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><line x1="22" y1="12" x2="18" y2="12"/><line x1="6" y1="12" x2="2" y2="12"/><line x1="12" y1="6" x2="12" y2="2"/><line x1="12" y1="22" x2="12" y2="18"/></svg>',
    "cpu":        '<svg aria-label="CPU icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="4" y="4" width="16" height="16" rx="2" ry="2"/><rect x="9" y="9" width="6" height="6"/><line x1="9" y1="1" x2="9" y2="4"/><line x1="15" y1="1" x2="15" y2="4"/><line x1="9" y1="20" x2="9" y2="23"/><line x1="15" y1="20" x2="15" y2="23"/><line x1="20" y1="9" x2="23" y2="9"/><line x1="20" y1="14" x2="23" y2="14"/><line x1="1" y1="9" x2="4" y2="9"/><line x1="1" y1="14" x2="4" y2="14"/></svg>',
    "zap":        '<svg aria-label="Zap icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2"/></svg>',
    "shield":     '<svg aria-label="Shield icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/></svg>',
    "grid":       '<svg aria-label="Grid icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="3" y="3" width="7" height="7"/><rect x="14" y="3" width="7" height="7"/><rect x="14" y="14" width="7" height="7"/><rect x="3" y="14" width="7" height="7"/></svg>',
    "database":   '<svg aria-label="Database icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><ellipse cx="12" cy="5" rx="9" ry="3"/><path d="M21 12c0 1.66-4 3-9 3s-9-1.34-9-3"/><path d="M3 5v14c0 1.66 4 3 9 3s9-1.34 9-3V5"/></svg>',
    "trending":   '<svg aria-label="Trending icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><polyline points="23 6 13.5 15.5 8.5 10.5 1 18"/><polyline points="17 6 23 6 23 12"/></svg>',
    "eye":        '<svg aria-label="Eye icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M1 12s4-8 11-8 11 8 11 8-4 8-11 8-11-8-11-8z"/><circle cx="12" cy="12" r="3"/></svg>',
    "play":       '<svg aria-label="Play icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><circle cx="12" cy="12" r="10"/><polygon points="10 8 16 12 10 16 10 8"/></svg>',
    "chevron-right": '<svg aria-label="Expand icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><polyline points="9 18 15 12 9 6"/></svg>',
    "sun":        '<svg aria-label="Light mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="5"/><line x1="12" y1="1" x2="12" y2="3"/><line x1="12" y1="21" x2="12" y2="23"/><line x1="4.22" y1="4.22" x2="5.64" y2="5.64"/><line x1="18.36" y1="18.36" x2="19.78" y2="19.78"/><line x1="1" y1="12" x2="3" y2="12"/><line x1="21" y1="12" x2="23" y2="12"/><line x1="4.22" y1="19.78" x2="5.64" y2="18.36"/><line x1="18.36" y1="5.64" x2="19.78" y2="4.22"/></svg>',
    "moon":       '<svg aria-label="Dark mode icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z"/></svg>',
    "download":   '<svg aria-label="Download icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/><polyline points="7 10 12 15 17 10"/><line x1="12" y1="15" x2="12" y2="3"/></svg>',
    "briefcase":  '<svg aria-label="Portfolio icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><rect x="2" y="7" width="20" height="14" rx="2" ry="2"/><path d="M16 21V5a2 2 0 0 0-2-2h-4a2 2 0 0 0-2 2v16"/></svg>',
    "compass":    '<svg aria-label="Regime icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5"><circle cx="12" cy="12" r="10"/><polygon points="16.24 7.76 14.12 14.12 7.76 16.24 9.88 9.88 16.24 7.76"/></svg>',
    "rocket":     '<svg aria-label="Strong Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="M4.5 16.5c-1.5 1.26-2 5-2 5s3.74-.5 5-2c.71-.84.7-2.13-.09-2.91a2.18 2.18 0 0 0-2.91-.09z"/><path d="m12 15-3-3a22 22 0 0 1 2-3.95A12.88 12.88 0 0 1 22 2c0 2.72-.78 7.5-3 11a22.35 22.35 0 0 1-4 2z"/><path d="M9 12H4s.55-3.03 2-4.5c1.62-1.63 5-2.5 5-2.5"/><path d="M12 15v5s3.03-.55 4.5-2c1.63-1.62 2.5-5 2.5-5"/></svg>',
    "trending-up": '<svg aria-label="Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 7 13.5 15.5 8.5 10.5 2 17"/><polyline points="16 7 22 7 22 13"/></svg>',
    "trending-down": '<svg aria-label="Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="22 17 13.5 8.5 8.5 13.5 2 7"/><polyline points="16 17 22 17 22 11"/></svg>',
    "arrow-up-right": '<svg aria-label="Weak Bull icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="17" x2="17" y2="7"/><polyline points="7 7 17 7 17 17"/></svg>',
    "arrow-down-right": '<svg aria-label="Weak Bear icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><line x1="7" y1="7" x2="17" y2="17"/><polyline points="17 7 17 17 7 17"/></svg>',
    "arrow-up":   '<svg aria-label="Up" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="19" x2="12" y2="5"/><polyline points="5 12 12 5 19 12"/></svg>',
    "arrow-down": '<svg aria-label="Down" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><line x1="12" y1="5" x2="12" y2="19"/><polyline points="19 12 12 19 5 12"/></svg>',
    "move-horizontal": '<svg aria-label="Chop icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><polyline points="18 8 22 12 18 16"/><polyline points="6 8 2 12 6 16"/><line x1="2" y1="12" x2="22" y2="12"/></svg>',
    "alert-triangle": '<svg aria-label="Crisis icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "help-circle": '<svg aria-label="Unknown icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/></svg>',
    "circle":     '<svg aria-label="Circle" role="img" viewBox="0 0 24 24" fill="currentColor" stroke="none"><circle cx="12" cy="12" r="10"/></svg>',
    "check-circle": '<svg aria-label="Check" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"><path d="M22 11.08V12a10 10 0 1 1-5.93-9.14"/><polyline points="22 4 12 14.01 9 11.01"/></svg>',
    "scale":      '<svg aria-label="Weighting icon" role="img" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5" stroke-linecap="round" stroke-linejoin="round"><path d="m16 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="m2 16 3-8 3 8c-.87.65-1.92 1-3 1s-2.13-.35-3-1Z"/><path d="M7 21h10"/><path d="M12 3v18"/><path d="M3 7h18"/></svg>',
}


def get_icon(name: str, size: int = 18, stroke_width: float = 1.5) -> str:
    """Return an SVG icon string with custom size and stroke width."""
    import re
    base_svg = ICONS.get(name, ICONS["chart"])

    # Clean existing attributes to avoid duplicates or stale values
    base_svg = re.sub(r'\s+width="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+height="[^"]*"', '', base_svg)
    base_svg = re.sub(r'\s+stroke-width="[^"]*"', '', base_svg)

    # Inject standardized attributes
    return base_svg.replace('<svg', f'<svg width="{size}" height="{size}" stroke-width="{stroke_width}"')


def render_section_header(
    title: str,
    description: str = "",
    icon: str = "chart",
    accent: str = "",
) -> None:
    """Render a styled section header with icon, title, and optional description.

    Args:
        title: Section title (rendered uppercase).
        description: Optional one-line description below title.
        icon: Key from ICONS dict.
        accent: CSS color class — "", "cyan", "emerald", "violet", "rose".
    """
    svg = get_icon(icon, size=16, stroke_width=1.8)
    icon_class = f"icon {accent}" if accent else "icon"
    hdr_class = f"section-hdr {accent}" if accent else "section-hdr"
    desc_html = f'<div class="desc">{html_mod.escape(description)}</div>' if description else ""
    st.markdown(
        f'<div class="{hdr_class}">'
        f'<div class="{icon_class}">{svg}</div>'
        f'<div class="text">'
        f'<h3>{html_mod.escape(title)}</h3>'
        f'{desc_html}'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def section_gap() -> None:
    """Insert vertical spacing between major sections."""
    st.markdown('<div class="section-gap"></div>', unsafe_allow_html=True)


def render_control_hint(text: str) -> None:
    """Render the canonical terse helper caption beneath a control.

    This is the single source of truth for the "sub-control hint" tier — the
    uppercase micro-caption used by e.g. the "Swayam basket · producer
    cross-section" and Signal-Horizon hints. Use it instead of ``st.caption``
    for control helper text so the sidebar/tab fine-print stays one coherent
    visual hierarchy. Keep the text terse and ``·``-separated.
    """
    st.markdown(
        f'<div class="control-hint">{html_mod.escape(text)}</div>',
        unsafe_allow_html=True,
    )


def render_metric_card(
    label: str,
    value: str,
    subtext: str = "",
    color_class: str = "neutral",
    tooltip: str = "",
    icon: str = "",
) -> None:
    """Render a terminal-styled metric card with optional tooltip.

    Args:
        label: Card label (rendered uppercase).
        value: Primary metric value.
        subtext: Optional secondary description below value.
        color_class: Semantic color — "neutral", "success", "danger", "warning", "info", "violet".
        tooltip: Optional hover explanation text.
        icon: Optional ICONS key — small icon inlined before the label.
    """
    tooltip_html = ""
    if tooltip:
        tooltip_html = (
            f'<div class="metric-tooltip" data-tooltip="{html_mod.escape(tooltip)}">'
            f'<svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2">'
            f'<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/>'
            f'<line x1="12" y1="17" x2="12.01" y2="17"/></svg>'
            f'<span class="metric-tooltip-text">{html_mod.escape(tooltip)}</span>'
            f'</div>'
        )

    sub_metric_html = f'<div class="sub-metric">{html_mod.escape(subtext)}</div>' if subtext else ""
    icon_html = f'<span class="card-icon">{get_icon(icon, size=12, stroke_width=2)}</span> ' if icon else ""
    st.markdown(
        f'<div class="metric-card {html_mod.escape(color_class)}">'
        f'<span class="label">{icon_html}{html_mod.escape(label)}</span>'
        f"<h2>{html_mod.escape(value)}</h2>"
        f"{sub_metric_html}"
        f"{tooltip_html}"
        f"</div>",
        unsafe_allow_html=True,
    )




def render_header(title: str, tagline: str) -> None:
    """Render the terminal masthead."""
    st.markdown(
        f'<div class="premium-header">'
        f'<span class="title">{html_mod.escape(title)}</span>'
        f'<span class="tagline">{html_mod.escape(tagline)}</span>'
        f"</div>",
        unsafe_allow_html=True,
    )


#: Instruments on the tape, in reading order: the broad risk complex, then
#: rates, then the dollar, then the commodity and volatility poles. Ordered
#: by what a desk checks first rather than alphabetically — a tape is scanned
#: peripherally, and a familiar running order is what makes that possible.
TICKER_INSTRUMENTS: tuple[str, ...] = (
    "US Large Cap (S&P 500)", "US Nasdaq 100", "US Small Cap (Russell 2000)",
    "Global Equity (ACWI)", "Japan Equity (EWJ)", "Eurozone Equity (EZU)",
    "Emerging Markets Equity", "India Equity",
    "US 10-Year Treasury Yield", "US Treasury Long (20Y+)",
    "US Corporate Investment Grade", "US High Yield Corporate",
    "Dollar Index", "EUR/USD", "USD/JPY", "USD/INR",
    "Gold", "Silver", "Copper", "Crude Oil", "Brent Crude", "Natural Gas",
    "Broad Commodity Index (DBC)", "Equity Volatility (VIX)",
)


#: Display-name overrides for the tape. Used only where the yfinance ticker
#: is not what a desk would call the instrument (a futures root with an "=F"
#: suffix, an index with a caret) — everywhere else the real ticker is the
#: right label, because that is what a tape shows.
_TAPE_ALIAS: dict[str, str] = {
    "GC=F": "GOLD", "SI=F": "SILVER", "HG=F": "COPPER", "PL=F": "PLAT",
    "CL=F": "WTI", "BZ=F": "BRENT", "NG=F": "NATGAS",
    "^VIX": "VIX", "^MOVE": "MOVE", "^TNX": "US10Y", "^TYX": "US30Y",
    "^FVX": "US5Y", "^IRX": "US3M",
    "DX-Y.NYB": "DXY", "EURUSD=X": "EURUSD", "JPY=X": "USDJPY",
    "INR=X": "USDINR", "CNY=X": "USDCNY",
}


def _tape_symbol(column: str) -> str:
    """Ticker for a macro column, as a tape would print it.

    Resolves the display name back through the config maps to its real symbol
    — "US Nasdaq 100" prints as QQQ, not as a name truncated mid-word. Falls
    back to a word-boundary-safe abbreviation for anything unmapped (a
    Google-Sheet series, a user column), because cutting "US Nasdaq 100" to
    "US Nasdaq 10" is worse than showing fewer words.
    """
    try:
        from core.config import (COMMODITY_TARGETS, GLOBAL_MACRO_MAP,
                                 MACRO_SYMBOLS_YF)
        lookup = {**GLOBAL_MACRO_MAP, **MACRO_SYMBOLS_YF, **COMMODITY_TARGETS}
    except Exception:
        lookup = {}
    tkr = lookup.get(column)
    if tkr:
        if tkr in _TAPE_ALIAS:
            return _TAPE_ALIAS[tkr]
        # Strip exchange suffixes (.NS/.L/.TO) and index carets for display.
        clean = tkr.lstrip("^").split(".")[0].replace("=X", "").replace("=F", "")
        if clean:
            return clean.upper()[:10]
    # Unmapped: prefer a trailing parenthetical, else whole words up to 12 chars.
    if "(" in column and column.rstrip().endswith(")"):
        return column.split("(")[-1].rstrip(")")[:10].upper()
    out = ""
    for word in column.split():
        if len(out) + len(word) + 1 > 12:
            break
        out = f"{out} {word}".strip()
    return (out or column[:12]).upper()


def render_ticker(frame, instruments: tuple[str, ...] = TICKER_INSTRUMENTS,
                  seconds_per_item: float = 3.6) -> None:
    """Render the running tape from the already-fetched macro panel.

    No additional network call: the panel behind the valuation engine already
    holds every one of these instruments, so the tape is a view of the data the
    run is using rather than a second, possibly disagreeing, source.

    The track is emitted TWICE and animated to -50%, which is what makes the
    loop seamless — at the moment the first copy leaves the viewport the second
    is exactly where the first began. Duration scales with item count so the
    scroll speed stays constant no matter how many instruments are listed;
    a tape that accelerates as you add symbols is unreadable.

    Direction is carried by an arrow glyph as well as by colour. Roughly 8% of
    men have red/green colour deficiency, and the sign of a move is the one
    reading here that must never be ambiguous.
    """
    if frame is None or not len(frame):
        return
    cols = [c for c in instruments if c in getattr(frame, "columns", ())]
    if not cols:
        return

    tail = frame[cols].tail(2)
    if len(tail) < 2:
        return
    prev, last = tail.iloc[0], tail.iloc[1]

    items: list[str] = []
    for c in cols:
        try:
            p1, p0 = float(last[c]), float(prev[c])
        except (TypeError, ValueError):
            continue
        if not (np.isfinite(p1) and np.isfinite(p0)) or p0 == 0:
            continue
        chg = (p1 / p0 - 1.0) * 100.0
        cls, arrow = (("up", "▲") if chg > 0.005 else
                      ("down", "▼") if chg < -0.005 else ("flat", "•"))
        sym = _tape_symbol(c)
        px = f"{p1:,.2f}" if abs(p1) < 10000 else f"{p1:,.0f}"
        items.append(
            f'<span class="tt-item">'
            f'<span class="tt-sym">{html_mod.escape(sym)}</span>'
            f'<span class="tt-px">{px}</span>'
            f'<span class="tt-chg {cls}" data-arrow="{arrow}">{abs(chg):.2f}%</span>'
            f'</span><span class="tt-sep">|</span>'
        )
    if not items:
        return

    run = "".join(items)
    duration = max(40.0, len(items) * float(seconds_per_item))
    st.markdown(
        f'<div class="ticker" role="marquee" aria-label="Live macro tape">'
        f'<div class="tt-track" style="--tt-duration:{duration:.0f}s">{run}{run}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


def render_info_box(title: str, content: str, color: str = "cyan") -> None:
    """Render an info box. ``color`` is applied as a modifier class (cyan / amber /
    emerald / rose / violet) so callers can theme it; was previously ignored."""
    st.markdown(
        f'<div class="info-box {html_mod.escape(color)}">'
        f"<h4>{html_mod.escape(title)}</h4>"
        f"<p>{html_mod.escape(content)}</p>"
        f"</div>",
        unsafe_allow_html=True,
    )


def render_interpretation_card(
    title: str,
    body: str,
    color: str = "neutral",
) -> None:
    """Render a state-aware interpretation card — terminal readout style.

    Args:
        title: Short state label (e.g. "NEUTRAL", "STRONG OVERSOLD").
        body: One-paragraph explanation (raw HTML allowed — caller is trusted).
        color: Semantic color — "neutral", "success", "danger", "warning", "info".
    """
    st.markdown(
        f'<div class="interp-card {html_mod.escape(color)}">'
        f'<div class="interp-title">{html_mod.escape(title)}</div>'
        f'<div class="interp-body">{body}</div>'
        f'</div>',
        unsafe_allow_html=True,
    )


# (``render_nishkarsh_signal_card`` lived here. It was a thin wrapper that
# called build_hero_verdict with a signature three rewrites out of date, had no
# callers anywhere in the app, and would have raised TypeError if one had
# appeared.)


# ── Data-table styling tokens ──────────────────────────────────────────
# render_data_table renders into an isolated components.v1.html iframe, which
# does NOT inherit the app's CSS variables — so the theme values it needs are
# mirrored here as literals. Any change to the corresponding --token in
# theme.css has to be made in both places; there is no way around that while
# the table lives in an iframe, and a stale colour here is the visible symptom.
_TABLE_TOKENS = {
    "ink_primary":   "#F8FAFC",   # --ink
    "ink_tertiary":  "#94A3B8",   # --ink-tertiary
    "border":        "rgba(255, 255, 255, 0.05)",   # --line
    "border_subtle": "rgba(255, 255, 255, 0.02)",  # --line-faint
    "amber":         "#F59E0B",   # --caution
    "emerald":       "#10B981",   # --long   (positive numeric cells)
    "rose":          "#EF4444",   # --short  (negative numeric cells)
    "amber_border":  "rgba(245, 158, 11, 0.35)",
    "amber_hover":   "rgba(245, 158, 11, 0.1)",
    "row_odd":       "rgba(255, 255, 255, 0.012)",
    "row_even":      "transparent",
}

#: Webfont the iframe must import for itself, for the same isolation reason.
_TABLE_FONTS = ("https://fonts.googleapis.com/css2?"
                "family=JetBrains+Mono:wght@400;500;600;700&display=swap")


# ═══════════════════════════════════════════════════════════════════════
#  HERO VERDICT — the conviction chain
# ═══════════════════════════════════════════════════════════════════════
#
# The system does not produce a "signal" that evidence then votes on. It makes
# ONE claim and attaches a series of conditions to it, every one of which can
# independently invalidate it:
#
#   the asset is mispriced          (FVO — the claim)
#   ...the mispricing reverts       (mean-reversion evidence, FVO)
#   ...its own internals agree      (Swayam breadth)
#   ...both engines converge        (agreement ratio + normalized consensus)
#   ...this has historically paid   (walk-forward OOS IC)
#   ...and it paid last time too    (precedent base rate)
#
# Every engine the app runs appears exactly once, and only where it has
# something to say that the others do not.
#
# So conviction is a PRODUCT of gates in [0, 1], not a sum of votes. That
# distinction is the entire redesign, and it matters for two reasons.
#
# A product cannot be rescued by piling on agreement: three enthusiastic
# confirmations and one broken precondition is not "3 - 1 = act smaller", it is
# "the precondition is broken". Additive scoring said the former. Every version
# of this card since it was written has had a table of +1/-2 point weights that
# nobody could derive from anything; that table is gone.
#
# And a product has a MINIMUM. Whichever gate is smallest is the binding
# constraint — the single specific reason conviction is not higher, which is
# the most useful sentence a card of this kind can produce and which no amount
# of vote-tallying can express. The card leads with it.
#
# Direction comes from FVO alone. It is the only component that makes a
# directional claim about the world ("this is cheap relative to the traded
# opportunity set"); breadth, reversion evidence and historical skill are all
# statements ABOUT that claim, not rival claims of their own. Averaging them
# into a "consensus" — which is what the headline used to be — produced a
# number whose sign nothing in particular was responsible for.


def _gate(value: float, lo: float, hi: float) -> float:
    """Map a raw reading onto [0, 1] with a soft floor and ceiling.

    ``lo`` is where the gate is fully shut, ``hi`` where it is fully open;
    between them it opens linearly. Never returns exactly 0 — a shut gate
    should collapse conviction, not erase the reading and its explanation
    with it.
    """
    if not np.isfinite(value):
        return 0.5
    if hi == lo:
        return 1.0
    return float(np.clip((value - lo) / (hi - lo), 0.02, 1.0))


#: Minimum DISTINCT analogs before the precedent base rate is treated as
#: probative. Below this, "% positive" is a handful of coin flips.
MIN_PRECEDENT_N = 8

#: Conviction tiers. Products of SIX gates concentrate hard toward zero, so
#: these are not evenly spaced: 0.30 already requires every gate to average
#: ~0.82, and 0.15 requires ~0.73.
_TIERS = (
    (0.30, "high", "HIGH CONVICTION", "act on it"),
    (0.15, "moderate", "MODERATE CONVICTION", "act at reduced size"),
    (0.06, "low", "LOW CONVICTION", "starter size at most"),
    (0.00, "standaside", "STAND ASIDE", "no actionable edge"),
)


def build_hero_verdict(
    *,
    fvo_signal: dict,
    swayam_breadth: dict | None,
    convergence: dict | None,
    wf_ic: float | None,
    wf_pos: float | None,
    wf_n: int | None,
    precedent: dict | None,
    n_divergences: int,
    horizon_days: int,
    div_window: int | None = None,
) -> dict:
    """Build the hero verdict from the conviction chain. Pure data in/out.

    Returns ``{signal, signal_class, direction, score, headline, conviction,
    binding, gates, trust, evidence, action}`` where ``gates`` is the ordered
    chain (each ``{name, value, label, detail}``) and ``binding`` names the
    smallest one. Rendering is entirely separate (``render_hero_card``), so
    these rules stay unit-testable — see research/test_hero_verdict.py.
    """
    # ── The claim: FVO's standardized mispricing ───────────────────────
    fvo = float(fvo_signal.get("fvo", 0.0) or 0.0)
    pct = float(fvo_signal.get("pct_mispricing", 0.0) or 0.0) * 100.0
    conf = float(fvo_signal.get("valuation_confidence", 0.0) or 0.0)
    xs = float(fvo_signal.get("xs_consistency", 0.0) or 0.0)
    mr = float(fvo_signal.get("mr_prob", 0.0) or 0.0)
    half_life = float(fvo_signal.get("gap_half_life", 0.0) or 0.0)

    # Direction is FVO's alone. A negative oscillator means the asset trades
    # below the level the cross-section implies — cheap, therefore bullish.
    if fvo <= -0.5:
        direction, verb = "bullish", "cheap"
    elif fvo >= 0.5:
        direction, verb = "bearish", "rich"
    else:
        direction, verb = "neutral", "fairly valued"

    gates: list[dict] = []

    # ── Gate 1: is it mispriced at all? ────────────────────────────────
    g_mag = _gate(abs(fvo), 0.5, 2.0)
    gates.append({
        "name": "mispricing", "value": g_mag,
        "label": f"{abs(fvo):.2f}σ {verb}" if direction != "neutral" else "within noise",
        "detail": (f"{abs(pct):.1f}% from fair value, {abs(fvo):.2f} predictive SDs."
                   if direction != "neutral"
                   else f"{abs(fvo):.2f} SDs from fair value — inside the engine's own uncertainty."),
    })

    # ── Gate 2: does the mispricing revert? ────────────────────────────
    g_conf = _gate(conf, 0.2, 0.8)
    gates.append({
        "name": "reversion", "value": g_conf,
        "label": f"confidence {conf:.2f}",
        "detail": (f"Mean-reversion evidence {mr:.2f}, cross-sectional agreement {xs:.2f}"
                   + (f", half-life {half_life:.0f}d." if half_life > 0 else ".")),
    })

    # ── Gate 3: do the asset's own internals agree? ────────────────────
    if swayam_breadth:
        net = (float(swayam_breadth.get("oversold_pct", 50.0))
               - float(swayam_breadth.get("overbought_pct", 50.0))) / 100.0
        aligned = net if direction == "bullish" else -net if direction == "bearish" else abs(net)
        g_breadth = _gate(aligned, -0.3, 0.4)
        gates.append({
            "name": "breadth", "value": g_breadth,
            "label": ("internals agree" if aligned > 0.1 else
                      "internals disagree" if aligned < -0.1 else "internals split"),
            "detail": f"Swayam net breadth {net:+.0%} across the view bank.",
        })
    else:
        g_breadth = 0.5
        gates.append({"name": "breadth", "value": 0.5, "label": "no breadth read",
                      "detail": "Swayam produced no overlapping breadth for this target."})

    # ── Gate 4: do the two engines converge? ───────────────────────────
    # Two readings, multiplied: HOW OFTEN they have agreed (agreement ratio)
    # and WHETHER the normalized consensus currently points the same way as
    # the FVO call. A high agreement ratio pointing the wrong way is not
    # convergence, which is why one number could not carry this gate.
    if convergence:
        agree_ratio = float(convergence.get("agreement_ratio", 0.5) or 0.5)
        cons = convergence.get("consensus")
        # Consensus is signed like the oscillator (negative = cheap), so it is
        # flipped to a bullish-positive convention before comparison.
        if cons is not None and np.isfinite(cons):
            cons_bull = -float(cons)
            aligned = (cons_bull if direction == "bullish" else
                       -cons_bull if direction == "bearish" else abs(cons_bull))
        else:
            aligned = 0.0
        g_conv = _gate(agree_ratio, 0.45, 0.85) * _gate(aligned, -0.25, 0.25)
        if aligned < -0.1:
            conv_label = "engines disagree"
        elif aligned > 0.1:
            conv_label = "engines converge"
        else:
            conv_label = "engines split"
        conv_detail = f"Cross-engine agreement {agree_ratio:.0%}" + (
            f", normalized consensus {cons:+.2f} "
            f"({'confirms' if aligned > 0.1 else 'contradicts' if aligned < -0.1 else 'neutral on'}"
            f" the {direction} call)."
            if cons is not None and np.isfinite(cons) else ".")
    else:
        g_conv = 0.5
        conv_label, conv_detail = ("no convergence read",
                                   "FVO and Swayam had no overlapping history to converge over.")
    gates.append({"name": "convergence", "value": g_conv,
                  "label": conv_label, "detail": conv_detail})

    # ── Gate 5: has this ever paid, out of sample? ─────────────────────
    if wf_ic is None:
        # Not "no edge" — no measurement. A system too young to have been
        # scored gets a discount, not a verdict.
        g_edge = 0.25
        edge_label, edge_detail = ("unvalidated",
                                   "Not enough scored history for a walk-forward read (~250+ dates).")
    else:
        g_edge = _gate(wf_ic, -0.02, 0.15)
        edge_label = ("edge holds" if wf_ic > 0.05 else
                      "edge marginal" if wf_ic > 0 else "no measured edge")
        edge_detail = f"Walk-forward OOS IC {wf_ic:+.3f}"
        if wf_pos is not None and wf_n:
            edge_detail += f" across {wf_n} windows, {round(wf_pos * wf_n)} positive."
        else:
            edge_detail += "."
    gates.append({"name": "edge", "value": g_edge,
                  "label": edge_label, "detail": edge_detail})

    # ── Gate 6: did it pay the last times this state occurred? ─────────
    p_n = int((precedent or {}).get("n", 0) or 0)
    p_pos = (precedent or {}).get("positive_pct")
    if precedent and p_n >= MIN_PRECEDENT_N and p_pos is not None:
        p_bull = float(p_pos) / 100.0
        agree = (p_bull if direction == "bullish" else
                 1.0 - p_bull if direction == "bearish" else 0.5)
        g_prec = _gate(agree, 0.35, 0.65)
        gates.append({
            "name": "precedent", "value": g_prec,
            "label": ("precedent agrees" if agree > 0.55 else
                      "precedent disagrees" if agree < 0.45 else "precedent split"),
            "detail": f"{float(p_pos):.0f}% of {p_n} distinct analogs rose over +{horizon_days}d.",
        })
    else:
        g_prec = 0.6
        gates.append({
            "name": "precedent", "value": 0.6, "label": "no usable precedent",
            "detail": f"Only {p_n} distinct analogs (need {MIN_PRECEDENT_N}) — "
                      f"too few to read as a base rate.",
        })

    # ── Conviction: the product, and the constraint that binds it ──────
    conviction = float(np.prod([g["value"] for g in gates]))
    binding = min(gates, key=lambda g: g["value"])

    level, label, prose = "standaside", "STAND ASIDE", "no actionable edge"
    for cut, lvl, lab, pr in _TIERS:
        if conviction >= cut:
            level, label, prose = lvl, lab, pr
            break
    if direction == "neutral":
        level, label, prose = "standaside", "STAND ASIDE", "no directional claim to act on"

    signal = "BUY" if direction == "bullish" else "SELL" if direction == "bearish" else "HOLD"
    if direction != "neutral" and level in ("high", "moderate"):
        signal = f"STRONG {signal}" if level == "high" else signal

    headline = (
        f"{signal} — {abs(pct):.1f}% {verb} versus the level the macro cross-section implies."
        if direction != "neutral"
        else "HOLD — trading within the engine's own uncertainty about fair value."
    )

    # The single most useful sentence the card produces: what is holding it back.
    if direction == "neutral":
        limit = "No directional claim: the gap is inside the engine's uncertainty band."
    elif binding["value"] >= 0.75:
        limit = "Nothing is materially limiting this — every condition holds."
    else:
        limit = f"Capped by {binding['name']}: {binding['detail']}"

    # ── Standing risk flag, outside the chain ──────────────────────────
    risk = None
    if n_divergences > 0:
        risk = (f"{n_divergences} FVO/Swayam divergence event"
                f"{'s' if n_divergences != 1 else ''}"
                + (f" in the last {div_window} sessions." if div_window else "."))

    trust = {
        "tier": ("solid" if (wf_ic or 0) >= 0.1 else
                 "modest" if (wf_ic or 0) >= 0.05 else
                 "marginal" if (wf_ic or 0) > 0 else
                 "no_edge" if wf_ic is not None else "uncalibrated"),
        "chip": ("SOLID EDGE" if (wf_ic or 0) >= 0.1 else
                 "MODEST EDGE" if (wf_ic or 0) >= 0.05 else
                 "MARGINAL" if (wf_ic or 0) > 0 else
                 "NO EDGE" if wf_ic is not None else "NO READ"),
        "oos_ic": wf_ic, "wf_pos": wf_pos, "wf_n": wf_n, "prose": edge_detail,
    }

    return {
        "signal": signal,
        "signal_class": ("buy" if direction == "bullish" else
                         "sell" if direction == "bearish" else "hold"),
        "direction": direction,
        "score": float(np.clip(-fvo / 3.0, -1.0, 1.0)),
        "conviction": conviction,
        "headline": headline,
        "binding": binding["name"] if binding["value"] < 0.75 else None,
        "limit": limit,
        "gates": gates,
        "risk": risk,
        "trust": trust,
        "action": {"level": level, "label": label, "prose": prose,
                   "conviction": conviction},
        "horizon_days": horizon_days,
    }


def render_hero_card(verdict: dict) -> None:
    """Render the verdict: claim, what limits it, then the chain behind both.

    The layout follows the logic rather than decorating it. A reader who stops
    after two lines has the decision (signal + conviction) and the single
    reason it is not stronger; a reader who continues gets every gate with the
    number behind it. The old card put five equal-weight evidence rows above a
    points total, which buried the one line that mattered among four that
    usually did not.
    """
    trust = verdict["trust"]
    chip_style = {
        "uncalibrated": ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"),
        "no_edge":      ("#FB7185", "rgba(251,113,133,0.12)"),
        "marginal":     ("#D4A853", "rgba(212,168,83,0.12)"),
        "modest":       ("#34D399", "rgba(52,211,153,0.12)"),
        "solid":        ("#34D399", "rgba(52,211,153,0.18)"),
    }.get(trust["tier"], ("var(--ink-tertiary)", "rgba(148,163,184,0.12)"))
    ic_text = (f"OOS IC {trust['oos_ic']:+.3f}" if trust.get("oos_ic") is not None
               else "no OOS read")
    if trust.get("wf_pos") is not None and trust.get("wf_n"):
        ic_text += f" &bull; {round(trust['wf_pos'] * trust['wf_n'])}/{trust['wf_n']} windows+"

    action = verdict["action"]
    conviction = float(verdict.get("conviction", 0.0))
    tier_color = {"high": "#34D399", "moderate": "#D4A853",
                  "low": "var(--ink-secondary)", "standaside": "var(--ink-tertiary)"
                  }.get(action["level"], "var(--ink-tertiary)")

    # ── Gate chain: one row each, bar width = how open the gate is ──────
    binding = verdict.get("binding")
    gate_rows = "".join(
        f'<div class="hero-gate{" binding" if g["name"] == binding else ""}">'
        f'<span class="hero-gate-name">{html_mod.escape(g["name"])}</span>'
        f'<span class="hero-gate-bar"><i style="width:{max(2, round(g["value"] * 100))}%;'
        f'background:{"#FB7185" if g["value"] < 0.35 else "#D4A853" if g["value"] < 0.7 else "#34D399"};">'
        f'</i></span>'
        f'<span class="hero-gate-label">{html_mod.escape(g["label"])}</span>'
        f'<span class="hero-gate-detail">{html_mod.escape(g["detail"])}</span>'
        f'</div>'
        for g in verdict["gates"]
    )

    risk_html = (
        f'<div class="hero-risk">{html_mod.escape(verdict["risk"])}</div>'
        if verdict.get("risk") else ""
    )

    st.markdown(
        f"""\
<div class="hero-card {html_mod.escape(verdict["signal_class"])}">
  <div class="hero-top">
    <div class="hero-signal-block">
      <div class="hero-eyebrow">Tattva &bull; {verdict["horizon_days"]}d horizon</div>
      <div class="hero-signal">{html_mod.escape(verdict["signal"])}</div>
    </div>
    <div class="hero-conviction-block">
      <span class="hero-chip" style="background:{chip_style[1]};color:{chip_style[0]};">\
{html_mod.escape(trust["chip"])} &bull; {ic_text}</span>
      <div class="hero-conviction" style="color:{tier_color};">\
{html_mod.escape(action["label"])} &middot; {conviction:.2f}</div>
      <div class="hero-conviction-sub">{html_mod.escape(action["prose"])}</div>
    </div>
  </div>
  <div class="hero-headline">{html_mod.escape(verdict["headline"])}</div>
  <div class="hero-limit">{html_mod.escape(verdict["limit"])}</div>
  <div class="hero-gates">{gate_rows}</div>
  {risk_html}
  <div class="hero-foot">Conviction is the product of the five gates above &mdash; \
the weakest one caps it, so a single broken condition is not outvoted by the rest.</div>
</div>
""",
        unsafe_allow_html=True,
    )


def _fmt_cell(value, precision: int) -> str:
    """Format one cell value for display (NaN → em dash; floats to `precision`).

    Dates render date-only: Tattva is a DAILY system, so a Timestamp's
    ``00:00:00`` time component is noise — never shown.
    """
    if value is None:
        return "—"
    # Date-only for any datetime-like (pd.Timestamp subclasses datetime.date).
    if isinstance(value, (pd.Timestamp, _dt.date)):
        try:
            if pd.isna(value):
                return "—"
        except (TypeError, ValueError):
            pass
        return value.strftime("%Y-%m-%d")
    if isinstance(value, float):
        if value != value:            # NaN
            return "—"
        return f"{value:,.{precision}f}"
    if isinstance(value, (int,)) and not isinstance(value, bool):
        return f"{value:,}"
    try:
        if pd.isna(value):
            return "—"
    except (TypeError, ValueError):
        pass
    return html_mod.escape(str(value))


# Column-name tokens that must stay UPPER-CASE when a raw column name is
# prettified into a professional header ("MSF_Osc" → "MSF Osc", not "Msf Osc").
# (Deliberately NOT including "OSC" — an oscillator column reads more
# professionally as "Osc" than "OSC", matching the source design.)
_HEADER_ACRONYMS = {
    "RSI", "MA", "MSF", "MMR", "VAP", "IC", "HR", "HMM", "GARCH", "CUSUM",
    "ADF", "KPSS", "DDM", "OU", "PCA", "US", "FX", "ID", "N", "T", "Z", "R2",
    "OHLC", "OHLCV", "ATR", "MACD", "EMA", "SMA",
}


def _prettify_header(name: str) -> str:
    """Turn a raw column/field name into a professional table header.

    ``divergence_type`` → ``Divergence Type``; ``MSF_Osc`` → ``MSF Osc``;
    ``Change_Point`` → ``Change Point``; ``val_ic`` → ``Val IC``. Already-clean
    headers ("Buy Avg Δ", "Period") pass through with only per-word acronym
    casing applied.
    """
    raw = str(name).replace("_", " ").strip()
    if not raw:
        return ""
    out = []
    for word in raw.split():
        up = word.upper()
        if up in _HEADER_ACRONYMS:
            out.append(up)
        elif word.isupper() and len(word) <= 4:   # keep short all-caps as-is
            out.append(word)
        elif any(ch.isdigit() for ch in word) and word.isupper():
            out.append(word)
        else:
            out.append(word[:1].upper() + word[1:])
        # Preserve non-alphanumeric tokens verbatim (Δ, %, etc.)
        if not word[:1].isalnum():
            out[-1] = word
    return " ".join(out)


def render_data_table(
    df: "pd.DataFrame",
    *,
    index_label: str | None = None,
    show_index: bool | None = None,
    max_rows: int | None = None,
    precision: int = 2,
    col_precision: dict[str, int] | None = None,
    sign_color_cols: "set[str] | None" = None,
    label_col: str | None = None,
    col_labels: dict[str, str] | None = None,
    max_height: int = 520,
    row_height: int = 42,
) -> None:
    """Render a DataFrame as an Obsidian-Quant signal table (Pragyam design).

    A theme-faithful replacement for ``st.dataframe`` across Tattva: a rounded
    glass card, uppercase amber-ruled header (sticky on scroll), zebra rows with
    an amber hover, right-aligned tabular numerics, and a bolder first "label"
    column. Wide tables scroll horizontally; long tables scroll vertically under
    a fixed ``max_height`` — so it is safe on both the 10-row divergence table
    and the full dataset viewer.

    Parameters
    ----------
    index_label : shown as the first column header when the index is rendered;
        also forces the index to render.
    show_index : override index rendering (default: auto — shown when the index
        is not a plain 0..N RangeIndex, i.e. it carries dates/labels).
    max_rows : cap to the LAST ``max_rows`` rows (tables are newest-relevant).
    precision / col_precision : default and per-column float precision.
    sign_color_cols : numeric columns whose values are tinted emerald/rose by
        sign (the "signal" colouring from Pragyam's per-signal columns).
    label_col : the column to style as the bold Space-Grotesk label (default:
        the index if shown, else the first column).
    col_labels : explicit header overrides ``{raw_name: display}``; any column
        not listed is auto-prettified (``MSF_Osc`` → ``MSF Osc``).
    """
    if df is None or getattr(df, "empty", True):
        st.caption("No rows to display.")
        return

    view = df.tail(max_rows).copy() if max_rows else df.copy()
    if isinstance(view.columns, pd.MultiIndex):
        view.columns = [" · ".join(str(x) for x in c) for c in view.columns]

    if show_index is None:
        show_index = index_label is not None or not isinstance(view.index, pd.RangeIndex)
    idx_header = (index_label or _prettify_header(view.index.name or "")) if show_index else ""
    col_labels = col_labels or {}

    def _header(c: str) -> str:
        return col_labels.get(c) or _prettify_header(c)

    cols = list(view.columns)
    numeric_cols = {c for c in cols if pd.api.types.is_numeric_dtype(view[c])}
    sign_cols = (sign_color_cols or set()) & numeric_cols
    col_precision = col_precision or {}
    # The label column: explicit, else the index (when shown), else first column.
    if label_col is None:
        label_col = "__index__" if show_index else (cols[0] if cols else None)

    t = _TABLE_TOKENS

    def _header_cells() -> str:
        cells = []
        if show_index:
            cells.append(f'<th class="lbl">{html_mod.escape(str(idx_header))}</th>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            cells.append(f'<th class="{cls}">{html_mod.escape(_header(c))}</th>')
        return "".join(cells)

    def _value_html(c: str, val) -> str:
        p = col_precision.get(c, precision)
        text = _fmt_cell(val, p)
        if c in sign_cols and text != "—":
            try:
                fv = float(val)
                color = (t["emerald"] if fv > 1e-12 else t["rose"] if fv < -1e-12
                         else t["ink_tertiary"])
                return f'<span style="color:{color};font-weight:600;">{text}</span>'
            except (TypeError, ValueError):
                pass
        return text

    body_rows = []
    for idx, row in view.iterrows():
        tds = []
        if show_index:
            tds.append(f'<td class="lbl">{_fmt_cell(idx, precision)}</td>')
        for c in cols:
            cls = "num" if c in numeric_cols and c != label_col else "lbl" if c == label_col else "txt"
            tds.append(f'<td class="{cls}">{_value_html(c, row[c])}</td>')
        body_rows.append(f"<tr>{''.join(tds)}</tr>")

    n_rows = len(view)
    content_h = 44 + n_rows * row_height + 28
    iframe_h = min(content_h, max_height + 28)

    table_html = f"""<!DOCTYPE html><html><head><style>
    @import url('{_TABLE_FONTS}');
    * {{ margin:0; padding:0; box-sizing:border-box; }}
    body {{ font-family:'IBM Plex Mono',monospace; background:transparent;
            color:{t['ink_primary']}; padding:2px; }}
    .tt-wrap {{ border-radius:10px; overflow:hidden; border:1px solid {t['border']};
                background:linear-gradient(145deg,rgba(17,24,39,0.45) 0%,rgba(17,24,39,0.40) 100%); }}
    /* Scrollbar matched to theme.css's global scroller (5px · ink-tertiary thumb ·
       transparent track · 3px radius) so the table scrolls like the plots do. */
    .tt-scroll {{ max-height:{max_height}px; overflow:auto;
                  scrollbar-width:thin; scrollbar-color:{t['ink_tertiary']} transparent; }}
    .tt-scroll::-webkit-scrollbar {{ width:5px; height:5px; }}
    .tt-scroll::-webkit-scrollbar-track {{ background:transparent; }}
    .tt-scroll::-webkit-scrollbar-thumb {{ background:{t['ink_tertiary']}; border-radius:3px; }}
    .tt-scroll::-webkit-scrollbar-corner {{ background:transparent; }}
    table {{ width:100%; border-collapse:collapse; }}
    thead th {{ position:sticky; top:0; z-index:2;
        background:linear-gradient(180deg,rgba(10,14,23,0.98) 0%,rgba(10,14,23,0.92) 100%);
        color:{t['ink_tertiary']}; font-size:0.62rem; font-weight:600;
        text-transform:uppercase; letter-spacing:0.1em; padding:0.7rem 0.75rem;
        border-bottom:2px solid {t['amber_border']}; text-align:left; white-space:nowrap; }}
    thead th.num {{ text-align:right; }}
    tbody tr {{ border-bottom:1px solid {t['border_subtle']}; transition:background 0.15s ease; }}
    tbody tr:nth-child(odd) {{ background:{t['row_odd']}; }}
    tbody tr:nth-child(even) {{ background:{t['row_even']}; }}
    tbody tr:hover {{ background:{t['amber_hover']}; }}
    tbody td {{ padding:0.6rem 0.75rem; color:{t['ink_primary']}; font-size:0.75rem;
                vertical-align:middle; white-space:nowrap; }}
    tbody td.num {{ text-align:right; font-variant-numeric:tabular-nums; }}
    tbody td.lbl {{ font-family:'Space Grotesk',sans-serif; font-weight:600;
                    font-size:0.76rem; letter-spacing:0.02em; color:{t['ink_primary']}; }}
    thead th.lbl {{ color:{t['amber']}; }}
    </style></head><body>
    <div class="tt-wrap"><div class="tt-scroll"><table>
    <thead><tr>{_header_cells()}</tr></thead>
    <tbody>{''.join(body_rows)}</tbody>
    </table></div></div></body></html>"""

    _components_html(table_html, height=iframe_h, scrolling=False)


def render_warning_box(title: str, content: str) -> None:
    """Render a themed alert/warning box."""
    st.markdown(
        f"""
        <div class="warning-box">
            <div class="icon"></div>
            <div>
                <div class="title">{html_mod.escape(title)}</div>
                <div class="content">{html_mod.escape(content)}</div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
