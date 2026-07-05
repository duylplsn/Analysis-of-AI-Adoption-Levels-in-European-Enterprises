import pandas as pd
import plotly.express as px
import streamlit as st

__all__ = [
    'insight_box',
    'missing_warning',
    'inject_css',
    '_esc',
    'render_kpi_card',
    'kpi_row',
    'render_panel_header',
    'render_decision_note',
    'render_warning_note',
    'page_top',
    'style_fig',
    'fix_year_axis',
    'show_chart',
    'short_industry_label',
    'unique_short_labels',
    'key_point_labels',
    'COLORS',
    'CHART_SEQUENCE',
    '_INDUSTRY_SHORT_RULES',
    '_fmt_pct',
    '_fmt_pp',
]

def insight_box(text: str):

    st.info(f"💡 **Insight:** {text}")

def missing_warning(label: str):

    st.warning(
        f"⚠️ Could not find or read `{label}`. "
        "This section is hidden - re-run the EDA pipeline / add the file to restore it."
    )

COLORS = {
    "bg": "#0F172A",
    "sidebar": "#111827",
    "card": "#1E293B",
    "card2": "#172033",
    "text": "#F8FAFC",
    "text2": "#CBD5E1",
    "cyan": "#38BDF8",
    "purple": "#A78BFA",
    "green": "#22C55E",
    "orange": "#F59E0B",
    "red": "#EF4444",
    "border": "#334155",
    "grid": "#1F2B3E",
}

CHART_SEQUENCE = ["#38BDF8", "#A78BFA", "#22C55E", "#F59E0B", "#EF4444", "#60A5FA", "#F472B6"]

def inject_css():

    st.markdown(
        f"""
        <style>
        /* ---- Page + base ---- */
        .stApp {{ background: {COLORS['bg']}; }}
        .block-container {{ padding-top: 1.6rem; padding-bottom: 2.5rem; max-width: 1400px; }}
        html, body, [class*="css"] {{ color: {COLORS['text2']}; }}
        h1, h2, h3, h4, h5 {{ color: {COLORS['text']} !important; letter-spacing: .2px; }}

        /* ---- Sidebar ---- */
        section[data-testid="stSidebar"] {{
            background: {COLORS['sidebar']};
            border-right: 1px solid {COLORS['border']};
        }}
        section[data-testid="stSidebar"] .stRadio label {{ color: {COLORS['text2']}; }}
        .sidebar-brand {{ font-size: 1.15rem; font-weight: 700; color: {COLORS['text']}; margin: .2rem 0 0; }}
        .sidebar-sub {{ font-size: .78rem; color: {COLORS['text2']}; opacity: .8; margin-bottom: .6rem; }}
        .sidebar-label {{ font-size: .7rem; text-transform: uppercase; letter-spacing: 1px;
            color: {COLORS['cyan']}; font-weight: 600; margin: .4rem 0 .2rem; }}

        /* ---- Bordered containers act as section panels ---- */
        div[data-testid="stVerticalBlockBorderWrapper"] {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']} !important;
            border-radius: 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,.35), 0 8px 24px rgba(0,0,0,.18);
        }}

        /* ---- KPI cards ---- */
        .kpi-card {{
            background: {COLORS['card']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['cyan']};
            border-radius: 14px;
            padding: 14px 16px;
            box-shadow: 0 6px 18px rgba(0,0,0,.20);
            min-height: 104px;
        }}
        .kpi-card .kpi-label {{ font-size: .74rem; text-transform: uppercase; letter-spacing: .6px;
            color: {COLORS['text2']}; font-weight: 600; }}
        .kpi-card .kpi-value {{ font-size: 1.7rem; font-weight: 700; color: {COLORS['text']};
            line-height: 1.25; margin-top: 2px; white-space: nowrap; overflow: hidden;
            text-overflow: ellipsis; }}
        .kpi-card .kpi-caption {{ font-size: .76rem; color: {COLORS['text2']}; margin-top: 2px;
            white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
        .kpi-cyan {{ border-left-color: {COLORS['cyan']}; }}
        .kpi-cyan .kpi-value {{ color: {COLORS['cyan']}; }}
        .kpi-purple {{ border-left-color: {COLORS['purple']}; }}
        .kpi-purple .kpi-value {{ color: {COLORS['purple']}; }}
        .kpi-green {{ border-left-color: {COLORS['green']}; }}
        .kpi-green .kpi-value {{ color: {COLORS['green']}; }}
        .kpi-orange {{ border-left-color: {COLORS['orange']}; }}
        .kpi-orange .kpi-value {{ color: {COLORS['orange']}; }}
        .kpi-red {{ border-left-color: {COLORS['red']}; }}
        .kpi-red .kpi-value {{ color: {COLORS['red']}; }}

        /* ---- Panel headers ---- */
        .panel-header {{ margin: 0 0 .4rem; }}
        .panel-header .panel-title {{ font-size: 1.02rem; font-weight: 700; color: {COLORS['text']}; }}
        .panel-header .panel-sub {{ display:block; font-size: .8rem; color: {COLORS['text2']}; opacity:.85; }}

        /* ---- Notes ---- */
        .decision-note {{
            background: {COLORS['card2']};
            border: 1px solid {COLORS['border']};
            border-left: 4px solid {COLORS['purple']};
            border-radius: 12px; padding: 12px 14px; color: {COLORS['text2']};
            font-size: .88rem; line-height: 1.5;
        }}
        .decision-note .note-title {{ color: {COLORS['purple']}; font-weight: 700;
            font-size: .76rem; text-transform: uppercase; letter-spacing: .6px; display:block; margin-bottom:4px; }}
        .warning-note {{
            background: rgba(245,158,11,.08);
            border: 1px solid rgba(245,158,11,.35);
            border-left: 4px solid {COLORS['orange']};
            border-radius: 12px; padding: 12px 14px; color: #FCD9A1; font-size: .86rem; line-height:1.5;
        }}

        .page-title {{ font-size: 1.55rem; font-weight: 800; color: {COLORS['text']}; margin-bottom: 0; }}
        .page-sub {{ font-size: .9rem; color: {COLORS['text2']}; opacity:.85; margin: 2px 0 .6rem; }}

        /* ---- Tables ---- */
        div[data-testid="stDataFrame"] {{ border: 1px solid {COLORS['border']}; border-radius: 12px; }}

        /* ---- Radio nav spacing ---- */
        section[data-testid="stSidebar"] div[role="radiogroup"] label {{
            padding: 2px 0; font-size: .92rem;
        }}
        </style>
        """,
        unsafe_allow_html=True,
    )

def _esc(x) -> str:
    return (str(x).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;"))

def render_kpi_card(label, value, caption="", accent="cyan", full=None, max_chars=16):

    v = "-" if value is None else str(value)
    show_caption = caption or ""

    if len(v) > max_chars:
        if not show_caption and full is None:
            full = v
        v = v[: max_chars - 1].rstrip() + "…"
        if full and not show_caption:
            show_caption = _esc(full)
    cap_html = f'<div class="kpi-caption">{_esc(show_caption)}</div>' if show_caption else ""
    st.markdown(
        f'<div class="kpi-card kpi-{accent}">'
        f'<div class="kpi-label">{_esc(label)}</div>'
        f'<div class="kpi-value" title="{_esc(full or value)}">{_esc(v)}</div>'
        f'{cap_html}</div>',
        unsafe_allow_html=True,
    )

def kpi_row(cards):

    cols = st.columns(len(cards))
    for col, c in zip(cols, cards):
        with col:
            render_kpi_card(c.get("label"), c.get("value"), c.get("caption", ""),
                            c.get("accent", "cyan"), c.get("full"))

def render_panel_header(title, subtitle=None):
    sub = f'<span class="panel-sub">{_esc(subtitle)}</span>' if subtitle else ""
    st.markdown(f'<div class="panel-header"><span class="panel-title">{_esc(title)}</span>{sub}</div>',
                unsafe_allow_html=True)

def render_decision_note(text, title="Decision note"):
    st.markdown(f'<div class="decision-note"><span class="note-title">{_esc(title)}</span>{text}</div>',
                unsafe_allow_html=True)

def render_warning_note(text):
    st.markdown(f'<div class="warning-note">⚠️ {text}</div>', unsafe_allow_html=True)

def page_top(title, subtitle):
    st.markdown(f'<div class="page-title">{_esc(title)}</div>'
                f'<div class="page-sub">{_esc(subtitle)}</div>', unsafe_allow_html=True)

def style_fig(fig, height=360, geo=False):

    fig.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color=COLORS["text2"], size=12),
        margin=dict(l=10, r=10, t=34, b=10),
        height=height,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0,
                    bgcolor="rgba(0,0,0,0)", font=dict(color=COLORS["text2"])),
        coloraxis_colorbar=dict(outlinewidth=0, tickfont=dict(color=COLORS["text2"])),

        title=dict(text="", font=dict(color=COLORS["text"])),
    )
    if geo:
        fig.update_geos(bgcolor="rgba(0,0,0,0)", lakecolor="rgba(0,0,0,0)",
                        landcolor="#0c1626", showland=True,
                        coastlinecolor=COLORS["border"], framecolor=COLORS["border"])
    else:
        fig.update_xaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"],
                         linecolor=COLORS["border"])
        fig.update_yaxes(gridcolor=COLORS["grid"], zerolinecolor=COLORS["border"],
                         linecolor=COLORS["border"])
    return fig

def fix_year_axis(fig, year_values=(2023, 2024, 2025)):

    vals = list(year_values)
    fig.update_xaxes(
        tickmode="array",
        tickvals=vals,
        ticktext=[str(v) for v in vals],
        tickformat="d",
        title_text="Year",
    )
    return fig

def show_chart(fig, height=360, geo=False, year_axis=False):
    style_fig(fig, height=height, geo=geo)
    if year_axis:
        fix_year_axis(fig)
    st.plotly_chart(fig, use_container_width=True, config={"displayModeBar": False})

_INDUSTRY_SHORT_RULES = [
    ("computer programming", "ICT & programming"),
    ("information and communication", "Information & communication"),
    ("telecommunic", "Telecom"),
    ("scientific research", "R&D"),
    ("professional, scientific", "Professional services"),
    ("professional", "Professional services"),
    ("publishing", "Media & publishing"),
    ("motion picture", "Media & publishing"),
    ("manufactur", "Manufacturing"),
    ("financ", "Finance & insurance"),
    ("insurance", "Finance & insurance"),
    ("wholesale", "Wholesale & retail"),
    ("retail", "Wholesale & retail"),
    ("transport", "Transport & logistics"),
    ("accommodation", "Accommodation & food"),
    ("administrative", "Administrative services"),
    ("real estate", "Real estate"),
    ("construction", "Construction"),
    ("electricity", "Energy & utilities"),
    ("water supply", "Energy & utilities"),
]

def short_industry_label(name, max_len=24):

    if not isinstance(name, str) or not name.strip():
        return "Other"
    low = name.lower()
    for key, short in _INDUSTRY_SHORT_RULES:
        if key in low:
            return short

    if len(name) <= max_len:
        return name
    return name[:max_len].rsplit(" ", 1)[0].rstrip(",; ") + "…"

def unique_short_labels(names):

    seen = {}
    out = []
    for n in names:
        s = short_industry_label(n)
        if s in seen and seen[s] != n:

            suffix = 2
            base = s
            while f"{base} ({suffix})" in seen.values():
                suffix += 1
            s = f"{base} ({suffix})"
        seen[s] = n
        out.append(s)
    return out

def key_point_labels(df, value_col, name_col="country", n_top=6, n_bottom=3, always=None):

    keep = set()
    d = df.dropna(subset=[value_col])
    if not d.empty:
        keep.update(d.nlargest(min(n_top, len(d)), value_col)[name_col].tolist())
        keep.update(d.nsmallest(min(n_bottom, len(d)), value_col)[name_col].tolist())
    if always:
        keep.update(always)
    return df[name_col].where(df[name_col].isin(keep), "")

def _fmt_pct(x, dec=1):
    return f"{x:.{dec}f}%" if pd.notna(x) else "-"

def _fmt_pp(x, dec=1):
    return f"{x:+.{dec}f} pp" if pd.notna(x) else "-"
