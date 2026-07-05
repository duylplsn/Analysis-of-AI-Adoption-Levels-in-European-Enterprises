import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(
    page_title="AI Business Pulse Europe - Decision Support",
    page_icon="🤖",
    layout="wide",
)

from pulse_data import *
from pulse_viz import *

fact_size = safe_read_csv("fact_size.csv")
fact_industry = safe_read_csv("fact_industry.csv")
agg_cy = safe_read_csv("agg_country_year.csv")
agg_sy = safe_read_csv("agg_eu_size_year.csv")
agg_iy = safe_read_csv("agg_eu_industry_year.csv")
growth = safe_read_csv("summary_country_growth.csv")
ranking = safe_read_csv("summary_ranking_change.csv")
sme_gap = safe_read_csv("summary_sme_gap.csv")
industry_2025 = safe_read_csv("summary_industry_2025.csv")
quality = safe_read_csv("data_quality_summary.csv")
kpi = safe_load_json("kpi.json")

raw_dii = safe_read_raw(RAW_FILES["Digital intensity"])
raw_cloud = safe_read_raw(RAW_FILES["Cloud adoption"])
raw_das = safe_read_raw(RAW_FILES["Data analytics"])

PAGES = [
    "Overview",
    "Countries",
    "Market Priority",
    "SME Gap",
    "Industries",
    "AI Barriers",
    "AI Use Cases",
    "AI Readiness",
    "Data Confidence",
]

inject_css()

st.sidebar.markdown('<div class="sidebar-brand">AI Business Pulse</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-sub">Enterprise AI Adoption Strategy</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-label">Navigation</div>', unsafe_allow_html=True)
page = st.sidebar.radio("Navigation", PAGES, index=0, label_visibility="collapsed")

st.sidebar.markdown('<div class="sidebar-label">Filters</div>', unsafe_allow_html=True)

if not agg_cy.empty and "year" in agg_cy.columns:
    years = sorted(agg_cy["year"].dropna().unique().astype(int).tolist())
else:
    years = [2023, 2024, 2025]
year_sel = st.sidebar.select_slider("Year", options=years, value=max(years))

if not fact_size.empty and "enterprise_size_label" in fact_size.columns:
    size_options = sorted(fact_size["enterprise_size_label"].dropna().unique().tolist())
else:
    size_options = []
size_sel = st.sidebar.multiselect("Enterprise size", size_options, default=size_options)

if not agg_cy.empty and "country_type" in agg_cy.columns:
    country_options = sorted(
        agg_cy.loc[agg_cy["country_type"] == "Country", "country"].dropna().unique().tolist()
    )
else:
    country_options = []

with st.sidebar.expander("Countries", expanded=False):
    country_sel = st.multiselect(
        "Select countries", country_options, label_visibility="collapsed",
        placeholder="All countries",
        help="Leave empty to include all countries.",
    )
    if country_sel:
        st.caption(f"{len(country_sel)} of {len(country_options)} selected")
    else:
        st.caption("Showing all countries")

st.sidebar.markdown('<div class="sidebar-label">Source</div>', unsafe_allow_html=True)
st.sidebar.caption(
    "Eurostat enterprise surveys, 2023-2025. Metric: % of enterprises using >=1 AI "
    "technology, plus digital intensity, cloud, and data-analytics indicators."
)

if page == PAGES[0]:
    page_top("How is AI adoption evolving across Europe?",
             "A 30-second read on where European enterprise AI adoption stands today.")

    eu_now = kpi.get("eu_avg_2025") if kpi else None
    eu_growth = kpi.get("eu_growth_2023_2025_pp") if kpi else None
    tc = (kpi.get("top_country") or {}) if kpi else {}
    fastest = None
    if not growth.empty and {"country", "growth"}.issubset(growth.columns):
        gg = growth.dropna(subset=["growth"])
        if not gg.empty:
            fastest = gg.loc[gg["growth"].idxmax()]
    n_countries = kpi.get("countries_covered") if kpi else None
    if n_countries is None and not agg_cy.empty:
        n_countries = int(agg_cy[agg_cy["country_type"] == "Country"]["country"].nunique())

    kpi_row([
        {"label": "EU AI Adoption", "value": _fmt_pct(eu_now) if eu_now is not None else "-",
         "caption": _fmt_pp(eu_growth) + " vs 2023" if eu_growth is not None else "", "accent": "cyan"},
        {"label": "Top AI Market", "value": tc.get("name", "-"),
         "caption": _fmt_pct(tc.get("value")) if tc.get("value") is not None else "", "accent": "purple"},
        {"label": "Fastest Growth", "value": fastest["country"] if fastest is not None else "-",
         "caption": _fmt_pp(fastest["growth"]) if fastest is not None else "", "accent": "green"},
        {"label": "Countries Tracked", "value": str(n_countries) if n_countries else "-",
         "caption": "2023-2025 coverage", "accent": "cyan"},
    ])

    st.write("")
    left, right = st.columns([2, 1])
    with left:
        with st.container(border=True):
            render_panel_header("AI adoption trend", "EU average, 2023-2025")
            eu_df = agg_cy[agg_cy["country_type"] == "Aggregate"] if not agg_cy.empty else pd.DataFrame()
            if not eu_df.empty:
                trend = eu_df.groupby("year", as_index=False)["ai_usage_rate"].mean()
                fig = px.line(trend, x="year", y="ai_usage_rate", markers=True,
                              labels={"ai_usage_rate": "AI adoption (%)", "year": "Year"},
                              color_discrete_sequence=[COLORS["cyan"]])
                fig.update_traces(line=dict(width=3))
                show_chart(fig, height=320, year_axis=True)
            else:
                render_warning_note("Could not read `agg_country_year.csv`.")
    with right:
        with st.container(border=True):
            render_panel_header("What this means")
            eu_df = agg_cy[agg_cy["country_type"] == "Aggregate"] if not agg_cy.empty else pd.DataFrame()
            if not eu_df.empty:
                trend = eu_df.groupby("year", as_index=False)["ai_usage_rate"].mean()
                delta = trend["ai_usage_rate"].iloc[-1] - trend["ai_usage_rate"].iloc[0] if len(trend) >= 2 else float("nan")
                render_decision_note(
                    f"Average EU AI adoption rose by about <b>{delta:+.1f} pp</b> over "
                    f"{int(trend['year'].min())}-{int(trend['year'].max())}. The trend is clearly upward "
                    "but from a low base - significant headroom remains for enterprise AI.",
                    title="Insight",
                )
            else:
                render_decision_note("Trend data is unavailable.", title="Insight")

    st.write("")
    with st.container(border=True):
        render_panel_header("How spread out is adoption across countries?",
                            "Distribution per year - the gap between leaders and laggards")
        box_df = agg_cy[agg_cy["country_type"] == "Country"].copy() if not agg_cy.empty else pd.DataFrame()
        if not box_df.empty:
            box_df["year"] = box_df["year"].astype(str)
            fig = px.box(box_df, x="year", y="ai_usage_rate", points="all", color="year",
                         color_discrete_sequence=CHART_SEQUENCE)
            fig.update_layout(showlegend=False, yaxis_title="AI adoption (%)", xaxis_title="Year")
            show_chart(fig, height=320)
            render_decision_note(
                "The distribution widens every year - the gap between leaders and laggards is "
                "<b>widening, not narrowing</b>, signalling a need for targeted support in weaker markets.",
                title="Insight",
            )
        else:
            render_warning_note("Could not read `agg_country_year.csv`.")

elif page == PAGES[1]:
    page_top("Which countries are gaining momentum?",
             "Leaders, laggards, fast movers, and ranking shifts across Europe.")

    df = pd.DataFrame()
    if not agg_cy.empty:
        df = agg_cy[(agg_cy["country_type"] == "Country") & (agg_cy["year"] == year_sel)].dropna(subset=["ai_usage_rate"])
        if country_sel:
            df = df[df["country"].isin(country_sel)]

    top_row = df.loc[df["ai_usage_rate"].idxmax()] if not df.empty else None
    bot_row = df.loc[df["ai_usage_rate"].idxmin()] if not df.empty else None
    fastest = None
    if not growth.empty and {"country", "growth"}.issubset(growth.columns):
        gg = growth.dropna(subset=["growth"])
        fastest = gg.loc[gg["growth"].idxmax()] if not gg.empty else None
    riser = None
    if not ranking.empty and "rank_change" in ranking.columns:
        rr = ranking.dropna(subset=["rank_change"])
        riser = rr.loc[rr["rank_change"].idxmax()] if not rr.empty else None

    kpi_row([
        {"label": f"Top Market ({year_sel})", "value": top_row["country"] if top_row is not None else "-",
         "caption": _fmt_pct(top_row["ai_usage_rate"]) if top_row is not None else "", "accent": "purple"},
        {"label": f"Lowest Market ({year_sel})", "value": bot_row["country"] if bot_row is not None else "-",
         "caption": _fmt_pct(bot_row["ai_usage_rate"]) if bot_row is not None else "", "accent": "red"},
        {"label": "Fastest Growth", "value": fastest["country"] if fastest is not None else "-",
         "caption": _fmt_pp(fastest["growth"]) if fastest is not None else "", "accent": "green"},
        {"label": "Biggest Rank Gain", "value": riser["country"] if riser is not None else "-",
         "caption": f"+{int(riser['rank_change'])} places" if riser is not None else "", "accent": "cyan"},
    ])

    st.write("")
    cmap, cnote = st.columns([2, 1])
    with cmap:
        with st.container(border=True):
            render_panel_header(f"AI adoption map ({year_sel})", "% of enterprises using AI")
            if not df.empty:
                fig = px.choropleth(df, locations="country", locationmode="country names",
                                    color="ai_usage_rate", hover_name="country",
                                    color_continuous_scale="Viridis", labels={"ai_usage_rate": "AI (%)"})
                fig.update_layout(geo=dict(scope="europe"))
                show_chart(fig, height=420, geo=True)
            else:
                render_warning_note("Could not read `agg_country_year.csv`.")
    with cnote:
        with st.container(border=True):
            render_panel_header("Momentum vs level")
            if not growth.empty and {"2025", "growth", "country"}.issubset(growth.columns):
                gdf = growth.dropna(subset=["2025", "growth"]).copy()
                fig = px.scatter(gdf, x="growth", y="2025", size="2025", size_max=30,
                                 color="growth", color_continuous_scale="Tealgrn",
                                 hover_name="country",
                                 labels={"growth": "Growth (pp)", "2025": "AI 2025 (%)"})
                show_chart(fig, height=420)
            else:
                render_warning_note("Could not read `summary_country_growth.csv`.")

    st.write("")
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            render_panel_header("Top 10 leaders")
            if not df.empty:
                top = df.nlargest(10, "ai_usage_rate").sort_values("ai_usage_rate")
                fig = px.bar(top, x="ai_usage_rate", y="country", orientation="h",
                             color="ai_usage_rate", color_continuous_scale="Viridis",
                             labels={"ai_usage_rate": "AI (%)", "country": ""})
                show_chart(fig, height=320)
    with c2:
        with st.container(border=True):
            render_panel_header("Bottom 10 laggards")
            if not df.empty:
                bot = df.nsmallest(10, "ai_usage_rate").sort_values("ai_usage_rate", ascending=False)
                fig = px.bar(bot, x="ai_usage_rate", y="country", orientation="h",
                             color="ai_usage_rate", color_continuous_scale="Reds_r",
                             labels={"ai_usage_rate": "AI (%)", "country": ""})
                show_chart(fig, height=320)

    st.write("")
    with st.container(border=True):
        render_panel_header("Who moved up or down the ranking the most?", "Rank change 2023-2025")
        if not ranking.empty and "rank_change" in ranking.columns:
            rk = ranking.dropna(subset=["rank_change"]).copy()
            movers = pd.concat([rk.nlargest(7, "rank_change"), rk.nsmallest(7, "rank_change")]) \
                .drop_duplicates(subset=["country"]).sort_values("rank_change")
            fig = px.bar(movers, x="rank_change", y="country", orientation="h",
                         color="rank_change", color_continuous_scale="RdYlGn",
                         labels={"rank_change": "Rank change", "country": ""})
            show_chart(fig, height=360)
            render_decision_note(
                "Risers are outpacing the field; fallers are being overtaken by faster movers even if "
                "their own adoption still grows. Rank change is a <b>relative momentum</b> signal.",
                title="Insight")
        else:
            render_warning_note("Could not read `summary_ranking_change.csv`.")

elif page == PAGES[2]:
    page_top("Which markets should decision-makers prioritize?",
             "Four opportunity groups by AI adoption level and growth momentum.")

    seg = build_segments(growth)
    if seg.empty:
        render_warning_note("Could not read `summary_country_growth.csv`.")
    else:
        med_adopt = seg.attrs.get("med_adopt")
        med_growth = seg.attrs.get("med_growth")
        counts = seg["segment"].value_counts()

        kpi_row([
            {"label": "AI Leaders", "value": str(int(counts.get("AI Leaders", 0))),
             "caption": "high level + high growth", "accent": "green"},
            {"label": "Catch-up Markets", "value": str(int(counts.get("Catch-up Markets", 0))),
             "caption": "low level + high growth", "accent": "cyan"},
            {"label": "Mature Markets", "value": str(int(counts.get("Mature Markets", 0))),
             "caption": "high level + low growth", "accent": "purple"},
            {"label": "Lagging Markets", "value": str(int(counts.get("Lagging Markets", 0))),
             "caption": "low level + low growth", "accent": "red"},
        ])

        st.write("")
        cmain, cside = st.columns([2, 1])
        seg_color = {"AI Leaders": COLORS["green"], "Mature Markets": COLORS["purple"],
                     "Catch-up Markets": COLORS["cyan"], "Lagging Markets": COLORS["red"]}
        with cmain:
            with st.container(border=True):
                render_panel_header("Opportunity matrix",
                                    f"Split at median adoption ~{med_adopt:.1f}% and median growth ~{med_growth:.1f} pp")

                seg_lbl = seg.copy()
                seg_lbl["label"] = key_point_labels(seg_lbl, "adoption_2025", n_top=6, n_bottom=4)
                fig = px.scatter(seg_lbl, x="growth", y="adoption_2025", color="segment",
                                 text="label", hover_name="country",
                                 size="adoption_2025", size_max=34, color_discrete_map=seg_color,
                                 labels={"growth": "Growth 2023-2025 (pp)", "adoption_2025": "AI 2025 (%)",
                                         "segment": "Market group"})
                fig.update_traces(textposition="top center", textfont_size=9)
                fig.add_hline(y=med_adopt, line_dash="dash", line_color=COLORS["border"])
                fig.add_vline(x=med_growth, line_dash="dash", line_color=COLORS["border"])
                show_chart(fig, height=420)
        with cside:
            with st.container(border=True):
                render_panel_header("How to read it")
                render_decision_note(
                    "Most countries sit on two diagonals: <b>high-and-rising</b> (AI Leaders) and "
                    "<b>low-and-slow</b> (Lagging Markets). Policy and investment should differ sharply "
                    "by group rather than apply one formula to all.", title="Insight")

        st.write("")
        ctable, crec = st.columns([1, 1])
        with ctable:
            with st.container(border=True):
                render_panel_header("Country segmentation")
                seg_table = seg[["country", "adoption_2025", "growth", "segment"]].rename(columns={
                    "country": "Country", "adoption_2025": "Adoption 2025 (%)",
                    "growth": "Growth (pp)", "segment": "Market group"}) \
                    .sort_values(["Market group", "Adoption 2025 (%)"], ascending=[True, False])
                st.dataframe(seg_table.style.format({"Adoption 2025 (%)": "{:.1f}", "Growth (pp)": "{:+.1f}"}),
                             use_container_width=True, hide_index=True, height=360)
        with crec:
            with st.container(border=True):
                render_panel_header("What should decision-makers do next?")
                pick = st.selectbox("Country", options=seg["country"].sort_values().tolist())
                row = seg[seg["country"] == pick].iloc[0]
                seg_name = row["segment"]
                m1, m2, m3 = st.columns(3)
                with m1:
                    render_kpi_card("Group", seg_name, accent="purple", max_chars=12)
                with m2:
                    render_kpi_card("Adoption", _fmt_pct(row["adoption_2025"]), accent="cyan")
                with m3:
                    render_kpi_card("Growth", _fmt_pp(row["growth"]), accent="green")
                st.write("")
                render_decision_note(SEGMENT_RECO.get(seg_name, "-"),
                                     title=f"Recommendation - {pick}")
                st.caption("Rule-based guidance, not absolute conclusions - combine with local context.")

elif page == PAGES[3]:
    page_top("Are SMEs being left behind?",
             "The adoption gap between large enterprises and small firms.")

    gap_df = pd.DataFrame()
    if not sme_gap.empty and "sme_ai_gap" in sme_gap.columns:
        gap_df = sme_gap.dropna(subset=["sme_ai_gap"]).copy().rename(columns={"sme_ai_gap": "gap"})
    elif not fact_size.empty:
        sme = fact_size[(fact_size["country_type"] == "Country") & (fact_size["year"] == year_sel)
                        & fact_size["enterprise_size_label"].isin(["Small (10-49)", "Large (250+)"])
                        ].dropna(subset=["ai_usage_rate"])
        if not sme.empty:
            piv = sme.pivot_table(index="country", columns="enterprise_size_label",
                                  values="ai_usage_rate", aggfunc="mean").reset_index()
            if {"Large (250+)", "Small (10-49)"}.issubset(piv.columns):
                piv["gap"] = piv["Large (250+)"] - piv["Small (10-49)"]
                gap_df = piv.dropna(subset=["gap"])

    eu_gap = None
    if not agg_sy.empty:
        latest = agg_sy[agg_sy["year"] == year_sel]
        lg = latest.loc[latest["enterprise_size_label"] == "Large (250+)", "ai_usage_rate"]
        sm = latest.loc[latest["enterprise_size_label"] == "Small (10-49)", "ai_usage_rate"]
        if not lg.empty and not sm.empty:
            eu_gap = float(lg.iloc[0]) - float(sm.iloc[0])
    widest = gap_df.nlargest(1, "gap").iloc[0] if not gap_df.empty else None
    avg_gap = gap_df["gap"].mean() if not gap_df.empty else None

    kpi_row([
        {"label": f"EU Large-Small Gap ({year_sel})", "value": _fmt_pp(eu_gap).replace("+", "") if eu_gap is not None else "-",
         "caption": "EU average", "accent": "orange"},
        {"label": "Widest SME Gap", "value": widest["country"] if widest is not None else "-",
         "caption": f"{widest['gap']:.0f} pp" if widest is not None else "", "accent": "red"},
        {"label": "Avg Country Gap", "value": f"{avg_gap:.0f} pp" if avg_gap is not None else "-",
         "caption": "across countries", "accent": "purple"},
    ])

    st.write("")
    cleft, cright = st.columns([2, 1])
    with cleft:
        with st.container(border=True):
            render_panel_header("AI adoption by enterprise size", "EU trend, 2023-2025")
            if not agg_sy.empty:
                d = agg_sy.copy()
                if size_sel:
                    d = d[d["enterprise_size_label"].isin(size_sel)]
                fig = px.line(d, x="year", y="ai_usage_rate", color="enterprise_size_label", markers=True,
                              color_discrete_sequence=CHART_SEQUENCE,
                              labels={"ai_usage_rate": "AI (%)", "year": "Year", "enterprise_size_label": "Size"})
                show_chart(fig, height=340, year_axis=True)
            else:
                render_warning_note("Could not read `agg_eu_size_year.csv`.")
    with cright:
        with st.container(border=True):
            render_panel_header("Why it matters")
            render_decision_note(
                "Large firms almost always adopt first. The gap can reflect barriers in <b>budget, data "
                "skills, tech talent</b>, or <b>AI implementation capacity</b>. Where the gap is widest is "
                "where SME support has the biggest potential impact.", title="Insight")

    st.write("")
    with st.container(border=True):
        render_panel_header("How wide is the SME adoption gap by country?", "Top 15 - Large minus Small")
        if not gap_df.empty:
            top_gap = gap_df.nlargest(15, "gap").sort_values("gap")
            fig = px.bar(top_gap, x="gap", y="country", orientation="h",
                         color="gap", color_continuous_scale="OrRd",
                         labels={"gap": "Large - Small (pp)", "country": ""})
            show_chart(fig, height=420)
        else:
            render_warning_note("Could not read `summary_sme_gap.csv`.")

elif page == PAGES[4]:
    page_top("Where can AI create business value?",
             "Industry adoption and the AI use cases that fit each sector.")

    ind_df = pd.DataFrame()
    if not agg_iy.empty and "year" in agg_iy.columns:
        ind_df = agg_iy[agg_iy["year"] == year_sel].dropna(subset=["ai_usage_rate"]).copy()
    if ind_df.empty and not industry_2025.empty:
        ind_df = industry_2025.dropna(subset=["ai_usage_rate"]).copy()

    if ind_df.empty:
        render_warning_note("Could not read industry data (`agg_eu_industry_year.csv` / `summary_industry_2025.csv`).")
    else:
        top_ind = ind_df.sort_values("ai_usage_rate", ascending=False).head(20).copy()

        top_ind["display"] = unique_short_labels(top_ind["industry_label"].tolist())
        leader = top_ind.iloc[0]
        kpi_row([
            {"label": "Top Industry", "value": short_industry_label(leader["industry_label"]),
             "caption": _fmt_pct(leader["ai_usage_rate"]), "full": leader["industry_label"],
             "accent": "purple"},
            {"label": "Industries Tracked", "value": str(int(ind_df["industry_label"].nunique())),
             "caption": f"year {year_sel}", "accent": "cyan"},
            {"label": "Median Industry AI", "value": _fmt_pct(ind_df["ai_usage_rate"].median()),
             "caption": "across sectors", "accent": "green"},
        ])

        st.write("")
        ctree, cnote = st.columns([2, 1])
        with ctree:
            with st.container(border=True):
                render_panel_header(f"Top sectors by AI adoption ({year_sel})")
                fig = px.treemap(top_ind, path=["display"], values="ai_usage_rate",
                                 color="ai_usage_rate", color_continuous_scale="Viridis")

                fig.update_traces(
                    hovertemplate="<b>%{label}</b><br>AI adoption: %{value:.1f}%<extra></extra>")
                show_chart(fig, height=400)
                st.caption("Sector names are shortened for readability; full Eurostat names appear "
                           "in the use-case table below.")
        with cnote:
            with st.container(border=True):
                render_panel_header("Where value concentrates")
                render_decision_note(
                    "<b>ICT, communication, and professional services</b> lead clearly, while traditional "
                    "manufacturing and local services trail. AI value concentrates where data is available "
                    "and processes are easy to digitize first.", title="Insight")

        if not agg_iy.empty and "year" in agg_iy.columns:
            with st.container(border=True):
                render_panel_header("Are leading industries still pulling ahead?", "Top 8 sectors over time")
                top_codes = top_ind["industry"].head(8).tolist() if "industry" in top_ind.columns else []
                trend = agg_iy[agg_iy["industry"].isin(top_codes)] if top_codes else pd.DataFrame()
                if not trend.empty:
                    fig = px.line(trend, x="year", y="ai_usage_rate", color="industry_label", markers=True,
                                  color_discrete_sequence=CHART_SEQUENCE,
                                  labels={"ai_usage_rate": "AI (%)", "year": "Year", "industry_label": "Industry"})
                    show_chart(fig, height=340, year_axis=True)

        with st.container(border=True):
            render_panel_header("What AI use cases fit each industry?", "Rule-based mapping with safe fallback")
            rows = []
            for _, r in top_ind.iterrows():
                category, cases = map_industry_usecases(r["industry_label"])
                rows.append({"Sector": short_industry_label(r["industry_label"]),
                             "Industry (Eurostat)": r["industry_label"],
                             "Adoption (%)": round(float(r["ai_usage_rate"]), 1),
                             "Use-case group": category,
                             "Suggested AI use cases": " - ".join(cases)})
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=380,
                         column_config={
                             "Industry (Eurostat)": st.column_config.TextColumn(width="large"),
                             "Suggested AI use cases": st.column_config.TextColumn(width="large"),
                         })

elif page == PAGES[5]:
    page_top("What prevents enterprises from adopting AI?",
             "The most common barriers to AI adoption, by enterprise size and industry.")

    bar_size = safe_read_barrier("size")
    bar_ind = safe_read_barrier("industry")
    bars = build_barriers(bar_size, bar_ind, year=2025)
    overall, by_size, by_industry, meta = bars["overall"], bars["by_size"], bars["by_industry"], bars["meta"]

    if overall.empty and by_size.empty and by_industry.empty:
        render_warning_note(
            "<b>Barrier data unavailable —</b> no usable AI-barrier indicators were found in "
            f"<code>{RAW}/</code>. Add <code>isoc_eb_ai_barriers_linear.csv</code> and/or "
            "<code>isoc_eb_ain2_barriers_linear.csv</code> (Eurostat AI-barrier datasets) to enable this page.")
    else:

        top_barrier = overall.iloc[0] if not overall.empty else None
        sme_barrier = None
        if not by_size.empty:
            small = by_size[by_size["size"] == "Small (10-49)"]
            if not small.empty:
                sme_barrier = small.loc[small["value"].idxmax()]
        cost_segment = None
        if not by_size.empty:
            cost = by_size[by_size["barrier"] == "Costs too high"]
            if not cost.empty:
                cost_segment = cost.loc[cost["value"].idxmax()]
        legal_concern = None
        if not by_size.empty:
            lp = by_size[by_size["barrier"].isin(["Legal uncertainty", "Privacy/data protection"])]
            if not lp.empty:
                legal_concern = lp.loc[lp["value"].idxmax()]

        kpi_row([
            {"label": "Top Barrier",
             "value": BARRIER_SHORT.get(top_barrier["barrier"], top_barrier["barrier"]) if top_barrier is not None else "Not available",
             "caption": _fmt_pct(top_barrier["value"]) + " of enterprises" if top_barrier is not None else "",
             "full": top_barrier["barrier"] if top_barrier is not None else None, "accent": "red"},
            {"label": "Highest SME Barrier",
             "value": BARRIER_SHORT.get(sme_barrier["barrier"], sme_barrier["barrier"]) if sme_barrier is not None else "Not available",
             "caption": _fmt_pct(sme_barrier["value"]) + " of small enterprises" if sme_barrier is not None else "",
             "full": sme_barrier["barrier"] if sme_barrier is not None else None, "accent": "orange"},
            {"label": "Most Cost-Sensitive Segment", "value": cost_segment["size"] if cost_segment is not None else "Not available",
             "caption": ("highest share citing cost: " + _fmt_pct(cost_segment["value"])) if cost_segment is not None else "",
             "accent": "purple"},
            {"label": "Strongest Legal/Privacy Concern", "value": legal_concern["size"] if legal_concern is not None else "Not available",
             "caption": (f"{BARRIER_SHORT.get(legal_concern['barrier'], legal_concern['barrier'])}: " + _fmt_pct(legal_concern["value"])) if legal_concern is not None else "",
             "accent": "cyan", "full": (legal_concern["barrier"] if legal_concern is not None else None)},
        ])

        st.write("")
        cbar, cnote = st.columns([2, 1])
        with cbar:
            with st.container(border=True):
                render_panel_header("Top AI adoption barriers",
                                    "Mean share of all enterprises reporting each reason (2025)")
                if not overall.empty:
                    o = overall.sort_values("value").copy()
                    o["label"] = o["barrier"].map(lambda b: BARRIER_SHORT.get(b, b))
                    fig = px.bar(o, x="value", y="label", orientation="h", custom_data=["barrier"],
                                 color="value", color_continuous_scale="Reds",
                                 labels={"value": "% of enterprises", "label": ""})
                    fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% of enterprises<extra></extra>")
                    show_chart(fig, height=360)
                    st.caption("Denominator: all enterprises (not only non-adopters). Short labels shown; "
                               "full Eurostat reasons appear on hover and in the table below.")
                else:
                    st.caption("Overall barrier ranking is not available in the data.")
        with cnote:
            with st.container(border=True):
                render_panel_header("What stands out")
                if top_barrier is not None:
                    render_decision_note(
                        f"<b>{top_barrier['barrier']}</b> appears to be the most reported obstacle "
                        f"(~{top_barrier['value']:.1f}% of enterprises). Skills and uncertainty tend to be "
                        "reported more than raw cost - pointing toward training, guidance, and governance.",
                        title="Insight")
                else:
                    render_decision_note("Barrier ranking is not available.", title="Insight")

        if not by_size.empty:
            st.write("")
            with st.container(border=True):
                render_panel_header("Barriers by enterprise size",
                                    "Share of enterprises reporting each reason, by size (2025)")
                bsz = by_size.copy()
                bsz["label"] = bsz["barrier"].map(lambda b: BARRIER_SHORT.get(b, b))
                order = [BARRIER_SHORT.get(b, b) for b in overall["barrier"].tolist()] if not overall.empty else None
                fig = px.bar(bsz, x="value", y="label", color="size", barmode="group",
                             orientation="h", color_discrete_sequence=CHART_SEQUENCE, custom_data=["barrier"],
                             category_orders={"label": order} if order else None,
                             labels={"value": "% of enterprises", "label": "", "size": "Size"})
                fig.update_traces(hovertemplate="<b>%{customdata[0]}</b><br>%{x:.1f}% of enterprises<extra></extra>")
                show_chart(fig, height=420)
                render_decision_note(
                    "Larger enterprises appear more likely to report <b>cost</b> and <b>legal/privacy</b> "
                    "concerns, while smaller enterprises appear more likely to report an <b>expertise gap</b> "
                    "- so SME support could lead with skills and low-code tooling.", title="Insight")

        if not by_industry.empty:
            st.write("")
            with st.container(border=True):
                render_panel_header("Barrier profile by industry",
                                    "Top sectors by AI adoption - share of enterprises reporting each barrier")
                top_secs = (by_industry.groupby("industry")["value"].sum()
                            .sort_values(ascending=False).head(8).index.tolist())
                prof = by_industry[by_industry["industry"].isin(top_secs)].copy()
                prof["Sector"] = unique_short_labels(prof["industry"].tolist())
                prof["label"] = prof["barrier"].map(lambda b: BARRIER_SHORT.get(b, b))
                fig = px.density_heatmap(
                    prof, x="label", y="Sector", z="value", histfunc="avg",
                    color_continuous_scale="Reds",
                    labels={"value": "% of enterprises", "label": "", "Sector": ""})
                fig.update_xaxes(tickangle=-30)
                show_chart(fig, height=420)
                st.caption("Sector and barrier names shortened for readability; full Eurostat names are in "
                           "the recommendation table below.")
        elif meta.get("industry_available"):
            st.caption("Industry-level barrier breakdown is present but has no usable rows for the selected scope.")

        st.write("")
        with st.container(border=True):
            render_panel_header("What can address each barrier?", "Rule-based guidance per barrier")
            if not overall.empty:
                rec_rows = []
                for _, r in overall.iterrows():
                    rec_rows.append({"Barrier": r["barrier"],
                                     "Share (%)": round(float(r["value"]), 1),
                                     "Recommended response": BARRIER_RECO.get(r["barrier"], "-")})
                st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True, height=340,
                             column_config={"Recommended response": st.column_config.TextColumn(width="large")})
            else:
                st.caption("Recommendations require the overall barrier ranking, which is unavailable.")
        st.caption("AI barrier indicators guide practical interpretation only - they are not causal proof "
                   "of why AI adoption differs across countries, sectors, or enterprise sizes.")

elif page == PAGES[6]:
    page_top("What do enterprises use AI for?",
             "The most common AI use purposes, by enterprise size and industry.")

    pur_size = safe_read_purpose("size")
    pur_ind = safe_read_purpose("industry")
    pur = build_purposes(pur_size, pur_ind, year=2025)
    overall, by_size, by_industry, meta = pur["overall"], pur["by_size"], pur["by_industry"], pur["meta"]

    if overall.empty and by_size.empty and by_industry.empty:
        render_warning_note(
            "<b>Purpose data unavailable —</b> no usable AI use-purpose indicators were found in "
            f"<code>{RAW}/</code>. Add <code>isoc_eb_ai_purpose_linear.csv</code> and/or "
            "<code>isoc_eb_ain2_purpose_linear.csv</code> (Eurostat AI use-purpose datasets) to enable this page.")
    else:
        denom = "share of AI-adopting enterprises"

        top_purpose = overall.iloc[0] if not overall.empty else None
        low_purpose = overall.iloc[-1] if not overall.empty else None

        def _purpose_val(name):
            if overall.empty:
                return None
            row = overall[overall["purpose"] == name]
            return row.iloc[0] if not row.empty else None
        sec = _purpose_val("ICT security")
        rnd = _purpose_val("R&D & innovation")

        kpi_row([
            {"label": "Most Common AI Purpose", "value": top_purpose["purpose"] if top_purpose is not None else "Not available",
             "caption": (_fmt_pct(top_purpose["value"]) + " of AI users") if top_purpose is not None else "",
             "accent": "cyan"},
            {"label": "Highest Security Use", "value": _fmt_pct(sec["value"]) if sec is not None else "Not available",
             "caption": "ICT security (of AI users)" if sec is not None else "", "accent": "purple"},
            {"label": "Strongest R&D / Innovation Use", "value": _fmt_pct(rnd["value"]) if rnd is not None else "Not available",
             "caption": "R&D & innovation (of AI users)" if rnd is not None else "", "accent": "green"},
            {"label": "Lowest AI Purpose Area", "value": low_purpose["purpose"] if low_purpose is not None else "Not available",
             "caption": (_fmt_pct(low_purpose["value"]) + " of AI users") if low_purpose is not None else "",
             "accent": "orange"},
        ])

        st.write("")
        cbar, cnote = st.columns([2, 1])
        with cbar:
            with st.container(border=True):
                render_panel_header("Top AI use purposes",
                                    f"Mean {denom} reporting each purpose (2025)")
                if not overall.empty:
                    o = overall.sort_values("value")
                    fig = px.bar(o, x="value", y="purpose", orientation="h",
                                 color="value", color_continuous_scale="Tealgrn",
                                 labels={"value": "% of AI adopters", "purpose": ""})
                    show_chart(fig, height=360)
                else:
                    st.caption("Overall purpose ranking is not available in the data.")
        with cnote:
            with st.container(border=True):
                render_panel_header("What stands out")
                if top_purpose is not None:
                    render_decision_note(
                        f"<b>{top_purpose['purpose']}</b> is the most common application "
                        f"(~{top_purpose['value']:.1f}% of AI adopters), with business administration and "
                        "ICT security close behind. Front-office and operational uses lead adoption.",
                        title="Insight")
                else:
                    render_decision_note("Purpose ranking is not available.", title="Insight")
                st.caption(f"Denominator: {denom} (enterprises using >=1 AI technology).")

        if not by_size.empty:
            st.write("")
            with st.container(border=True):
                render_panel_header("AI purposes by enterprise size",
                                    f"{denom.capitalize()} reporting each purpose, by size")
                order = overall["purpose"].tolist() if not overall.empty else None
                fig = px.bar(by_size, x="value", y="purpose", color="size", barmode="group",
                             orientation="h", color_discrete_sequence=CHART_SEQUENCE,
                             category_orders={"purpose": order} if order else None,
                             labels={"value": "% of AI adopters", "purpose": "", "size": "Size"})
                show_chart(fig, height=420)
                render_decision_note(
                    "Larger enterprises appear more likely to report operational uses such as production "
                    "and logistics, while front-office uses such as marketing are widespread across sizes.",
                    title="Insight")

        if not by_industry.empty:
            st.write("")
            with st.container(border=True):
                render_panel_header("AI purpose profile by industry",
                                    "Top sectors by AI use - share of AI adopters per purpose")
                top_secs = (by_industry.groupby("industry")["value"].sum()
                            .sort_values(ascending=False).head(8).index.tolist())
                prof = by_industry[by_industry["industry"].isin(top_secs)].copy()
                prof["Sector"] = unique_short_labels(prof["industry"].tolist())
                fig = px.density_heatmap(
                    prof, x="purpose", y="Sector", z="value", histfunc="avg",
                    color_continuous_scale="Tealgrn",
                    labels={"value": "% of AI adopters", "purpose": "", "Sector": ""})
                fig.update_xaxes(tickangle=-30)
                show_chart(fig, height=420)
                st.caption("Sector names shortened for readability; full Eurostat names are in the recommendation table below.")
        elif meta.get("industry_available"):
            st.caption("Industry-level purpose breakdown is present but has no usable rows for the selected scope.")

        st.write("")
        with st.container(border=True):
            render_panel_header("Which AI use cases fit each purpose?", "Practical use cases per purpose area")
            if not overall.empty:
                rows = []
                for _, r in overall.iterrows():
                    rows.append({"AI purpose": r["purpose"],
                                 "Share of AI users (%)": round(float(r["value"]), 1),
                                 "Practical use cases": " - ".join(purpose_usecases(r["purpose"]))})
                st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True, height=340,
                             column_config={"Practical use cases": st.column_config.TextColumn(width="large")})
            else:
                st.caption("Recommendations require the overall purpose ranking, which is unavailable.")
        st.caption("AI purpose indicators describe how AI adopters apply AI; they are not causal proof of "
                   "why adoption differs across countries, sectors, or enterprise sizes.")

elif page == PAGES[7]:
    page_top("Which countries are ready to scale AI adoption?",
             "A composite readiness score plus the gaps between digital capability and AI use.")

    raw_map = {"Digital intensity": raw_dii, "Cloud adoption": raw_cloud, "Data analytics": raw_das}
    for label, fname in RAW_FILES.items():
        if raw_map[label].empty:
            render_warning_note(
                f"Dataset for <b>{label}</b> (<code>{fname}</code>) is missing from <code>{RAW}/</code>. "
                "The readiness score will use the remaining available indicators.")

    rd = build_readiness(agg_cy, raw_dii, raw_cloud, raw_das)
    if rd.empty:
        render_warning_note("AI adoption data or the readiness datasets are unavailable.")
    else:
        best = rd.loc[rd["readiness"].idxmax()] if rd["readiness"].notna().any() else None
        cmax = rd.loc[rd["cloud_to_ai_gap"].idxmax()] if rd["cloud_to_ai_gap"].notna().any() else None
        dmax = rd.loc[rd["data_to_ai_gap"].idxmax()] if rd["data_to_ai_gap"].notna().any() else None
        kpi_row([
            {"label": "Highest Readiness", "value": best["country"] if best is not None else "-",
             "caption": f"score {best['readiness']:.2f}" if best is not None else "", "accent": "cyan"},
            {"label": "Largest Cloud-to-AI Gap", "value": cmax["country"] if cmax is not None else "-",
             "caption": _fmt_pp(cmax["cloud_to_ai_gap"]) if cmax is not None else "", "accent": "purple"},
            {"label": "Largest Data-to-AI Gap", "value": dmax["country"] if dmax is not None else "-",
             "caption": _fmt_pp(dmax["data_to_ai_gap"]) if dmax is not None else "", "accent": "green"},
        ])

        st.write("")
        csc, cnote = st.columns([2, 1])
        with csc:
            with st.container(border=True):
                render_panel_header("Readiness vs current adoption", "Top-left = ready but under-adopted")
                sc = rd.dropna(subset=["ai_2025", "readiness"]).copy()
                if not sc.empty:

                    sc["label"] = key_point_labels(sc, "readiness", n_top=6, n_bottom=4)
                    fig = px.scatter(sc, x="ai_2025", y="readiness", text="label", hover_name="country",
                                     size="readiness", size_max=30, color="readiness",
                                     color_continuous_scale="Viridis",
                                     labels={"ai_2025": "AI 2025 (%)", "readiness": "Readiness score"})
                    fig.update_traces(textposition="top center", textfont_size=9)
                    fig.add_hline(y=sc["readiness"].median(), line_dash="dash", line_color=COLORS["border"])
                    fig.add_vline(x=sc["ai_2025"].median(), line_dash="dash", line_color=COLORS["border"])
                    show_chart(fig, height=420)
        with cnote:
            with st.container(border=True):
                render_panel_header("How to read the score")
                render_decision_note(
                    "AI Readiness Score = mean of min-max normalized <b>digital intensity</b>, <b>cloud "
                    "adoption</b>, <b>data analytics</b>, and <b>AI growth</b>. A large positive Cloud-to-AI "
                    "or Data-to-AI gap means the digital plumbing exists but is not yet converted into AI use - "
                    "a prime target for AI tooling rather than infrastructure.", title="Insight")

        st.write("")
        cg, dg = st.columns(2)
        with cg:
            with st.container(border=True):
                render_panel_header("Top 10 Cloud-to-AI Gap")
                g = rd.dropna(subset=["cloud_to_ai_gap"]).nlargest(10, "cloud_to_ai_gap").sort_values("cloud_to_ai_gap")
                if not g.empty:
                    fig = px.bar(g, x="cloud_to_ai_gap", y="country", orientation="h",
                                 color="cloud_to_ai_gap", color_continuous_scale="Blues",
                                 labels={"cloud_to_ai_gap": "Cloud - AI (pp)", "country": ""})
                    show_chart(fig, height=320)
                else:
                    st.caption("Cloud data unavailable for the gap chart.")
        with dg:
            with st.container(border=True):
                render_panel_header("Top 10 Data-to-AI Gap")
                g = rd.dropna(subset=["data_to_ai_gap"]).nlargest(10, "data_to_ai_gap").sort_values("data_to_ai_gap")
                if not g.empty:
                    fig = px.bar(g, x="data_to_ai_gap", y="country", orientation="h",
                                 color="data_to_ai_gap", color_continuous_scale="Greens",
                                 labels={"data_to_ai_gap": "Data - AI (pp)", "country": ""})
                    show_chart(fig, height=320)
                else:
                    st.caption("Data-analytics data unavailable for the gap chart.")

        st.write("")
        crank, crec = st.columns([1, 1])
        with crank:
            with st.container(border=True):
                render_panel_header("AI Readiness ranking")
                table = rd.sort_values("readiness", ascending=False)[
                    ["country", "readiness", "ai_2025", "cloud_to_ai_gap", "data_to_ai_gap"]].rename(columns={
                        "country": "Country", "readiness": "Readiness", "ai_2025": "AI 2025 (%)",
                        "cloud_to_ai_gap": "Cloud-AI (pp)", "data_to_ai_gap": "Data-AI (pp)"})
                st.dataframe(table.style.format({"Readiness": "{:.2f}", "AI 2025 (%)": "{:.1f}",
                                                 "Cloud-AI (pp)": "{:+.1f}", "Data-AI (pp)": "{:+.1f}"}, na_rep="-"),
                             use_container_width=True, hide_index=True, height=380)
        with crec:
            with st.container(border=True):
                render_panel_header("Country recommendations", "Based on readiness & gaps")
                med = {"ai_2025": rd["ai_2025"].median(), "readiness": rd["readiness"].median(),
                       "cloud_2025": rd["cloud_2025"].median(), "das_2025": rd["das_2025"].median(),
                       "dii_2025": rd["dii_2025"].median()}
                rec_rows = []
                for _, r in rd.sort_values("readiness", ascending=False).iterrows():
                    rec_rows.append({"Country": r["country"],
                                     "Readiness": round(r["readiness"], 2) if pd.notna(r["readiness"]) else None,
                                     "Recommendation": readiness_recommendation(r, med)})
                st.dataframe(pd.DataFrame(rec_rows), use_container_width=True, hide_index=True, height=380,
                             column_config={"Recommendation": st.column_config.TextColumn(width="large")})
        st.caption(READINESS_RECO_HELP)

elif page == PAGES[8]:
    page_top("Can we trust the data?",
             "Coverage, missing values, and observation flags behind every decision.")

    n_datasets = int(quality["dataset"].nunique()) if (not quality.empty and "dataset" in quality.columns) else 0
    avg_missing = None
    worst = None
    if not quality.empty and "missing_rate_percent" in quality.columns:
        avg_missing = pd.to_numeric(quality["missing_rate_percent"], errors="coerce").mean()
        wi = pd.to_numeric(quality["missing_rate_percent"], errors="coerce").idxmax()
        worst = quality.loc[wi] if pd.notna(wi) else None
    rd = build_readiness(agg_cy, raw_dii, raw_cloud, raw_das)
    n_ai = int(agg_cy[(agg_cy["country_type"] == "Country") & (agg_cy["year"] == 2025)]["country"].nunique()) \
        if not agg_cy.empty else 0
    n_full = 0
    if not rd.empty:
        n_full = len(rd.dropna(subset=["dii_2025", "cloud_2025", "das_2025", "ai_2025", "ai_2023"]))

    n_ai_datasets = n_datasets
    n_readiness_present = sum(1 for d in (raw_dii, raw_cloud, raw_das) if not d.empty)
    n_tracked = n_ai_datasets + n_readiness_present

    kpi_row([
        {"label": "Tracked Datasets", "value": str(n_tracked) if n_tracked else "-",
         "caption": f"{n_ai_datasets} AI adoption + {n_readiness_present} readiness", "accent": "cyan"},
        {"label": "Avg Missing Rate", "value": f"{avg_missing:.1f}%" if avg_missing is not None else "-",
         "caption": "AI adoption datasets", "accent": "orange"},
        {"label": "Readiness Coverage", "value": f"{n_full}/{n_ai}" if n_ai else "-",
         "caption": "countries with full data", "accent": "green"},
        {"label": "Highest Missing", "value": FRIENDLY_DATASET.get(worst["dataset"], worst["dataset"]) if worst is not None else "-",
         "caption": f"{worst['missing_rate_percent']:.1f}%" if worst is not None else "", "accent": "red"},
    ])

    st.write("")
    cleft, cright = st.columns([2, 1])
    with cleft:
        with st.container(border=True):
            render_panel_header("Missing values by dataset", "Lower is better")
            if not quality.empty and {"dataset", "missing_rate_percent"}.issubset(quality.columns):
                qd = quality.sort_values("missing_rate_percent").copy()
                qd["Dataset"] = qd["dataset"].map(lambda x: FRIENDLY_DATASET.get(x, x))
                fig = px.bar(qd, x="missing_rate_percent", y="Dataset", orientation="h",
                             color="missing_rate_percent", color_continuous_scale="OrRd",
                             labels={"missing_rate_percent": "Missing rate (%)", "Dataset": ""})
                show_chart(fig, height=300)
            else:
                render_warning_note("Could not read `data_quality_summary.csv`.")
    with cright:
        with st.container(border=True):
            render_panel_header("Coverage check")
            if not quality.empty and {"dataset", "missing_rate_percent"}.issubset(quality.columns):
                try:
                    m = quality.set_index("dataset")["missing_rate_percent"]
                    ind_miss, size_miss = m.get("ai_by_industry"), m.get("ai_by_enterprise_size")
                    if ind_miss is not None and size_miss is not None and ind_miss > size_miss:
                        render_warning_note(
                            f"<b>Data-quality note —</b> industry data is missing more often "
                            f"({ind_miss:.1f}%) than size data ({size_miss:.1f}%), so industry-level "
                            "comparisons should be read with a little more caution.")
                except Exception:
                    pass
            if n_ai and n_full < n_ai:
                render_warning_note(
                    f"<b>Data-quality note —</b> the AI Readiness Score is computed on <b>{n_full}</b> "
                    f"countries, fewer than the <b>{n_ai}</b> in the AI adoption data. This is expected "
                    "when some readiness indicators are missing; rankings near the median may shift slightly.")
            if (quality.empty or n_full >= n_ai) and not (not quality.empty and "ai_by_industry" in quality.get("dataset", pd.Series()).values):
                st.caption("Coverage looks consistent across the tracked datasets.")

    st.write("")
    with st.container(border=True):
        render_panel_header("AI Readiness dataset coverage", "Do the new datasets span 2023-2025?")
        needed_years = {2023, 2024, 2025}
        rmap = {
            "Digital intensity (isoc_e_dii)": (raw_dii, "high digital intensity", ["very"]),
            "Cloud adoption (isoc_cicce_use)": (raw_cloud,
                                                "Enterprises using paid cloud computing services used over the internet", None),
            "Data analytics (isoc_eb_das)": (raw_das, "Data analytics for the enterprise is performed by", None),
        }
        rows = []
        for label, (raw_df, contains, exclude) in rmap.items():
            if raw_df.empty:
                rows.append({"Dataset": label, "Available": "Missing", "Covers 2023-2025": "-",
                             "Missing years": "all", "Countries (2025)": 0, "Missing rate 2025 (%)": None})
                continue
            tidy = _extract_indicator(raw_df, contains, exclude)
            yrs = set(tidy["year"].unique().tolist()) if not tidy.empty else set()
            missing_years = sorted(needed_years - yrs)
            c2025 = tidy[tidy["year"] == 2025] if not tidy.empty else pd.DataFrame()
            rows.append({
                "Dataset": label, "Available": "Yes",
                "Covers 2023-2025": "Yes" if needed_years.issubset(yrs) else f"Partial ({len(needed_years & yrs)}/3)",
                "Missing years": ", ".join(map(str, missing_years)) if missing_years else "none",
                "Countries (2025)": int(c2025["country"].nunique()) if not c2025.empty else 0,
                "Missing rate 2025 (%)": round(100 * c2025["value"].isna().mean(), 1) if not c2025.empty else None,
            })
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.write("")
    with st.container(border=True):
        render_panel_header("AI Barriers dataset coverage", "Optional barrier datasets used by the AI Barriers page")
        b_size = safe_read_barrier("size")
        b_ind = safe_read_barrier("industry")
        b = build_barriers(b_size, b_ind, year=2025)
        bmeta = b["meta"]
        if not bmeta or (not bmeta.get("size_available") and not bmeta.get("industry_available")):
            render_warning_note(
                "<b>Data-quality note —</b> AI-barrier datasets were not found in "
                f"<code>{RAW}/</code>. The AI Barriers page is hidden/empty until they are added.")
        else:
            dim = ("Both size & industry" if bmeta.get("by_size_dim") and bmeta.get("by_industry_dim")
                   else "Enterprise size" if bmeta.get("by_size_dim")
                   else "Industry" if bmeta.get("by_industry_dim") else "-")
            brow = [{
                "Dataset": "AI barriers (size + industry)",
                "Available": "Yes",
                "Years covered": ", ".join(map(str, bmeta.get("years", []))) or "-",
                "Countries (2025)": bmeta.get("countries", 0),
                "Missing rate 2025 (%)": bmeta.get("missing_rate"),
                "Breakdown": dim,
            }]
            st.dataframe(pd.DataFrame(brow), use_container_width=True, hide_index=True)
        st.caption("AI barrier indicators are used to guide practical interpretation. They should not be "
                   "interpreted as causal proof of why AI adoption differs across countries, sectors, or "
                   "enterprise sizes.")

    st.write("")
    with st.container(border=True):
        render_panel_header("AI Use Cases dataset coverage", "Optional purpose datasets used by the AI Use Cases page")
        p_size = safe_read_purpose("size")
        p_ind = safe_read_purpose("industry")
        pp = build_purposes(p_size, p_ind, year=2025)
        pmeta = pp["meta"]
        if not pmeta or (not pmeta.get("size_available") and not pmeta.get("industry_available")):
            render_warning_note(
                "<b>Data-quality note —</b> AI use-purpose datasets were not found in "
                f"<code>{RAW}/</code>. The AI Use Cases page is hidden/empty until they are added.")
        else:
            dim = ("Both size & industry" if pmeta.get("by_size_dim") and pmeta.get("by_industry_dim")
                   else "Enterprise size" if pmeta.get("by_size_dim")
                   else "Industry" if pmeta.get("by_industry_dim") else "-")
            prow = [{
                "Dataset": "AI use purposes (size + industry)",
                "Available": "Yes",
                "Years covered": ", ".join(map(str, pmeta.get("years", []))) or "-",
                "Countries (2025)": pmeta.get("countries", 0),
                "Missing rate 2025 (%)": pmeta.get("missing_rate"),
                "Breakdown": dim,
            }]
            st.dataframe(pd.DataFrame(prow), use_container_width=True, hide_index=True)
        st.caption("AI purpose indicators describe how AI adopters apply AI. They should not be interpreted "
                   "as causal proof of why AI adoption differs across countries, sectors, or enterprise sizes.")

    st.write("")
    cflag, climit = st.columns([1, 1])
    with cflag:
        with st.container(border=True):
            render_panel_header("Observation flags", "Eurostat data-quality markers")
            flag_frames = []
            for name, dfX in [("By size", fact_size), ("By industry", fact_industry)]:
                if not dfX.empty and "obs_flag" in dfX.columns:
                    f = dfX["obs_flag"].dropna()
                    f = f[f.astype(str).str.strip() != ""]
                    if not f.empty:
                        vc = f.value_counts().reset_index()
                        vc.columns = ["obs_flag", "count"]
                        vc.insert(0, "dataset", name)
                        flag_frames.append(vc)
            if flag_frames:
                st.dataframe(pd.concat(flag_frames, ignore_index=True),
                             use_container_width=True, hide_index=True)
                st.caption("`b` = break in series, `e` = estimated, `p` = provisional, `d` = definition differs.")
            else:
                st.caption("No significant observation flags found in the available data.")
    with climit:
        with st.container(border=True):
            render_panel_header("Limitations")
            render_decision_note(
                "Time window is short (2023-2025), so no long-term forecasting is performed. The readiness "
                "score is a rule-based composite of normalized indicators and may cover fewer countries when "
                "inputs are missing.<br><br><b>Scores and segments are designed for relative comparison and "
                "decision support. They do not prove causality.</b>", title="Decision note")
