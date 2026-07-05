import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

__all__ = [
    'OUT_ROOT',
    'DATA',
    'KPI',
    'RAW',
    'SIZE_TOTAL',
    'NACE_TOTAL',
    'RAW_FILES',
    '_classify_country_type',
    '_map_size_label',
    '_shorten',
    '_read_raw_pipeline',
    '_base_clean',
    '_load_size',
    '_load_industry',
    '_quality_report',
    '_sme_gap',
    '_country_growth',
    '_ranking_change',
    '_industry_2025',
    '_eu_size_2025',
    '_build_kpis',
    'run_pipeline',
    'ensure_processed',
    'MAIN_INDICATOR',
    'MAIN_UNIT',
    'EU_NAME',
    'YEAR_MIN',
    'YEAR_MAX',
    'SIZE_ORDER',
    'SIZE_COMPARE_ORDER',
    'SIZE_PATH',
    'IND_PATH',
    '_PIPELINE_COLS',
    'safe_read_csv',
    'safe_read_raw',
    'safe_load_json',
    'build_segments',
    'SEGMENT_RECO',
    'map_industry_usecases',
    'USECASE_RULES',
    'USECASE_FALLBACK',
    '_minmax',
    '_extract_indicator',
    'build_readiness',
    'readiness_recommendation',
    'READINESS_RECO_HELP',
    '_short_barrier',
    'safe_read_barrier',
    'build_barriers',
    'FRIENDLY_DATASET',
    'BARRIER_FILES',
    'BARRIER_UNIT',
    'BARRIER_MARKER',
    'BARRIER_MAP',
    'BARRIER_SIZES',
    'BARRIER_RECO',
    'BARRIER_SHORT',
    '_short_purpose',
    'purpose_usecases',
    'safe_read_purpose',
    'build_purposes',
    'PURPOSE_FILES',
    'PURPOSE_UNIT',
    'PURPOSE_PREFIX',
    'PURPOSE_MAP',
    'PURPOSE_SIZES',
]

OUT_ROOT = Path(os.getenv("AI_OUT_DIR", "AI_Business_Pulse_EDA_v3"))
DATA = OUT_ROOT / "data"
KPI = OUT_ROOT / "kpi"
RAW = Path(os.getenv("AI_RAW_DIR", "data_raw"))

SIZE_TOTAL = "10 persons employed or more"
NACE_TOTAL = ("All activities (except agriculture, forestry and fishing, "
              "and mining and quarrying), without financial sector")
RAW_FILES = {
    "Digital intensity": "isoc_e_dii_linear.csv",
    "Cloud adoption": "isoc_cicce_use_linear.csv",
    "Data analytics": "isoc_eb_das_linear.csv",
}

MAIN_INDICATOR = ("Enterprises using at least one of the AI technologies: "
                  "AI_TTM, AI_TSR, AI_TNLG, AI_TIR, AI_TML, AI_TPA, AI_TAR, E_AI_TPVSG")
MAIN_UNIT = "Percentage of enterprises"
EU_NAME = "European Union - 27 countries (from 2020)"
YEAR_MIN = 2023
YEAR_MAX = 2025
SIZE_ORDER = ("All enterprises (10+)", "Small (10-49)", "Medium (50-249)", "Large (250+)")
SIZE_COMPARE_ORDER = ("Small (10-49)", "Medium (50-249)", "Large (250+)")

SIZE_PATH = RAW / "isoc_eb_ai_linear.csv"
IND_PATH = RAW / "isoc_eb_ain2_linear.csv"
_PIPELINE_COLS = ["size_emp", "nace_r2", "indic_is", "unit", "geo",
                  "TIME_PERIOD", "OBS_VALUE", "OBS_FLAG", "CONF_STATUS"]

def _classify_country_type(country):
    c = str(country)
    if "European Union" in c or "Euro area" in c:
        return "Aggregate"
    return "Country"

def _map_size_label(size_emp):
    s = str(size_emp)
    if "10 persons employed or more" in s:
        return "All enterprises (10+)"
    if "10 to 49" in s:
        return "Small (10-49)"
    if "50 to 249" in s:
        return "Medium (50-249)"
    if "250 persons" in s or "250 or more" in s:
        return "Large (250+)"
    if "10 to 249" in s:
        return "SMEs (10-249)"
    if "0 to 9" in s or "0 to 1" in s or "2 to 9" in s:
        return "Micro enterprises (<10)"
    return s

def _shorten(text, max_len=55):
    text = str(text)
    return text if len(text) <= max_len else text[: max_len - 1] + "\u2026"

def _read_raw_pipeline(path):
    return pd.read_csv(path, usecols=lambda c: c in _PIPELINE_COLS)

def _base_clean(df):
    out = df.loc[
        (df["indic_is"] == MAIN_INDICATOR)
        & (df["unit"] == MAIN_UNIT)
        & (df["TIME_PERIOD"].between(YEAR_MIN, YEAR_MAX))
    ].rename(columns={
        "geo": "country", "TIME_PERIOD": "year", "OBS_VALUE": "ai_usage_rate",
        "OBS_FLAG": "obs_flag", "CONF_STATUS": "conf_status",
    })
    out["country_type"] = out["country"].map(_classify_country_type)
    out["ai_usage_rate"] = pd.to_numeric(out["ai_usage_rate"], errors="coerce")
    out["year"] = out["year"].astype(int)
    return out

def _load_size():
    clean = _base_clean(_read_raw_pipeline(SIZE_PATH))
    clean["enterprise_size_label"] = clean["size_emp"].map(_map_size_label)
    cols = ["country", "country_type", "year", "size_emp", "enterprise_size_label",
            "ai_usage_rate", "obs_flag", "conf_status"]
    full = clean[cols].reset_index(drop=True)
    main = full[full["enterprise_size_label"].isin(SIZE_ORDER)].copy()
    return main

def _load_industry():
    clean = _base_clean(_read_raw_pipeline(IND_PATH))
    clean = clean.rename(columns={"nace_r2": "industry"})
    clean["industry_label"] = clean["industry"].map(_shorten)
    clean = clean[clean["size_emp"] == "10 persons employed or more"].copy()
    cols = ["country", "country_type", "year", "industry", "industry_label",
            "size_emp", "ai_usage_rate", "obs_flag", "conf_status"]
    return clean[cols].reset_index(drop=True)

def _quality_report(df, name):
    return {
        "dataset": name, "rows": int(len(df)), "columns": int(df.shape[1]),
        "years": sorted(df["year"].dropna().unique().tolist()),
        "countries": int(df["country"].nunique()),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_ai_usage_rate": int(df["ai_usage_rate"].isna().sum()),
        "missing_rate_percent": round(df["ai_usage_rate"].isna().mean() * 100, 2),
        "min_ai_usage_rate": float(np.nanmin(df["ai_usage_rate"])),
        "max_ai_usage_rate": float(np.nanmax(df["ai_usage_rate"])),
        "values_below_0": int((df["ai_usage_rate"] < 0).sum()),
        "values_above_100": int((df["ai_usage_rate"] > 100).sum()),
    }

def _sme_gap(size_main):
    base = size_main[
        (size_main["country_type"] == "Country")
        & (size_main["year"] == YEAR_MAX)
        & (size_main["enterprise_size_label"].isin(["Small (10-49)", "Large (250+)"]))
    ].dropna(subset=["ai_usage_rate"])
    pivot = base.pivot_table(index="country", columns="enterprise_size_label",
                             values="ai_usage_rate", aggfunc="mean").reset_index()
    pivot["sme_ai_gap"] = pivot["Large (250+)"] - pivot["Small (10-49)"]
    pivot = pivot.dropna(subset=["sme_ai_gap"]).copy()
    return pivot.sort_values("sme_ai_gap", ascending=False)

def _country_growth(country_all):
    wide = country_all.pivot_table(index="country", columns="year",
                                   values="ai_usage_rate", aggfunc="mean").reset_index()
    wide.columns = [str(c) for c in wide.columns]
    needed = [str(YEAR_MIN), str(YEAR_MAX)]
    wide = wide.dropna(subset=needed).copy()
    wide["growth"] = wide[str(YEAR_MAX)] - wide[str(YEAR_MIN)]
    wide["growth_group"] = pd.cut(wide["growth"], bins=[-100, 0, 5, 10, 100],
                                  labels=["Declining", "Slow", "Moderate", "Fast"])
    wide["abs_level"] = wide[str(YEAR_MAX)].clip(lower=1)
    return wide

def _ranking_change(country_all):
    wide = country_all.pivot_table(index="country", columns="year",
                                   values="ai_usage_rate", aggfunc="mean")
    wide = wide.dropna(subset=[YEAR_MIN, YEAR_MAX]).copy()
    wide["rank_%d" % YEAR_MIN] = wide[YEAR_MIN].rank(ascending=False, method="min").astype(int)
    wide["rank_%d" % YEAR_MAX] = wide[YEAR_MAX].rank(ascending=False, method="min").astype(int)
    wide["rank_change"] = wide["rank_%d" % YEAR_MIN] - wide["rank_%d" % YEAR_MAX]
    return wide.reset_index()

def _industry_2025(industry_clean):
    eu_ind = industry_clean[
        (industry_clean["country"] == EU_NAME) & (industry_clean["year"] == YEAR_MAX)
    ].dropna(subset=["ai_usage_rate"]).copy()
    if eu_ind.empty:
        eu_ind = (industry_clean[
            (industry_clean["country_type"] == "Country") & (industry_clean["year"] == YEAR_MAX)
        ].dropna(subset=["ai_usage_rate"])
            .groupby(["industry", "industry_label"], as_index=False)["ai_usage_rate"].mean())
    summary = (eu_ind.groupby(["industry", "industry_label"], as_index=False)["ai_usage_rate"]
               .mean().sort_values("ai_usage_rate", ascending=False))
    return summary

def _eu_size_2025(size_main):
    eu_size = size_main[
        (size_main["country"] == EU_NAME)
        & (size_main["enterprise_size_label"].isin(SIZE_COMPARE_ORDER))
    ].dropna(subset=["ai_usage_rate"]).copy()
    return eu_size[eu_size["year"] == YEAR_MAX].copy()

def _build_kpis(eu_all, country_2025, eu_size_2025, industry_summary):
    eu_2025 = eu_all.loc[eu_all["year"] == YEAR_MAX, "ai_usage_rate"]
    eu_2023 = eu_all.loc[eu_all["year"] == YEAR_MIN, "ai_usage_rate"]
    eu_avg = float(eu_2025.mean()) if not eu_2025.empty else None
    growth = (float(eu_2025.mean() - eu_2023.mean())
              if not eu_2025.empty and not eu_2023.empty else None)
    top_country = country_2025.nlargest(1, "ai_usage_rate").iloc[0]
    bot_country = country_2025.nsmallest(1, "ai_usage_rate").iloc[0]
    top_industry = industry_summary.iloc[0]
    leader_size = (eu_size_2025.sort_values("ai_usage_rate", ascending=False).iloc[0]
                   if not eu_size_2025.empty else None)
    return {
        "year": YEAR_MAX,
        "eu_avg_2025": round(eu_avg, 2) if eu_avg is not None else None,
        "eu_growth_2023_2025_pp": round(growth, 2) if growth is not None else None,
        "top_country": {"name": top_country["country"],
                        "value": round(float(top_country["ai_usage_rate"]), 2)},
        "bottom_country": {"name": bot_country["country"],
                           "value": round(float(bot_country["ai_usage_rate"]), 2)},
        "top_industry": {"name": str(top_industry["industry_label"]),
                         "value": round(float(top_industry["ai_usage_rate"]), 2)},
        "leading_size": ({"name": str(leader_size["enterprise_size_label"]),
                          "value": round(float(leader_size["ai_usage_rate"]), 2)}
                         if leader_size is not None else None),
        "countries_covered": int(country_2025["country"].nunique()),
        "industries_covered": int(industry_summary["industry"].nunique()),
    }

def run_pipeline():
    DATA.mkdir(parents=True, exist_ok=True)
    KPI.mkdir(parents=True, exist_ok=True)
    size_main = _load_size()
    industry_clean = _load_industry()
    quality = pd.DataFrame([
        _quality_report(size_main, "ai_by_enterprise_size"),
        _quality_report(industry_clean, "ai_by_industry"),
    ])
    quality.to_csv(DATA / "data_quality_summary.csv", index=False)
    all_ent = size_main[size_main["enterprise_size_label"] == "All enterprises (10+)"]
    country_all = all_ent[all_ent["country_type"] == "Country"].copy()
    eu_all = all_ent[all_ent["country"] == EU_NAME].copy()
    country_2025 = country_all[country_all["year"] == YEAR_MAX].dropna(subset=["ai_usage_rate"])
    _sme_gap(size_main).to_csv(DATA / "summary_sme_gap.csv", index=False)
    _country_growth(country_all).to_csv(DATA / "summary_country_growth.csv", index=False)
    _ranking_change(country_all).to_csv(DATA / "summary_ranking_change.csv", index=False)
    industry_summary = _industry_2025(industry_clean)
    industry_summary.to_csv(DATA / "summary_industry_2025.csv", index=False)
    eu_size_2025 = _eu_size_2025(size_main)
    size_main.to_csv(DATA / "fact_size.csv", index=False)
    industry_clean.to_csv(DATA / "fact_industry.csv", index=False)
    (size_main[size_main["enterprise_size_label"] == "All enterprises (10+)"]
        .groupby(["country", "country_type", "year"], as_index=False)["ai_usage_rate"].mean()
        .to_csv(DATA / "agg_country_year.csv", index=False))
    (size_main[size_main["country"] == EU_NAME]
        .groupby(["enterprise_size_label", "year"], as_index=False)["ai_usage_rate"].mean()
        .to_csv(DATA / "agg_eu_size_year.csv", index=False))
    (industry_clean[industry_clean["country"] == EU_NAME]
        .groupby(["industry", "industry_label", "year"], as_index=False)["ai_usage_rate"].mean()
        .to_csv(DATA / "agg_eu_industry_year.csv", index=False))
    kpi = _build_kpis(eu_all, country_2025, eu_size_2025, industry_summary)
    (KPI / "kpi.json").write_text(json.dumps(kpi, ensure_ascii=False, indent=2), encoding="utf-8")
    return True

@st.cache_resource(show_spinner="Building dataset from data_raw...")
def ensure_processed():
    return run_pipeline()

@st.cache_data(show_spinner=False)
def safe_read_csv(name):
    ensure_processed()
    p = DATA / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def safe_read_raw(name):
    p = RAW / name
    if not p.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(p)
    except Exception:
        return pd.DataFrame()

@st.cache_data(show_spinner=False)
def safe_load_json(name):
    ensure_processed()
    p = KPI / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}

SEGMENT_RECO = {
    "AI Leaders": (
        "Ready for **advanced AI**: deep automation, generative AI in products, and "
        "especially **AI governance** (oversight, ethics, model-risk control). This is "
        "where to invest in high-end solutions and build market leadership."
    ),
    "Mature Markets": (
        "Adoption is already high but slowing. The opportunity is in **optimizing** "
        "existing AI systems, **compliance** with new regulation, and upgrading / "
        "integrating what is already deployed rather than expanding the count."
    ),
    "Catch-up Markets": (
        "Growing fast from a low base - high potential. Prioritize **low-cost, "
        "easy-to-deploy basic AI tools** for SMEs (chatbots, document automation, sales "
        "analytics). This is the group where quick impact is easiest."
    ),
    "Lagging Markets": (
        "Both level and pace are low. Prioritize a **digital foundation before AI**: "
        "cloud/data infrastructure, process digitization, and **digital skills** for the "
        "workforce. AI only pays off once that foundation is in place."
    ),
}

@st.cache_data(show_spinner=False)
def build_segments(growth_df: pd.DataFrame) -> pd.DataFrame:

    if growth_df.empty or "2025" not in growth_df.columns or "growth" not in growth_df.columns:
        return pd.DataFrame()

    df = growth_df[["country", "2025", "growth"]].dropna().copy()
    df = df.rename(columns={"2025": "adoption_2025"})
    if df.empty:
        return df

    med_adopt = df["adoption_2025"].median()
    med_growth = df["growth"].median()

    def classify(row):
        high_adopt = row["adoption_2025"] >= med_adopt
        high_growth = row["growth"] >= med_growth
        if high_adopt and high_growth:
            return "AI Leaders"
        if high_adopt and not high_growth:
            return "Mature Markets"
        if not high_adopt and high_growth:
            return "Catch-up Markets"
        return "Lagging Markets"

    df["segment"] = df.apply(classify, axis=1)
    df.attrs["med_adopt"] = med_adopt
    df.attrs["med_growth"] = med_growth
    return df

USECASE_RULES = [
    (("information and communication", "telecommunic", "computer programming",
      "ict", "publishing", "software", "communication technology"),
     "Information & Communication",
     ["AI product development", "Process automation", "Data services"]),

    (("manufactur", "manufacture"),
     "Manufacturing",
     ["Predictive maintenance", "Quality inspection (computer vision)", "Production planning"]),

    (("financ", "insurance", "credit", "banking"),
     "Finance & Insurance",
     ["Fraud detection", "Credit scoring", "Risk analytics"]),

    (("transport", "logistic", "warehous", "postal", "storage", "delivery"),
     "Logistics & Transportation",
     ["Route optimization", "Warehouse optimization", "Delivery time prediction"]),

    (("retail", "wholesale", "trade", "accommodation", "food service", "motor vehicle"),
     "Retail & Trade",
     ["Demand forecasting", "Customer segmentation", "Personalized marketing"]),

    (("professional", "scientific", "technical", "legal", "accounting",
      "consult", "administrative", "real estate", "research"),
     "Professional Services",
     ["Document automation", "Knowledge management", "Decision support"]),
]

USECASE_FALLBACK = ["Automation", "Data analytics", "Operational decision support"]

def map_industry_usecases(label: str):

    if not isinstance(label, str):
        return ("Other", USECASE_FALLBACK)
    low = label.lower()
    for keywords, category, cases in USECASE_RULES:
        if any(k in low for k in keywords):
            return (category, cases)
    return ("Other", USECASE_FALLBACK)

def _minmax(s: pd.Series) -> pd.Series:

    lo, hi = s.min(), s.max()
    if pd.isna(lo) or pd.isna(hi) or hi <= lo:
        return pd.Series(np.where(s.notna(), 0.5, np.nan), index=s.index)
    return (s - lo) / (hi - lo)

@st.cache_data(show_spinner=False)
def _extract_indicator(raw_df: pd.DataFrame, contains: str, exclude=None) -> pd.DataFrame:

    if raw_df.empty:
        return pd.DataFrame(columns=["country", "year", "value"])
    needed = {"size_emp", "nace_r2", "indic_is", "geo", "TIME_PERIOD", "OBS_VALUE"}
    if not needed.issubset(raw_df.columns):
        return pd.DataFrame(columns=["country", "year", "value"])

    d = raw_df[(raw_df["size_emp"] == SIZE_TOTAL) & (raw_df["nace_r2"] == NACE_TOTAL)].copy()
    ind = d["indic_is"].astype(str)
    mask = ind.str.contains(contains, case=False, regex=False)
    for ex in (exclude or []):
        mask &= ~ind.str.contains(ex, case=False, regex=False)
    d = d[mask]
    if d.empty:
        return pd.DataFrame(columns=["country", "year", "value"])

    g = d.groupby(["geo", "TIME_PERIOD"], as_index=False)["OBS_VALUE"].mean()
    g = g.rename(columns={"geo": "country", "TIME_PERIOD": "year", "OBS_VALUE": "value"})
    return g

@st.cache_data(show_spinner=False)
def build_readiness(agg_country_year: pd.DataFrame,
                    dii_df: pd.DataFrame,
                    cloud_df: pd.DataFrame,
                    das_df: pd.DataFrame) -> pd.DataFrame:

    if agg_country_year.empty:
        return pd.DataFrame()

    ai = agg_country_year[agg_country_year["country_type"] == "Country"].copy()
    if ai.empty:
        return pd.DataFrame()
    ai_p = ai.pivot_table(index="country", columns="year", values="ai_usage_rate")

    out = pd.DataFrame(index=ai_p.index)
    out["ai_2023"] = ai_p.get(2023)
    out["ai_2025"] = ai_p.get(2025)

    def year_value(tidy: pd.DataFrame, yr: int) -> pd.Series:
        if tidy.empty:
            return pd.Series(dtype="float64")
        sub = tidy[tidy["year"] == yr]
        return sub.set_index("country")["value"]

    cloud_t = _extract_indicator(
        cloud_df, "Enterprises using paid cloud computing services used over the internet")
    das_t = _extract_indicator(
        das_df, "Data analytics for the enterprise is performed by")
    dii_t = _extract_indicator(
        dii_df, "high digital intensity", exclude=["very"])

    out["dii_2025"] = year_value(dii_t, 2025)
    out["cloud_2025"] = year_value(cloud_t, 2025)
    out["das_2025"] = year_value(das_t, 2025)

    out["ai_growth"] = out["ai_2025"] - out["ai_2023"]
    out["cloud_to_ai_gap"] = out["cloud_2025"] - out["ai_2025"]
    out["data_to_ai_gap"] = out["das_2025"] - out["ai_2025"]

    comp = ["dii_2025", "cloud_2025", "das_2025", "ai_growth"]
    norm_cols = []
    for c in comp:
        nc = "n_" + c
        out[nc] = _minmax(out[c]) if c in out.columns else np.nan
        norm_cols.append(nc)

    out["readiness"] = out[norm_cols].mean(axis=1, skipna=True)

    out = out[out["ai_2025"].notna()].copy()
    out.index.name = "country"
    return out.reset_index()

READINESS_RECO_HELP = (
    "Recommendations are rule-based, derived from each country's position relative to the "
    "median on AI adoption, readiness, cloud, data analytics, and digital intensity."
)

def readiness_recommendation(row, med) -> str:

    ai_high = pd.notna(row["ai_2025"]) and row["ai_2025"] >= med["ai_2025"]
    read_high = pd.notna(row["readiness"]) and row["readiness"] >= med["readiness"]
    cloud_high = pd.notna(row["cloud_2025"]) and row["cloud_2025"] >= med["cloud_2025"]
    das_high = pd.notna(row["das_2025"]) and row["das_2025"] >= med["das_2025"]
    dii_low = pd.notna(row["dii_2025"]) and row["dii_2025"] < med["dii_2025"]

    if ai_high and read_high:
        return "Mature AI market - suitable for advanced AI, governance, compliance, and optimization."
    if read_high and not ai_high:
        return "Ready but under-adopted market - readiness is high while AI adoption lags."
    if cloud_high and not ai_high:
        return "Opportunity for cloud-based AI tools / AI SaaS."
    if das_high and not ai_high:
        return "Opportunity for predictive analytics, ML, and decision-support tools."
    if dii_low and not ai_high:
        return "Basic digital transformation should come first."
    return "Mixed signals - assess case by case using the metrics above."

FRIENDLY_DATASET = {
    "ai_by_industry": "AI by Industry",
    "ai_by_enterprise_size": "AI by Enterprise Size",
}

BARRIER_FILES = {
    "size": ("isoc_eb_ai_barriers_linear.csv", "isoc_eb_ai_linear.csv"),
    "industry": ("isoc_eb_ain2_barriers_linear.csv", "isoc_eb_ain2_linear.csv"),
}

BARRIER_UNIT = "Percentage of enterprises"
BARRIER_MARKER = "do not use AI technologies, because"

BARRIER_MAP = [
    ("lack of relevant expertise", "Lack of expertise"),
    ("costs seem too high", "Costs too high"),
    ("incompatibility with existing", "Incompatible systems"),
    ("availability or quality of the necessary data", "Data availability/quality"),
    ("data protection and privacy", "Privacy/data protection"),
    ("legal consequences", "Legal uncertainty"),
    ("ethical considerations", "Ethical concerns"),
    ("are not useful for enterprise", "AI not useful / no need"),
]

BARRIER_SIZES = {
    "From 10 to 49 persons employed": "Small (10-49)",
    "From 50 to 249 persons employed": "Medium (50-249)",
    "250 persons employed or more": "Large (250+)",
}

BARRIER_RECO = {
    "Lack of expertise": "AI training, consulting support, and low-code/no-code AI tools.",
    "Costs too high": "Affordable AI packages, shared AI services, and SME-friendly subscriptions.",
    "Data availability/quality": "Data-readiness assessment, data governance, and an analytics foundation first.",
    "Privacy/data protection": "AI governance, compliance support, and risk assessment.",
    "Legal uncertainty": "AI governance, compliance support, and risk assessment.",
    "Ethical concerns": "AI governance, compliance support, and risk assessment.",
    "Incompatible systems": "System-integration support, cloud-based AI tools, and API-based services.",
    "AI not useful / no need": "Awareness building, sector-specific use-case demos, and ROI examples.",
}

BARRIER_SHORT = {
    "Lack of expertise": "Expertise gap",
    "Costs too high": "Costs",
    "Incompatible systems": "System fit",
    "Data availability/quality": "Data quality",
    "Privacy/data protection": "Privacy",
    "Legal uncertainty": "Legal risk",
    "Ethical concerns": "Ethics",
    "AI not useful / no need": "No need",
}

def _short_barrier(indicator: str):

    if not isinstance(indicator, str):
        return None
    low = indicator.lower()
    for key, short in BARRIER_MAP:
        if key in low:
            return short
    return None

@st.cache_data(show_spinner=False)
def safe_read_barrier(kind: str) -> pd.DataFrame:

    preferred, fallback = BARRIER_FILES[kind]
    cols = ["size_emp", "nace_r2", "indic_is", "unit", "geo", "TIME_PERIOD", "OBS_VALUE"]
    for name in (preferred, fallback):
        p = RAW / name
        if p.exists():
            try:
                return pd.read_csv(p, usecols=lambda c: c in cols)
            except Exception:
                try:
                    return pd.read_csv(p)
                except Exception:
                    return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def build_barriers(size_raw: pd.DataFrame, ind_raw: pd.DataFrame, year: int = 2025) -> dict:

    out = {"overall": pd.DataFrame(), "by_size": pd.DataFrame(),
           "by_industry": pd.DataFrame(), "meta": {}}

    def prep(df):
        if df.empty or not {"indic_is", "unit", "OBS_VALUE", "geo", "TIME_PERIOD"}.issubset(df.columns):
            return pd.DataFrame()
        d = df[(df["unit"] == BARRIER_UNIT)
               & (df["indic_is"].astype(str).str.contains(BARRIER_MARKER, case=False))].copy()
        if d.empty:
            return d
        d["barrier"] = d["indic_is"].map(_short_barrier)
        return d.dropna(subset=["barrier"])

    s = prep(size_raw)
    i = prep(ind_raw)

    meta = {"size_available": not s.empty, "industry_available": not i.empty}

    if not s.empty:
        years = sorted(int(y) for y in s["TIME_PERIOD"].dropna().unique())
        meta["years"] = years
        meta["countries"] = int(s[s["TIME_PERIOD"] == year]["geo"].nunique())
        meta["missing_rate"] = round(100 * s[s["TIME_PERIOD"] == year]["OBS_VALUE"].isna().mean(), 1) \
            if (s["TIME_PERIOD"] == year).any() else None

        base = s[(s["nace_r2"] == NACE_TOTAL) & (s["TIME_PERIOD"] == year)]
        tot = base[base["size_emp"] == SIZE_TOTAL]
        if not tot.empty:
            out["overall"] = (tot.groupby("barrier", as_index=False)["OBS_VALUE"].mean()
                              .rename(columns={"OBS_VALUE": "value"})
                              .sort_values("value", ascending=False))
        bysz = base[base["size_emp"].isin(BARRIER_SIZES)].copy()
        if not bysz.empty:
            bysz["size"] = bysz["size_emp"].map(BARRIER_SIZES)
            out["by_size"] = (bysz.groupby(["size", "barrier"], as_index=False)["OBS_VALUE"].mean()
                              .rename(columns={"OBS_VALUE": "value"}))

    if not i.empty:
        ind = i[(i["size_emp"] == SIZE_TOTAL) & (i["TIME_PERIOD"] == year)
                & (i["nace_r2"] != NACE_TOTAL)].copy()
        if not ind.empty:
            out["by_industry"] = (ind.groupby(["nace_r2", "barrier"], as_index=False)["OBS_VALUE"].mean()
                                  .rename(columns={"OBS_VALUE": "value", "nace_r2": "industry"}))

    meta["by_size_dim"] = not out["by_size"].empty
    meta["by_industry_dim"] = not out["by_industry"].empty
    out["meta"] = meta
    return out

PURPOSE_FILES = {
    "size": ("isoc_eb_ai_purpose_linear.csv", "isoc_eb_ai_linear__1_.csv"),
    "industry": ("isoc_eb_ain2_purpose_linear.csv", "isoc_eb_ain2_linear__1_.csv"),
}

PURPOSE_UNIT = "Percentage of the enterprises using at least one AI technologies"
PURPOSE_PREFIX = "enterprises using ai technologies for"

PURPOSE_MAP = [
    ("for marketing or sales", "Marketing & sales",
     ["Customer segmentation", "Personalized marketing", "Lead scoring", "Campaign optimization"]),
    ("for organisation of business administration processes or management", "Business admin & mgmt",
     ["Workflow automation", "Document processing", "Internal decision support"]),
    ("for accounting, controlling or finance management", "Accounting & finance",
     ["Invoice processing", "Anomaly detection", "Financial control", "Reporting automation"]),
    ("for production processes", "Production",
     ["Predictive maintenance", "Quality inspection", "Production planning"]),
    ("for ICT security", "ICT security",
     ["Threat detection", "Fraud monitoring", "Cybersecurity automation"]),
    ("for research and development (R&D) or innovation activity", "R&D & innovation",
     ["Product design support", "Simulation", "Idea generation", "Research automation"]),
    ("for logistics", "Logistics",
     ["Route optimization", "Warehouse optimization", "Delivery-time prediction"]),
    ("for human resources management or recruiting", "HR & recruiting",
     ["CV screening support", "Workforce planning", "Internal knowledge assistants"]),
]
PURPOSE_SIZES = BARRIER_SIZES

def _short_purpose(indicator: str):

    if not isinstance(indicator, str):
        return None
    low = indicator.lower()
    if "at least" in low:
        return None
    for key, short, _ in PURPOSE_MAP:
        if key.lower() in low:
            return short
    return None

def purpose_usecases(short_label: str):

    for _, short, cases in PURPOSE_MAP:
        if short == short_label:
            return cases
    return ["Automation", "Data analytics", "Decision support"]

@st.cache_data(show_spinner=False)
def safe_read_purpose(kind: str) -> pd.DataFrame:

    preferred, fallback = PURPOSE_FILES[kind]
    cols = ["size_emp", "nace_r2", "indic_is", "unit", "geo", "TIME_PERIOD", "OBS_VALUE"]
    for name in (preferred, fallback):
        p = RAW / name
        if p.exists():
            try:
                return pd.read_csv(p, usecols=lambda c: c in cols)
            except Exception:
                try:
                    return pd.read_csv(p)
                except Exception:
                    return pd.DataFrame()
    return pd.DataFrame()

@st.cache_data(show_spinner=False)
def build_purposes(size_raw: pd.DataFrame, ind_raw: pd.DataFrame, year: int = 2025) -> dict:

    out = {"overall": pd.DataFrame(), "by_size": pd.DataFrame(),
           "by_industry": pd.DataFrame(), "meta": {}}

    def prep(df):
        if df.empty or not {"indic_is", "unit", "OBS_VALUE", "geo", "TIME_PERIOD"}.issubset(df.columns):
            return pd.DataFrame()
        d = df[(df["unit"] == PURPOSE_UNIT)
               & (df["indic_is"].astype(str).str.lower().str.startswith(PURPOSE_PREFIX))].copy()
        if d.empty:
            return d
        d["purpose"] = d["indic_is"].map(_short_purpose)
        return d.dropna(subset=["purpose"])

    s = prep(size_raw)
    i = prep(ind_raw)
    meta = {"size_available": not s.empty, "industry_available": not i.empty}

    if not s.empty:
        meta["years"] = sorted(int(y) for y in s["TIME_PERIOD"].dropna().unique())
        meta["countries"] = int(s[s["TIME_PERIOD"] == year]["geo"].nunique())
        meta["missing_rate"] = round(100 * s[s["TIME_PERIOD"] == year]["OBS_VALUE"].isna().mean(), 1) \
            if (s["TIME_PERIOD"] == year).any() else None

        base = s[(s["nace_r2"] == NACE_TOTAL) & (s["TIME_PERIOD"] == year)]
        tot = base[base["size_emp"] == SIZE_TOTAL]
        if not tot.empty:
            out["overall"] = (tot.groupby("purpose", as_index=False)["OBS_VALUE"].mean()
                              .rename(columns={"OBS_VALUE": "value"})
                              .sort_values("value", ascending=False))
        bysz = base[base["size_emp"].isin(PURPOSE_SIZES)].copy()
        if not bysz.empty:
            bysz["size"] = bysz["size_emp"].map(PURPOSE_SIZES)
            out["by_size"] = (bysz.groupby(["size", "purpose"], as_index=False)["OBS_VALUE"].mean()
                              .rename(columns={"OBS_VALUE": "value"}))

    if not i.empty:
        ind = i[(i["size_emp"] == SIZE_TOTAL) & (i["TIME_PERIOD"] == year)
                & (i["nace_r2"] != NACE_TOTAL)].copy()
        if not ind.empty:
            out["by_industry"] = (ind.groupby(["nace_r2", "purpose"], as_index=False)["OBS_VALUE"].mean()
                                  .rename(columns={"OBS_VALUE": "value", "nace_r2": "industry"}))

    meta["by_size_dim"] = not out["by_size"].empty
    meta["by_industry_dim"] = not out["by_industry"].empty
    out["meta"] = meta
    return out
