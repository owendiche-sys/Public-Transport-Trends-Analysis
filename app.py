from __future__ import annotations

import io
import os
from dataclasses import dataclass
from html import escape
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split


# =========================================================
# Page config
# =========================================================
st.set_page_config(
    page_title="Public Transport Demand Intelligence Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_2019 = "2019data0.csv"
DEFAULT_2022 = "2022data0.csv"

BUS_COLS = ["Bus pax number peak", "Bus pax number offpeak"]
TRAM_COLS = ["Tram pax number peak", "Tram pax number offpeak"]
METRO_COLS = ["Metro pax number peak", "Metro pax number offpeak"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTH_ORDER = list(range(1, 13))
MONTH_LABELS = {
    1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr",
    5: "May", 6: "Jun", 7: "Jul", 8: "Aug",
    9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec",
}


# =========================================================
# Styling
# =========================================================
def inject_css() -> None:
    st.markdown(
        """
        <style>
          :root{
            --bg:#f6f8fc;
            --panel:#ffffff;
            --text:#111827;
            --muted:#6b7280;
            --border:rgba(17,24,39,0.10);
            --shadow:0 10px 30px rgba(15,23,42,0.06);
            --radius:18px;
            --accent:#1d4ed8;
            --accent-soft:rgba(29,78,216,0.08);
          }

          html, body, [data-testid="stAppViewContainer"]{
            background:var(--bg) !important;
            color:var(--text) !important;
          }

          [data-testid="stHeader"]{
            background:rgba(246,248,252,0.82);
          }

          [data-testid="stSidebar"]{
            background:#ffffff !important;
            border-right:1px solid var(--border);
          }

          .block-container{
            padding-top:1.2rem;
            padding-bottom:2.4rem;
            max-width:1400px;
          }

          .hero{
            background:linear-gradient(135deg, #ffffff 0%, #f9fbff 100%);
            border:1px solid var(--border);
            border-radius:24px;
            padding:24px 24px 18px 24px;
            box-shadow:var(--shadow);
            margin-bottom:18px;
          }

          .hero-title{
            font-size:30px;
            font-weight:800;
            letter-spacing:-0.02em;
            margin:0 0 8px 0;
            color:var(--text);
          }

          .hero-sub{
            margin:0;
            font-size:15px;
            line-height:1.6;
            color:var(--muted);
            max-width:980px;
          }

          .hero-strip{
            margin-top:14px;
            padding:10px 12px;
            border-radius:14px;
            background:var(--accent-soft);
            border:1px solid rgba(29,78,216,0.12);
            color:#1e3a8a;
            font-size:13px;
          }

          .section-title{
            font-size:18px;
            font-weight:800;
            color:var(--text);
            margin:0 0 10px 0;
          }

          .card{
            background:var(--panel);
            border:1px solid var(--border);
            border-radius:var(--radius);
            box-shadow:var(--shadow);
            padding:16px;
          }

          .card-title{
            margin:0 0 6px 0;
            font-size:16px;
            font-weight:800;
            color:var(--text);
          }

          .card-sub{
            margin:0 0 12px 0;
            color:var(--muted);
            font-size:13px;
            line-height:1.5;
          }

          .kpi-grid{
            display:grid;
            grid-template-columns:repeat(6, minmax(0, 1fr));
            gap:12px;
            margin-bottom:10px;
          }

          @media (max-width:1280px){
            .kpi-grid{ grid-template-columns:repeat(3, minmax(0, 1fr)); }
          }

          @media (max-width:700px){
            .kpi-grid{ grid-template-columns:repeat(2, minmax(0, 1fr)); }
          }

          @media (max-width:540px){
            .kpi-grid{ grid-template-columns:repeat(1, minmax(0, 1fr)); }
          }

          .kpi-card{
            background:var(--panel);
            border:1px solid var(--border);
            border-radius:16px;
            box-shadow:var(--shadow);
            padding:14px;
            min-height:108px;
          }

          .kpi-label{
            font-size:12px;
            color:var(--muted);
            margin-bottom:8px;
          }

          .kpi-value{
            font-size:24px;
            font-weight:800;
            line-height:1.1;
            color:var(--text);
            margin-bottom:8px;
          }

          .kpi-note{
            font-size:12px;
            color:var(--muted);
            line-height:1.45;
          }

          .insight-list{
            margin:0;
            padding-left:18px;
          }

          .insight-list li{
            margin-bottom:8px;
            line-height:1.55;
            color:var(--text);
          }

          .badge-row{
            display:flex;
            flex-wrap:wrap;
            gap:8px;
            margin-top:6px;
          }

          .badge{
            display:inline-block;
            padding:6px 10px;
            border-radius:999px;
            background:#f8fafc;
            border:1px solid var(--border);
            color:var(--text);
            font-size:12px;
          }

          .divider{
            height:1px;
            background:var(--border);
            margin:10px 0 14px 0;
          }

          .js-plotly-plot .plotly .modebar{
            opacity:0.08;
          }

          .js-plotly-plot .plotly:hover .modebar{
            opacity:1;
          }
        </style>
        """,
        unsafe_allow_html=True,
    )


inject_css()


# =========================================================
# Formatting helpers
# =========================================================
def fmt_number(x: Optional[float], digits: int = 0) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}"


def fmt_pct(x: Optional[float], digits: int = 1) -> str:
    if x is None or pd.isna(x):
        return "N/A"
    return f"{x:,.{digits}f}%"


def safe_divide(a: float, b: float) -> float:
    if b == 0 or pd.isna(b):
        return np.nan
    return a / b


def esc(text: str) -> str:
    return escape(str(text))


# =========================================================
# UI helpers
# =========================================================
def render_section_title(text: str) -> None:
    st.markdown(f"<div class='section-title'>{esc(text)}</div>", unsafe_allow_html=True)


def render_kpis(items: List[Tuple[str, str, str]]) -> None:
    blocks: List[str] = []
    for label, value, note in items:
        blocks.append(
            "<div class='kpi-card'>"
            f"<div class='kpi-label'>{esc(label)}</div>"
            f"<div class='kpi-value'>{esc(value)}</div>"
            f"<div class='kpi-note'>{esc(note)}</div>"
            "</div>"
        )
    st.markdown("<div class='kpi-grid'>" + "".join(blocks) + "</div>", unsafe_allow_html=True)


def render_list_card(title: str, items: List[str], subtitle: str = "") -> None:
    html = "<div class='card'>"
    html += f"<div class='card-title'>{esc(title)}</div>"
    if subtitle:
        html += f"<div class='card-sub'>{esc(subtitle)}</div>"
    html += "<ul class='insight-list'>"
    for item in items:
        html += f"<li>{esc(item)}</li>"
    html += "</ul></div>"
    st.markdown(html, unsafe_allow_html=True)


def render_badges(items: List[str]) -> None:
    html = "<div class='badge-row'>"
    for item in items:
        html += f"<span class='badge'>{esc(item)}</span>"
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)


# =========================================================
# CSV loading
# =========================================================
@dataclass
class DataMeta:
    source_2019: str
    source_2022: str


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1", "cp1252"):
        try:
            return b.decode(enc)
        except Exception:
            continue
    return b.decode(errors="ignore")


def _detect_delimiter(sample_text: str) -> str:
    lines = [ln for ln in sample_text.splitlines() if ln.strip()]
    if not lines:
        return ","
    header = lines[0]
    candidates = [",", ";", "\t", "|"]
    counts = {c: header.count(c) for c in candidates}
    best = max(counts, key=counts.get)
    return best if counts[best] > 0 else ","


def safe_read_csv(file_bytes: bytes) -> pd.DataFrame:
    sample = _decode_bytes(file_bytes[:4096])
    sep = _detect_delimiter(sample)

    for try_sep in [sep, ",", ";", "\t", "|"]:
        try:
            df = pd.read_csv(io.BytesIO(file_bytes), sep=try_sep, low_memory=False)
            if df.shape[1] > 1:
                return df
        except Exception:
            continue

    return pd.read_csv(io.BytesIO(file_bytes), low_memory=False)


@st.cache_data(show_spinner=False)
def load_two_files(
    use_upload_2019: bool,
    upload_2019_bytes: Optional[bytes],
    upload_2019_name: Optional[str],
    use_upload_2022: bool,
    upload_2022_bytes: Optional[bytes],
    upload_2022_name: Optional[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
    if use_upload_2019 and upload_2019_bytes is not None:
        df2019 = safe_read_csv(upload_2019_bytes)
        src2019 = f"Uploaded: {upload_2019_name or '2019.csv'}"
    else:
        if not os.path.exists(DEFAULT_2019):
            raise FileNotFoundError(f"Missing {DEFAULT_2019} next to app.py, or upload it in the sidebar.")
        with open(DEFAULT_2019, "rb") as f:
            df2019 = safe_read_csv(f.read())
        src2019 = f"Default: {DEFAULT_2019}"

    if use_upload_2022 and upload_2022_bytes is not None:
        df2022 = safe_read_csv(upload_2022_bytes)
        src2022 = f"Uploaded: {upload_2022_name or '2022.csv'}"
    else:
        if not os.path.exists(DEFAULT_2022):
            raise FileNotFoundError(f"Missing {DEFAULT_2022} next to app.py, or upload it in the sidebar.")
        with open(DEFAULT_2022, "rb") as f:
            df2022 = safe_read_csv(f.read())
        src2022 = f"Default: {DEFAULT_2022}"

    return df2019, df2022, DataMeta(source_2019=src2019, source_2022=src2022)


# =========================================================
# Data preparation
# =========================================================
def detect_datetime_col_2022(df: pd.DataFrame) -> Optional[str]:
    for c in df.columns:
        key = c.lower().strip()
        if key in {"date and time", "datetime", "date_time", "timestamp"}:
            return c
    for c in df.columns:
        key = c.lower()
        if "date" in key and "time" in key:
            return c
    return None


@st.cache_data(show_spinner=False)
def build_2019_daily(df2019: pd.DataFrame) -> pd.DataFrame:
    d = df2019.copy()

    if "Date" not in d.columns:
        raise ValueError("2019 dataset must contain a 'Date' column.")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    present_bus = [c for c in BUS_COLS if c in d.columns]
    present_tram = [c for c in TRAM_COLS if c in d.columns]
    present_metro = [c for c in METRO_COLS if c in d.columns]
    present_all = present_bus + present_tram + present_metro

    if not present_all:
        raise ValueError("2019 dataset is missing passenger count columns for Bus, Tram, and Metro.")

    for c in present_all:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["Bus total"] = d[present_bus].sum(axis=1, skipna=True) if present_bus else 0.0
    d["Tram total"] = d[present_tram].sum(axis=1, skipna=True) if present_tram else 0.0
    d["Metro total"] = d[present_metro].sum(axis=1, skipna=True) if present_metro else 0.0
    d["Total passengers"] = d[["Bus total", "Tram total", "Metro total"]].sum(axis=1, skipna=True)

    d["Day"] = d["Date"].dt.day_name()
    d["Day"] = pd.Categorical(d["Day"], categories=WEEKDAY_ORDER, ordered=True)
    d["Month"] = d["Date"].dt.month
    d["Month label"] = d["Month"].map(MONTH_LABELS)
    d["Day index"] = np.arange(1, len(d) + 1)

    return d


@st.cache_data(show_spinner=False)
def build_2022_daily_expanded(df2022: pd.DataFrame, total_annual_passengers: float) -> Tuple[pd.DataFrame, float]:
    d = df2022.copy()
    dt_col = detect_datetime_col_2022(d)
    if dt_col is None:
        raise ValueError("2022 dataset must contain a datetime column such as 'Date and time'.")

    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).copy()
    if d.empty:
        raise ValueError("2022 dataset has no valid datetime rows after parsing.")

    d["Date"] = d[dt_col].dt.date
    daily = d.groupby("Date").size().reset_index(name="Sample")

    sample_total = float(daily["Sample"].sum())
    if sample_total <= 0:
        raise ValueError("2022 dataset has no valid sample rows after preprocessing.")

    expansion_factor = float(total_annual_passengers) / sample_total
    daily["Total passengers"] = daily["Sample"] * expansion_factor

    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)
    daily["Day"] = daily["Date"].dt.day_name()
    daily["Day"] = pd.Categorical(daily["Day"], categories=WEEKDAY_ORDER, ordered=True)
    daily["Month"] = daily["Date"].dt.month
    daily["Month label"] = daily["Month"].map(MONTH_LABELS)
    daily["Day index"] = np.arange(1, len(daily) + 1)

    return daily, expansion_factor


@st.cache_data(show_spinner=False)
def clean_2022_raw(df2022: pd.DataFrame) -> pd.DataFrame:
    d = df2022.copy()
    dt_col = detect_datetime_col_2022(d)
    if dt_col is not None:
        d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")

    if "Mode" in d.columns:
        d["Mode"] = d["Mode"].astype(str).str.strip().str.title()

    if "Distance" in d.columns:
        d["Distance"] = pd.to_numeric(d["Distance"], errors="coerce")

    if "Price" in d.columns:
        d["Price"] = pd.to_numeric(d["Price"], errors="coerce")

    return d


# =========================================================
# Analytics helpers
# =========================================================
def fourier_smooth(y: np.ndarray, n_terms: int = 8) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 10:
        return y
    n = len(y)
    fft_vals = np.fft.rfft(y)
    keep = max(1, int(n_terms) + 1)
    fft_vals[keep:] = 0
    return np.fft.irfft(fft_vals, n=n)


@st.cache_data(show_spinner=False)
def weekday_profile(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Day")["Total passengers"].mean().reindex(WEEKDAY_ORDER)


@st.cache_data(show_spinner=False)
def month_profile(df: pd.DataFrame) -> pd.Series:
    return df.groupby("Month")["Total passengers"].mean().reindex(MONTH_ORDER)


@st.cache_data(show_spinner=False)
def compute_mode_shares_2019(df2019_daily: pd.DataFrame) -> Dict[str, float]:
    bus_total = float(pd.to_numeric(df2019_daily["Bus total"], errors="coerce").sum()) if "Bus total" in df2019_daily.columns else 0.0
    tram_total = float(pd.to_numeric(df2019_daily["Tram total"], errors="coerce").sum()) if "Tram total" in df2019_daily.columns else 0.0
    metro_total = float(pd.to_numeric(df2019_daily["Metro total"], errors="coerce").sum()) if "Metro total" in df2019_daily.columns else 0.0

    total_all = bus_total + tram_total + metro_total
    if total_all <= 0:
        return {"Bus": 0.0, "Tram": 0.0, "Metro": 0.0}

    return {
        "Bus": bus_total / total_all * 100.0,
        "Tram": tram_total / total_all * 100.0,
        "Metro": metro_total / total_all * 100.0,
    }


@st.cache_data(show_spinner=False)
def compute_mode_shares_2022(df2022: pd.DataFrame) -> Dict[str, float]:
    d = df2022.copy()
    if "Mode" not in d.columns:
        return {"Bus": 0.0, "Tram": 0.0, "Metro": 0.0}

    d["Mode"] = d["Mode"].astype(str).str.strip().str.title()
    shares = d["Mode"].value_counts(normalize=True) * 100.0
    return {
        "Bus": float(shares.get("Bus", 0.0)),
        "Tram": float(shares.get("Tram", 0.0)),
        "Metro": float(shares.get("Metro", 0.0)),
    }


@st.cache_data(show_spinner=False)
def compute_season_percentages_2022(df2022: pd.DataFrame) -> Dict[str, float]:
    d = df2022.copy()
    dt_col = detect_datetime_col_2022(d)
    if dt_col is None:
        return {"Spring": np.nan, "Summer": np.nan, "Autumn": np.nan, "Winter": np.nan}

    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).copy()
    if d.empty:
        return {"Spring": np.nan, "Summer": np.nan, "Autumn": np.nan, "Winter": np.nan}

    d["Month"] = d[dt_col].dt.month
    total = len(d)

    return {
        "Spring": float(d["Month"].isin([3, 4, 5]).mean() * 100.0),
        "Summer": float(d["Month"].isin([6, 7, 8]).mean() * 100.0),
        "Autumn": float(d["Month"].isin([9, 10, 11]).mean() * 100.0),
        "Winter": float(d["Month"].isin([12, 1, 2]).mean() * 100.0),
    }


def metro_frame(df2022: pd.DataFrame) -> pd.DataFrame:
    d = df2022.copy()
    needed = {"Mode", "Distance", "Price"}
    if not needed.issubset(set(d.columns)):
        return pd.DataFrame(columns=["Distance", "Price"])

    d["Mode"] = d["Mode"].astype(str).str.strip().str.title()
    d["Distance"] = pd.to_numeric(d["Distance"], errors="coerce")
    d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
    d = d[d["Mode"] == "Metro"].dropna(subset=["Distance", "Price"]).copy()
    return d


@st.cache_resource(show_spinner=False)
def fit_metro_model(df2022: pd.DataFrame, seed: int = 42) -> Optional[Dict]:
    metro = metro_frame(df2022)
    if len(metro) < 30:
        return None

    x = metro[["Distance"]].copy()
    y = metro["Price"].copy()

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=0.25, random_state=int(seed)
    )

    model = LinearRegression()
    model.fit(x_train, y_train)

    pred_train = model.predict(x_train)
    pred_test = model.predict(x_test)

    full_line_x = np.linspace(float(x["Distance"].min()), float(x["Distance"].max()), 200)
    full_line_y = model.predict(full_line_x.reshape(-1, 1))

    result = {
        "model": model,
        "metro": metro,
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "pred_train": pred_train,
        "pred_test": pred_test,
        "slope": float(model.coef_[0]),
        "intercept": float(model.intercept_),
        "train_mae": float(mean_absolute_error(y_train, pred_train)),
        "test_mae": float(mean_absolute_error(y_test, pred_test)),
        "train_r2": float(r2_score(y_train, pred_train)),
        "test_r2": float(r2_score(y_test, pred_test)),
        "line_x": full_line_x,
        "line_y": full_line_y,
        "n": int(len(metro)),
    }
    return result


def build_data_driven_insights(
    d2019_daily: pd.DataFrame,
    d2022_daily: pd.DataFrame,
    df2022_raw: pd.DataFrame,
) -> List[str]:
    insights: List[str] = []

    total_2019 = float(d2019_daily["Total passengers"].sum())
    total_2022 = float(d2022_daily["Total passengers"].sum())
    avg_2019 = float(d2019_daily["Total passengers"].mean())
    avg_2022 = float(d2022_daily["Total passengers"].mean())

    change_pct = safe_divide(avg_2022 - avg_2019, avg_2019)
    insights.append(
        f"Average daily passenger volume is {fmt_number(avg_2019, 0)} in 2019 versus an estimated {fmt_number(avg_2022, 0)} in 2022, a change of {fmt_pct(change_pct)}."
    )

    peak_day_2019 = weekday_profile(d2019_daily).idxmax()
    peak_day_2022 = weekday_profile(d2022_daily).idxmax()
    insights.append(
        f"The busiest weekday changes from {peak_day_2019} in 2019 to {peak_day_2022} in 2022 based on average daily passengers."
    )

    month_2019 = month_profile(d2019_daily)
    month_2022 = month_profile(d2022_daily)
    if month_2019.notna().any() and month_2022.notna().any():
        peak_month_2019 = MONTH_LABELS[int(month_2019.idxmax())]
        peak_month_2022 = MONTH_LABELS[int(month_2022.idxmax())]
        insights.append(
            f"Seasonality remains visible, with the strongest monthly average appearing in {peak_month_2019} for 2019 and {peak_month_2022} for 2022."
        )

    sh2019 = compute_mode_shares_2019(d2019_daily)
    sh2022 = compute_mode_shares_2022(df2022_raw)
    biggest_gap_mode = max(
        ["Bus", "Tram", "Metro"],
        key=lambda m: abs(sh2022.get(m, 0.0) - sh2019.get(m, 0.0))
    )
    insights.append(
        f"The largest mode-share difference between the two datasets appears in {biggest_gap_mode}, indicating a meaningful usage mix shift between 2019 totals and 2022 trip proportions."
    )

    seasonal = compute_season_percentages_2022(df2022_raw)
    seasonal_rank = pd.Series(
        {"Spring": seasonal["Spring"], "Summer": seasonal["Summer"], "Autumn": seasonal["Autumn"], "Winter": seasonal["Winter"]}
    ).sort_values(ascending=False)
    if seasonal_rank.notna().any():
        insights.append(
            f"In the 2022 sample, {seasonal_rank.index[0]} contributes the largest share of observed trips."
        )

    insights.append(
        f"Across the available days, cumulative passenger volume equals {fmt_number(total_2019, 0)} in 2019 and an estimated {fmt_number(total_2022, 0)} in 2022 after sample expansion."
    )

    return insights[:6]


def build_model_driven_insights(model_artifacts: Optional[Dict], scenario_distance: float) -> List[str]:
    if model_artifacts is None:
        return ["A stable Metro fare model is not available because the 2022 data does not contain enough valid Metro rows with both Distance and Price."]

    pred_price = float(model_artifacts["model"].predict(np.array([[scenario_distance]]))[0])

    return [
        f"The Metro fare model estimates a slope of {model_artifacts['slope']:.3f}, meaning price increases by roughly {model_artifacts['slope']:.3f} units for each extra kilometre.",
        f"Holdout performance reaches test MAE {model_artifacts['test_mae']:.3f} and test R² {model_artifacts['test_r2']:.3f}.",
        f"The intercept is {model_artifacts['intercept']:.3f}, representing the fitted base component for very short trips.",
        f"For a scenario distance of {scenario_distance:.1f} km, the model predicts a Metro fare of {pred_price:.2f}.",
    ]


# =========================================================
# Sidebar inputs
# =========================================================
with st.sidebar:
    st.markdown("### Data loading")
    use_up_2019 = st.toggle("Use uploaded 2019 file", value=False)
    up_2019 = st.file_uploader("Upload 2019 CSV", type=["csv"], key="up2019", disabled=not use_up_2019)

    use_up_2022 = st.toggle("Use uploaded 2022 file", value=False)
    up_2022 = st.file_uploader("Upload 2022 CSV", type=["csv"], key="up2022", disabled=not use_up_2022)

    st.markdown("### 2022 scaling")
    total_annual = st.number_input(
        "Total annual passengers (2022)",
        min_value=1_000_000.0,
        max_value=10_000_000_000.0,
        value=369_382_078.0,
        step=1_000_000.0,
        help="Used to scale 2022 sample counts into estimated daily totals.",
    )

    st.markdown("### Analysis controls")
    smoothing_terms = st.slider("Fourier smoothing terms", min_value=0, max_value=30, value=8, step=1)
    metro_scenario_distance = st.slider("Metro fare scenario distance (km)", min_value=0.5, max_value=60.0, value=10.0, step=0.5)
    random_seed = st.number_input("Random seed", min_value=0, max_value=9999, value=42, step=1)


# =========================================================
# Load and prepare data
# =========================================================
try:
    df2019_raw, df2022_raw, meta = load_two_files(
        use_upload_2019=use_up_2019,
        upload_2019_bytes=up_2019.getvalue() if up_2019 is not None else None,
        upload_2019_name=up_2019.name if up_2019 is not None else None,
        use_upload_2022=use_up_2022,
        upload_2022_bytes=up_2022.getvalue() if up_2022 is not None else None,
        upload_2022_name=up_2022.name if up_2022 is not None else None,
    )

    d2019_daily = build_2019_daily(df2019_raw)
    df2022_clean = clean_2022_raw(df2022_raw)
    d2022_daily, expansion_factor = build_2022_daily_expanded(
        df2022_clean,
        total_annual_passengers=float(total_annual),
    )
    metro_artifacts = fit_metro_model(df2022_clean, seed=int(random_seed))
except Exception as e:
    st.error("The data could not be loaded or prepared.")
    st.write(str(e))
    st.stop()


# =========================================================
# Hero and KPIs
# =========================================================
hero_text = (
    "This dashboard compares 2019 observed public-transport demand with a 2022 demand estimate built "
    "from sampled trip records and an annual scaling assumption. It focuses on demand level changes, "
    "weekday and seasonal shifts, mode composition, and Metro fare behaviour."
)

scope_text = (
    f"2019 source: {meta.source_2019} | 2022 source: {meta.source_2022} | "
    f"2022 expansion factor: {fmt_number(expansion_factor, 3)}"
)

st.markdown(
    f"""
    <div class="hero">
      <div class="hero-title">Public Transport Demand Intelligence Dashboard</div>
      <p class="hero-sub">{esc(hero_text)}</p>
      <div class="hero-strip">{esc(scope_text)}</div>
    </div>
    """,
    unsafe_allow_html=True,
)

avg_2019 = float(d2019_daily["Total passengers"].mean())
avg_2022 = float(d2022_daily["Total passengers"].mean())
change_pct = safe_divide(avg_2022 - avg_2019, avg_2019)

mode_2019 = compute_mode_shares_2019(d2019_daily)
mode_2022 = compute_mode_shares_2022(df2022_clean)

render_kpis(
    [
        ("Average daily passengers (2019)", fmt_number(avg_2019, 0), "Observed daily total from Bus, Tram, and Metro counts"),
        ("Average daily passengers (2022)", fmt_number(avg_2022, 0), "Estimated daily total after sample expansion"),
        ("Estimated daily change", fmt_pct(change_pct), "Change in average daily passengers from 2019 to 2022"),
        ("Largest 2019 mode", max(mode_2019, key=mode_2019.get), f"{fmt_pct(mode_2019[max(mode_2019, key=mode_2019.get)] / 100)} of 2019 passenger totals"),
        ("Largest 2022 mode", max(mode_2022, key=mode_2022.get), f"{fmt_pct(mode_2022[max(mode_2022, key=mode_2022.get)] / 100)} of 2022 trip records"),
        ("Metro fare slope", f"{metro_artifacts['slope']:.3f}" if metro_artifacts else "N/A", "Estimated fare increase per kilometre for Metro journeys"),
    ]
)


# =========================================================
# Tabs
# =========================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs(
    [
        "Executive summary",
        "Demand patterns",
        "Mode and seasonality",
        "Metro fare analytics",
        "Data appendix",
    ]
)


# =========================================================
# Executive summary
# =========================================================
with tab1:
    left, right = st.columns([1.1, 0.9], gap="large")

    with left:
        render_list_card(
            "1. Data-driven insights",
            build_data_driven_insights(d2019_daily, d2022_daily, df2022_clean),
            "Passenger volume, seasonality, weekday effects, and mode composition based on the loaded datasets.",
        )

        st.write("")
        render_section_title("Smoothed daily passenger comparison")

        daily_plot = pd.DataFrame(
            {
                "Day index": d2019_daily["Day index"],
                "2019 smoothed": fourier_smooth(d2019_daily["Total passengers"].to_numpy(dtype=float), smoothing_terms),
            }
        )
        daily_plot_2 = pd.DataFrame(
            {
                "Day index": d2022_daily["Day index"],
                "2022 smoothed": fourier_smooth(d2022_daily["Total passengers"].to_numpy(dtype=float), smoothing_terms),
            }
        )

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=daily_plot["Day index"],
                y=daily_plot["2019 smoothed"],
                mode="lines",
                name="2019 smoothed",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=daily_plot_2["Day index"],
                y=daily_plot_2["2022 smoothed"],
                mode="lines",
                name="2022 smoothed",
            )
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Day index within dataset",
            yaxis_title="Daily passengers",
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        )
        st.plotly_chart(fig, use_container_width=True)

    with right:
        render_list_card(
            "2. Model-driven insights",
            build_model_driven_insights(metro_artifacts, metro_scenario_distance),
            "Metro fare behaviour from the linear regression layer, used as the model-driven complement to the demand analysis.",
        )

        st.write("")
        render_section_title("Weekday comparison")
        weekday_df = pd.DataFrame(
            {
                "Day": WEEKDAY_ORDER,
                "2019": weekday_profile(d2019_daily).values,
                "2022": weekday_profile(d2022_daily).values,
            }
        ).melt(id_vars="Day", var_name="Dataset", value_name="Average daily passengers")

        fig = px.bar(
            weekday_df,
            x="Day",
            y="Average daily passengers",
            color="Dataset",
            barmode="group",
            category_orders={"Day": WEEKDAY_ORDER},
        )
        fig.update_layout(
            height=420,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Day of week",
            yaxis_title="Average daily passengers",
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Demand patterns
# =========================================================
with tab2:
    render_section_title("Daily demand profile")
    show_raw = st.toggle("Show raw daily series", value=False)

    fig = go.Figure()

    if show_raw:
        fig.add_trace(
            go.Scatter(
                x=d2019_daily["Day index"],
                y=d2019_daily["Total passengers"],
                mode="lines",
                name="2019 raw",
                line=dict(width=1),
                opacity=0.35,
            )
        )
        fig.add_trace(
            go.Scatter(
                x=d2022_daily["Day index"],
                y=d2022_daily["Total passengers"],
                mode="lines",
                name="2022 raw",
                line=dict(width=1),
                opacity=0.35,
            )
        )

    fig.add_trace(
        go.Scatter(
            x=d2019_daily["Day index"],
            y=fourier_smooth(d2019_daily["Total passengers"].to_numpy(dtype=float), smoothing_terms),
            mode="lines",
            name="2019 smoothed",
            line=dict(width=3),
        )
    )
    fig.add_trace(
        go.Scatter(
            x=d2022_daily["Day index"],
            y=fourier_smooth(d2022_daily["Total passengers"].to_numpy(dtype=float), smoothing_terms),
            mode="lines",
            name="2022 smoothed",
            line=dict(width=3),
        )
    )

    fig.update_layout(
        height=460,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Day index within dataset",
        yaxis_title="Daily passengers",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig, use_container_width=True)

    st.write("")
    c1, c2 = st.columns([1.0, 1.0], gap="large")

    with c1:
        render_section_title("Monthly average comparison")
        monthly_df = pd.DataFrame(
            {
                "Month": MONTH_ORDER,
                "2019": month_profile(d2019_daily).values,
                "2022": month_profile(d2022_daily).values,
            }
        ).melt(id_vars="Month", var_name="Dataset", value_name="Average daily passengers")
        monthly_df["Month label"] = monthly_df["Month"].map(MONTH_LABELS)

        fig = px.line(
            monthly_df,
            x="Month",
            y="Average daily passengers",
            color="Dataset",
            markers=True,
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis=dict(
                title="Month",
                tickmode="array",
                tickvals=MONTH_ORDER,
                ticktext=[MONTH_LABELS[m] for m in MONTH_ORDER],
            ),
            yaxis_title="Average daily passengers",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        render_section_title("Demand distribution")
        dist_df = pd.concat(
            [
                pd.DataFrame({"Dataset": "2019", "Daily passengers": d2019_daily["Total passengers"]}),
                pd.DataFrame({"Dataset": "2022", "Daily passengers": d2022_daily["Total passengers"]}),
            ],
            ignore_index=True,
        )
        fig = px.box(
            dist_df,
            x="Dataset",
            y="Daily passengers",
            points="outliers",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Dataset",
            yaxis_title="Daily passengers",
        )
        st.plotly_chart(fig, use_container_width=True)


# =========================================================
# Mode and seasonality
# =========================================================
with tab3:
    c1, c2 = st.columns([1.0, 1.0], gap="large")

    with c1:
        render_section_title("Mode share comparison")
        mode_df = pd.DataFrame(
            {
                "Mode": ["Bus", "Tram", "Metro"],
                "2019": [mode_2019["Bus"], mode_2019["Tram"], mode_2019["Metro"]],
                "2022": [mode_2022["Bus"], mode_2022["Tram"], mode_2022["Metro"]],
            }
        ).melt(id_vars="Mode", var_name="Dataset", value_name="Share")

        fig = px.bar(
            mode_df,
            x="Mode",
            y="Share",
            color="Dataset",
            barmode="group",
            text_auto=".1f",
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
            xaxis_title="Mode",
            yaxis_title="Share of journeys (%)",
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        render_section_title("2022 seasonal composition")
        seasonal = compute_season_percentages_2022(df2022_clean)
        season_df = pd.DataFrame(
            {
                "Season": ["Spring", "Summer", "Autumn", "Winter"],
                "Share": [seasonal["Spring"], seasonal["Summer"], seasonal["Autumn"], seasonal["Winter"]],
            }
        ).dropna()

        fig = px.pie(
            season_df,
            names="Season",
            values="Share",
            hole=0.52,
        )
        fig.update_layout(
            height=380,
            margin=dict(l=10, r=10, t=20, b=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    st.write("")
    render_section_title("Weekday shift from 2019 to 2022")
    weekday_diff = pd.DataFrame(
        {
            "Day": WEEKDAY_ORDER,
            "2019": weekday_profile(d2019_daily).values,
            "2022": weekday_profile(d2022_daily).values,
        }
    )
    weekday_diff["2022 minus 2019"] = weekday_diff["2022"] - weekday_diff["2019"]

    fig = px.bar(
        weekday_diff,
        x="Day",
        y="2022 minus 2019",
        category_orders={"Day": WEEKDAY_ORDER},
    )
    fig.update_layout(
        height=380,
        margin=dict(l=10, r=10, t=20, b=10),
        xaxis_title="Day of week",
        yaxis_title="Passenger difference",
    )
    st.plotly_chart(fig, use_container_width=True)

    display_weekday = weekday_diff.copy()
    display_weekday["2019"] = display_weekday["2019"].map(lambda x: fmt_number(x, 0))
    display_weekday["2022"] = display_weekday["2022"].map(lambda x: fmt_number(x, 0))
    display_weekday["2022 minus 2019"] = display_weekday["2022 minus 2019"].map(lambda x: fmt_number(x, 0))
    st.dataframe(display_weekday, use_container_width=True, hide_index=True)


# =========================================================
# Metro fare analytics
# =========================================================
with tab4:
    if metro_artifacts is None:
        st.info("Metro fare analytics require a 2022 dataset with enough Metro rows containing both Distance and Price.")
    else:
        render_kpis(
            [
                ("Metro sample size", fmt_number(metro_artifacts["n"], 0), "Valid Metro journeys with Distance and Price"),
                ("Slope", f"{metro_artifacts['slope']:.3f}", "Estimated fare increase per kilometre"),
                ("Intercept", f"{metro_artifacts['intercept']:.3f}", "Estimated base component"),
                ("Train MAE", f"{metro_artifacts['train_mae']:.3f}", "Average absolute training error"),
                ("Test MAE", f"{metro_artifacts['test_mae']:.3f}", "Average absolute holdout error"),
                ("Test R²", f"{metro_artifacts['test_r2']:.3f}", "Explained variance on the holdout split"),
            ]
        )

        c1, c2 = st.columns([1.0, 1.0], gap="large")

        with c1:
            render_section_title("Observed Metro fares and fitted line")
            scatter_df = metro_artifacts["metro"].copy()

            fig = go.Figure()
            fig.add_trace(
                go.Scatter(
                    x=scatter_df["Distance"],
                    y=scatter_df["Price"],
                    mode="markers",
                    name="Observed Metro journeys",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=metro_artifacts["line_x"],
                    y=metro_artifacts["line_y"],
                    mode="lines",
                    name="Linear fit",
                )
            )
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
                xaxis_title="Trip distance (km)",
                yaxis_title="Price",
            )
            st.plotly_chart(fig, use_container_width=True)

        with c2:
            render_section_title("Actual vs predicted holdout fares")
            eval_df = pd.DataFrame(
                {
                    "Actual price": metro_artifacts["y_test"].to_numpy(),
                    "Predicted price": metro_artifacts["pred_test"],
                    "Distance": metro_artifacts["x_test"]["Distance"].to_numpy(),
                }
            )
            fig = px.scatter(
                eval_df,
                x="Actual price",
                y="Predicted price",
                hover_data=["Distance"],
            )
            fig.update_layout(
                height=400,
                margin=dict(l=10, r=10, t=20, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)

        st.write("")
        render_section_title("Metro fare scenario")
        predicted_price = float(
            metro_artifacts["model"].predict(np.array([[metro_scenario_distance]]))[0]
        )

        render_kpis(
            [
                ("Scenario distance", f"{metro_scenario_distance:.1f} km", "Input selected in the sidebar"),
                ("Predicted Metro fare", f"{predicted_price:.2f}", "Linear model estimate"),
                ("Model equation", f"{metro_artifacts['slope']:.3f} × distance + {metro_artifacts['intercept']:.3f}", "Regression form"),
                ("Distance range in data", f"{metro_artifacts['metro']['Distance'].min():.1f} to {metro_artifacts['metro']['Distance'].max():.1f} km", "Observed range used for fitting"),
                ("Median observed fare", f"{metro_artifacts['metro']['Price'].median():.2f}", "Central observed Metro fare"),
                ("Median observed distance", f"{metro_artifacts['metro']['Distance'].median():.1f} km", "Central observed Metro trip length"),
            ]
        )


# =========================================================
# Data appendix
# =========================================================
with tab5:
    render_section_title("Source and preparation summary")
    render_badges(
        [
            meta.source_2019,
            meta.source_2022,
            f"2019 prepared days: {fmt_number(len(d2019_daily), 0)}",
            f"2022 prepared days: {fmt_number(len(d2022_daily), 0)}",
            f"2022 expansion factor: {fmt_number(expansion_factor, 3)}",
        ]
    )

    st.write("")
    a1, a2 = st.columns([1.0, 1.0], gap="large")

    with a1:
        render_section_title("2019 missingness")
        miss_2019 = df2019_raw.isna().mean().sort_values(ascending=False).reset_index()
        miss_2019.columns = ["Column", "Missing share"]
        miss_2019["Missing share"] = miss_2019["Missing share"].map(lambda x: fmt_pct(x))
        st.dataframe(miss_2019, use_container_width=True, hide_index=True)

    with a2:
        render_section_title("2022 missingness")
        miss_2022 = df2022_raw.isna().mean().sort_values(ascending=False).reset_index()
        miss_2022.columns = ["Column", "Missing share"]
        miss_2022["Missing share"] = miss_2022["Missing share"].map(lambda x: fmt_pct(x))
        st.dataframe(miss_2022, use_container_width=True, hide_index=True)

    st.write("")
    render_section_title("Prepared daily comparison table")
    prepared_table = pd.DataFrame(
        {
            "Metric": [
                "Total passengers across available days",
                "Average daily passengers",
                "Median daily passengers",
                "Minimum daily passengers",
                "Maximum daily passengers",
            ],
            "2019": [
                fmt_number(d2019_daily["Total passengers"].sum(), 0),
                fmt_number(d2019_daily["Total passengers"].mean(), 0),
                fmt_number(d2019_daily["Total passengers"].median(), 0),
                fmt_number(d2019_daily["Total passengers"].min(), 0),
                fmt_number(d2019_daily["Total passengers"].max(), 0),
            ],
            "2022": [
                fmt_number(d2022_daily["Total passengers"].sum(), 0),
                fmt_number(d2022_daily["Total passengers"].mean(), 0),
                fmt_number(d2022_daily["Total passengers"].median(), 0),
                fmt_number(d2022_daily["Total passengers"].min(), 0),
                fmt_number(d2022_daily["Total passengers"].max(), 0),
            ],
        }
    )
    st.dataframe(prepared_table, use_container_width=True, hide_index=True)

    st.write("")
    with st.expander("Preview prepared 2019 daily data"):
        st.dataframe(d2019_daily.head(100), use_container_width=True, hide_index=True)

    with st.expander("Preview prepared 2022 daily data"):
        st.dataframe(d2022_daily.head(100), use_container_width=True, hide_index=True)

    with st.expander("Preview raw 2022 data"):
        st.dataframe(df2022_raw.head(100), use_container_width=True, hide_index=True)