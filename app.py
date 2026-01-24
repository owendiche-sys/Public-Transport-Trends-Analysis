# app.py
# Public Transport Trend Analysis (2019 vs 2022)
# Default datasets expected next to app.py:
#   - 2019data0.csv
#   - 2022data0.csv
#
# Non-negotiables:
# - Light theme only
# - Card UI styling (rounded cards, subtle border/shadow)
# - Clean professional labels (no emojis)
# - Sidebar radio navigation
# - KPI cards ONLY on Summary page
# - Insights page with exact structure:
#   1) Data-driven insights
#   2) Model-driven insights
# - Stand-alone wording (no notebook references)
#
# Data loading rules:
# - Default loads the local files above
# - Optional sidebar toggles to upload CSVs instead
# - No URL loading

from __future__ import annotations

import io
import os
from dataclasses import dataclass
from typing import Optional, Tuple, Dict, List

import numpy as np
import pandas as pd
import streamlit as st

from sklearn.linear_model import LinearRegression


DEFAULT_2019 = "2019data0.csv"
DEFAULT_2022 = "2022data0.csv"


# ----------------------------
# UI (light + cards)
# ----------------------------
def inject_css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #fafafa; color: #111827; }
          section[data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid rgba(15, 23, 42, 0.08);
          }

          .card {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 16px 16px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .card h3 { margin: 0 0 8px 0; }
          .muted { color: rgba(17, 24, 39, 0.70); font-size: 0.92rem; }

          .kpi-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
          }
          .kpi {
            background: #ffffff;
            border: 1px solid rgba(15, 23, 42, 0.10);
            border-radius: 14px;
            padding: 14px 14px;
            box-shadow: 0 1px 2px rgba(0,0,0,0.04);
          }
          .kpi .label { color: rgba(17, 24, 39, 0.70); font-size: 0.85rem; margin-bottom: 6px; }
          .kpi .value { font-size: 1.35rem; font-weight: 700; color: #111827; line-height: 1.1; }
          .kpi .sub { margin-top: 6px; color: rgba(17, 24, 39, 0.65); font-size: 0.85rem; }

          @media (max-width: 1100px) { .kpi-grid { grid-template-columns: repeat(2, minmax(0, 1fr)); } }
          @media (max-width: 600px) { .kpi-grid { grid-template-columns: repeat(1, minmax(0, 1fr)); } }
        </style>
        """,
        unsafe_allow_html=True,
    )


def kpi_card(label: str, value: str, sub: str = "") -> str:
    sub_html = f'<div class="sub">{sub}</div>' if sub else ""
    return f"""
      <div class="kpi">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        {sub_html}
      </div>
    """


def card(title: str, body_html: str) -> None:
    st.markdown(
        f"""
        <div class="card">
          <h3>{title}</h3>
          <div class="muted">{body_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


# ----------------------------
# Robust CSV loading (encoding + delimiter)
# ----------------------------
@dataclass
class DataMeta:
    source_2019: str
    source_2022: str


def _decode_bytes(b: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin1"):
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
    txt = _decode_bytes(file_bytes[:4096])
    sep = _detect_delimiter(txt)

    # Try detected delimiter
    try:
        df = pd.read_csv(io.BytesIO(file_bytes), sep=sep)
    except Exception:
        df = None

    # Fallbacks
    if df is None:
        for s in [";", ",", "\t", "|"]:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=s)
                break
            except Exception:
                df = None

    if df is None:
        df = pd.read_csv(io.BytesIO(file_bytes))

    # Retry if it collapsed into 1 column but looks delimited
    if df.shape[1] == 1:
        col0 = df.columns[0]
        sample_val = str(df.iloc[0, 0]) if len(df) else ""
        if ";" in col0 or ";" in sample_val:
            try:
                df = pd.read_csv(io.BytesIO(file_bytes), sep=";")
            except Exception:
                pass

    return df


@st.cache_data(show_spinner=False)
def load_two_files(
    use_upload_2019: bool,
    upload_2019,
    use_upload_2022: bool,
    upload_2022,
) -> Tuple[pd.DataFrame, pd.DataFrame, DataMeta]:
    # 2019
    if use_upload_2019 and upload_2019 is not None:
        b = upload_2019.read()
        df2019 = safe_read_csv(b)
        src2019 = f"Uploaded: {upload_2019.name}"
    else:
        if not os.path.exists(DEFAULT_2019):
            raise FileNotFoundError(f"Missing {DEFAULT_2019} next to app.py (or upload it in the sidebar).")
        with open(DEFAULT_2019, "rb") as f:
            df2019 = safe_read_csv(f.read())
        src2019 = f"Default: {DEFAULT_2019}"

    # 2022
    if use_upload_2022 and upload_2022 is not None:
        b = upload_2022.read()
        df2022 = safe_read_csv(b)
        src2022 = f"Uploaded: {upload_2022.name}"
    else:
        if not os.path.exists(DEFAULT_2022):
            raise FileNotFoundError(f"Missing {DEFAULT_2022} next to app.py (or upload it in the sidebar).")
        with open(DEFAULT_2022, "rb") as f:
            df2022 = safe_read_csv(f.read())
        src2022 = f"Default: {DEFAULT_2022}"

    return df2019, df2022, DataMeta(source_2019=src2019, source_2022=src2022)


# ----------------------------
# Core transforms (based on your script)
# ----------------------------
BUS_COLS = ["Bus pax number peak", "Bus pax number offpeak"]
TRAM_COLS = ["Tram pax number peak", "Tram pax number offpeak"]
METRO_COLS = ["Metro pax number peak", "Metro pax number offpeak"]

WEEKDAY_ORDER = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]


def fourier_smooth(y: np.ndarray, n_terms: int = 8) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    if y.size < 10:
        return y
    N = len(y)
    fft_vals = np.fft.rfft(y)
    keep = max(1, int(n_terms) + 1)
    fft_vals[keep:] = 0
    return np.fft.irfft(fft_vals, n=N)


@st.cache_data(show_spinner=False)
def build_2019_daily(df2019: pd.DataFrame) -> pd.DataFrame:
    d = df2019.copy()
    if "Date" not in d.columns:
        raise ValueError("2019 dataset must contain a 'Date' column.")

    d["Date"] = pd.to_datetime(d["Date"], errors="coerce")
    d = d.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    needed = [c for c in (BUS_COLS + TRAM_COLS + METRO_COLS) if c in d.columns]
    if len(needed) == 0:
        raise ValueError("2019 dataset is missing passenger count columns for Bus/Tram/Metro.")

    for c in needed:
        d[c] = pd.to_numeric(d[c], errors="coerce")

    d["Total passengers"] = d[needed].sum(axis=1, skipna=True)
    d["Day"] = d["Date"].dt.day_name()
    d["Day"] = pd.Categorical(d["Day"], categories=WEEKDAY_ORDER, ordered=True)
    return d


@st.cache_data(show_spinner=False)
def build_2022_daily_expanded(df2022: pd.DataFrame, total_annual_passengers: float) -> Tuple[pd.DataFrame, float]:
    """
    Expands sample counts to annual estimated totals using:
      expansion_factor = total_annual_passengers / total_sample
      daily_total = daily_sample * expansion_factor
    """
    d = df2022.copy()

    # Find datetime column
    dt_col = None
    for c in d.columns:
        if c.lower().strip() in {"date and time", "datetime", "date_time", "timestamp"}:
            dt_col = c
            break
    if dt_col is None:
        # heuristic
        for c in d.columns:
            if "date" in c.lower() and "time" in c.lower():
                dt_col = c
                break
    if dt_col is None:
        raise ValueError("2022 dataset must contain a datetime column like 'Date and time'.")

    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).reset_index(drop=True)
    d["Date"] = d[dt_col].dt.date

    daily = d.groupby("Date").size().reset_index(name="Sample")
    sample_total = float(daily["Sample"].sum())
    if sample_total <= 0:
        raise ValueError("2022 dataset has no valid sample rows after datetime parsing.")

    expansion_factor = float(total_annual_passengers) / sample_total
    daily["Total passengers"] = daily["Sample"] * expansion_factor

    daily["Date"] = pd.to_datetime(daily["Date"], errors="coerce")
    daily = daily.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)

    daily["Day"] = daily["Date"].dt.day_name()
    daily["Day"] = pd.Categorical(daily["Day"], categories=WEEKDAY_ORDER, ordered=True)

    return daily, expansion_factor


def compute_mode_shares_2019(df2019_daily: pd.DataFrame) -> Dict[str, float]:
    d = df2019_daily.copy()

    totals: Dict[str, float] = {}

    for label, cols in [("Bus", BUS_COLS), ("Tram", TRAM_COLS), ("Metro", METRO_COLS)]:
        present = [c for c in cols if c in d.columns]
        if not present:
            totals[label] = 0.0
            continue

        # Convert each column to numeric safely, then sum all values
        numeric_block = d[present].apply(pd.to_numeric, errors="coerce")
        totals[label] = float(numeric_block.to_numpy(dtype=float, na_value=np.nan).sum())

    total_all = float(sum(totals.values()))
    if not np.isfinite(total_all) or total_all <= 0:
        return {"Bus": 0.0, "Tram": 0.0, "Metro": 0.0}

    return {k: (v / total_all) * 100.0 for k, v in totals.items()}



def compute_mode_shares_2022(df2022: pd.DataFrame) -> Dict[str, float]:
    d = df2022.copy()
    if "Mode" not in d.columns:
        return {"Bus": 0.0, "Tram": 0.0, "Metro": 0.0}
    shares = d["Mode"].value_counts(normalize=True) * 100.0
    return {
        "Bus": float(shares.get("Bus", 0.0)),
        "Tram": float(shares.get("Tram", 0.0)),
        "Metro": float(shares.get("Metro", 0.0)),
    }


def compute_season_percentages_2022(df2022: pd.DataFrame) -> Dict[str, float]:
    d = df2022.copy()

    dt_col = None
    for c in d.columns:
        if c.lower().strip() in {"date and time", "datetime", "date_time", "timestamp"}:
            dt_col = c
            break
    if dt_col is None:
        for c in d.columns:
            if "date" in c.lower() and "time" in c.lower():
                dt_col = c
                break
    if dt_col is None:
        return {"Spring": float("nan"), "Summer": float("nan"), "Autumn": float("nan")}

    d[dt_col] = pd.to_datetime(d[dt_col], errors="coerce")
    d = d.dropna(subset=[dt_col]).copy()
    if len(d) == 0:
        return {"Spring": float("nan"), "Summer": float("nan"), "Autumn": float("nan")}

    d["Month"] = d[dt_col].dt.month
    total = len(d)
    spring = int(d[d["Month"].isin([3, 4, 5])].shape[0])
    summer = int(d[d["Month"].isin([6, 7, 8])].shape[0])
    autumn = int(d[d["Month"].isin([9, 10, 11])].shape[0])
    return {
        "Spring": (spring / total) * 100.0,
        "Summer": (summer / total) * 100.0,
        "Autumn": (autumn / total) * 100.0,
    }


def fit_metro_price_distance(df2022: pd.DataFrame) -> Optional[Dict[str, float]]:
    d = df2022.copy()
    needed = {"Mode", "Distance", "Price"}
    if not needed.issubset(set(d.columns)):
        return None

    metro = d[d["Mode"] == "Metro"].copy()
    if len(metro) < 30:
        return None

    metro["Distance"] = pd.to_numeric(metro["Distance"], errors="coerce")
    metro["Price"] = pd.to_numeric(metro["Price"], errors="coerce")
    metro = metro.dropna(subset=["Distance", "Price"])
    if len(metro) < 30:
        return None

    X = metro["Distance"].values.reshape(-1, 1)
    y = metro["Price"].values
    model = LinearRegression()
    model.fit(X, y)
    return {"slope": float(model.coef_[0]), "intercept": float(model.intercept_), "n": int(len(metro))}


# ----------------------------
# Plot helpers
# ----------------------------
def plot_daily_trends(d2019: pd.DataFrame, d2022: pd.DataFrame, n_terms: int) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    y2019 = d2019["Total passengers"].to_numpy(dtype=float)
    y2022 = d2022["Total passengers"].to_numpy(dtype=float)
    x2019 = np.arange(1, len(y2019) + 1)
    x2022 = np.arange(1, len(y2022) + 1)

    y2019_s = fourier_smooth(y2019, n_terms=n_terms)
    y2022_s = fourier_smooth(y2022, n_terms=n_terms)

    fig = plt.figure(figsize=(12, 6))
    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:,.0f}"))

    ax.scatter(x2019, y2019, s=22, alpha=0.6, label="2019", marker="o")
    ax.scatter(x2022, y2022, s=22, alpha=0.6, label="2022", marker="x")

    ax.plot(x2019, y2019_s, linewidth=2.5, label="2019 (Fourier)")
    ax.plot(x2022, y2022_s, linewidth=2.5, label="2022 (Fourier)")

    ax.set_xlabel("Day index within dataset")
    ax.set_ylabel("Total daily passengers")
    ax.set_title("Daily public transport passengers (2019 vs 2022)")
    ax.grid(True, linestyle="--", alpha=0.25)
    ax.legend(loc="upper center", ncol=4, frameon=True, edgecolor="black")

    st.pyplot(fig, clear_figure=True)


def plot_weekday_bars(avg2019: pd.Series, avg2022: pd.Series, seasonal: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter

    fig = plt.figure(figsize=(12, 6))
    x = np.arange(len(WEEKDAY_ORDER))
    width = 0.35

    plt.bar(x - width / 2, avg2019.values, width, label="2019", edgecolor="black")
    plt.bar(x + width / 2, avg2022.values, width, label="2022", edgecolor="black")

    plt.xticks(x, WEEKDAY_ORDER)
    plt.xlabel("Day of week")
    plt.ylabel("Average daily passengers")
    plt.title("Average daily passengers by day of week (2019 vs 2022)")

    ax = plt.gca()
    ax.ticklabel_format(style="plain", axis="y")
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))

    plt.legend(frameon=True, edgecolor="black")

    # Seasonal box (Spring/Summer/Autumn)
    text_box = (
        f"Spring share: {seasonal.get('Spring', float('nan')):.2f}%\n"
        f"Summer share: {seasonal.get('Summer', float('nan')):.2f}%\n"
        f"Autumn share: {seasonal.get('Autumn', float('nan')):.2f}%"
    )
    plt.text(
        0.72, 0.80, text_box,
        transform=ax.transAxes,
        fontsize=12,
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.85),
    )

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_price_distance(df2022: pd.DataFrame) -> None:
    import matplotlib.pyplot as plt

    if not {"Mode", "Distance", "Price"}.issubset(set(df2022.columns)):
        st.info("Price vs distance requires Mode, Distance, and Price columns in the 2022 dataset.")
        return

    d = df2022[df2022["Mode"] == "Metro"].copy()
    d["Distance"] = pd.to_numeric(d["Distance"], errors="coerce")
    d["Price"] = pd.to_numeric(d["Price"], errors="coerce")
    d = d.dropna(subset=["Distance", "Price"])
    if len(d) < 30:
        st.info("Not enough Metro rows to fit a stable line.")
        return

    X = d["Distance"].values.reshape(-1, 1)
    y = d["Price"].values

    model = LinearRegression()
    model.fit(X, y)
    a = float(model.coef_[0])
    b = float(model.intercept_)

    x_line = np.linspace(float(X.min()), float(X.max()), 150).reshape(-1, 1)
    y_line = model.predict(x_line)

    fig = plt.figure(figsize=(12, 6))
    plt.scatter(d["Distance"], d["Price"], s=18, alpha=0.6, label="Metro journeys", marker="x")
    plt.plot(x_line.ravel(), y_line, linewidth=2.5, label="Linear fit")

    plt.xlabel("Trip length (km)")
    plt.ylabel("Price")
    plt.title("Price vs trip length for 2022 Metro journeys")
    plt.legend(frameon=True, edgecolor="black", loc="upper left")

    plt.text(
        0.50, 0.15,
        f"Fit: Price = {a:.3f} × Distance + {b:.3f}",
        transform=plt.gca().transAxes,
        fontsize=12,
        ha="center",
        va="center",
        bbox=dict(facecolor="white", edgecolor="black", alpha=0.90),
    )

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


def plot_mode_shares(sh2019: Dict[str, float], sh2022: Dict[str, float]) -> None:
    import matplotlib.pyplot as plt

    labels = ["Bus", "Tram", "Metro"]
    vals2019 = [float(sh2019.get(k, 0.0)) for k in labels]
    vals2022 = [float(sh2022.get(k, 0.0)) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig = plt.figure(figsize=(12, 6))
    plt.bar(x - width / 2, vals2019, width, label="2019", edgecolor="black")
    plt.bar(x + width / 2, vals2022, width, label="2022", edgecolor="black")

    plt.xlabel("Mode of transport")
    plt.ylabel("Percentage of journeys (%)")
    plt.title("Fraction of journeys by transport mode (2019 vs 2022)")
    plt.xticks(x, labels)
    plt.legend(frameon=True, edgecolor="black")
    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)


# ----------------------------
# Pages
# ----------------------------
def page_summary(d2019: pd.DataFrame, d2022_daily: pd.DataFrame, meta: DataMeta, expansion_factor: float) -> None:
    st.title("Public Transport Trend Dashboard")
    st.caption(f"2019 source: {meta.source_2019} | 2022 source: {meta.source_2022}")

    # KPI cards (ONLY here)
    tot2019 = float(pd.to_numeric(d2019["Total passengers"], errors="coerce").sum())
    tot2022 = float(pd.to_numeric(d2022_daily["Total passengers"], errors="coerce").sum())
    mean2019 = float(pd.to_numeric(d2019["Total passengers"], errors="coerce").mean())
    mean2022 = float(pd.to_numeric(d2022_daily["Total passengers"], errors="coerce").mean())

    kpis_html = f"""
    <div class="kpi-grid">
      {kpi_card("2019 days", f"{len(d2019):,}", "Daily totals computed")}
      {kpi_card("2022 days", f"{len(d2022_daily):,}", "Expanded from sample")}
      {kpi_card("Total passengers (2019)", f"{tot2019:,.0f}", f"Average per day: {mean2019:,.0f}")}
      {kpi_card("Expansion factor (2022)", f"{expansion_factor:,.3f}", f"Average per day: {mean2022:,.0f}")}
    </div>
    """
    st.markdown(kpis_html, unsafe_allow_html=True)

    st.write("")
    c1, c2 = st.columns([1.2, 1.0], gap="large")
    with c1:
        card(
            "What this dashboard does",
            """
            • Computes total daily passengers for 2019 using Bus/Tram/Metro peak and off-peak counts.<br>
            • Converts 2022 trip-level records into daily sample counts, then scales them using a configurable annual total.<br>
            • Visualises daily trends, day-of-week patterns, mode share differences, and metro fare-distance behaviour.
            """.strip(),
        )
    with c2:
        card(
            "Data health checks",
            f"""
            • 2019 missing totals: {int(d2019["Total passengers"].isna().sum()):,}<br>
            • 2022 missing totals: {int(d2022_daily["Total passengers"].isna().sum()):,}<br>
            • 2022 expansion is based on sample size and the annual total you set in the sidebar
            """.strip(),
        )


def page_daily_trends(d2019: pd.DataFrame, d2022: pd.DataFrame) -> None:
    st.title("Daily Trends")
    st.caption("Compare daily passenger totals with optional Fourier smoothing.")

    st.markdown('<div class="card"><h3>Controls</h3>', unsafe_allow_html=True)
    n_terms = st.slider("Fourier smoothing terms", min_value=0, max_value=30, value=8, step=1)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Daily totals (2019 vs 2022)</h3>', unsafe_allow_html=True)
    plot_daily_trends(d2019, d2022, n_terms=n_terms)
    st.markdown("</div>", unsafe_allow_html=True)


def page_weekday_patterns(d2019: pd.DataFrame, d2022: pd.DataFrame, seasonal: Dict[str, float]) -> None:
    st.title("Day-of-Week Patterns")
    st.caption("Compare average passenger totals across weekdays for 2019 and 2022.")

    avg2019 = d2019.groupby("Day")["Total passengers"].mean().reindex(WEEKDAY_ORDER)
    avg2022 = d2022.groupby("Day")["Total passengers"].mean().reindex(WEEKDAY_ORDER)

    st.markdown('<div class="card"><h3>Average daily passengers by weekday</h3>', unsafe_allow_html=True)
    plot_weekday_bars(avg2019, avg2022, seasonal=seasonal)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Weekday table</h3>', unsafe_allow_html=True)
    table = pd.DataFrame({"2019": avg2019.values, "2022": avg2022.values}, index=WEEKDAY_ORDER)
    table = table.reset_index().rename(columns={"index": "Day"})
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_price_distance(df2022_raw: pd.DataFrame) -> None:
    st.title("Metro Price vs Distance")
    st.caption("Fit a linear model to 2022 Metro journeys (Price vs Distance).")

    st.markdown('<div class="card"><h3>Scatter and fitted line</h3>', unsafe_allow_html=True)
    plot_price_distance(df2022_raw)
    st.markdown("</div>", unsafe_allow_html=True)


def page_mode_share(d2019_daily: pd.DataFrame, df2022_raw: pd.DataFrame) -> None:
    st.title("Mode Share Comparison")
    st.caption("Compare mode shares between 2019 passenger totals and 2022 trip proportions.")

    sh2019 = compute_mode_shares_2019(d2019_daily)
    sh2022 = compute_mode_shares_2022(df2022_raw)

    st.markdown('<div class="card"><h3>Mode share chart</h3>', unsafe_allow_html=True)
    plot_mode_shares(sh2019, sh2022)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>Mode share table</h3>', unsafe_allow_html=True)
    table = pd.DataFrame(
        {
            "Mode": ["Bus", "Tram", "Metro"],
            "2019 share (%)": [sh2019["Bus"], sh2019["Tram"], sh2019["Metro"]],
            "2022 share (%)": [sh2022["Bus"], sh2022["Tram"], sh2022["Metro"]],
        }
    )
    st.dataframe(table, use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


def page_insights(
    d2019_daily: pd.DataFrame,
    d2022_daily: pd.DataFrame,
    df2022_raw: pd.DataFrame,
    seasonal: Dict[str, float],
) -> None:
    st.title("Insights")
    st.caption("Key findings based on observed data patterns and model behavior.")

    # 1) Data-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("1. Data-driven insights")

    tot2019 = float(d2019_daily["Total passengers"].sum())
    tot2022 = float(d2022_daily["Total passengers"].sum())

    avg2019 = float(d2019_daily["Total passengers"].mean())
    avg2022 = float(d2022_daily["Total passengers"].mean())

    st.write(f"Total passengers for 2019 (summed over available days): {tot2019:,.0f}.")
    st.write(f"Total passengers for 2022 (expanded estimate over available days): {tot2022:,.0f}.")
    st.write(f"Average daily passengers: 2019 = {avg2019:,.0f}, 2022 = {avg2022:,.0f}.")

    # Weekday contrast
    wk2019 = d2019_daily.groupby("Day")["Total passengers"].mean().reindex(WEEKDAY_ORDER)
    wk2022 = d2022_daily.groupby("Day")["Total passengers"].mean().reindex(WEEKDAY_ORDER)
    if wk2019.notna().any() and wk2022.notna().any():
        peak_day_2019 = wk2019.idxmax()
        peak_day_2022 = wk2022.idxmax()
        st.write(f"Peak weekday by average passengers: 2019 = {peak_day_2019}, 2022 = {peak_day_2022}.")

    # Seasonal composition (from 2022 trips)
    if all(np.isfinite([seasonal.get("Spring", np.nan), seasonal.get("Summer", np.nan), seasonal.get("Autumn", np.nan)])):
        st.write(
            f"2022 seasonal composition of trips: Spring = {seasonal['Spring']:.2f}%, "
            f"Summer = {seasonal['Summer']:.2f}%, Autumn = {seasonal['Autumn']:.2f}%."
        )

    # Mode shares
    sh2019 = compute_mode_shares_2019(d2019_daily)
    sh2022 = compute_mode_shares_2022(df2022_raw)
    st.write(
        "Mode shares differ between 2019 passenger totals and 2022 trip proportions. "
        "This highlights changes in usage mix and/or how each dataset represents demand."
    )
    st.dataframe(
        pd.DataFrame(
            {
                "Mode": ["Bus", "Tram", "Metro"],
                "2019 share (%)": [sh2019["Bus"], sh2019["Tram"], sh2019["Metro"]],
                "2022 share (%)": [sh2022["Bus"], sh2022["Tram"], sh2022["Metro"]],
            }
        ),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    # 2) Model-driven insights
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.header("2. Model-driven insights")

    # Fourier smoothing choice impact (simple model-ish tradeoff)
    st.subheader("Smoothing sensitivity")
    terms = st.slider("Smoothing terms for comparison", 0, 30, 8, 1, key="ins_terms")
    y2019 = d2019_daily["Total passengers"].to_numpy(dtype=float)
    y2022 = d2022_daily["Total passengers"].to_numpy(dtype=float)
    y2019_s = fourier_smooth(y2019, n_terms=terms)
    y2022_s = fourier_smooth(y2022, n_terms=terms)

    # Define a simple volatility metric: mean absolute day-to-day change (smoothed vs raw)
    def mad_change(arr: np.ndarray) -> float:
        if len(arr) < 2:
            return float("nan")
        return float(np.nanmean(np.abs(np.diff(arr))))

    raw_vol_2019 = mad_change(y2019)
    sm_vol_2019 = mad_change(y2019_s)
    raw_vol_2022 = mad_change(y2022)
    sm_vol_2022 = mad_change(y2022_s)

    st.write(
        "Fourier smoothing reduces short-term fluctuations. "
        "Higher term counts preserve more variation, while lower term counts emphasise long-run seasonality."
    )
    st.dataframe(
        pd.DataFrame(
            [
                ["2019", raw_vol_2019, sm_vol_2019],
                ["2022", raw_vol_2022, sm_vol_2022],
            ],
            columns=["Dataset", "Raw mean absolute daily change", "Smoothed mean absolute daily change"],
        ),
        use_container_width=True,
        hide_index=True,
    )

    # Fare model for Metro
    st.subheader("Fare-distance relationship for Metro journeys")
    fit = fit_metro_price_distance(df2022_raw)
    if fit is None:
        st.write("A stable metro fare-distance fit is not available due to missing columns or insufficient Metro rows.")
    else:
        st.write(
            f"Linear fit on {fit['n']:,} Metro journeys: "
            f"Price = {fit['slope']:.3f} × Distance + {fit['intercept']:.3f}."
        )
        st.write(
            "The slope approximates how much price increases per additional kilometre. "
            "The intercept approximates the base fare component at very short distances."
        )

    st.markdown("</div>", unsafe_allow_html=True)


def page_data(df2019_raw: pd.DataFrame, df2022_raw: pd.DataFrame) -> None:
    st.title("Data")
    st.caption("Inspect raw inputs used for the analysis.")

    st.markdown('<div class="card"><h3>2019 raw preview</h3>', unsafe_allow_html=True)
    st.dataframe(df2019_raw.head(200), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")
    st.markdown('<div class="card"><h3>2022 raw preview</h3>', unsafe_allow_html=True)
    st.dataframe(df2022_raw.head(200), use_container_width=True, hide_index=True)
    st.markdown("</div>", unsafe_allow_html=True)


# ----------------------------
# App
# ----------------------------
def main() -> None:
    st.set_page_config(page_title="Public Transport Trends", layout="wide", initial_sidebar_state="expanded")
    inject_css()

    st.sidebar.title("Navigation")
    page = st.sidebar.radio(
        "Go to",
        ["Summary", "Daily Trends", "Day-of-Week", "Metro Price vs Distance", "Mode Share", "Insights", "Data"],
        index=0,
    )

    st.sidebar.write("")
    st.sidebar.subheader("Data loading")

    use_up_2019 = st.sidebar.toggle("Use uploaded 2019 file instead of default", value=False)
    up_2019 = st.sidebar.file_uploader("Upload 2019 CSV", type=["csv"], key="up2019")

    use_up_2022 = st.sidebar.toggle("Use uploaded 2022 file instead of default", value=False)
    up_2022 = st.sidebar.file_uploader("Upload 2022 CSV", type=["csv"], key="up2022")

    st.sidebar.write("")
    st.sidebar.subheader("2022 scaling")
    total_annual = st.sidebar.number_input(
        "Total annual passengers (2022)",
        min_value=1_000_000.0,
        max_value=10_000_000_000.0,
        value=369_382_078.0,
        step=1_000_000.0,
        help="Used to scale 2022 sample counts into estimated daily totals.",
    )

    try:
        df2019_raw, df2022_raw, meta = load_two_files(use_up_2019, up_2019, use_up_2022, up_2022)
        d2019_daily = build_2019_daily(df2019_raw)
        d2022_daily, expansion_factor = build_2022_daily_expanded(df2022_raw, total_annual_passengers=float(total_annual))
        seasonal = compute_season_percentages_2022(df2022_raw)
    except Exception as e:
        st.error("Data could not be loaded or processed.")
        st.write(str(e))
        return

    if page == "Summary":
        page_summary(d2019_daily, d2022_daily, meta, expansion_factor)
    elif page == "Daily Trends":
        page_daily_trends(d2019_daily, d2022_daily)
    elif page == "Day-of-Week":
        page_weekday_patterns(d2019_daily, d2022_daily, seasonal)
    elif page == "Metro Price vs Distance":
        page_price_distance(df2022_raw)
    elif page == "Mode Share":
        page_mode_share(d2019_daily, df2022_raw)
    elif page == "Insights":
        page_insights(d2019_daily, d2022_daily, df2022_raw, seasonal)
    elif page == "Data":
        page_data(df2019_raw, df2022_raw)


if __name__ == "__main__":
    main()
