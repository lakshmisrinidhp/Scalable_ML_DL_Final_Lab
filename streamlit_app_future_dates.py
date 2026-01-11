# streamlit_app_flightsfuture.py
# ============================================================
# JFK Live Delay Predictor (Two-Stage) — FUTURE SCHEDULES MVP
# Data:
#   - Aviationstack: /v1/flightsFuture (future departures; date must be > 7 days ahead)
#   - Open-Meteo: hourly forecast (may not fully cover far future dates)
# Models:
#   - Hopsworks Model Registry: Stage1 classifier + Stage2 regressor + metadata.json
#
# Run:
#   streamlit run streamlit_app_flightsfuture.py
# ============================================================

import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

from dotenv import load_dotenv
import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import hopsworks
import pytz
from datetime import datetime, date, timedelta

load_dotenv()

# =========================
# CONFIG — FILL THESE
# =========================
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "Flight_Predictor_JFK")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

AVIATIONSTACK_API_KEY = os.getenv("AVIATIONSTACK_API_KEY")
AVIATIONSTACK_BASE_URL = os.getenv("AVIATIONSTACK_BASE_URL", "https://api.aviationstack.com/v1")

# Model names in Hopsworks registry
DEFAULT_STAGE1_NAME = "jfk_delay_stage1_classifier"
DEFAULT_STAGE2_NAME = "jfk_delay_stage2_regressor"
DEFAULT_META_NAME   = "jfk_delay_two_stage_metadata"

# JFK coords for Open-Meteo
JFK_LAT = 40.6413
JFK_LON = -73.7781
TZ_NY = "America/New_York"

if not HOPSWORKS_API_KEY:
    st.error("Set HOPSWORKS_API_KEY in your environment or .env file.")
    st.stop()

if not AVIATIONSTACK_API_KEY:
    st.error("Set AVIATIONSTACK_API_KEY in your environment or .env file.")
    st.stop()


# ============================================================
# Two-stage predictor
# ============================================================
@dataclass
class TwoStageDelayPredictor:
    stage1_model: Any
    stage2_model: Any
    feature_cols: List[str]
    threshold: float

    @staticmethod
    def _find_first_model_file(path: str) -> str:
        # Finds first .pkl or .joblib inside a downloaded model folder
        for root, _, files in os.walk(path):
            for f in files:
                if f.endswith(".pkl") or f.endswith(".joblib"):
                    return os.path.join(root, f)
        raise FileNotFoundError(f"No .pkl/.joblib found under: {path}")

    @classmethod
    def load_from_hopsworks(
        cls,
        project,
        stage1_name: str,
        stage2_name: str,
        metadata_name: str,
        stage1_version: Optional[int] = None,
        stage2_version: Optional[int] = None,
        metadata_version: Optional[int] = None,
        fallback_threshold: float = 0.40,
        fallback_feature_cols: Optional[List[str]] = None,
    ):
        mr = project.get_model_registry()

        stage1_obj = mr.get_model(stage1_name, version=stage1_version) if stage1_version else mr.get_model(stage1_name)
        stage2_obj = mr.get_model(stage2_name, version=stage2_version) if stage2_version else mr.get_model(stage2_name)
        meta_obj   = mr.get_model(metadata_name, version=metadata_version) if metadata_version else mr.get_model(metadata_name)

        # Use a unique temp dir each time to avoid overwrite collisions
        tmp_dir = tempfile.mkdtemp(prefix="hs_models_")
        stage1_dir = os.path.join(tmp_dir, "stage1")
        stage2_dir = os.path.join(tmp_dir, "stage2")
        meta_dir   = os.path.join(tmp_dir, "meta")
        os.makedirs(stage1_dir, exist_ok=True)
        os.makedirs(stage2_dir, exist_ok=True)
        os.makedirs(meta_dir, exist_ok=True)

        stage1_path = stage1_obj.download(stage1_dir)
        stage2_path = stage2_obj.download(stage2_dir)
        meta_path   = meta_obj.download(meta_dir)

        stage1_file = cls._find_first_model_file(stage1_path)
        stage2_file = cls._find_first_model_file(stage2_path)
        stage1_model = joblib.load(stage1_file)
        stage2_model = joblib.load(stage2_file)

        # Load metadata.json if present
        threshold = fallback_threshold
        feature_cols = fallback_feature_cols

        meta_json = None
        for root, _, files in os.walk(meta_path):
            for f in files:
                if f.endswith(".json"):
                    meta_json = os.path.join(root, f)
                    break

        if meta_json and os.path.exists(meta_json):
            with open(meta_json, "r") as fp:
                md = json.load(fp)

            if "live_threshold" in md:
                threshold = float(md["live_threshold"])
            elif "threshold" in md:
                threshold = float(md["threshold"])
            elif "decision_rule" in md and isinstance(md["decision_rule"], dict) and "live_threshold" in md["decision_rule"]:
                threshold = float(md["decision_rule"]["live_threshold"])

            if "feature_cols" in md and isinstance(md["feature_cols"], list):
                feature_cols = md["feature_cols"]

        if feature_cols is None:
            raise ValueError("feature_cols missing. Put feature_cols in metadata JSON or pass fallback_feature_cols.")

        return cls(stage1_model=stage1_model, stage2_model=stage2_model, feature_cols=feature_cols, threshold=threshold)

    def predict_dataframe(self, X: pd.DataFrame, include_proba: bool = True) -> pd.DataFrame:
        X = X.copy()

        missing = [c for c in self.feature_cols if c not in X.columns]
        if missing:
            raise ValueError(f"Missing required feature columns: {missing}")

        X_model = X[self.feature_cols]
        p = self.stage1_model.predict_proba(X_model)[:, 1]
        is_delayed = (p >= self.threshold).astype(int)

        pred_delay = np.zeros(len(X_model), dtype=float)
        if is_delayed.sum() > 0:
            pred_delay[is_delayed == 1] = self.stage2_model.predict(X_model[is_delayed == 1])

        pred_delay = np.clip(pred_delay, 0, None)

        out = pd.DataFrame({
            "p_delayed": p,
            "pred_is_delayed": is_delayed,
            "pred_delay_min": pred_delay
        })
        return out if include_proba else out.drop(columns=["p_delayed"])


# ============================================================
# Aviationstack — flightsFuture
# ============================================================
@st.cache_data(ttl=65)  # free plan: 1 request per 60s
def fetch_aviationstack_flights_future_jfk(api_key: str, future_date: str, limit: int = 100, offset: int = 0) -> dict:
    """
    GET /v1/flightsFuture
    Required:
      - iataCode=JFK
      - type=departure
      - date=YYYY-MM-DD (must be > 7 days from today)
    """
    url = f"{AVIATIONSTACK_BASE_URL}/flightsFuture"
    params = {
        "access_key": api_key,
        "iataCode": "JFK",
        "type": "departure",
        "date": future_date,
        "limit": int(limit),
        "offset": int(offset),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def normalize_aviationstack_data(payload: dict) -> List[dict]:
    """
    flightsFuture may return:
      - data: list[list[dict]]
      - data: list[dict]
      - errors: {success:false, error:"No Record Found"}
    """
    if not isinstance(payload, dict):
        return []

    if payload.get("success") is False and payload.get("error"):
        return []

    data = payload.get("data", [])
    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], dict):
            return data
        if isinstance(data[0], list):
            flat = []
            for chunk in data:
                if isinstance(chunk, list):
                    flat.extend([x for x in chunk if isinstance(x, dict)])
            return flat
    return []


def parse_flightsfuture_to_df(rows: List[dict], future_date: str) -> pd.DataFrame:
    out = []

    for r in rows:
        dep = r.get("departure") or {}
        arr = r.get("arrival") or {}
        airline = r.get("airline") or {}
        flight = r.get("flight") or {}
        codeshared = r.get("codeshared") or {}

        dep_iata = (dep.get("iataCode") or "").strip().upper()
        dest = (arr.get("iataCode") or "").strip().upper()
        sched_time = (dep.get("scheduledTime") or "").strip()  # "HH:MM"

        if dep_iata != "JFK" or not dest or not sched_time:
            continue
        if ":" not in sched_time:
            continue

        dt_str = f"{future_date} {sched_time}"
        sched_naive = pd.to_datetime(dt_str, errors="coerce")
        if pd.isna(sched_naive):
            continue

        sched_dep_local = sched_naive.tz_localize(
            TZ_NY, ambiguous="NaT", nonexistent="shift_forward"
        )
        if pd.isna(sched_dep_local):
            continue

        sched_hour_local = sched_dep_local.floor("h")

        # Prefer codeshare airline/flight (usually the marketing carrier)
        reporting_airline = (airline.get("iataCode") or "").strip().upper()
        if isinstance(codeshared, dict) and codeshared.get("airline"):
            cs_air = (codeshared.get("airline") or {}).get("iataCode")
            if cs_air:
                reporting_airline = str(cs_air).strip().upper()

        flight_iata = (flight.get("iataNumber") or "").strip().upper()
        if isinstance(codeshared, dict) and codeshared.get("flight"):
            cs_flt = (codeshared.get("flight") or {}).get("iataNumber")
            if cs_flt:
                flight_iata = str(cs_flt).strip().upper()

        if not reporting_airline and flight_iata:
            reporting_airline = flight_iata[:2]

        day_of_week = int(sched_dep_local.weekday()) + 1  # 1=Mon..7=Sun

        out.append({
            "flight_iata": flight_iata,
            "reporting_airline": reporting_airline,
            "dest": dest,
            "sched_dep_local": sched_dep_local,
            "sched_hour_local": sched_hour_local,
            "month": int(sched_dep_local.month),
            "day_of_week": int(day_of_week),
        })

    df = pd.DataFrame(out)
    if not df.empty:
        df = df.sort_values("sched_dep_local").reset_index(drop=True)
    return df


# ============================================================
# Open-Meteo hourly forecast (NY local)
# ============================================================
@st.cache_data(ttl=15 * 60)
def fetch_openmeteo_hourly_forecast_jfk(tz: str) -> pd.DataFrame:
    """
    Fetch hourly forecast for JFK with timezone=America/New_York.
    Important: pd.to_datetime(list) -> DatetimeIndex; we convert to Series for .dt.
    """
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": JFK_LAT,
        "longitude": JFK_LON,
        "hourly": ",".join([
            "weathercode",
            "windspeed_10m",
            "windgusts_10m",
            "temperature_2m",
            "precipitation",
            "snowfall",
        ]),
        "timezone": tz
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    data = r.json()

    hourly = data.get("hourly", {})
    time_list = hourly.get("time", [])
    times = pd.Series(pd.to_datetime(time_list, errors="coerce"))  # Series -> .dt works

    df = pd.DataFrame({
        "weather_hour_local": times.dt.floor("h"),
        "weather_jfk_hourly_fg_weather_code": pd.to_numeric(hourly.get("weathercode", []), errors="coerce"),
        # km/h -> m/s
        "weather_jfk_hourly_fg_wind_speed_ms": pd.to_numeric(hourly.get("windspeed_10m", []), errors="coerce") / 3.6,
        "weather_jfk_hourly_fg_wind_gust_ms": pd.to_numeric(hourly.get("windgusts_10m", []), errors="coerce") / 3.6,
        "weather_jfk_hourly_fg_temp_c": pd.to_numeric(hourly.get("temperature_2m", []), errors="coerce"),
        "weather_jfk_hourly_fg_precip_mm": pd.to_numeric(hourly.get("precipitation", []), errors="coerce"),
        "weather_jfk_hourly_fg_snowfall_cm": pd.to_numeric(hourly.get("snowfall", []), errors="coerce"),
    })

    # Make sure weather_hour_local is tz-aware NY
    df["weather_hour_local"] = pd.to_datetime(df["weather_hour_local"], errors="coerce")
    if df["weather_hour_local"].dt.tz is None:
        df["weather_hour_local"] = df["weather_hour_local"].dt.tz_localize(TZ_NY, ambiguous="NaT", nonexistent="shift_forward")

    df = df.dropna(subset=["weather_hour_local"])
    return df


# ============================================================
# Helpers: distance + weather join + model feature frame
# ============================================================
def add_distance(df_flights: pd.DataFrame, dist_lookup: Dict[str, float], default_distance: float = 1000.0) -> pd.DataFrame:
    df = df_flights.copy()
    df["distance"] = df["dest"].map(dist_lookup).astype(float)
    df["distance"] = df["distance"].fillna(float(default_distance))
    return df


def add_weather(df_flights: pd.DataFrame, df_weather: pd.DataFrame) -> pd.DataFrame:
    df = df_flights.copy()
    wx = df_weather.copy()

    # Coerce both join keys to same tz-aware dtype
    df["sched_hour_local"] = pd.to_datetime(df["sched_hour_local"], errors="coerce")
    wx["weather_hour_local"] = pd.to_datetime(wx["weather_hour_local"], errors="coerce")

    # Ensure both tz-aware in NY
    if df["sched_hour_local"].dt.tz is None:
        df["sched_hour_local"] = df["sched_hour_local"].dt.tz_localize(TZ_NY, ambiguous="NaT", nonexistent="shift_forward")
    else:
        df["sched_hour_local"] = df["sched_hour_local"].dt.tz_convert(TZ_NY)

    if wx["weather_hour_local"].dt.tz is None:
        wx["weather_hour_local"] = wx["weather_hour_local"].dt.tz_localize(TZ_NY, ambiguous="NaT", nonexistent="shift_forward")
    else:
        wx["weather_hour_local"] = wx["weather_hour_local"].dt.tz_convert(TZ_NY)

    df = df.merge(
        wx,
        left_on="sched_hour_local",
        right_on="weather_hour_local",
        how="left",
    ).drop(columns=["weather_hour_local"], errors="ignore")

    return df


def build_model_features(df_joined: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df_joined.copy()
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    # Normalize string cols expected by OHE
    if "reporting_airline" in df.columns:
        df["reporting_airline"] = df["reporting_airline"].astype(str)
    if "dest" in df.columns:
        df["dest"] = df["dest"].astype(str)

    return df[feature_cols].copy()


def nice_delay_label(mins: float) -> str:
    if mins <= 0:
        return "On time"
    if mins < 15:
        return f"{mins:.0f} min (small)"
    if mins < 45:
        return f"{mins:.0f} min (moderate)"
    return f"{mins:.0f} min (high)"


# ============================================================
# Streamlit UI
# ============================================================
st.set_page_config(page_title="JFK Future Delay Predictor", page_icon="🛫", layout="wide")
st.title("🛫 JFK Departure Delay Predictor — Future Schedules (flightsFuture)")
st.caption(
    "Uses Aviationstack **/v1/flightsFuture** (future date > 7 days) + Open-Meteo hourly forecast (if available) "
    "+ Hopsworks two-stage model."
)

with st.sidebar:
    st.header("Settings")

    # Future date picker
    ny = pytz.timezone(TZ_NY)
    today_ny = datetime.now(ny).date()
    default_future = today_ny + timedelta(days=8)  # must be > 7 days
    future_date = st.date_input("Future date (>= 8 days from today)", value=default_future)

    flights_limit = st.slider("Max rows to fetch", 20, 200, 100, 20)
    st.caption("Free plan note: /flightsFuture is rate-limited (often 1 request / 60s).")

    st.divider()
    stage1_name = st.text_input("Stage 1 model name", value=DEFAULT_STAGE1_NAME)
    stage2_name = st.text_input("Stage 2 model name", value=DEFAULT_STAGE2_NAME)
    meta_name   = st.text_input("Metadata model name", value=DEFAULT_META_NAME)

    st.divider()
    ui_threshold = st.slider(
        "Delay probability threshold (UI)",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Higher threshold -> fewer flights flagged as delayed."
    )

    refresh = st.button("🔄 Fetch & Predict", type="primary")


if not refresh:
    st.info("Choose a future date (>= 8 days) and click **Fetch & Predict**.")
    st.stop()

# Validate future date rule
min_allowed = today_ny + timedelta(days=8)
if future_date < min_allowed:
    st.error(f"flightsFuture requires a date > 7 days ahead. Pick **{min_allowed}** or later.")
    st.stop()

future_date_str = future_date.strftime("%Y-%m-%d")

# ============================================================
# Hopsworks login + model load (cached)
# ============================================================
@st.cache_resource(show_spinner="Connecting to Hopsworks…")
def get_hopsworks_project():
    return hopsworks.login(project=HOPSWORKS_PROJECT, api_key_value=HOPSWORKS_API_KEY)

@st.cache_resource(show_spinner="Loading models from Hopsworks…")
def load_predictor(_project, s1, s2, meta):
    return TwoStageDelayPredictor.load_from_hopsworks(
        project=_project,
        stage1_name=s1,
        stage2_name=s2,
        metadata_name=meta,
    )

project = get_hopsworks_project()
predictor = load_predictor(project, stage1_name, stage2_name, meta_name)

# Apply UI threshold
predictor.threshold = float(ui_threshold)

# ============================================================
# Distance lookup: simplest MVP
# - If you already have a dest->distance dict, paste it here.
# - Otherwise we use a constant default.
# ============================================================
DIST_LOOKUP: Dict[str, float] = {}  # optional: {"AUS": 1521.0, "PIT": 340.0, ...}
DEFAULT_DISTANCE = 1000.0

# ============================================================
# Fetch future schedules
# ============================================================
with st.spinner(f"Fetching future departures… (Aviationstack /v1/flightsFuture, date={future_date_str})"):
    try:
        payload = fetch_aviationstack_flights_future_jfk(
            api_key=AVIATIONSTACK_API_KEY,
            future_date=future_date_str,
            limit=int(flights_limit),
            offset=0
        )
    except Exception as e:
        st.error(f"Aviationstack request failed: {e}")
        st.stop()

rows = normalize_aviationstack_data(payload)

# Debug if empty
if not rows:
    st.warning("No records returned. Possible reasons:\n"
               "- Your plan doesn't include flightsFuture data for JFK\n"
               "- The date is not >7 days ahead\n"
               "- Aviationstack returned 'No Record Found'\n")
    st.code(json.dumps(payload, indent=2)[:1500])
    st.stop()

df_flights = parse_flightsfuture_to_df(rows, future_date_str)
if df_flights.empty:
    st.warning("Records returned, but none could be parsed into flights. Showing payload sample:")
    st.code(json.dumps(rows[0], indent=2)[:1500])
    st.stop()

# ============================================================
# Weather
# ============================================================
with st.spinner("Fetching hourly weather… (Open-Meteo)"):
    try:
        df_weather = fetch_openmeteo_hourly_forecast_jfk(TZ_NY)
    except Exception as e:
        st.warning(f"Open-Meteo failed: {e}. Continuing without weather (model will impute).")
        df_weather = pd.DataFrame(columns=["weather_hour_local"])

# Join distance + weather
df_joined = add_distance(df_flights, DIST_LOOKUP, default_distance=DEFAULT_DISTANCE)
if not df_weather.empty:
    df_joined = add_weather(df_joined, df_weather)

# Build model features
X_live = build_model_features(df_joined, predictor.feature_cols)

# Predict
with st.spinner("Running predictions…"):
    preds = predictor.predict_dataframe(X_live, include_proba=True)

df_out = df_joined.reset_index(drop=True).join(preds)

# Friendly columns
df_out["sched_dep_local_str"] = df_out["sched_dep_local"].dt.strftime("%Y-%m-%d %H:%M %Z")
df_out["p_delayed_pct"] = (df_out["p_delayed"] * 100).round(1)
df_out["delay_label"] = df_out["pred_delay_min"].apply(nice_delay_label)

# ============================================================
# UI — summary + table + simple charts
# ============================================================
c1, c2, c3, c4 = st.columns(4)
c1.metric("Flights", len(df_out))
c2.metric("Threshold", f"{predictor.threshold:.2f}")
c3.metric("Predicted delayed rate", f"{(df_out['pred_is_delayed'].mean()*100):.1f}%")
c4.metric("Avg predicted delay", f"{df_out['pred_delay_min'].mean():.1f} min")

st.divider()
st.subheader(f"Future JFK departures on {future_date_str} (predicted)")

display_cols = [
    "sched_dep_local_str",
    "flight_iata",
    "reporting_airline",
    "dest",
    "p_delayed_pct",
    "pred_delay_min",
    "delay_label",
    "distance",
    "weather_jfk_hourly_fg_weather_code",
    "weather_jfk_hourly_fg_wind_speed_ms",
    "weather_jfk_hourly_fg_precip_mm",
]
for col in display_cols:
    if col not in df_out.columns:
        df_out[col] = np.nan

df_show = df_out[display_cols].copy()
df_show = df_show.sort_values(["pred_delay_min", "p_delayed_pct"], ascending=[False, False])

st.dataframe(df_show, use_container_width=True, hide_index=True)

st.divider()
st.subheader("Quick visuals")

left, right = st.columns(2)
with left:
    st.write("Predicted delay distribution (minutes)")
    bins = [0, 1, 5, 10, 15, 30, 45, 60, 90, 120, 180, 240]
    hist = np.histogram(df_out["pred_delay_min"].fillna(0), bins=bins)
    hist_df = pd.DataFrame({
        "bin": [f"{bins[i]}–{bins[i+1]}" for i in range(len(bins) - 1)],
        "count": hist[0]
    })
    st.bar_chart(hist_df.set_index("bin"))

with right:
    st.write("Average predicted delay by scheduled hour (NY local)")
    tmp = df_out.copy()
    tmp["hour"] = tmp["sched_dep_local"].dt.hour
    avg_by_hour = tmp.groupby("hour")["pred_delay_min"].mean().reset_index()
    st.line_chart(avg_by_hour.set_index("hour"))

st.divider()
st.caption(
    "Notes:\n"
    "- flightsFuture is for dates **more than 7 days** ahead.\n"
    "- Open-Meteo forecast might not cover far future dates; weather may be missing and imputed.\n"
    "- If everything looks 'delayed', raise threshold (0.45–0.60). If everything looks 'on time', lower it (0.25–0.40)."
)
