# streamlit_app_live_airlabs.py
# ============================================================
# JFK Live Delay Predictor (Two-Stage) — LIVE SCHEDULES MVP
# Data:
#   - AirLabs: /api/v9/schedules (upcoming JFK departures)
#   - Open-Meteo: hourly forecast (America/New_York)
# Models:
#   - Hopsworks Model Registry: Stage1 classifier + Stage2 regressor + metadata.json
#
# Run:
#   streamlit run streamlit_app_live_airlabs.py
# ============================================================

import os
import json
import tempfile
from dataclasses import dataclass
from typing import List, Optional, Dict, Any

import numpy as np
import pandas as pd
import requests
import streamlit as st
import joblib
import hopsworks
import pytz
from datetime import datetime

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

if load_dotenv:
    load_dotenv()

load_dotenv()

# =========================
# CONFIG
# =========================
HOPSWORKS_PROJECT = os.getenv("HOPSWORKS_PROJECT", "Flight_Predictor_JFK")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")

AIRLABS_API_KEY = os.getenv("AIRLABS_API_KEY")
AIRLABS_BASE_URL = os.getenv("AIRLABS_BASE_URL", "https://airlabs.co/api/v9")
AIRLABS_DEP_IATA = os.getenv("AIRLABS_DEP_IATA", "JFK").strip().upper()

# Model names in Hopsworks registry
DEFAULT_STAGE1_NAME = "jfk_delay_stage1_classifier"
DEFAULT_STAGE2_NAME = "jfk_delay_stage2_regressor"
DEFAULT_META_NAME   = "jfk_delay_two_stage_metadata"
DEFAULT_FV_NAME = "jfk_delay_weather_fv"
DEFAULT_FV_VERSION = 1

# JFK coords for Open-Meteo
JFK_LAT = 40.6413
JFK_LON = -73.7781
TZ_NY = "America/New_York"


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
# AirLabs — schedules (single endpoint, cached)
# ============================================================
@st.cache_data(ttl=10 * 60)
def fetch_airlabs_schedules(api_key: str, dep_iata: str, limit: int = 100) -> dict:
    url = f"{AIRLABS_BASE_URL}/schedules"
    params = {
        "api_key": api_key,
        "dep_iata": dep_iata,
        "limit": int(limit),
    }
    r = requests.get(url, params=params, timeout=30)
    r.raise_for_status()
    return r.json()


def normalize_airlabs_data(payload: dict) -> List[dict]:
    if not isinstance(payload, dict):
        return []

    if payload.get("error"):
        return []

    data = payload.get("response")
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        for key in ("data", "schedules", "items", "results"):
            items = data.get(key)
            if isinstance(items, list):
                return items
    return []


def parse_airlabs_schedules_to_df(rows: List[dict], min_departure_local: Optional[pd.Timestamp] = None) -> pd.DataFrame:
    out = []

    for r in rows:
        dep_iata = (r.get("dep_iata") or "").strip().upper()
        if dep_iata and dep_iata != AIRLABS_DEP_IATA:
            continue

        dest = (r.get("arr_iata") or "").strip().upper()
        flight_iata = (r.get("flight_iata") or "").strip().upper()
        reporting_airline = (r.get("airline_iata") or "").strip().upper()

        dep_time_utc = r.get("dep_time_utc")
        dep_time_local = r.get("dep_time")

        sched_dep_local = None
        if dep_time_utc:
            ts = pd.to_datetime(dep_time_utc, errors="coerce", utc=True)
            if pd.notna(ts):
                sched_dep_local = ts.tz_convert(TZ_NY)
        if sched_dep_local is None and dep_time_local:
            ts = pd.to_datetime(dep_time_local, errors="coerce")
            if pd.notna(ts):
                if ts.tzinfo is None:
                    sched_dep_local = ts.tz_localize(TZ_NY, ambiguous="NaT", nonexistent="shift_forward")
                else:
                    sched_dep_local = ts.tz_convert(TZ_NY)

        if sched_dep_local is None or pd.isna(sched_dep_local):
            continue
        if min_departure_local is not None and sched_dep_local < min_departure_local:
            continue

        if not reporting_airline and flight_iata:
            reporting_airline = flight_iata[:2]
        if not flight_iata:
            flight_number = r.get("flight_number")
            if flight_number and reporting_airline:
                flight_iata = f"{reporting_airline}{flight_number}".upper()

        if not dest:
            continue

        sched_hour_local = sched_dep_local.floor("h")
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
        df = df.drop_duplicates(subset=["flight_iata", "sched_dep_local"])
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
    df["weather_code"] = df["weather_jfk_hourly_fg_weather_code"]
    df["wind_speed_ms"] = df["weather_jfk_hourly_fg_wind_speed_ms"]
    df["wind_gust_ms"] = df["weather_jfk_hourly_fg_wind_gust_ms"]
    df["temp_c"] = df["weather_jfk_hourly_fg_temp_c"]
    df["precip_mm"] = df["weather_jfk_hourly_fg_precip_mm"]
    df["snowfall_cm"] = df["weather_jfk_hourly_fg_snowfall_cm"]

    # Make sure weather_hour_local is tz-aware NY
    df["weather_hour_local"] = pd.to_datetime(df["weather_hour_local"], errors="coerce")
    if df["weather_hour_local"].dt.tz is None:
        df["weather_hour_local"] = df["weather_hour_local"].dt.tz_localize(TZ_NY, ambiguous="NaT", nonexistent="shift_forward")

    df = df.dropna(subset=["weather_hour_local"])
    return df


# ============================================================
# Helpers: distance + weather join + model feature frame
# ============================================================
@st.cache_resource(ttl=24 * 60 * 60)
def build_dest_distance_lookup(_project, fv_name: str, fv_version: int) -> Dict[str, float]:
    fs = _project.get_feature_store()
    fv = fs.get_feature_view(fv_name, version=fv_version)
    df = fv.get_batch_data()
    lookup = (
        df.groupby("dest")["distance"]
          .median()
          .dropna()
          .to_dict()
    )
    return lookup


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
st.set_page_config(page_title="JFK Live Delay Predictor", page_icon="🛫", layout="wide")
st.title("JFK Live Departure Delay Predictor 🛫")
st.caption(
    "Uses AirLabs **/api/v9/schedules** (JFK departures) + Open-Meteo hourly forecast + Hopsworks two-stage model. "
    "Airlabs responses are cached for at least 10 minutes to protect the free-tier quota."
)

if load_dotenv is None:
    st.warning("python-dotenv not installed; set env vars manually or install python-dotenv.")

if not HOPSWORKS_API_KEY:
    st.error("Set HOPSWORKS_API_KEY in your environment or .env file.")
    st.stop()

if not AIRLABS_API_KEY:
    st.error("Set AIRLABS_API_KEY in your environment or .env file.")
    st.stop()

if "airlabs_payload" not in st.session_state:
    st.session_state["airlabs_payload"] = None
if "airlabs_last_fetch" not in st.session_state:
    st.session_state["airlabs_last_fetch"] = None

with st.sidebar:
    ui_threshold = st.slider(
        "Delay probability threshold (UI)",
        min_value=0.20,
        max_value=0.90,
        value=0.40,
        step=0.05,
        help="Higher threshold -> fewer flights flagged as delayed."
    )

    refresh = st.button("🔄 Refresh flights & predict", type="primary")

    if st.session_state["airlabs_last_fetch"]:
        last_fetch = st.session_state["airlabs_last_fetch"].strftime("%Y-%m-%d %H:%M %Z")
        st.caption(f"Last refresh: {last_fetch}")

if refresh:
    with st.spinner("Fetching upcoming JFK departures… (AirLabs schedules)"):
        try:
            payload = fetch_airlabs_schedules(
                api_key=AIRLABS_API_KEY,
                dep_iata=AIRLABS_DEP_IATA,
                limit=75,
            )
        except Exception as e:
            st.error(f"AirLabs request failed: {e}")
            st.stop()

    st.session_state["airlabs_payload"] = payload
    st.session_state["airlabs_last_fetch"] = datetime.now(pytz.timezone(TZ_NY))

payload = st.session_state["airlabs_payload"]
if payload is None:
    st.info("Click **Refresh flights & predict** to fetch live departures.")
    st.stop()

rows = normalize_airlabs_data(payload)
if not rows:
    st.warning("No records returned from AirLabs. Showing payload sample:")
    st.code(json.dumps(payload, indent=2)[:1500])
    st.stop()

min_departure = pd.Timestamp.now(tz=TZ_NY)
df_flights = parse_airlabs_schedules_to_df(rows, min_departure_local=min_departure)
if df_flights.empty:
    st.warning("No upcoming JFK departures after time filter; showing all schedules returned by AirLabs.")
    df_flights = parse_airlabs_schedules_to_df(rows, min_departure_local=None)

if df_flights.empty:
    st.warning("No JFK departures parsed from AirLabs. Showing payload sample:")
    st.code(json.dumps(rows[:3], indent=2)[:1500])
    st.stop()

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
predictor = load_predictor(project, DEFAULT_STAGE1_NAME, DEFAULT_STAGE2_NAME, DEFAULT_META_NAME)

# Apply UI threshold
predictor.threshold = float(ui_threshold)

# ============================================================
# Distance lookup from Feature View (cached 24h)
# ============================================================
DEFAULT_DISTANCE = 1000.0
try:
    DIST_LOOKUP = build_dest_distance_lookup(project, DEFAULT_FV_NAME, DEFAULT_FV_VERSION)
except Exception as e:
    st.warning(f"Distance lookup failed: {e}. Using fallback distance={DEFAULT_DISTANCE}.")
    DIST_LOOKUP = {}

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

weather_box = None
if "sched_hour_local" in df_out.columns:
    next_hour = df_out["sched_hour_local"].min()
    if pd.notna(next_hour):
        wx_row = df_out[df_out["sched_hour_local"] == next_hour].head(1)
        if not wx_row.empty:
            weather_box = {
                "hour": next_hour,
                "weather_code": wx_row.get("weather_jfk_hourly_fg_weather_code", pd.Series([np.nan])).iloc[0],
                "wind_speed_ms": wx_row.get("weather_jfk_hourly_fg_wind_speed_ms", pd.Series([np.nan])).iloc[0],
                "wind_gust_ms": wx_row.get("weather_jfk_hourly_fg_wind_gust_ms", pd.Series([np.nan])).iloc[0],
                "temp_c": wx_row.get("weather_jfk_hourly_fg_temp_c", pd.Series([np.nan])).iloc[0],
                "precip_mm": wx_row.get("weather_jfk_hourly_fg_precip_mm", pd.Series([np.nan])).iloc[0],
                "snowfall_cm": wx_row.get("weather_jfk_hourly_fg_snowfall_cm", pd.Series([np.nan])).iloc[0],
            }

st.divider()
st.subheader("Upcoming JFK departures (predicted)")
st.caption(
    "Column guide: Scheduled (NY) = local departure time, Flight = IATA code, Airline = carrier code, "
    "Destination = arrival airport, Delay chance (%) = probability from the classifier, "
    "Predicted delay (min) = regressor output (0 if below threshold)."
)

if weather_box:
    st.info(
        "JFK weather near the next scheduled hour\n"
        f"- Hour (NY): {weather_box['hour']:%Y-%m-%d %H:%M %Z}\n"
        f"- Weather code: {weather_box['weather_code']}\n"
        f"- Temp (C): {weather_box['temp_c']:.1f}\n"
        f"- Wind speed (m/s): {weather_box['wind_speed_ms']:.1f}\n"
        f"- Wind gust (m/s): {weather_box['wind_gust_ms']:.1f}\n"
        f"- Precip (mm): {weather_box['precip_mm']:.1f}\n"
        f"- Snowfall (cm): {weather_box['snowfall_cm']:.1f}"
    )

display_cols = [
    "sched_dep_local_str",
    "flight_iata",
    "reporting_airline",
    "dest",
    "p_delayed_pct",
    "pred_delay_min",
]
for col in display_cols:
    if col not in df_out.columns:
        df_out[col] = np.nan

df_show = df_out[display_cols].copy()
df_show = df_show.sort_values(["pred_delay_min", "p_delayed_pct"], ascending=[False, False])
df_show = df_show.rename(columns={
    "sched_dep_local_str": "Scheduled (NY)",
    "flight_iata": "Flight",
    "reporting_airline": "Airline",
    "dest": "Destination",
    "p_delayed_pct": "Delay chance (%)",
    "pred_delay_min": "Predicted delay (min)",
})

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
    "- AirLabs schedules are cached for 10 minutes; refresh only when needed.\n"
    "- Open-Meteo provides hourly forecasts; missing weather will be imputed by the model pipeline.\n"
    "- If everything looks 'delayed', raise threshold (0.45–0.60). If everything looks 'on time', lower it (0.25–0.40)."
)
