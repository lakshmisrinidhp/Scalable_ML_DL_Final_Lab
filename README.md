# JFK Departure Delay Prediction (Live)

Live demo: https://scalablemldlfinallab-spcv7qu66j8vkkuaxdt3kp.streamlit.app/

This project predicts departure delays for JFK flights using a two-stage ML model:
1) Stage 1 classifier estimates delay probability.
2) Stage 2 regressor predicts delay minutes for flights flagged as delayed.

The app combines live schedules, weather forecasts, and historical features to provide a simple, shareable demo.

## What This App Does
- Fetches upcoming JFK departures from AirLabs (single request per refresh).
- Joins hourly weather from Open-Meteo for JFK.
- Builds a destination-to-distance lookup from a Hopsworks Feature View.
- Runs two-stage inference and shows results in a Streamlit UI.

## Key Features
- Manual refresh only (no auto-polling).
- API responses cached for 10 minutes to protect free-tier quotas.
- Distance lookup cached for 24 hours.
- Clear UI with summary metrics and a flight table.

## Main App Files
- `streamlit_app_live_airlabs.py`: Live demo app using AirLabs schedules.
- `streamlit_app_future_dates.py`: Future-date schedules demo (Aviationstack).
- `streamlit_app.py` / `streamlit_app1.py`: Earlier live prototypes.

## Requirements
- Python 3.10+ (Streamlit Cloud recommended: 3.11)
- `pip install -r requirements.txt`

## Environment Variables
Create a `.env` file locally (never commit it):

```
HOPSWORKS_PROJECT=Flight_Predictor_JFK
HOPSWORKS_API_KEY=YOUR_HOPSWORKS_KEY
AIRLABS_API_KEY=YOUR_AIRLABS_KEY
AIRLABS_BASE_URL=https://airlabs.co/api/v9
```

Optional (used by other files):
```
AVIATIONSTACK_API_KEY=YOUR_AVIATIONSTACK_KEY
AVIATIONSTACK_BASE_URL=https://api.aviationstack.com/v1
```

## Run Locally
```
streamlit run streamlit_app_live_airlabs.py
```

## Streamlit Cloud Deployment
1) Push the repo to GitHub.
2) In Streamlit Cloud, set the secrets in TOML format:

```
HOPSWORKS_PROJECT = "Flight_Predictor_JFK"
HOPSWORKS_API_KEY = "YOUR_HOPSWORKS_KEY"
AIRLABS_API_KEY = "YOUR_AIRLABS_KEY"
AIRLABS_BASE_URL = "https://airlabs.co/api/v9"
```

3) If needed, add `runtime.txt` with:
```
python-3.11
```

## Outputs Shown in the App
- Flights count, predicted delayed rate, and average predicted delay.
- Flight table with scheduled time, airline, destination, delay probability, and delay minutes.
- A small weather summary box to explain possible delay drivers.

## Notes
- AirLabs schedules can be stale; the app falls back to show all schedules if no future flights pass the time filter.
- Weather coverage can be missing for some hours; the model will impute missing values.
