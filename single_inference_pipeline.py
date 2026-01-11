# ============================================================
# Two-stage inference wrapper (NEW NOTEBOOK)
# Loads Stage 1 + Stage 2 + metadata from Hopsworks Model Registry
# Then predicts delay minutes with gating:
#   p = stage1.predict_proba(X)[:,1]
#   if p >= threshold -> stage2.predict(X) else 0
# ============================================================

import os
import json
import glob
import joblib
import numpy as np
import pandas as pd
import hopsworks
from dotenv import load_dotenv

load_dotenv()


class TwoStageDelayPredictor:
    """
    Two-stage predictor for JFK delay minutes:
      - Stage 1: classifier pipeline (must support predict_proba)
      - Stage 2: regressor pipeline
      - Metadata: threshold + feature_cols (optional but recommended)
    """

    def __init__(self, stage1_model, stage2_model, threshold: float, feature_cols=None):
        self.stage1_model = stage1_model
        self.stage2_model = stage2_model
        self.threshold = float(threshold)
        self.feature_cols = feature_cols  # list or None

    # ----------------------------
    # Internal helpers
    # ----------------------------
    @staticmethod
    def _download_model_artifact(model_obj, local_dir: str) -> str:
        """
        Downloads the model artifacts to local_dir and returns the local directory path.
        """
        os.makedirs(local_dir, exist_ok=True)
        # Hopsworks Model object supports .download()
        # It returns the download path (folder).
        path = model_obj.download(local_dir)
        return path

    @staticmethod
    def _find_first_pkl(folder: str) -> str:
        """
        Finds a .pkl (or .joblib) inside folder (recursively) and returns its path.
        """
        candidates = glob.glob(os.path.join(folder, "**", "*.pkl"), recursive=True)
        if not candidates:
            candidates = glob.glob(os.path.join(folder, "**", "*.joblib"), recursive=True)
        if not candidates:
            raise FileNotFoundError(f"No .pkl/.joblib file found under: {folder}")
        # pick the most recently modified as a safe heuristic
        candidates.sort(key=lambda p: os.path.getmtime(p), reverse=True)
        return candidates[0]

    @staticmethod
    def _load_metadata_from_folder(folder: str):
        """
        Loads metadata JSON if present. Returns dict or None.
        """
        candidates = glob.glob(os.path.join(folder, "**", "*.json"), recursive=True)
        if not candidates:
            return None
        # prefer files that look like our metadata
        preferred = [p for p in candidates if "metadata" in os.path.basename(p).lower()]
        path = preferred[0] if preferred else candidates[0]
        with open(path, "r") as f:
            return json.load(f)

    @classmethod
    def load_from_hopsworks(
        cls,
        project,
        stage1_name: str,
        stage2_name: str,
        metadata_name: str = None,
        stage1_version: int = None,
        stage2_version: int = None,
        metadata_version: int = None,
        fallback_threshold: float = None,
        fallback_feature_cols=None,
        download_dir: str = "hs_models_download",
    ):
        """
        Loads models + metadata from Hopsworks Model Registry.

        - If version is None, loads the latest version.
        - If metadata is missing, uses fallback_threshold / fallback_feature_cols.
        """
        mr = project.get_model_registry()

        # ---- Stage 1 ----
        stage1_obj = mr.get_model(stage1_name, version=stage1_version) if stage1_version else mr.get_model(stage1_name)
        stage1_path = cls._download_model_artifact(stage1_obj, os.path.join(download_dir, "stage1"))
        stage1_file = cls._find_first_pkl(stage1_path)
        stage1_model = joblib.load(stage1_file)

        # ---- Stage 2 ----
        stage2_obj = mr.get_model(stage2_name, version=stage2_version) if stage2_version else mr.get_model(stage2_name)
        stage2_path = cls._download_model_artifact(stage2_obj, os.path.join(download_dir, "stage2"))
        stage2_file = cls._find_first_pkl(stage2_path)
        stage2_model = joblib.load(stage2_file)

        # ---- Metadata (optional) ----
        meta = None
        if metadata_name:
            meta_obj = mr.get_model(metadata_name, version=metadata_version) if metadata_version else mr.get_model(metadata_name)
            meta_path = cls._download_model_artifact(meta_obj, os.path.join(download_dir, "metadata"))
            meta = cls._load_metadata_from_folder(meta_path)

        # Resolve threshold + feature_cols
        threshold = None
        feature_cols = None

        if meta:
            # Common keys we used earlier
            if "decision_rule" in meta and "best_threshold" in meta["decision_rule"]:
                threshold = meta["decision_rule"]["best_threshold"]
            elif "best_threshold" in meta:
                threshold = meta["best_threshold"]

            if "feature_cols" in meta:
                feature_cols = meta["feature_cols"]

        if threshold is None:
            if fallback_threshold is None:
                raise ValueError(
                    "Threshold not found in metadata and no fallback_threshold provided. "
                    "Provide metadata_name or pass fallback_threshold."
                )
            threshold = fallback_threshold

        if feature_cols is None:
            feature_cols = fallback_feature_cols  # may be None; user can pass df already filtered

        print("✅ Loaded TwoStageDelayPredictor")
        print("   Stage 1 file:", stage1_file)
        print("   Stage 2 file:", stage2_file)
        print("   Threshold:", threshold)
        print("   Feature cols:", "from metadata" if meta and "feature_cols" in (meta or {}) else ("provided" if feature_cols else "not set"))

        return cls(stage1_model=stage1_model, stage2_model=stage2_model, threshold=threshold, feature_cols=feature_cols)

    # ----------------------------
    # Prediction API
    # ----------------------------
    def _prepare_X(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.feature_cols is not None:
            missing = [c for c in self.feature_cols if c not in df.columns]
            if missing:
                raise ValueError(f"Input df is missing required feature columns: {missing}")
            X = df[self.feature_cols].copy()
        else:
            # Assume df already contains exactly the features expected by the pipelines
            X = df.copy()

        # Ensure categoricals are strings for OneHotEncoder stability
        for col in ["reporting_airline", "dest"]:
            if col in X.columns:
                X[col] = X[col].astype(str)

        return X

    def predict_proba_delayed(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns P(delay) from Stage 1.
        """
        X = self._prepare_X(df)
        proba = self.stage1_model.predict_proba(X)[:, 1]
        return proba

    def predict_delay_minutes(self, df: pd.DataFrame) -> np.ndarray:
        """
        Returns final predicted delay minutes using gating.
        """
        X = self._prepare_X(df)
        proba = self.stage1_model.predict_proba(X)[:, 1]

        yhat = np.zeros(len(X), dtype=float)
        idx = proba >= self.threshold
        if np.any(idx):
            yhat[idx] = self.stage2_model.predict(X.loc[idx])

        # Safety: no negative delays
        yhat = np.clip(yhat, 0, None)
        return yhat

    def predict_dataframe(self, df: pd.DataFrame, include_proba: bool = True) -> pd.DataFrame:
        """
        Returns df + prediction columns (and optionally probability).
        """
        out = df.copy()
        proba = self.predict_proba_delayed(df)
        pred = self.predict_delay_minutes(df)

        if include_proba:
            out["p_delayed"] = proba
        out["pred_delay_min"] = pred
        out["pred_is_delayed"] = (proba >= self.threshold).astype(int)

        return out


# ============================================================
# Example usage in your NEW notebook
# ============================================================

# 1) Login (replace with your values or environment variables)
PROJECT_NAME = os.getenv("HOPSWORKS_PROJECT", "Flight_Predictor_JFK")
HOPSWORKS_API_KEY = os.getenv("HOPSWORKS_API_KEY")
project = hopsworks.login(project=PROJECT_NAME, api_key_value=HOPSWORKS_API_KEY)

# 2) Load from Model Registry
# Use your actual registered names here:
STAGE1_NAME = "jfk_delay_stage1_classifier"
STAGE2_NAME = "jfk_delay_stage2_regressor"
META_NAME   = "jfk_delay_two_stage_metadata"  # set to None if you didn't register metadata

# If you didn't register metadata, set fallback_threshold and fallback_feature_cols
# fallback_threshold = 0.80
# fallback_feature_cols = [...]  # exact list used in training

predictor = TwoStageDelayPredictor.load_from_hopsworks(
    project=project,
    stage1_name=STAGE1_NAME,
    stage2_name=STAGE2_NAME,
    metadata_name=META_NAME,           # set None if not available
    stage1_version=None,               # None = latest
    stage2_version=None,               # None = latest
    metadata_version=None,             # None = latest
    fallback_threshold=0.80,           # used only if metadata missing
    fallback_feature_cols=None,        # optional: set if metadata missing
    download_dir="hs_models_download"
)

# 3) Predict on any dataframe that has the required feature columns
# df_live_features = ...
# preds_df = predictor.predict_dataframe(df_live_features, include_proba=True)
# preds_df.head()

print("Stage 1 type:", type(predictor.stage1_model))
print("Stage 2 type:", type(predictor.stage2_model))
print("Threshold:", predictor.threshold)
print("Feature cols set?:", predictor.feature_cols is not None)
if predictor.feature_cols is not None:
    print("Num feature cols:", len(predictor.feature_cols))
    print("First 10 feature cols:", predictor.feature_cols[:10])





assert hasattr(predictor.stage1_model, "predict_proba"), "Stage 1 model has no predict_proba()"
print("✅ Stage 1 supports predict_proba()")





fs = project.get_feature_store()
fv = fs.get_feature_view("jfk_delay_weather_fv", version=1)

df_batch = fv.get_batch_data()
print("FV batch shape:", df_batch.shape)
print("FV columns (first 20):", df_batch.columns[:20].tolist())

# Build the exact input dataframe for the wrapper
# If you saved feature_cols in metadata, we’ll use them automatically.
# Otherwise, select the known training features explicitly:

if predictor.feature_cols is None:
    feature_cols = [
        "month",
        "day_of_week",
        "reporting_airline",
        "dest",
        "distance",
        "weather_jfk_hourly_fg_weather_code",
        "weather_jfk_hourly_fg_wind_speed_ms",
        "weather_jfk_hourly_fg_wind_gust_ms",
        "weather_jfk_hourly_fg_temp_c",
        "weather_jfk_hourly_fg_precip_mm",
        "weather_jfk_hourly_fg_snowfall_cm",
        # do NOT include visibility (all missing)
    ]
else:
    feature_cols = predictor.feature_cols

X_small = df_batch[feature_cols].head(20).copy()
preds = predictor.predict_dataframe(X_small, include_proba=True)

preds[["p_delayed", "pred_is_delayed", "pred_delay_min"]].head(20)



X_small = df_batch[predictor.feature_cols].head(10).copy()
preds_df = predictor.predict_dataframe(X_small, include_proba=True)

print(preds_df[["p_delayed", "pred_is_delayed", "pred_delay_min"]])
print("Pred delay min summary:", preds_df["pred_delay_min"].describe())



X = df_batch[predictor.feature_cols].copy()
proba = predictor.stage1_model.predict_proba(X)[:, 1]

top_idx = np.argsort(proba)[-10:]  # top 10 highest probabilities
X_top = X.iloc[top_idx]

preds_top = predictor.predict_dataframe(X_top, include_proba=True)
print(preds_top[["p_delayed", "pred_is_delayed", "pred_delay_min"]])
