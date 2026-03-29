"""
=============================================================================
 STEP 2: Shared Preprocessing & Feature Engineering
=============================================================================
 This module is imported by each of the 4 separate pipelines.
 It provides: loading, cleaning, feature engineering, and label creation.
"""

import pandas as pd
import numpy as np

# ══════════════════════════════════════════════════════════════════════════
#  LOAD & CLEAN
# ══════════════════════════════════════════════════════════════════════════

def load_raw(path="data/raw/coastal_weather_raw.csv"):
    df = pd.read_csv(path, parse_dates=["date"])
    df.sort_values(["station", "date"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


def clean(df):
    """Interpolate small gaps, drop remaining NaN on critical cols."""
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    num_cols = [c for c in num_cols if c not in ("latitude", "longitude")]

    for col in num_cols:
        df[col] = df.groupby("station")[col].transform(
            lambda s: s.interpolate(method="linear", limit=7).ffill().bfill()
        )

    df.dropna(subset=["temperature_2m_mean", "precipitation_sum"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ══════════════════════════════════════════════════════════════════════════
#  FEATURE ENGINEERING
# ══════════════════════════════════════════════════════════════════════════

def add_temporal_features(df):
    """Cyclical + calendar features."""
    df["year"]      = df["date"].dt.year
    df["month"]     = df["date"].dt.month
    df["day"]       = df["date"].dt.day
    df["dayofyear"] = df["date"].dt.dayofyear
    df["weekofyear"] = df["date"].dt.isocalendar().week.astype(int)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["doy_sin"]   = np.sin(2 * np.pi * df["dayofyear"] / 365)
    df["doy_cos"]   = np.cos(2 * np.pi * df["dayofyear"] / 365)
    return df


def add_lag_features(df, cols, lags=(1, 3, 7, 14, 30)):
    """Per-station lag features."""
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = df.groupby("station")[col].shift(lag)
    return df


def add_rolling_features(df, cols, windows=(7, 14, 30)):
    """Per-station rolling mean & std."""
    for col in cols:
        if col not in df.columns:
            continue
        for w in windows:
            grp = df.groupby("station")[col]
            df[f"{col}_roll{w}_mean"] = grp.transform(
                lambda s: s.rolling(w, min_periods=1).mean()
            )
            df[f"{col}_roll{w}_std"] = grp.transform(
                lambda s: s.rolling(w, min_periods=1).std()
            )
    return df


def add_derived(df):
    """Extra engineered columns."""
    if {"temperature_2m_max", "temperature_2m_min"}.issubset(df.columns):
        df["temp_range"] = df["temperature_2m_max"] - df["temperature_2m_min"]
    if {"relative_humidity_2m_max", "relative_humidity_2m_min"}.issubset(df.columns):
        df["humidity_mean"] = (df["relative_humidity_2m_max"] + df["relative_humidity_2m_min"]) / 2
    return df


# ══════════════════════════════════════════════════════════════════════════
#  EXTREME WEATHER LABELS (Meteorological rule-based)
# ══════════════════════════════════════════════════════════════════════════

def add_labels(df):
    """
    Create binary labels for each extreme weather type.
    These are used as classification targets in separate pipelines.
    """
    # Heatwave: max temp ≥ 42°C  (Pakistan coastal threshold)
    df["label_heatwave"] = (df["temperature_2m_max"] >= 42).astype(int)

    # Heavy Rainfall: precipitation > 30 mm/day  (PMD heavy rain threshold)
    df["label_heavy_rain"] = (df["precipitation_sum"] >= 30).astype(int)

    # Storm Risk: wind gusts ≥ 60 km/h  (tropical storm level for coastal areas)
    if "windgusts_10m_max" in df.columns:
        df["label_storm"] = (df["windgusts_10m_max"] >= 60).astype(int)
    else:
        df["label_storm"] = (df["windspeed_10m_max"] >= 40).astype(int)

    return df


# ══════════════════════════════════════════════════════════════════════════
#  TIME-SERIES TRAIN / VAL / TEST SPLIT
# ══════════════════════════════════════════════════════════════════════════

def time_split(df, train_end=2019, val_end=2021):
    """
    Train: 2000–2019,  Val: 2020–2021,  Test: 2022–2024
    Simulates real-world forecasting — no data leakage.
    """
    train = df[df["date"].dt.year <= train_end].copy()
    val   = df[(df["date"].dt.year > train_end) & (df["date"].dt.year <= val_end)].copy()
    test  = df[df["date"].dt.year > val_end].copy()
    return train, val, test


# ══════════════════════════════════════════════════════════════════════════
#  FULL PREPROCESSING PIPELINE
# ══════════════════════════════════════════════════════════════════════════

def full_preprocess(raw_path="data/raw/coastal_weather_raw.csv"):
    """Run the complete preprocessing pipeline, return clean DataFrame."""
    df = load_raw(raw_path)
    df = clean(df)
    df = add_temporal_features(df)
    df = add_lag_features(df, [
        "temperature_2m_mean", "temperature_2m_max",
        "precipitation_sum", "windspeed_10m_max",
        "windgusts_10m_max", "pressure_msl_mean",
    ])
    df = add_rolling_features(df, [
        "temperature_2m_mean", "precipitation_sum",
        "windspeed_10m_max",
    ])
    df = add_derived(df)
    df = add_labels(df)
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df
