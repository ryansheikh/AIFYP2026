"""
=============================================================================
 STEP 1: Download Historical Weather Data — Pakistan Coastal Stations
 Source: Open-Meteo Historical Weather API (FREE, no API key)
=============================================================================
 Run:    python 01_download_data.py
 Output: data/raw/coastal_weather_raw.csv
"""

import requests
import pandas as pd
import os
import time

# ── Pakistan Coastal Stations ────────────────────────────────────────────
STATIONS = {
    "Karachi":  {"lat": 24.86, "lon": 67.01},
    "Thatta":   {"lat": 24.75, "lon": 67.92},
    "Badin":    {"lat": 24.63, "lon": 68.84},
    "Ormara":   {"lat": 25.21, "lon": 64.64},
    "Pasni":    {"lat": 25.26, "lon": 63.47},
    "Gwadar":   {"lat": 25.12, "lon": 62.33},
    "Jiwani":   {"lat": 25.05, "lon": 61.80},
    "Turbat":   {"lat": 26.00, "lon": 63.05},
}

BASE_URL   = "https://archive-api.open-meteo.com/v1/archive"
START_DATE = "2000-01-01"
END_DATE   = "2024-12-31"

# All daily variables needed across the 4 separate pipelines
DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
    "relative_humidity_2m_max", "relative_humidity_2m_min",
    "dewpoint_2m_mean", "pressure_msl_mean",
]


def fetch_station(name, lat, lon):
    params = {
        "latitude": lat, "longitude": lon,
        "start_date": START_DATE, "end_date": END_DATE,
        "daily": ",".join(DAILY_VARS),
        "timezone": "Asia/Karachi",
    }
    print(f"  ⬇  {name} ({lat}, {lon}) …")
    r = requests.get(BASE_URL, params=params, timeout=120)
    r.raise_for_status()
    d = r.json()["daily"]
    df = pd.DataFrame(d)
    df.rename(columns={"time": "date"}, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df["station"] = name
    df["latitude"], df["longitude"] = lat, lon
    print(f"     ✓ {len(df)} days")
    return df


def main():
    os.makedirs("data/raw", exist_ok=True)
    print("=" * 60)
    print(f" Downloading {len(STATIONS)} coastal stations  ({START_DATE} → {END_DATE})")
    print("=" * 60)

    frames = []
    for name, c in STATIONS.items():
        frames.append(fetch_station(name, c["lat"], c["lon"]))
        time.sleep(1)

    out = pd.concat(frames, ignore_index=True)
    out.to_csv("data/raw/coastal_weather_raw.csv", index=False)
    print(f"\n ✓ Saved {len(out):,} rows → data/raw/coastal_weather_raw.csv")


if __name__ == "__main__":
    main()
