"""
=============================================================================
 RUN THIS ON YOUR LAPTOP (one time only)
 Downloads 25 years of weather data → data.csv
 Then push data.csv to GitHub alongside app.py
=============================================================================
 Usage:  python download_data.py
 Output: data.csv  (~15 MB)
"""

import requests
import pandas as pd
import time

STATIONS = {
    "Karachi": (24.86, 67.01),
    "Thatta":  (24.75, 67.92),
    "Badin":   (24.63, 68.84),
    "Ormara":  (25.21, 64.64),
    "Pasni":   (25.26, 63.47),
    "Gwadar":  (25.12, 62.33),
    "Jiwani":  (25.05, 61.80),
    "Turbat":  (26.00, 63.05),
}

BASE_URL = "https://archive-api.open-meteo.com/v1/archive"

DAILY_VARS = [
    "temperature_2m_max", "temperature_2m_min", "temperature_2m_mean",
    "apparent_temperature_max", "apparent_temperature_min",
    "precipitation_sum", "rain_sum",
    "windspeed_10m_max", "windgusts_10m_max", "winddirection_10m_dominant",
    "shortwave_radiation_sum", "et0_fao_evapotranspiration",
]

# Download in 5-year chunks to stay within API limits
CHUNKS = [
    ("2000-01-01", "2004-12-31"),
    ("2005-01-01", "2009-12-31"),
    ("2010-01-01", "2014-12-31"),
    ("2015-01-01", "2019-12-31"),
    ("2020-01-01", "2024-12-31"),
]


def main():
    all_frames = []

    for name, (lat, lon) in STATIONS.items():
        print(f"\n{'='*50}")
        print(f"  Station: {name} ({lat}, {lon})")
        print(f"{'='*50}")

        for start, end in CHUNKS:
            success = False
            for attempt in range(5):  # up to 5 retries
                try:
                    print(f"  Downloading {start} → {end} (attempt {attempt+1}) ...", end=" ")
                    r = requests.get(BASE_URL, params={
                        "latitude": lat,
                        "longitude": lon,
                        "start_date": start,
                        "end_date": end,
                        "daily": ",".join(DAILY_VARS),
                        "timezone": "Asia/Karachi",
                    }, timeout=180)
                    r.raise_for_status()

                    d = pd.DataFrame(r.json()["daily"])
                    d.rename(columns={"time": "date"}, inplace=True)
                    d["date"] = pd.to_datetime(d["date"])
                    d["station"] = name
                    d["latitude"] = lat
                    d["longitude"] = lon
                    all_frames.append(d)
                    print(f"✓ {len(d)} days")
                    success = True
                    break

                except requests.exceptions.HTTPError as e:
                    if "429" in str(e):
                        wait = 15 * (attempt + 1)  # 15s, 30s, 45s, 60s, 75s
                        print(f"Rate limited. Waiting {wait}s ...")
                        time.sleep(wait)
                    else:
                        print(f"Error: {e}")
                        break
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(5)

            if not success:
                print(f"  ⚠️  FAILED: {name} ({start}–{end})")

            # Wait between each chunk to avoid rate limits
            time.sleep(5)

        # Wait between stations
        print(f"  Done with {name}. Waiting 10s before next station...")
        time.sleep(10)

    # Save
    if all_frames:
        result = pd.concat(all_frames, ignore_index=True)
        result.to_csv("data.csv", index=False)
        print(f"\n{'='*50}")
        print(f"  ✓ SAVED: data.csv")
        print(f"  Rows: {len(result):,}")
        print(f"  Stations: {result['station'].nunique()}")
        print(f"  Date range: {result['date'].min().date()} → {result['date'].max().date()}")
        print(f"  File size: {result.memory_usage(deep=True).sum()/1e6:.1f} MB")
        print(f"{'='*50}")
        print(f"\n  Next step: push data.csv + app.py + requirements.txt to GitHub")
    else:
        print("  ❌ No data downloaded!")


if __name__ == "__main__":
    main()
