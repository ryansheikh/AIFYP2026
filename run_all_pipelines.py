"""
=============================================================================
 MASTER RUNNER — Executes all 4 pipelines sequentially
=============================================================================
 Run: python run_all_pipelines.py
"""

import subprocess
import sys
import time

PIPELINES = [
    ("Pipeline 1 — Temperature Forecasting", "pipeline_1_temperature.py"),
    ("Pipeline 2 — Heatwave Detection",      "pipeline_2_heatwave.py"),
    ("Pipeline 3 — Heavy Rainfall",           "pipeline_3_rainfall.py"),
    ("Pipeline 4 — Storm Risk",               "pipeline_4_storm.py"),
]


def main():
    print("\n" + "█" * 65)
    print("  RUNNING ALL 4 SEPARATE PIPELINES")
    print("  Each pipeline: Train → Evaluate → Uncertainty → SHAP")
    print("█" * 65)

    start = time.time()

    for name, script in PIPELINES:
        print(f"\n\n{'▓' * 65}")
        print(f"  STARTING: {name}")
        print(f"{'▓' * 65}")
        result = subprocess.run([sys.executable, script], check=True)

    elapsed = time.time() - start
    print(f"\n\n{'█' * 65}")
    print(f"  ✓ ALL 4 PIPELINES COMPLETE  ({elapsed/60:.1f} minutes)")
    print(f"{'█' * 65}")
    print(f"""
  Output Structure:
  models/
  ├── temperature/     (model, quantile models, SHAP, uncertainty)
  ├── heatwave/        (model, calibrated, SHAP, reliability diagram)
  ├── rainfall/        (model, calibrated, SHAP, reliability diagram)
  └── storm/           (model, calibrated, SHAP, reliability diagram)

  Next: streamlit run dashboard.py
""")


if __name__ == "__main__":
    main()
