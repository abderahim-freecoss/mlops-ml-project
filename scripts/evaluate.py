import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
from sklearn.metrics import classification_report

from scripts.train import load_cfg
from src.data import load_dataset


def main():
    cfg = load_cfg()
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))

    model = joblib.load(artifacts_dir / "model.joblib")
    X, y = load_dataset(cfg)
    predictions = model.predict(X)

    report = classification_report(y, predictions, output_dict=True)
    report_path = artifacts_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")

    print(f"Evaluate OK: {report_path}")


if __name__ == "__main__":
    main()