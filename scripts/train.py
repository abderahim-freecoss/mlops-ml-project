import json
import sys
from pathlib import Path

# Permet d'importer src quand le script est lancé depuis PowerShell.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import yaml
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from src.data import load_dataset
from src.features import build_numeric_preprocess
from src.model import build_model


def load_cfg(path="config/train.yaml"):
    with open(path, "r", encoding="utf-8-sig") as config_file:
        return yaml.safe_load(config_file)


def save_confusion_matrix(y_true, y_pred, out_path):
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.imshow(matrix)
    ax.set_title("Matrice de confusion")
    ax.set_xlabel("Classe prédite")
    ax.set_ylabel("Classe réelle")

    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center")

    fig.tight_layout()
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def main():
    cfg = load_cfg()
    artifacts_dir = Path(cfg.get("artifacts_dir", "artifacts"))
    artifacts_dir.mkdir(exist_ok=True)

    X, y = load_dataset(cfg)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=float(cfg["split"]["test_size"]),
        random_state=int(cfg["split"]["random_state"]),
        stratify=y,
    )

    pipeline = Pipeline(steps=[
        ("preprocess", build_numeric_preprocess()),
        ("model", build_model(cfg)),
    ])
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro")),
    }

    joblib.dump(pipeline, artifacts_dir / "model.joblib")
    (artifacts_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    save_confusion_matrix(y_test, predictions, artifacts_dir / "confusion_matrix.png")

    print("Train OK:", metrics)


if __name__ == "__main__":
    main()