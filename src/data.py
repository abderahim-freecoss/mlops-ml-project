import pandas as pd
from sklearn.datasets import load_iris


def load_dataset(cfg: dict):
    """Charge le jeu Iris ou lit un fichier CSV selon la configuration."""
    kind = cfg["data"].get("kind", "iris")
    if kind == "iris":
        return load_iris(return_X_y=True, as_frame=True)

    df = pd.read_csv(cfg["data"]["path"])
    target = cfg["data"]["target"]
    X = df.drop(columns=[target])
    y = df[target]
    return X, y