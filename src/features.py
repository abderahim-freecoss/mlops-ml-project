from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler


def _clip(X):
    """Borne les valeurs standardisées entre -3 et 3."""
    return X.clip(-3, 3)


def build_numeric_preprocess():
    """Applique l'imputation, la standardisation et le clipping."""
    return Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("clip", FunctionTransformer(_clip)),
    ])
