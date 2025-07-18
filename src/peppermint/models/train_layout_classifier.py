import pandas as pd, joblib
from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils import compute_sample_weight

def train(feature_path: Path, model_out: Path):
    df = pd.read_parquet(feature_path)
    X = df.drop(columns=["label"])
    y = df["label"]
    w = compute_sample_weight("balanced", y)

    num = X.select_dtypes("number").columns
    pre = ColumnTransformer([("num", StandardScaler(), num)])
    gb  = HistGradientBoostingClassifier(
              learning_rate=0.05, max_depth=3,
              max_iter=800, early_stopping=True, random_state=42)
    from sklearn.pipeline import Pipeline
    pipe = Pipeline([("prep", pre), ("gb", gb)])
    Xtr, Xva, ytr, yva, wtr, wva = train_test_split(
        X, y, w, test_size=0.2, stratify=y, random_state=42)
    pipe.fit(Xtr, ytr, gb__sample_weight=wtr)

    joblib.dump(pipe, model_out)
