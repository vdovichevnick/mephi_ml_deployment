import os
import json
import logging
import pickle
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    classification_report, f1_score, precision_score,
    recall_score, roc_auc_score, confusion_matrix
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/Users/nikitavdovichev/Documents/5 курс/ВнедрениеМоделейМЛ/data/UCI_Credit_Card.csv",
)
MODELS_DIR = os.path.join(os.path.dirname(__file__))
RANDOM_STATE = 42
TEST_SIZE = 0.2

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default.payment.next.month"


def load_data(path: str) -> pd.DataFrame:
    logger.info(f"Загрузка данных из {path}")
    df = pd.read_csv(path)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df[df[TARGET_COL] != TARGET_COL].reset_index(drop=True)
    df = df.astype(float)
    logger.info(f"Загружено строк: {len(df)}, признаков: {len(FEATURE_COLS)}")
    return df


def preprocess(df: pd.DataFrame):
    X = df[FEATURE_COLS].values
    y = df[TARGET_COL].values.astype(int)
    return X, y


def evaluate(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    metrics = {
        "model": model_name,
        "f1": round(f1_score(y_test, y_pred), 4),
        "precision": round(precision_score(y_test, y_pred), 4),
        "recall": round(recall_score(y_test, y_pred), 4),
        "roc_auc": round(roc_auc_score(y_test, y_proba), 4),
    }
    logger.info(f"[{model_name}] Метрики: {metrics}")
    return metrics


def save_model(model, path: str):
    with open(path, "wb") as f:
        pickle.dump(model, f)
    logger.info(f"Модель сохранена: {path}")


def save_metrics(metrics: dict, path: str):
    with open(path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Метрики сохранены: {path}")


def train_v1(X_train, y_train) -> Pipeline:
    """v1 — LogisticRegression с StandardScaler."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(
            C=1.0, max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def train_v2(X_train, y_train) -> Pipeline:
    """v2 — GradientBoostingClassifier."""
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200, max_depth=4,
            learning_rate=0.05, subsample=0.8,
            random_state=RANDOM_STATE,
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def main():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Датасет не найден по пути: {DATA_PATH}\n"
            "Укажите путь через переменную окружения DATA_PATH=<путь>"
        )

    df = load_data(DATA_PATH)
    X, y = preprocess(df)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")
    logger.info(f"Доля дефолтов в train: {y_train.mean():.3f}")

    logger.info("Обучение модели v1 (LogisticRegression)...")
    model_v1 = train_v1(X_train, y_train)
    metrics_v1 = evaluate(model_v1, X_test, y_test, "v1_LogisticRegression")
    save_model(model_v1, os.path.join(MODELS_DIR, "model_v1.pkl"))
    save_metrics(metrics_v1, os.path.join(MODELS_DIR, "metrics_v1.json"))

    logger.info("Обучение модели v2 (GradientBoosting)...")
    model_v2 = train_v2(X_train, y_train)
    metrics_v2 = evaluate(model_v2, X_test, y_test, "v2_GradientBoosting")
    save_model(model_v2, os.path.join(MODELS_DIR, "model_v2.pkl"))
    save_metrics(metrics_v2, os.path.join(MODELS_DIR, "metrics_v2.json"))

    feature_meta = {"feature_cols": FEATURE_COLS, "target_col": TARGET_COL}
    with open(os.path.join(MODELS_DIR, "feature_meta.json"), "w") as f:
        json.dump(feature_meta, f, indent=2)

    logger.info("Обучение завершено. Файлы сохранены в models/")


if __name__ == "__main__":
    main()