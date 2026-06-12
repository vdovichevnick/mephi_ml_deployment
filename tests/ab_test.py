import json
import os
import sys
import urllib.request
import urllib.error

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from statsmodels.stats.proportion import proportions_ztest

# Конфигурация
API_PORT = os.environ.get("API_PORT", "5010")
BASE_URL = f"http://localhost:{API_PORT}"

DATA_PATH = os.environ.get(
    "DATA_PATH",
    "/Users/nikitavdovichev/Documents/5 курс/ВнедрениеМоделейМЛ/data/UCI_Credit_Card.csv",
)

FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]
TARGET_COL = "default.payment.next.month"

# Сколько строк отправить на API (None = весь тест-сет)
SAMPLE_SIZE = int(os.environ.get("SAMPLE_SIZE", "1000"))
RANDOM_STATE = 42


# Загрузка данных
def load_test_data():
    if not os.path.exists(DATA_PATH):
        print(f"[ERROR] Датасет не найден: {DATA_PATH}")
        print("Укажите путь через переменную окружения DATA_PATH=<путь>")
        sys.exit(1)

    df = pd.read_csv(DATA_PATH)
    if "ID" in df.columns:
        df = df.drop(columns=["ID"])
    df = df[df[TARGET_COL] != TARGET_COL].reset_index(drop=True)
    df = df.astype(float)

    _, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE,
                                  stratify=df[TARGET_COL].astype(int))

    if SAMPLE_SIZE and SAMPLE_SIZE < len(df_test):
        df_test = df_test.sample(n=SAMPLE_SIZE, random_state=RANDOM_STATE)

    print(f"Тестовая выборка: {len(df_test)} строк "
          f"(дефолтов: {int(df_test[TARGET_COL].sum())}, "
          f"{df_test[TARGET_COL].mean():.1%})")
    return df_test


# Запрос к API
def predict_via_api(row: dict, version: str) -> int:
    payload = json.dumps({col: row[col] for col in FEATURE_COLS}).encode()
    req = urllib.request.Request(
        f"{BASE_URL}/predict?version={version}",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=5) as resp:
        return json.loads(resp.read())["prediction"]


def collect_predictions(df: pd.DataFrame, version: str) -> np.ndarray:
    preds = []
    total = len(df)
    for i, (_, row) in enumerate(df.iterrows()):
        pred = predict_via_api(row.to_dict(), version)
        preds.append(pred)
        if (i + 1) % 50 == 0 or (i + 1) == total:
            print(f"  [{version}] {i + 1}/{total} запросов отправлено...")
    return np.array(preds)


# Bootstrap CI для F1
def bootstrap_f1_ci(y_true: np.ndarray, y_pred: np.ndarray,
                    n_iter: int = 1000, ci: float = 0.95) -> tuple:
    scores = []
    n = len(y_true)
    rng = np.random.default_rng(RANDOM_STATE)
    for _ in range(n_iter):
        idx = rng.integers(0, n, size=n)
        if y_true[idx].sum() == 0:
            continue
        scores.append(f1_score(y_true[idx], y_pred[idx], zero_division=0))
    alpha = (1 - ci) / 2
    return (round(float(np.percentile(scores, alpha * 100)), 4),
            round(float(np.percentile(scores, (1 - alpha) * 100)), 4))


# Z-тест для двух пропорций
def z_test(y_true: np.ndarray,
           pred_v1: np.ndarray, pred_v2: np.ndarray) -> tuple:

    # Считаем "правильные" предсказания дефолта (TP) как прокси для F1
    tp_v1 = int(((pred_v1 == 1) & (y_true == 1)).sum())
    tp_v2 = int(((pred_v2 == 1) & (y_true == 1)).sum())
    n_positives = int(y_true.sum())

    stat, pvalue = proportions_ztest(
        count=[tp_v1, tp_v2],
        nobs=[n_positives, n_positives],
        alternative="smaller",  # H1: v2 обнаруживает больше дефолтов
    )
    return round(float(stat), 4), round(float(pvalue), 4)


# Бизнес-метрики 
def expected_loss_reduction(y_true: np.ndarray,
                             pred_v1: np.ndarray, pred_v2: np.ndarray,
                             avg_debt: float = 30000, lgd: float = 0.45) -> float:
    fn_v1 = int(((pred_v1 == 0) & (y_true == 1)).sum())
    fn_v2 = int(((pred_v2 == 0) & (y_true == 1)).sum())
    return round((fn_v1 - fn_v2) * avg_debt * lgd, 2)


def approval_rate(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    approved = int(((y_pred == 0)).sum())
    return round(approved / len(y_pred), 4)


# Вывод результатов
def print_results(y_true, pred_v1, pred_v2):
    f1_v1 = round(f1_score(y_true, pred_v1, zero_division=0), 4)
    f1_v2 = round(f1_score(y_true, pred_v2, zero_division=0), 4)
    pr_v1 = round(precision_score(y_true, pred_v1, zero_division=0), 4)
    pr_v2 = round(precision_score(y_true, pred_v2, zero_division=0), 4)
    rc_v1 = round(recall_score(y_true, pred_v1, zero_division=0), 4)
    rc_v2 = round(recall_score(y_true, pred_v2, zero_division=0), 4)

    ci_v1 = bootstrap_f1_ci(y_true, pred_v1)
    ci_v2 = bootstrap_f1_ci(y_true, pred_v2)

    stat, pvalue = z_test(y_true, pred_v1, pred_v2)

    elr = expected_loss_reduction(y_true, pred_v1, pred_v2)
    ar_v1 = approval_rate(y_true, pred_v1)
    ar_v2 = approval_rate(y_true, pred_v2)

    sep = "─" * 52

    print(f"\n{sep}")
    print("  РЕЗУЛЬТАТЫ A/B-ТЕСТА")
    print(sep)
    print(f"  {'Метрика':<28} {'v1 (LR)':>8}  {'v2 (GBM)':>8}")
    print(sep)
    print(f"  {'F1-score':<28} {f1_v1:>8.4f}  {f1_v2:>8.4f}")
    print(f"  {'Precision':<28} {pr_v1:>8.4f}  {pr_v2:>8.4f}")
    print(f"  {'Recall':<28} {rc_v1:>8.4f}  {rc_v2:>8.4f}")
    print(f"  {'95% Bootstrap CI (F1)':<28} {str(ci_v1):>8}  {str(ci_v2):>8}")
    print(f"  {'Approval Rate':<28} {ar_v1:>8.4f}  {ar_v2:>8.4f}")
    print(sep)
    print(f"  Z-тест: stat={stat}, p-value={pvalue}")
    print(f"  Δ F1 (v2 - v1): {round(f1_v2 - f1_v1, 4):+.4f}")
    if elr >= 0:
        print(f"  Экономия от перехода на v2: +{elr:,.0f} ₽  (v2 пропускает меньше дефолтов)")
    else:
        print(f"  Дополнительные потери от перехода на v2: {abs(elr):,.0f} ₽  (v2 пропускает больше дефолтов)")
    print(sep)

    # Критерии успешности
    c1 = pvalue < 0.05
    c2 = (f1_v2 - f1_v1) > 0.02
    c3 = pr_v2 >= 0.60
    c4 = ci_v2[0] > f1_v1  # нижняя граница CI v2 выше точечного F1 v1

    print("\n  Критерии успешности:")
    print(f"  {'[OK]' if c1 else '[--]'} p-value < 0.05              ({pvalue})")
    print(f"  {'[OK]' if c2 else '[--]'} F1(v2) > F1(v1) + 0.02     ({f1_v2} > {f1_v1 + 0.02:.4f})")
    print(f"  {'[OK]' if c3 else '[--]'} Precision(v2) >= 0.60       ({pr_v2})")
    print(f"  {'[OK]' if c4 else '[--]'} 95% CI v2 не пересекается   ({ci_v2[0]} > {f1_v1})")
    print(sep)

    if all([c1, c2, c3, c4]):
        print("  ВЕРДИКТ: v2 побеждает → переводить в production")
    elif sum([c1, c2, c3, c4]) >= 2:
        print("  ВЕРДИКТ: результаты смешанные → продлить тест")
    else:
        print("  ВЕРДИКТ: v2 не показала улучшения → оставить v1")
    print(f"{sep}\n")


def main():
    print(f"API: {BASE_URL}")
    print(f"Данные: {DATA_PATH}\n")

    # Проверка доступности сервиса
    try:
        req = urllib.request.Request(f"{BASE_URL}/health")
        with urllib.request.urlopen(req, timeout=3) as resp:
            health = json.loads(resp.read())
        print(f"Сервис доступен: {health}")
    except Exception as e:
        print(f"[ERROR] Сервис недоступен: {e}")
        print(f"Запустите: docker compose up  или  python app/api.py")
        sys.exit(1)

    df_test = load_test_data()
    y_true = df_test[TARGET_COL].values.astype(int)

    print("\nСобираем предсказания v1...")
    pred_v1 = collect_predictions(df_test, "v1")

    print("\nСобираем предсказания v2...")
    pred_v2 = collect_predictions(df_test, "v2")

    print_results(y_true, pred_v1, pred_v2)


if __name__ == "__main__":
    main()