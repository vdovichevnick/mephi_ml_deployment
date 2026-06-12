import json
import logging
import os
import pickle
from typing import Dict, List

import numpy as np

logger = logging.getLogger(__name__)

# Путь к папке models/ относительно этого файла (app/../models)
MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# Порядок признаков должен совпадать с тем, в котором обучалась модель.
# Изменение порядка без переобучения сломает предсказания.
FEATURE_COLS = [
    "LIMIT_BAL", "SEX", "EDUCATION", "MARRIAGE", "AGE",
    "PAY_0", "PAY_2", "PAY_3", "PAY_4", "PAY_5", "PAY_6",
    "BILL_AMT1", "BILL_AMT2", "BILL_AMT3", "BILL_AMT4", "BILL_AMT5", "BILL_AMT6",
    "PAY_AMT1", "PAY_AMT2", "PAY_AMT3", "PAY_AMT4", "PAY_AMT5", "PAY_AMT6",
]


class ModelHandler:
    """Загружает модель из .pkl и выполняет предсказание.

    Args:
        version: "v1" или "v2". По умолчанию "v1".
    """

    def __init__(self, version: str = "v1"):
        self.version = version
        self.model = None
        self._load()

    def _load(self):
        path = os.path.join(MODELS_DIR, f"model_{self.version}.pkl")
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Файл модели не найден: {path}\n"
                "Запустите models/train_model.py для обучения модели."
            )
        with open(path, "rb") as f:
            self.model = pickle.load(f)
        logger.info(f"Модель {self.version} загружена из {path}")

    def predict(self, features: np.ndarray) -> Dict:
        """Возвращает предсказание и вероятность."""
        prediction = int(self.model.predict(features)[0])
        # predict_proba возвращает [P(класс=0), P(класс=1)] — берём вероятность дефолт
        probability = float(self.model.predict_proba(features)[0][1])
        return {
            "prediction": prediction,
            "probability": round(probability, 4),
            "model_version": self.version,
        }

    @staticmethod
    def preprocess_input(data: Dict) -> np.ndarray:
        """Преобразует JSON-словарь в numpy array."""
        try:
            features = np.array(
                [float(data[col]) for col in FEATURE_COLS]
            ).reshape(1, -1)
        except KeyError as e:
            raise ValueError(f"Отсутствует обязательный признак: {e}")
        except (TypeError, ValueError) as e:
            raise ValueError(f"Некорректный тип данных: {e}")
        return features

    @staticmethod
    def get_feature_names() -> List[str]:
        return FEATURE_COLS