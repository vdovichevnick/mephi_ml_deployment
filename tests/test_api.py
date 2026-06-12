import json
import os
import urllib.request
import urllib.error

# Адрес сервиса берётся из переменной окружения, по умолчанию localhost:5010
BASE_URL = f"http://localhost:{os.environ.get('API_PORT', '5010')}"

# Вспомогательные функции HTTP
def get(path):
    req = urllib.request.Request(f"{BASE_URL}{path}")
    with urllib.request.urlopen(req) as resp:
        return resp.status, json.loads(resp.read())


def post(path, payload):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        f"{BASE_URL}{path}",
        data=data,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())

# Тестовые данные
# Клиент с хорошей историей платежей — ожидаем prediction=0 (нет дефолта)
SAMPLE_PAYLOAD_NO_DEFAULT = {
    "LIMIT_BAL": 200000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 29,
    "PAY_0": -1, "PAY_2": -1, "PAY_3": -1, "PAY_4": -1, "PAY_5": -1, "PAY_6": -1,
    "BILL_AMT1": 1800, "BILL_AMT2": 1500, "BILL_AMT3": 1200,
    "BILL_AMT4": 1000, "BILL_AMT5": 800,  "BILL_AMT6": 700,
    "PAY_AMT1": 1800, "PAY_AMT2": 1500, "PAY_AMT3": 1200,
    "PAY_AMT4": 1000, "PAY_AMT5": 800,  "PAY_AMT6": 700,
}

# Клиент с просрочками и нулевыми платежами — ожидаем prediction=1 (дефолт)
SAMPLE_PAYLOAD_DEFAULT = {
    "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
    "PAY_0": 2, "PAY_2": 2, "PAY_3": 2, "PAY_4": 0, "PAY_5": 0, "PAY_6": 2,
    "BILL_AMT1": 19000, "BILL_AMT2": 18500, "BILL_AMT3": 18000,
    "BILL_AMT4": 17000, "BILL_AMT5": 16000, "BILL_AMT6": 15000,
    "PAY_AMT1": 0, "PAY_AMT2": 0, "PAY_AMT3": 0,
    "PAY_AMT4": 0, "PAY_AMT5": 0, "PAY_AMT6": 0,
}

# Неполный набор признаков — сервис должен вернуть 400
MISSING_FEATURE_PAYLOAD = {"LIMIT_BAL": 100000, "SEX": 1}

# Тесты
class TestHealthEndpoint:
    def test_health_returns_200(self):
        status, data = get("/health")
        assert status == 200

    def test_health_response_structure(self):
        # Проверяем что ответ содержит нужные поля и модели загружены
        _, data = get("/health")
        assert "status" in data
        assert "models_loaded" in data
        assert isinstance(data["models_loaded"], list)


class TestPredictEndpoint:
    def test_predict_missing_features(self):
        # Запрос без обязательных признаков → 400 Bad Request
        status, _ = post("/predict", MISSING_FEATURE_PAYLOAD)
        assert status == 400

    def test_predict_invalid_version(self):
        # Несуществующая версия модели → 400 Bad Request
        status, _ = post("/predict?version=v99", SAMPLE_PAYLOAD_NO_DEFAULT)
        assert status == 400

    def test_predict_success_v1(self):
        # Базовая модель: проверяем структуру и диапазон значений ответа
        status, data = post("/predict?version=v1", SAMPLE_PAYLOAD_NO_DEFAULT)
        assert status == 200
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["model_version"] == "v1"

    def test_predict_success_v2(self):
        # Новая модель: те же проверки что и для v1
        status, data = post("/predict?version=v2", SAMPLE_PAYLOAD_DEFAULT)
        assert status == 200
        assert data["prediction"] in (0, 1)
        assert 0.0 <= data["probability"] <= 1.0
        assert data["model_version"] == "v2"