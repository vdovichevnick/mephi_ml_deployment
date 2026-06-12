import logging
import os

from flask import Flask, jsonify, request
from model_handler import ModelHandler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Обе модели загружаются один раз в память при запуске процесса.
MODELS = {}
for v in ("v1", "v2"):
    try:
        MODELS[v] = ModelHandler(version=v)
        logger.info(f"Модель {v} загружена")
    except FileNotFoundError as e:
        logger.warning(f"Модель {v} не найдена: {e}")


# Эндпоинты
@app.route("/health", methods=["GET"])
def health():
    """GET /health — возвращает статус сервиса и список загруженных моделей."""
    loaded = list(MODELS.keys())
    status = "healthy" if loaded else "degraded"
    code = 200 if loaded else 503
    return jsonify({"status": status, "models_loaded": loaded}), code


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict?version=v1|v2

    Body (JSON):
    {
        "LIMIT_BAL": 20000, "SEX": 2, "EDUCATION": 2, "MARRIAGE": 1, "AGE": 24,
        "PAY_0": 2, "PAY_2": 2, "PAY_3": -1, "PAY_4": -1, "PAY_5": -2, "PAY_6": -2,
        "BILL_AMT1": 3913, "BILL_AMT2": 3102, "BILL_AMT3": 689,
        "BILL_AMT4": 0,    "BILL_AMT5": 0,    "BILL_AMT6": 0,
        "PAY_AMT1": 0,     "PAY_AMT2": 689,   "PAY_AMT3": 0,
        "PAY_AMT4": 0,     "PAY_AMT5": 0,     "PAY_AMT6": 0
    }

    Response 200:
    {"prediction": 1, "probability": 0.8342, "model_version": "v1"}

    Codes: 200 OK | 400 Bad Request | 503 Model Not Loaded
    """
    # Валидация параметра версии
    version = request.args.get("version", "v1")
    if version not in ("v1", "v2"):
        return jsonify({"error": "version должен быть 'v1' или 'v2'"}), 400
    # Проверяем что модель успешно загрузилась при старте
    if version not in MODELS:
        return jsonify({"error": f"Модель {version} не загружена. Запустите train_model.py"}), 503
    # Парсим тело запроса — silent=True возвращает None вместо исключения
    data = request.get_json(silent=True)
    if data is None:
        return jsonify({"error": "Тело запроса должно быть валидным JSON"}), 400

    try:
        features = ModelHandler.preprocess_input(data)
        result = MODELS[version].predict(features)
        logger.info(f"Предсказание: version={version} result={result}")
        return jsonify(result), 200
    except ValueError as e:
        # ValueError бросается из preprocess_input при отсутствующих признаках
        return jsonify({"error": str(e)}), 400
    except Exception as e:
        logger.error(f"Ошибка предсказания: {e}", exc_info=True)
        return jsonify({"error": "Внутренняя ошибка сервера"}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5010))
    app.run(host="0.0.0.0", port=port, debug=False)