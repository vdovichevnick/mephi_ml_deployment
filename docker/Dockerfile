FROM python:3.12-slim

# Метаданные образа
LABEL maintainer="Nikita Vdovichev"
LABEL description="Credit Card Default Prediction Service"
LABEL version="1.0"

# Рабочая директория внутри контейнера
WORKDIR /app

# Устанавливаем curl — нужен для HEALTHCHECK ниже
RUN apt-get update && apt-get install -y --no-install-recommends \
        curl \
    && rm -rf /var/lib/apt/lists/*

# Сначала копируем только requirements.txt и устанавливаем зависимости.
# Это отдельный слой — Docker закеширует его и не будет переустанавливать
# пакеты при каждом изменении кода (пока requirements.txt не меняется)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код сервиса и обученные модели
COPY app/ ./app/
COPY models/ ./models/

# Создаём непривилегированного пользователя и передаём ему права на /app.
# Запуск от root в контейнере — плохая практика безопасности
RUN adduser --disabled-password --gecos "" appuser && \
    chown -R appuser:appuser /app
USER appuser

# Переменные окружения для сервиса
ENV PORT=5010
ENV FLASK_DEBUG=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONPATH=/app/app

# Сообщаем Docker какой порт слушает контейнер
EXPOSE 5010

# Периодически проверяем что сервис живой.
# Docker перезапустит контейнер если /health не отвечает 3 раза подряд
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:5010/health || exit 1

# Запускаем через Gunicorn — WSGI-сервер.
# --workers 2        : два процесса-воркера для параллельных запросов
# --access-logfile - : логи запросов в stdout (видны через docker logs)
# --error-logfile -  : логи ошибок туда же
CMD ["gunicorn", "--workers", "2", "--bind", "0.0.0.0:5010", \
     "--timeout", "60", "--access-logfile", "-", "--error-logfile", "-", \
     "app.api:app"]