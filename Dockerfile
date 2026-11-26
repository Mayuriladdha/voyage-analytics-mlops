FROM python:3.12-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /app

COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

COPY src ./src
COPY api ./api
COPY models ./models

ENV MODEL_DIR=/app/models PORT=5000
EXPOSE 5000
HEALTHCHECK CMD python -c "import urllib.request as u; u.urlopen('http://localhost:5000/health')"
CMD ["gunicorn", "--chdir", "api", "--bind", "0.0.0.0:5000", "--workers", "2", "app:app"]
