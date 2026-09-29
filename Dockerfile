# ---- Stage 1: train the model ----
FROM python:3.11-slim AS trainer
WORKDIR /build
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY train.py .
COPY data/ data/
RUN python train.py

# ---- Stage 2: serve the model ----
FROM python:3.11-slim
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
WORKDIR /srv
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && useradd --create-home appuser
COPY app/ app/
COPY --from=trainer /build/model/ model/
USER appuser
EXPOSE 8000
# Render injects $PORT; default to 8000 for local runs.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
