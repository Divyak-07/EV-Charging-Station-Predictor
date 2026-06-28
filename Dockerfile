# ── Build Stage ───────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for lightgbm/xgboost (build only)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libgomp1 && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ── Production Stage ──────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Runtime deps only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 && \
    rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy application code
COPY ev_ml_predictor.py .
COPY ev_campus_analyzer.py .
COPY main.py .
COPY model_comparison.py .
COPY web/ web/

# Copy trained model files
RUN mkdir -p output
COPY output/*.joblib output/

# Create uploads directory
RUN mkdir -p web/uploads

# Expose port
EXPOSE 5000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD python -c "import requests; requests.get('http://localhost:5000/api/health', timeout=5)" || exit 1

# Production server
CMD ["gunicorn", "--bind", "0.0.0.0:5000", "--workers", "2", "--timeout", "120", "web.app:app"]
