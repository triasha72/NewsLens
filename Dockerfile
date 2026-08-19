FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1 \
    VECLIB_MAXIMUM_THREADS=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY src ./src

RUN python -m pip install --upgrade pip && \
    python -m pip install ".[recsys,serving]"

RUN useradd \
    --create-home \
    --uid 10001 \
    newslens

USER newslens

EXPOSE 8000

HEALTHCHECK \
    --interval=30s \
    --timeout=3s \
    --start-period=30s \
    --retries=3 \
    CMD python -c \
    "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=2)"

CMD [
    "uvicorn",
    "newslens.api.production:app",
    "--host",
    "0.0.0.0",
    "--port",
    "8000",
    "--workers",
    "1"
]
