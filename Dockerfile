# syntax=docker/dockerfile:1

# --- build -----------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /build
COPY requirements.txt .
RUN python -m venv /opt/venv \
 && /opt/venv/bin/pip install --upgrade pip \
 && /opt/venv/bin/pip install -r requirements.txt

# --- runtime ---------------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PATH="/opt/venv/bin:$PATH" \
    # Without this, `python scripts/foo.py` puts /srv/scripts on sys.path
    # instead of /srv, and every `from app...` import fails.
    PYTHONPATH=/srv

RUN apt-get update \
 && apt-get install -y --no-install-recommends curl \
 && rm -rf /var/lib/apt/lists/* \
 && useradd --create-home --uid 10001 alertradar

COPY --from=builder /opt/venv /opt/venv

WORKDIR /srv
COPY --chown=alertradar:alertradar app ./app
COPY --chown=alertradar:alertradar scripts ./scripts
COPY --chown=alertradar:alertradar data ./data

USER alertradar
EXPOSE 8000

# Overridden by docker-compose for the worker service.
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
