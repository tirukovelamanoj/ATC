# Serving image: the game plus a trained policy, WITHOUT torch.
# torch is ~2.5GB and would not fit a small instance; onnxruntime is ~50MB and
# runs this policy in about 0.2ms, so training stays on a laptop and only the
# exported .onnx ships.
FROM python:3.13-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependencies are installed from pyproject alone, against a STUB package, so a
# code edit never invalidates this layer. Copying src/ before the install meant
# every one-line change to the UI reinstalled onnxruntime from scratch.
COPY pyproject.toml README.md ./
RUN mkdir -p src/atc && touch src/atc/__init__.py \
 && pip install --no-cache-dir ".[ai]" \
 && rm -rf src

# real source last: this is the only layer a code change rebuilds
COPY src/ ./src/
# PYTHONPATH so the real tree wins over the stub left in site-packages
ENV PYTHONPATH=/app/src

COPY configs/ ./configs/
COPY models/ ./models/
COPY examples/ ./examples/

# Koyeb injects PORT; default to 8000 for local `docker run`
ENV ATC_CONFIG=/app/configs/arcade_m1.json \
    ATC_MODEL=/app/models/policy.onnx \
    ATC_EXAMPLE=/app/examples/agent.py \
    ATC_MAX_GAMES=4 \
    ATC_LOG_LEVEL=INFO \
    PORT=8000
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=20s \
  CMD python -c "import urllib.request,os; urllib.request.urlopen(f'http://127.0.0.1:{os.environ.get(\"PORT\",8000)}/health')"

CMD ["sh", "-c", "uvicorn atc.arcade.server:app --host 0.0.0.0 --port ${PORT:-8000}"]
