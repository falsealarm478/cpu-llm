FROM ubuntu:24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
       python3 python3-venv ca-certificates libnuma1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m venv /opt/venv

ENV PATH="/opt/venv/bin:$PATH"

WORKDIR /app
COPY requirements.txt /app/requirements.txt

RUN python -m pip install \
      --no-cache-dir \
      --no-deps \
      --only-binary=:all: \
      --extra-index-url https://download.pytorch.org/whl/cpu \
      -r requirements.txt \
    && python -m pip check

RUN test -f /opt/venv/lib/libiomp5.so

RUN useradd --create-home --uid 10001 app \
    && mkdir -p /home/app/.cache/huggingface \
    && chown -R app:app /home/app

ENV HOME="/home/app" \
    HF_HOME="/home/app/.cache/huggingface" \
    LD_PRELOAD="/opt/venv/lib/libiomp5.so" \
    VLLM_CPU_KVCACHE_SPACE="1" \
    VLLM_CPU_OMP_THREADS_BIND="nobind" \
    OMP_NUM_THREADS="2" \
    PYTHONUNBUFFERED="1"

# privacy-logging-v1
COPY privacy_logging.py logging.json /app/
ENV PYTHONPATH="/app" \
    VLLM_LOGGING_CONFIG_PATH="/app/logging.json" \
    VLLM_DEBUG_LOG_API_SERVER_RESPONSE="0"

# model-storage-v1
COPY download_model.py start_server.py /app/
RUN mkdir -p /models && chown 10001:10001 /models
ENV MODEL_ROOT="/models" PYTHONDONTWRITEBYTECODE="1"

USER 10001:10001

EXPOSE 8000

ENTRYPOINT ["python", "/app/start_server.py"]
