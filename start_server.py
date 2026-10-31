"""Download once, switch to offline mode, then replace PID 1 with vLLM."""
import json
import logging.config
import os
import sys
from pathlib import Path
from download_model import prepare_from_env


def integer(name, default):
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError("non_positive_setting")
    return value


def server_args(model_path):
    dtype = os.environ.get("DTYPE", "float32")
    if dtype not in {"float32", "bfloat16", "float16"}:
        raise ValueError("invalid_dtype")
    return [
        "vllm", "serve", str(model_path),
        "--served-model-name", os.environ.get("SERVED_MODEL_NAME", "local-model"),
        "--dtype", dtype,
        "--max-model-len", str(integer("MAX_MODEL_LEN", 512)),
        "--max-num-seqs", str(integer("MAX_NUM_SEQS", 1)),
        "--max-num-batched-tokens", str(integer("MAX_NUM_BATCHED_TOKENS", 512)),
        "--kv-cache-memory-bytes", str(integer("KV_CACHE_MEMORY_BYTES", 134217728)),
        "--enforce-eager", "--host", "0.0.0.0", "--port", "8000",
        "--no-enable-log-requests", "--disable-uvicorn-access-log",
        "--log-config-file", "/app/logging.json",
        "--middleware", "privacy_logging.RequestAuditMiddleware",
    ]


def configure_inference_env():
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1",
                      VLLM_NO_USAGE_STATS="1", VLLM_DEBUG_LOG_API_SERVER_RESPONSE="0",
                      VLLM_LOGGING_CONFIG_PATH="/app/logging.json")
    os.environ.pop("HF_TOKEN", None)
    # vLLM 0.27.0 applies this legacy variable AFTER parsing the CLI and it
    # would override --kv-cache-memory-bytes, including our Dockerfile default.
    os.environ.pop("VLLM_CPU_KVCACHE_SPACE", None)


def main():
    if not os.environ.get("VLLM_API_KEY", "").strip():
        raise ValueError("missing_api_key")
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
    logging.config.dictConfig(json.loads(Path("/app/logging.json").read_text()))
    # Validate inference settings before a potentially expensive download.
    server_args("/placeholder")
    path, reused = prepare_from_env()
    print(json.dumps({"event": "model_ready", "reused": reused}), flush=True)
    configure_inference_env()
    os.execvp("vllm", server_args(path))


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        # Do not print exception messages: HF failures may contain credentials.
        print(json.dumps({"event": "startup_failed", "error_type": type(exc).__name__}),
              file=sys.stderr, flush=True)
        sys.exit(1)
