"""Pinned HF download with a verified, network-free warm-start path."""
import hashlib
import json
import os
import re
from pathlib import Path

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-135M-Instruct"
DEFAULT_REVISION = "12fd25f77366fa6b3b4b768ec3050bf629380bac"
PATTERNS = ["config.json", "generation_config.json", "tokenizer*", "special_tokens_map.json",
            "vocab*", "merges.txt", "*.safetensors", "*.safetensors.index.json", "chat_template.jinja"]


class ModelStateError(RuntimeError):
    pass


def digest(path):
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            checksum.update(block)
    return checksum.hexdigest()


def safe_file(directory, name):
    candidate = directory / name
    if not candidate.resolve().is_relative_to(directory.resolve()) or candidate.is_symlink():
        raise ModelStateError("unsafe_model_path")
    return candidate


def check_layout(directory):
    for name in ("config.json", "tokenizer_config.json"):
        if not (directory / name).is_file():
            raise ModelStateError("missing_model_metadata")
    if not any((directory / name).is_file() for name in ("tokenizer.json", "tokenizer.model", "vocab.json")):
        raise ModelStateError("missing_tokenizer")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        shards = set(json.loads(index.read_text())["weight_map"].values())
        if not shards or any(not safe_file(directory, shard).is_file() for shard in shards):
            raise ModelStateError("missing_weight_shard")
    elif not (directory / "model.safetensors").is_file():
        raise ModelStateError("missing_weights")


def verify_manifest(directory, manifest, model_id, revision):
    if manifest.get("model_id") != model_id or manifest.get("revision") != revision:
        raise ModelStateError("manifest_identity_mismatch")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files:
        raise ModelStateError("invalid_manifest")
    for name, expected in files.items():
        path = safe_file(directory, name)
        if not path.is_file() or digest(path) != expected:
            raise ModelStateError("model_integrity_failed")
    check_layout(directory)


def hf_download(model_id, revision, directory, token):
    # Deliberately imported only on the cold path.
    from huggingface_hub import snapshot_download
    snapshot_download(repo_id=model_id, revision=revision, local_dir=str(directory),
                      allow_patterns=PATTERNS, token=token or False, max_workers=2)


def ensure_model(root, model_id, revision, token=None, downloader=hf_download):
    if not re.fullmatch(r"[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+", model_id):
        raise ModelStateError("invalid_model_id")
    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ModelStateError("revision_must_be_commit_sha")
    identity = hashlib.sha256(f"{model_id}@{revision}".encode()).hexdigest()[:24]
    directory = Path(root) / ("model-" + identity)
    directory.mkdir(parents=True, exist_ok=True)
    ready = directory / ".ready.json"
    if ready.exists():
        # No HF call or import, including on corrupt/missing cached files.
        verify_manifest(directory, json.loads(ready.read_text()), model_id, revision)
        return directory, True

    downloader(model_id, revision, directory, token)
    check_layout(directory)
    files = {}
    for path in sorted(directory.rglob("*")):
        relative = path.relative_to(directory)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if path.is_file():
            checked = safe_file(directory, relative.as_posix())
            files[relative.as_posix()] = digest(checked)
    manifest = {"model_id": model_id, "revision": revision, "files": files}
    verify_manifest(directory, manifest, model_id, revision)
    pending = directory / ".ready.json.tmp"
    with pending.open("w") as stream:
        json.dump(manifest, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, ready)
    return directory, False


def prepare_from_env():
    # This lock is for Linux containers; it is released automatically on a crash.
    import fcntl
    root = Path(os.environ.get("MODEL_ROOT", "/models"))
    root.mkdir(parents=True, exist_ok=True)
    with (root / ".download.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        return ensure_model(root, os.environ.get("MODEL_ID", DEFAULT_MODEL),
                            os.environ.get("MODEL_REVISION", DEFAULT_REVISION),
                            os.environ.get("HF_TOKEN"))


if __name__ == "__main__":
    import sys
    os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"
    try:
        path, reused = prepare_from_env()
        print(json.dumps({"event": "model_ready", "reused": reused}), flush=True)
    except Exception as exc:
        print(json.dumps({"event": "model_prepare_failed", "error_type": type(exc).__name__}),
              file=sys.stderr, flush=True)
        sys.exit(1)
