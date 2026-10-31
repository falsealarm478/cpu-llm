"""Apply a Secret over stdin; keep real values out of files and command arguments."""
import argparse
import json
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--profile", default="llm-dev")
args = parser.parse_args()
values = {}
for raw in Path(".env").read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#"):
        continue
    name, separator, value = line.partition("=")
    if separator and name in {"VLLM_API_KEY", "HF_TOKEN"}:
        values[name] = value
if not values.get("VLLM_API_KEY", "").strip():
    raise SystemExit("Missing VLLM_API_KEY in .env; no secret created.")
values.setdefault("HF_TOKEN", "")
secret = {"apiVersion": "v1", "kind": "Secret", "type": "Opaque",
          "metadata": {"name": "llm-secrets", "namespace": "llm-demo"}, "stringData": values}
command = ["minikube", "-p", args.profile, "kubectl", "--"]
subprocess.run(command + ["apply", "-f", "k8s/namespace.yaml"], check=True)
subprocess.run(command + ["apply", "--server-side", "--field-manager=llm-secret-setup", "-f", "-"],
               input=json.dumps(secret), text=True, check=True)
print("Secret applied. Values were not printed. HF_TOKEN may be empty for the public demo model.")
