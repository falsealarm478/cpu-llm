"""Live checks against Docker or Kubernetes through localhost. Never print secrets/logs."""
import argparse
import json
import subprocess
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import ProxyHandler, Request, build_opener

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--kubernetes", action="store_true", help="Read pod logs through Minikube")
parser.add_argument("--profile", default="llm-dev")
parser.add_argument("--namespace", default="llm-demo")
args = parser.parse_args()

key = next((line.split("=", 1)[1].strip() for line in Path(".env").read_text().splitlines()
            if line.startswith("VLLM_API_KEY=")), None)
if not key:
    raise SystemExit("Missing VLLM_API_KEY in .env")
marker = "PRIVATE_" + uuid.uuid4().hex
since = datetime.now(timezone.utc).isoformat()
log_command = (
    ["minikube", "-p", args.profile, "kubectl", "--", "-n", args.namespace,
     "logs", "deployment/llm-api", "-c", "inference", "--since-time=" + since]
    if args.kubernetes else ["sudo", "docker", "logs", "--since", since, "llm-api"]
)
opener = build_opener(ProxyHandler({}))
normal = {"model": "local-model", "messages": [{"role": "user", "content":
          "Say hello in one sentence. Test marker: " + marker}], "max_tokens": 16, "temperature": 0}
response_texts = []


def request(label, payload, token, expected):
    headers = {"Content-Type": "application/json", "X-Request-Id": marker}
    if token is not None:
        headers["Authorization"] = "Bearer " + token
    req = Request("http://127.0.0.1:8000/v1/chat/completions?private=" + marker,
                  data=json.dumps(payload).encode(), headers=headers)
    try:
        response = opener.open(req, timeout=120)
    except HTTPError as exc:
        response = exc
    with response:
        status = response.code
        body = response.read().decode()
    if status not in expected:
        raise SystemExit(f"FAIL {label}: HTTP {status}; expected {sorted(expected)} (body withheld)")
    if status == 200 and not payload.get("stream"):
        text = json.loads(body)["choices"][0]["message"].get("content")
        if isinstance(text, str) and text.strip():
            response_texts.append(text.strip())
    if status == 200 and payload.get("stream"):
        parts = []
        for line in body.splitlines():
            if line.startswith("data: ") and line != "data: [DONE]":
                choices = json.loads(line[6:]).get("choices", [])
                if choices:
                    text = choices[0].get("delta", {}).get("content")
                    if text:
                        parts.append(text)
        if parts:
            response_texts.append("".join(parts).strip())
    print(f"PASS {label}: HTTP {status}")


request("missing token", normal, None, {401})
request("wrong token", normal, marker, {401})
request("valid request", normal, key, {200})
request("invalid message", {**normal, "messages": [{"role": "user", "content": {"private": marker}}]},
        key, {400, 422})
request("streaming request", {**normal, "stream": True}, key, {200})

finished = []
for attempt in range(5):
    logs = subprocess.run(log_command,
                          capture_output=True, check=True, text=True)
    combined = logs.stdout + logs.stderr
    for secret in [marker, key] + [text for text in response_texts if len(text) >= 8]:
        if secret in combined:
            raise SystemExit("FAIL: private request/response/token content found in container logs; value withheld")
    finished = []
    for line in combined.splitlines():
        start = line.find("{")
        if start < 0:
            continue
        try:
            event = json.loads(line[start:])
        except json.JSONDecodeError:
            continue
        if event.get("event") == "request_finished" and event.get("endpoint") == "chat_completions":
            finished.append(event)
    statuses = [e.get("status") for e in finished]
    if statuses.count(401) >= 2 and statuses.count(200) >= 2 and any(s in (400, 422) for s in statuses):
        break
    time.sleep(0.2)
else:
    raise SystemExit("FAIL: expected audit events were not found in container logs")
print("PASS: audit events found for successful, rejected, invalid and streaming requests")
print("PASS: test markers, token and checked response texts absent from stdout/stderr logs")
print("These checks cover the exercised requests, not every possible crash or third-party output.")
