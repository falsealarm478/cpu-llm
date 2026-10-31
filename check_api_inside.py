"""Run inside a container/pod. No secrets are printed."""
import json
import os
from urllib.request import ProxyHandler, Request, build_opener

payload = {"model": os.environ.get("SERVED_MODEL_NAME", "local-model"), "messages": [
    {"role": "user", "content": "What is the capital of France?"}], "max_tokens": 32, "temperature": 0}
request = Request("http://127.0.0.1:8000/v1/chat/completions", data=json.dumps(payload).encode(),
                  headers={"Content-Type": "application/json",
                           "Authorization": "Bearer " + os.environ["VLLM_API_KEY"]})
with build_opener(ProxyHandler({})).open(request, timeout=120) as response:
    result = json.load(response)
    print("HTTP:", response.status)
    print("Answer:", result["choices"][0]["message"]["content"])
