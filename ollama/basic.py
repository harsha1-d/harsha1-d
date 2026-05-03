import json
from typing import Any

import requests

SERVER_URL = "http://127.0.0.1:11434"
DEFAULT_MODEL = "llama3:8b"
REQUEST_TIMEOUT = 120


def call_ollama(prompt: str, model: str = DEFAULT_MODEL) -> str:
    url = f"{SERVER_URL}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    response = requests.post(url, json=payload, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return parse_ollama_response(data)


def parse_ollama_response(data: Any) -> str:
    if isinstance(data, dict):
        if "response" in data:
            return str(data["response"])
        if "output" in data:
            output = data["output"]
            if isinstance(output, list):
                return "".join(str(item) for item in output)
            return str(output)
        if "choices" in data:
            parts = []
            for choice in data["choices"]:
                content = choice.get("content")
                if isinstance(content, list):
                    parts.extend(str(item) for item in content)
                elif content is not None:
                    parts.append(str(content))
            if parts:
                return "".join(parts)
    return json.dumps(data, indent=2)


def main() -> None:
    prompt = (
        "Write a short Tamil melody song in English with a gentle, lyrical tone."
    )
    print("Prompt:", prompt)
    print("-" * 60)
    try:
        output = call_ollama(prompt)
        print(output)
    except requests.exceptions.RequestException as exc:
        print("Ollama request failed:", exc)


if __name__ == "__main__":
    main()