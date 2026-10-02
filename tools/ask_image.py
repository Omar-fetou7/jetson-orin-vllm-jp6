"""Send images + a prompt to an OpenAI-compatible server and print metrics.

    python tools/ask_image.py photo.jpg --prompt "What is happening?"
    python tools/ask_image.py f1.jpg f2.jpg f3.jpg --prompt "Describe the motion."

Uses streaming so it can measure time to first token (prefill + vision
encoding) separately from decode speed. Standard library only.
"""

import argparse
import base64
import json
import mimetypes
import time
import urllib.request


def image_part(path):
    mime = mimetypes.guess_type(path)[0] or "image/jpeg"
    data = base64.b64encode(open(path, "rb").read()).decode("ascii")
    return {"type": "image_url", "image_url": {"url": f"data:{mime};base64,{data}"}}


def first_model(base_url):
    with urllib.request.urlopen(f"{base_url}/v1/models", timeout=10) as r:
        return json.load(r)["data"][0]["id"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("images", nargs="+")
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None, help="defaults to the served model")
    parser.add_argument("--max-tokens", type=int, default=512)
    args = parser.parse_args()

    model = args.model or first_model(args.base_url)
    content = [image_part(p) for p in args.images] + [{"type": "text", "text": args.prompt}]
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": args.max_tokens,
        "temperature": 0,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    request = urllib.request.Request(
        f"{args.base_url}/v1/chat/completions",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )

    start = time.perf_counter()
    first_token_at = None
    text = []
    usage = {}
    with urllib.request.urlopen(request, timeout=600) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            chunk = json.loads(line[5:])
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                piece = delta.get("content") or delta.get("reasoning_content") or ""
                if piece and first_token_at is None:
                    first_token_at = time.perf_counter()
                text.append(piece)
    end = time.perf_counter()

    print("".join(text).strip())
    out_tokens = usage.get("completion_tokens", 0)
    ttft = (first_token_at or end) - start
    decode = max(end - (first_token_at or end), 1e-9)
    print("\n--- metrics")
    print(f"input tokens      : {usage.get('prompt_tokens')}")
    print(f"output tokens     : {out_tokens}")
    print(f"time to 1st token : {ttft:.2f} s   (vision encoding + prefill + queue)")
    print(f"decode speed      : {out_tokens / decode:.1f} tokens/s")
    print(f"total             : {end - start:.2f} s")


if __name__ == "__main__":
    main()
