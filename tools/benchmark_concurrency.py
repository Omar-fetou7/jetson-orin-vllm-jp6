"""Measure per-request and aggregate throughput under concurrent load.

    python tools/benchmark_concurrency.py --requests 40 --concurrency 8
    python tools/benchmark_concurrency.py --image frame.jpg --requests 40 --concurrency 8

Per-request tokens/s falls as concurrency rises, while aggregate tokens/s
rises until the GPU saturates. Standard library only.
"""

import argparse
import base64
import json
import statistics
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor


def one_request(base_url, model, prompt, image_b64, max_tokens):
    content = [{"type": "text", "text": prompt}]
    if image_b64:
        content.insert(0, {"type": "image_url",
                           "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}})
    payload = {"model": model, "messages": [{"role": "user", "content": content}],
               "max_tokens": max_tokens, "temperature": 0,
               # force a full-length answer so every request does the same work
               "ignore_eos": True}
    request = urllib.request.Request(f"{base_url}/v1/chat/completions",
                                     data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=900) as response:
        body = json.load(response)
    seconds = time.perf_counter() - start
    usage = body.get("usage", {})
    return seconds, usage.get("prompt_tokens", 0), usage.get("completion_tokens", 0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", default=None)
    parser.add_argument("--requests", type=int, default=40)
    parser.add_argument("--concurrency", type=int, default=8)
    parser.add_argument("--max-tokens", type=int, default=256)
    parser.add_argument("--image", default=None)
    parser.add_argument("--prompt", default="Describe the scene in detail.")
    args = parser.parse_args()

    model = args.model
    if model is None:
        with urllib.request.urlopen(f"{args.base_url}/v1/models", timeout=10) as r:
            model = json.load(r)["data"][0]["id"]
    image_b64 = base64.b64encode(open(args.image, "rb").read()).decode() if args.image else None

    start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [pool.submit(one_request, args.base_url, model, args.prompt, image_b64,
                               args.max_tokens) for _ in range(args.requests)]
        results, failures = [], 0
        for future in futures:
            try:
                results.append(future.result())
            except Exception as error:  # report, keep measuring the rest
                failures += 1
                print("request failed:", error)
    wall = time.perf_counter() - start

    latencies = [r[0] for r in results]
    out_tokens = sum(r[2] for r in results)
    per_request = [r[2] / r[0] for r in results if r[0] > 0]
    print(f"successful / failed     : {len(results)} / {failures}")
    print(f"total input tokens      : {sum(r[1] for r in results)}")
    print(f"total output tokens     : {out_tokens}")
    print(f"wall time               : {wall:.1f} s")
    print(f"aggregate output tok/s  : {out_tokens / wall:.1f}")
    print(f"requests/s              : {len(results) / wall:.2f}")
    if results:
        print(f"latency mean / max      : {statistics.mean(latencies):.2f} / {max(latencies):.2f} s")
        print(f"per-request tok/s (mean): {statistics.mean(per_request):.1f}")


if __name__ == "__main__":
    main()
