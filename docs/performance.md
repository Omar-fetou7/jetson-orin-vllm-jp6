# Performance tuning on an AGX Orin

Numbers below come from an AGX Orin 64 GB (JetPack 6.2.1) serving a ~2B-parameter vision-language model in BF16. Treat them as orders of magnitude for your own measurements.

## Measure the right things

| Metric | What it tells you | How |
|---|---|---|
| Time to first token (TTFT) | Vision encoding + prefill + queueing | `tools/ask_image.py` (streaming) |
| Decode tokens/s per request | Generation speed one user sees | `tools/ask_image.py` |
| Aggregate tokens/s | Total generation capacity | `tools/benchmark_concurrency.py` |
| GPU / memory controller load, power, temperature | Hardware saturation | `sudo tegrastats`, `jtop` |
| Running / waiting requests, KV-cache use | Scheduler state | `curl -s localhost:8000/metrics` |

## Findings

**Eager mode costs a lot.** `--enforce-eager` disables `torch.compile` and CUDA Graphs. Removing it raised decode from ~16 to ~30 tokens/s. Startup gets slower and uses a little more memory while graphs are captured.

**Power mode.** `sudo nvpmodel -m <id>` selects the power profile (see `/etc/nvpmodel.conf` for IDs) and `sudo jetson_clocks` pins the clocks. Moving from 30 W to 50 W raised the GPU clock from 612 to 816 MHz. During inference the GPU sat at ~99% and stayed around 40 °C. `jetson_clocks --restore` (or a reboot) returns to normal clocks.

**Concurrency trades latency for throughput.** 80 requests at concurrency 8 produced ~300 aggregate tokens/s, while each request slowed down. For live video, prefer a small number of in-flight requests and drop stale frames over queueing them.

**Prefill is the video bottleneck.** Ten full-HD frames per request (~9,000 input tokens) sent once per second overwhelmed the Orin. It prefilled only ~900 input tokens/s, so queues grew to minutes and decode dropped below 1 token/s, while the KV cache was only a few percent used. vLLM schedules at most `max_num_batched_tokens` (2,048 by default here) prompt tokens per engine step, so long prompts are split into chunks and compete with decoding.

Practical rule: at one request per second, keep each request under ~2,000 input tokens.

- Resize frames: visual tokens scale with image area. At 640×352 a frame was ~220 tokens; at 1280×704, ~880.
- Use 3–4-frame windows rather than 10.
- Send windows less often if you need more frames.

**Structured output.** `response_format` with a JSON schema makes vLLM constrain decoding, so short yes/no JSON answers are ~35 tokens and finish fast. Reasoning ("thinking") modes generate hundreds of tokens before the final answer. With a token cap, some requests never reach the JSON at all, so budget generously or skip reasoning for real-time use.

## Memory

`--gpu-memory-utilization` is the fraction of total (unified) memory vLLM may claim. After loading weights, it gives almost everything else to the KV cache. On a 64 GB Orin, 0.75 reserves ~46 GiB even for a 2B model.

Estimate what you actually need:

```text
KV bytes per token = layers × 2 (K and V) × kv_heads × head_dim × bytes_per_value
```

For a 28-layer model with 8 KV heads of dimension 128 in BF16, that is 112 KiB per token. Fifteen concurrent requests of ~1,400 tokens need only ~2.3 GiB of KV cache. A budget of 0.30–0.40 (18–25 GiB) left plenty of room for other processes on the robot.
