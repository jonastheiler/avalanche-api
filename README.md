# avalanche-api

A REST API that reads a Swiss avalanche bulletin and returns its avalanche problems as
JSON: type, danger level, aspects and elevations. Behind it runs Qwen3-1.7B with a
[LoRA adapter](https://huggingface.co/jth00/qwen3-1.7b-avalanche-lora) that matches Claude
Haiku on this task. How it was trained and compared:
[avalanche-llm](https://github.com/jonastheiler/avalanche-llm).

## Run it

```bash
uv sync
uv run uvicorn avalanche_api.app:app
```

Open http://127.0.0.1:8000/docs, then **POST /extract**, *Try it out*, *Execute*. The first
start downloads the model (3.4 GB) and needs about 8 GB of RAM. On a laptop CPU an
answer takes about 30 seconds, up to a minute for long bulletins.

```bash
curl -X POST http://127.0.0.1:8000/extract -H "Content-Type: application/json" \
  -d '{"text": "Gliding avalanches are possible on steep grassy slopes."}'
```

With Docker, the image has the models baked in (6.5 GB) and runs on any server:

```bash
docker build -t avalanche-api .
docker run -p 8000:8000 avalanche-api
```

Tests: `uv run pytest` (fake model, a second) and `uv run pytest -m slow` (real model).

## Load test

`uv run python loadtest.py --concurrency 8` sends 16 real bulletins (4 per language) to a
running server. `MAX_BATCH_SIZE` sets how many requests share one pass through the model.
On a laptop CPU (float32, torch using 4 threads), before prefix caching (below):

| batch size | requests at once | p50 latency | p95 latency | requests/min |
|---|---|---|---|---|
| 1 | 1 | 27 s | 58 s | 1.9 |
| 1 | 8 | 251 s | 278 s | 1.9 |
| 8 | 8 | 226 s | 229 s | 2.1 |

Batching gains only 10%. Timing the two phases of one request shows why:

- **Prefill** (reading the ~550-token prompt) takes 13 s for one bulletin and 108 s for 8.
  It is limited by compute, so batching saves nothing.
- **Decode** (writing the ~50–140-token answer) produces 3.3 tokens/s alone and 8.7 tokens/s
  for a batch of 8. It is limited by reading the weights from memory, which a batch shares.
  But a batch only finishes when its longest answer does.

The bigger lever is the prompt: 377 of its ~450–600 tokens are the system prompt, the same
for every request. So the server reads it once at startup and keeps its attention keys and
values; each request then prefills only its own bulletin. Prefill for one bulletin, timed
in the same process (on a slower day for the laptop than the table above):

| bulletin | tokens read without cache | with cache | prefill without cache | with cache |
|---|---|---|---|---|
| typical | ~450 | ~75 | 26 s | 5 s |
| long | 544 | 167 | 33 s | 10 s |

## Key ideas

- **The repos share an artifact and a contract, not code.** The adapter comes from the
  Hugging Face Hub at a pinned commit. The schema (`schema.py`) and the system prompt
  (`prompt.py`) are copied from avalanche-llm.
- **Train/serve skew fails silently.** A prompt that drifts from training raises no error,
  it just gives worse answers. The slow test checks the real model on a known bulletin.
- **Load once, answer many times.** FastAPI's lifespan loads the model at startup, and the
  LoRA weights are merged into the base model so inference runs one plain model.
- **Never return unchecked model output.** Every answer is parsed into the Pydantic schema:
  invalid answers become a 502 with the raw output.
- **The schema is the documentation.** FastAPI turns the Pydantic models, their field
  descriptions and an example bulletin into Swagger UI at `/docs`.
- **Dynamic batching** (`batcher.py`): requests that arrive within 50 ms share one batch.
- **Prefix caching** (`model.py`): the shared system prompt is computed once. Padding goes
  between the cached prefix and each bulletin, and the attention mask hides it, so the
  answers are the same as uncached, which the slow test checks.
- **Measure before optimising.** The load test showed batching barely helps on a CPU, and
  timing prefill and decode showed why.
- **Inject the model.** `create_app(load=...)` lets the tests swap in a fake model, so
  they run in a second without torch.
