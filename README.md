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
answer takes 10 to 30 seconds.

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
- **Inject the model.** `create_app(load=...)` lets the tests swap in a fake model, so
  they run in a second without torch.
