"""The HTTP API: POST /extract turns an avalanche bulletin into structured avalanche problems.

uv run uvicorn avalanche_api.app:app    # then open http://127.0.0.1:8000/docs

MAX_BATCH_SIZE (default 8) caps how many requests run through the model together; 1 turns
batching off.
"""

import os
from collections.abc import Callable
from contextlib import asynccontextmanager
from typing import Protocol

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse, RedirectResponse
from pydantic import ValidationError

from avalanche_api.batcher import Batcher
from avalanche_api.schema import Bulletin, ExtractRequest, ModelOutputError

DESCRIPTION = """\
Reads the avalanche problem descriptions of a Swiss avalanche bulletin (SLF) and returns
each problem's type, danger level, aspects and elevations as JSON.

The model is [Qwen3-1.7B with a LoRA adapter](https://huggingface.co/jth00/qwen3-1.7b-avalanche-lora)
fine-tuned on 1,000 bulletins. On bulletins from winter 25/26 it scores the same as Claude
Haiku: type + level F1 0.612. How it was trained and compared:
[avalanche-llm](https://github.com/jonastheiler/avalanche-llm).

Open **POST /extract**, click *Try it out*, then *Execute*. On a laptop CPU an
answer takes about 30 seconds, up to a minute for long bulletins.
"""


class Model(Protocol):
    def generate(self, text: str) -> str: ...


def load_model() -> Model:
    # Imported here so the tests, which use a fake model, never import torch.
    from avalanche_api.model import Extractor

    max_size = int(os.environ.get("MAX_BATCH_SIZE", "8"))
    return Batcher(Extractor().generate_batch, max_size=max_size)


def create_app(load: Callable[[], Model] = load_model) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.model = load()  # Once at startup: loading takes far longer than an answer.
        yield

    app = FastAPI(
        title="Avalanche bulletin API", version="0.1.0", description=DESCRIPTION, lifespan=lifespan
    )

    @app.get("/", include_in_schema=False)
    def root() -> RedirectResponse:
        return RedirectResponse("/docs")

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    # A plain def, not async def: FastAPI runs it in a thread pool, so a request waiting for
    # its batch doesn't block the server, and other requests can join that batch.
    @app.post(
        "/extract",
        response_model=Bulletin,
        responses={502: {"model": ModelOutputError, "description": "Invalid model answer"}},
    )
    def extract(body: ExtractRequest, request: Request):
        output = request.app.state.model.generate(body.text)
        # Never pass on unchecked model output: it must parse into the schema.
        try:
            return Bulletin.model_validate_json(output)
        except ValidationError:
            error = ModelOutputError(
                detail="The model's answer is not a valid Bulletin.", output=output
            )
            return JSONResponse(status_code=502, content=error.model_dump())

    return app


app = create_app()
