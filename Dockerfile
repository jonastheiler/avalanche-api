FROM python:3.12-slim
COPY --from=ghcr.io/astral-sh/uv:0.9.10 /uv /bin/uv

# Hugging Face Spaces runs the container as user 1000.
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/app/.venv/bin:$PATH \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy
WORKDIR /home/user/app

# Dependencies first, the code second: editing the code doesn't reinstall torch.
COPY --chown=user pyproject.toml uv.lock ./
RUN uv sync --frozen --no-dev --no-install-project
COPY --chown=user README.md ./
COPY --chown=user src ./src
RUN uv sync --frozen --no-dev

# Bake the models into the image: a restart doesn't download 3.4 GB again, and the
# offline flag makes a missing file fail loudly instead of fetching something else.
RUN python -c "from avalanche_api.model import download; download()"
ENV HF_HUB_OFFLINE=1

EXPOSE 7860
CMD ["uvicorn", "avalanche_api.app:app", "--host", "0.0.0.0", "--port", "7860"]
