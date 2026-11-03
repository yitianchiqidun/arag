# The assistant's HTTP service and demo page (stage 9). One image for both services.
#
# - CPU torch: the container only encodes queries; the docs index is built outside and mounted
#   read-only (README, "Docker").
# - TiDB is not in the image: start it on the host with `python -m research_agent.tidb start`.
# - Two stages: uv and the build tools stay in the first one.

FROM python:3.12-slim AS build
ENV UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=never \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"
RUN pip install --no-cache-dir uv==0.12.23
WORKDIR /app
RUN uv venv /app/.venv
# CPU wheel first, so the project's `torch>=2.4` is already satisfied and no CUDA wheel is
# pulled; --no-sources ignores the cu128 index pinned in [tool.uv.sources] for local use.
RUN uv pip install --no-cache torch==2.11.0 --index-url https://download.pytorch.org/whl/cpu
COPY pyproject.toml README.md ./
COPY src src
COPY research_agent research_agent
COPY scripts scripts
RUN uv pip install --no-cache --no-sources -e ".[full,research,agent,tidb,service,viewer]" \
    && python -c "import torch; assert torch.version.cuda is None, torch.__version__" \
    && find /app/.venv -name "__pycache__" -prune -exec rm -rf {} + \
    && find /app/.venv -path "*/site-packages/*/tests" -type d -prune -exec rm -rf {} +
# tiktoken downloads its encodings on first use; fetch them now so the container needs no
# network for them.
ENV TIKTOKEN_CACHE_DIR=/app/.tiktoken
RUN python -c "import tiktoken; tiktoken.encoding_for_model('gpt-4o'); tiktoken.get_encoding('cl100k_base')"

FROM python:3.12-slim
LABEL project=tidb-ops-assistant
ENV PYTHONUNBUFFERED=1 \
    VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH" \
    TIKTOKEN_CACHE_DIR=/app/.tiktoken \
    HF_HOME=/models \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1 \
    EMBEDDING_DEVICE=cpu
WORKDIR /app
COPY --from=build /app /app
COPY apps apps
COPY configs configs
EXPOSE 8000 8501
CMD ["uvicorn", "research_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
