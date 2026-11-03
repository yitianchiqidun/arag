# The assistant's HTTP service and demo page (stage 9). TiDB is not in the image: start it on
# the host with `python -m research_agent.tidb start` (see README, "Docker").
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    HF_HOME=/models \
    PATH="/app/.venv/bin:$PATH"

RUN pip install --no-cache-dir uv==0.12.23

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src src
COPY research_agent research_agent
# Resolved from pyproject.toml (uv.lock is not tracked); torch comes from the CUDA 12.8 index
# set in [tool.uv.sources].
RUN uv sync --no-dev \
    --extra full --extra research --extra agent --extra tidb --extra service --extra viewer
COPY apps apps
COPY configs configs

EXPOSE 8000 8501
CMD ["uvicorn", "research_agent.service.app:app", "--host", "0.0.0.0", "--port", "8000"]
