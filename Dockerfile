FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

COPY pyproject.toml README.md LICENSE /app/
COPY lad_mcp_server /app/lad_mcp_server

RUN python -m pip install --no-cache-dir -U pip && \
    python -m pip install --no-cache-dir .

ENTRYPOINT ["lad-mcp-server"]

