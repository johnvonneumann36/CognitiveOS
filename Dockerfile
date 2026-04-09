FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    COGNITIVEOS_DB_PATH=/app/data/cognitiveos.db \
    COGNITIVEOS_MEMORY_OUTPUT_PATH=/app/MEMORY.MD

WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY docs ./docs
COPY data/.gitkeep ./data/.gitkeep

RUN pip install --no-cache-dir .

EXPOSE 8000

CMD ["cognitiveos-mcp", "--transport", "streamable-http", "--host", "0.0.0.0", "--port", "8000", "--path", "/mcp"]
