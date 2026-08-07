FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential git && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
RUN pip install --no-cache-dir -e .

EXPOSE 7860

ENV OLLAMA_HOST=http://ollama:11434
ENV MVPDR_AGENT_BACKEND=ollama
ENV OLLAMA_MODEL=qwen3:8b

CMD ["python", "app.py", "--config", "configs/plantdoc_plus.yaml", "--zero_shot", "--port", "7860", "--share", "false"]
