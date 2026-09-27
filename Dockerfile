FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Install python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code and dataset
COPY . .

ENV HOST=0.0.0.0
ENV PORT=8080

EXPOSE 8080

CMD ["sh", "-c", "uvicorn bot:app --host 0.0.0.0 --port ${PORT:-8080}"]
