FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY server/ server/
COPY notebooks/model/ models/

EXPOSE 12345

ENV HOST=0.0.0.0 PORT=12345 MODEL_DIR=/app/models DATA_DIR=/app/data

CMD ["python", "-u", "server/server.py"]