FROM pytorch/pytorch:2.14.0-cuda13.0-cudnn9-runtime

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_NO_CACHE_DIR=1
ENV PIP_BREAK_SYSTEM_PACKAGES=1
ENV PATH="/usr/games:${PATH}"
ENV CHESSFORMER_MODEL_DIR=/app/model

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        stockfish \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

RUN pip install --no-cache-dir python-chess datasets safetensors tqdm flask berserk

COPY . .

RUN mkdir -p "$CHESSFORMER_MODEL_DIR" && chmod 777 "$CHESSFORMER_MODEL_DIR"

EXPOSE 5000
CMD ["python", "train.py"]
