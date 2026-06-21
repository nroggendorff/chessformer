FROM public.ecr.aws/deep-learning-containers/pytorch-training:2.10.0-gpu-py313-cu130-ubuntu22.04-sagemaker

RUN apt-get update && apt-get install -y stockfish && rm -rf /var/lib/apt/lists/*

WORKDIR /app/

RUN pip install --no-cache-dir python-chess tqdm flask

COPY . .

CMD ["python", "train.py"]
