FROM python:3.10-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    unzip tar && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /kaggle/working

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir kaggle

COPY . .

RUN mkdir /root/.kaggle

RUN --mount=type=secret,id=kkey,target=/root/.kaggle/kaggle.json \
    mkdir -p /kaggle/input/chesscom-user-games-60000-games && \
    mkdir -p /kaggle/input/stockfish/other/binary/1 && \
    kaggle datasets download -p /kaggle/input/chesscom-user-games-60000-games adityajha1504/chesscom-user-games-60000-games && \
    unzip /kaggle/input/chesscom-user-games-60000-games/*.zip -d /kaggle/input/chesscom-user-games-60000-games/ && \
    rm -f /kaggle/input/chesscom-user-games-60000-games/*.zip && \
    kaggle models instances versions download -p /kaggle/input/stockfish/other/binary/1 nroggendorff/stockfish/other/binary/1 && \
    tar -xvzf /kaggle/input/stockfish/other/binary/1/*.tar.gz -C /kaggle/input/stockfish/other/binary/1/ && \
    rm -f /kaggle/input/stockfish/other/binary/1/*.tar.gz

EXPOSE 8888

CMD ["jupyter", "notebook", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root"]
