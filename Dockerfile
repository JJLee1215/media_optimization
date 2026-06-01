FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu

RUN pip3 install numpy pandas scikit-learn matplotlib seaborn jupyterlab xgboost \
    fastapi uvicorn[standard]