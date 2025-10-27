# Multi-stage: builder
FROM python:3.10-slim AS builder
WORKDIR /install

COPY requirements.txt .

RUN apt-get update \
    && apt-get install -y --no-install-recommends python3-venv ca-certificates wget \
    && python -m venv /opt/venv \
    && /opt/venv/bin/pip install --upgrade pip setuptools wheel

# Зависимости без torch
RUN /opt/venv/bin/pip install --no-cache-dir \
    chess>=1.0.0 \
    numpy>=1.21.0 \
    pandas>=1.3.0 \
    tqdm>=4.0.0 \
    scikit-learn>=1.0.0 \
    fastapi>=0.100.0 \
    uvicorn[standard]>=0.20.0 \
    python-multipart>=0.0.6 \
    requests>=2.25.0

# CPU-only PyTorch
RUN /opt/venv/bin/pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch torchvision

# Multi-stage: runtime
FROM python:3.10-slim
WORKDIR /app

# Копируем только виртуальное окружение
COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Системные зависимости для PyTorch
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY src/ ./src/
COPY notebook/best_chess_model.pt ./notebook/

EXPOSE 8000
CMD ["uvicorn", "src.main:app", "--host", "0.0.0.0", "--port", "8000"]