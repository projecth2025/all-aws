# Use NVIDIA CUDA base image
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# Install Python 3.11
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3-pip \
    libgl1 \
    libglib2.0-0 \
    libgomp1 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# Runtime environment
ENV PYTHONUNBUFFERED=1 \
    PORT=8080 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/opt/paddle_models \
    TMPDIR=/tmp

# Install PaddlePaddle GPU from official repository (NOT PyPI)
# CUDA 12.2 is compatible with CUDA 11.8 packages
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir paddlepaddle-gpu==3.2.0 -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# Install other Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Create directories
RUN mkdir -p /opt/paddle_models /tmp

# App code
COPY main.py .

# Health check with longer start period for model loading
HEALTHCHECK --interval=30s --timeout=3s --start-period=120s \
    CMD python -c "import requests; requests.get('http://localhost:8080/health', timeout=2)"

# Single-worker server
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8080", "--workers", "1", "--timeout-keep-alive", "120"]