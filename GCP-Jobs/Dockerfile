# ---------------------------------------------------------------------
# BASE IMAGE (UNCHANGED)
# ---------------------------------------------------------------------
FROM nvidia/cuda:12.2.0-runtime-ubuntu22.04

# ---------------------------------------------------------------------
# SYSTEM DEPS + PYTHON 3.11 (UNCHANGED)
# ---------------------------------------------------------------------
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

RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

WORKDIR /app

# ---------------------------------------------------------------------
# ENV (RUNTIME GPU ENABLED)
# ---------------------------------------------------------------------
ENV PYTHONUNBUFFERED=1 \
    CUDA_VISIBLE_DEVICES=0 \
    NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility \
    HOME=/opt/paddle_models \
    TMPDIR=/tmp

# ---------------------------------------------------------------------
# INSTALL PADDLE GPU (UNCHANGED, REQUIRED FOR RUNTIME)
# ---------------------------------------------------------------------
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir paddlepaddle-gpu==3.2.0 \
        -i https://www.paddlepaddle.org.cn/packages/stable/cu118/

# ---------------------------------------------------------------------
# PYTHON DEPS
# ---------------------------------------------------------------------
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ---------------------------------------------------------------------
# CREATE DIRECTORIES
# ---------------------------------------------------------------------
RUN mkdir -p /opt/paddle_models /tmp

# ---------------------------------------------------------------------
# APP CODE
# ---------------------------------------------------------------------
COPY main.py .

# ---------------------------------------------------------------------
# ENTRYPOINT (CLOUD RUN JOB)
# ---------------------------------------------------------------------
CMD ["python", "main.py"]