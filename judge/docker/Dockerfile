FROM nvidia/cuda:12.8.1-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 python3-pip python3-venv python3-dev \
    git wget curl ffmpeg libsndfile1 libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Create venv so pip doesn't complain about externally-managed
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# PyTorch — CUDA 12.8 wheels (compatible with host driver >=595).
# Versions pinned for reproducibility.
RUN pip install --no-cache-dir \
    torch==2.10.0 torchvision==0.25.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu128

# Transformers from source (Gemma 4 support requires >=4.53.0.dev0).
# Pinned to the commit SHA reflected in the currently-built image.
# timm required for Gemma 4 vision tower (MobileNetV5 via TimmWrapper).
RUN pip install --no-cache-dir \
    "transformers @ git+https://github.com/huggingface/transformers.git@a29df2d916e3b820aecd19d3b5a877abc523ba3c" \
    accelerate==1.13.0 \
    sentencepiece==0.2.1 \
    protobuf==6.33.6 \
    pillow==12.1.1 \
    soundfile==0.13.1 \
    librosa==0.11.0 \
    scipy==1.17.1 \
    numpy==2.2.6 \
    timm==1.0.26

# vLLM (pinned)
RUN pip install --no-cache-dir vllm==0.19.1

# Evaluation helpers
RUN pip install --no-cache-dir tqdm==4.67.3 jsonlines==4.0.0

# Server deps for baseline/solution
RUN pip install --no-cache-dir fastapi==0.136.0 "uvicorn[standard]==0.44.0"

# Workspace
WORKDIR /workspace

# Volumes mounted at runtime:
#   /data     — prompt data & assets (agent + judge subtrees)
#   /solution — agent solution
#   /judge    — judge scripts + baseline
