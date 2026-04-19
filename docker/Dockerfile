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

# PyTorch — CUDA 12.8 wheels (compatible with host driver >=595)
RUN pip install --no-cache-dir \
    torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu128

# Transformers from source (Gemma 4 support requires >=4.53.0.dev0)
# timm required for Gemma 4 vision tower (MobileNetV5 via TimmWrapper)
RUN pip install --no-cache-dir \
    "transformers @ git+https://github.com/huggingface/transformers.git@main" \
    accelerate sentencepiece protobuf pillow soundfile librosa scipy numpy \
    timm

# vLLM (latest release)
RUN pip install --no-cache-dir vllm

# Evaluation helpers
RUN pip install --no-cache-dir tqdm jsonlines

# Workspace
WORKDIR /workspace

# Volumes mounted at runtime:
#   /models   — Gemma 4 2B weights
#   /data     — prompt data & assets
#   /solution — agent solution
#   /judge    — judge scripts & eval data
#   /baseline — baseline scripts
