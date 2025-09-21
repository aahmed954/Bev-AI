# Advanced Avatar System - Optimized for RTX 4090
FROM nvidia/cuda:12.1-devel-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV CUDA_HOME=/usr/local/cuda
ENV PATH="${CUDA_HOME}/bin:${PATH}"
ENV LD_LIBRARY_PATH="${CUDA_HOME}/lib64:${LD_LIBRARY_PATH}"

# Install system dependencies
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-dev \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    ninja-build \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    libgstreamer1.0-0 \
    gstreamer1.0-plugins-base \
    gstreamer1.0-plugins-good \
    gstreamer1.0-plugins-bad \
    gstreamer1.0-plugins-ugly \
    libgstreamer-plugins-base1.0-dev \
    libavcodec-dev \
    libavformat-dev \
    libavutil-dev \
    libswscale-dev \
    libavresample-dev \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    libgtk2.0-dev \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

# Create working directory
WORKDIR /app

# Copy requirements first for better caching
COPY requirements-avatar.txt /app/
RUN pip3 install --no-cache-dir -r requirements-avatar.txt

# Install PyTorch with CUDA support (RTX 4090 optimized)
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install Gaussian Splatting (gsplat)
RUN pip3 install gsplat

# Install Bark AI for advanced TTS
RUN pip3 install git+https://github.com/suno-ai/bark.git

# Install additional ML/AI packages
RUN pip3 install \
    transformers>=4.35.0 \
    accelerate \
    bitsandbytes \
    xformers \
    opencv-python \
    soundfile \
    librosa \
    pydub \
    speechrecognition \
    pyaudio

# Install FastAPI and async components
RUN pip3 install \
    fastapi \
    uvicorn[standard] \
    websockets \
    redis \
    pydantic \
    numpy \
    scipy \
    scikit-learn

# Copy avatar system code
COPY . /app/

# Create directories for models and cache
RUN mkdir -p /app/models/metahuman \
    /app/models/emotion \
    /app/models/bark \
    /app/cache/audio \
    /app/cache/renders \
    /app/logs

# Set permissions
RUN chmod +x /app/*.py

# Download and cache models on build (optional, can be done at runtime)
# RUN python3 -c "from transformers import pipeline; pipeline('text-classification', model='j-hartmann/emotion-english-distilroberta-base')"

# Expose WebSocket port
EXPOSE 8091

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8091/health || exit 1

# Set GPU memory allocation for optimal performance
ENV CUDA_VISIBLE_DEVICES=0
ENV TORCH_CUDA_ARCH_LIST="8.9"  # RTX 4090 architecture
ENV PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:512

# Entry point
CMD ["python3", "advanced_avatar_controller.py"]