# Person Re-Identification with Siamese Network
# Dockerfile for containerized training and inference

# Use PyTorch base image with CUDA support
FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

LABEL maintainer="Person ReID Team"
LABEL description="Person Re-Identification with Siamese Network"

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    wget \
    unzip \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Install UV package manager
RUN curl -fsSL https://astral.sh/uv/install.sh | sh
ENV PATH="$PATH:/root/.cargo/bin"

# Copy requirements first to leverage Docker cache
COPY requirements.txt .

# Install Python dependencies
RUN uv pip install --system -r requirements.txt

# Copy project files
COPY . .

# Create necessary directories
RUN mkdir -p data checkpoints results logs

# Make scripts executable
RUN chmod +x scripts/*.py scripts/*.sh

# Set environment variables
ENV PYTHONPATH="$PYTHONPATH:/app"
ENV PYTHONUNBUFFERED=1

# Default command to show help
CMD ["/bin/bash", "-c", "make help"]
