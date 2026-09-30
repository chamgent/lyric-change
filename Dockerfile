FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone upstream SoulX-Singer (pinned to a known-good commit)
ARG SOULX_COMMIT=81aeb3ae772c70093c3de74dc23c92d983801ae4
RUN git clone https://github.com/Soul-AILab/SoulX-Singer.git /app/SoulX-Singer \
    && git -C /app/SoulX-Singer checkout ${SOULX_COMMIT}

# Install Python dependencies
COPY requirements.txt constraints.txt /app/
RUN pip install --no-cache-dir -r requirements.txt -c constraints.txt

# Copy application source code
COPY core/ /app/core/
COPY app.py /app/app.py
COPY run_pipeline.py /app/run_pipeline.py
COPY download_models.py /app/download_models.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

# Smoke test: fail the build early if the torch stack / upstream imports are broken
RUN python -c "import torch, torchaudio; assert torch.__version__.startswith('2.2.0'), torch.__version__; import core.lyric_replace, core.soulx_engine, core.soulx_preprocess"

EXPOSE 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
