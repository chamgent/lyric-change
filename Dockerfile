FROM pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    GRADIO_SERVER_NAME="0.0.0.0" \
    GRADIO_SERVER_PORT=7860 \
    HF_ENDPOINT="https://hf-mirror.com"

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Clone upstream SoulX-Singer
RUN git clone https://github.com/Soul-AILab/SoulX-Singer.git /app/SoulX-Singer

# Install Python dependencies
COPY requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy application source code
COPY core/ /app/core/
COPY app.py /app/app.py
COPY run_pipeline.py /app/run_pipeline.py
COPY download_models.py /app/download_models.py
COPY docker-entrypoint.sh /app/docker-entrypoint.sh

RUN chmod +x /app/docker-entrypoint.sh

EXPOSE 7860

ENTRYPOINT ["/app/docker-entrypoint.sh"]
CMD ["python", "app.py"]
