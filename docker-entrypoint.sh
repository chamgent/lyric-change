#!/bin/bash
set -e

MODELS_DIR="/app/SoulX-Singer/pretrained_models"

# Automatically download models on first run if not mounted or empty
if [ ! -d "$MODELS_DIR/SoulX-Singer" ] || [ ! -d "$MODELS_DIR/SoulX-Singer-Preprocess" ]; then
    echo "=========================================================="
    echo " Pretrained models not found. Downloading automatically..."
    echo "=========================================================="
    python /app/download_models.py
fi

exec "$@"
