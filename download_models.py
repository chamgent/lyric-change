# -*- coding: utf-8 -*-
"""Download pretrained models for SoulX-Singer and Preprocessing."""
import os
import sys

# Support mirror endpoints in mainland China if HF is blocked
if "HF_ENDPOINT" not in os.environ:
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

from huggingface_hub import snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
base_models = os.path.join(BASE, "SoulX-Singer", "pretrained_models")
os.makedirs(base_models, exist_ok=True)

print("=== 1. Downloading SoulX-Singer SVS model ===", flush=True)
snapshot_download(
    repo_id="Soul-AILab/SoulX-Singer",
    local_dir=os.path.join(base_models, "SoulX-Singer"),
    resume_download=True,
)

print("=== 2. Downloading SoulX-Singer Preprocess models (ASR, RMVPE, ROSVOT) ===", flush=True)
snapshot_download(
    repo_id="Soul-AILab/SoulX-Singer-Preprocess",
    local_dir=os.path.join(base_models, "SoulX-Singer-Preprocess"),
    resume_download=True,
)

print("=== All models downloaded successfully! ===", flush=True)
