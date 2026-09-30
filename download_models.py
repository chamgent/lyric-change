# -*- coding: utf-8 -*-
"""Download pretrained models for SoulX-Singer and Preprocessing.

下载源：若设置了 HF_ENDPOINT 则只用它；否则先试 hf-mirror.com（国内快），
失败再回退到 huggingface.co。hf-mirror 对境外 IP 会 308 跳转到 huggingface.co，
而 huggingface_hub 不跟随跨域跳转，所以境外服务器必须能回退。
"""
import os
import sys

from huggingface_hub import get_hf_file_metadata, hf_hub_url, snapshot_download

BASE = os.path.dirname(os.path.abspath(__file__))
base_models = os.path.join(BASE, "SoulX-Singer", "pretrained_models")
os.makedirs(base_models, exist_ok=True)

if os.environ.get("HF_ENDPOINT"):
    ENDPOINTS = [os.environ["HF_ENDPOINT"]]
else:
    ENDPOINTS = ["https://hf-mirror.com", "https://huggingface.co"]


def pick_endpoint(repo_id):
    """返回第一个能正常访问的下载源。

    不能靠 snapshot_download 抛异常来判断：连不上时它会静默返回 local_dir 里已有的
    （可能是下了一半的）文件，回退永远不会触发。也不能用 model_info 探测：它是 GET，
    会跟随 hf-mirror 的跳转而成功。这里用与文件下载相同的 HEAD 请求，并像
    hf_hub_download 一样要求返回 commit_hash / etag。
    """
    for endpoint in ENDPOINTS:
        try:
            meta = get_hf_file_metadata(
                hf_hub_url(repo_id, ".gitattributes", endpoint=endpoint), timeout=15
            )
            if meta.commit_hash and meta.etag:
                return endpoint
            reason = "响应缺少 commit_hash/etag（可能被跳转到其他域名）"
        except Exception as e:
            reason = type(e).__name__
        print(f"    {endpoint} 不可用: {reason}", file=sys.stderr, flush=True)
    sys.exit(f"所有下载源均不可用: {ENDPOINTS}（可设置 HF_ENDPOINT 指定）")


def download(repo_id, local_dir):
    endpoint = pick_endpoint(repo_id)
    print(f"    from {endpoint}", flush=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=local_dir,
        endpoint=endpoint,
        resume_download=True,
    )


print("=== 1. Downloading SoulX-Singer SVS model ===", flush=True)
download("Soul-AILab/SoulX-Singer", os.path.join(base_models, "SoulX-Singer"))

print("=== 2. Downloading SoulX-Singer Preprocess models (ASR, RMVPE, ROSVOT) ===", flush=True)
download(
    "Soul-AILab/SoulX-Singer-Preprocess",
    os.path.join(base_models, "SoulX-Singer-Preprocess"),
)

print("=== 3. Downloading NLTK data for English G2P ===", flush=True)
sys.path.insert(0, BASE)
from core.nltk_data import ensure_english_g2p_data  # noqa: E402

if not ensure_english_g2p_data():
    sys.exit("NLTK 数据下载失败（英文注音需要），请检查网络后重试")

print("=== All models downloaded successfully! ===", flush=True)
