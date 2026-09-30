#!/bin/bash
set -e

MODELS_DIR="/app/SoulX-Singer/pretrained_models"
OUTPUTS_DIR="/app/outputs"

# 以挂载目录在宿主机上的属主身份运行，避免在 ./outputs、./pretrained_models 里
# 留下 root 属主的文件。宿主机目录不存在时 Docker 会以 root 创建，此时保持 root。
if [ "$(id -u)" = "0" ]; then
    RUN_UID=0
    RUN_GID=0
    for d in "$OUTPUTS_DIR" "$MODELS_DIR"; do
        if [ -d "$d" ] && [ "$(stat -c %u "$d")" != "0" ]; then
            RUN_UID="$(stat -c %u "$d")"
            RUN_GID="$(stat -c %g "$d")"
            break
        fi
    done
    if [ "$RUN_UID" != "0" ]; then
        # 另一个挂载目录若是 Docker 刚以 root 创建的空目录，交给同一属主，否则无法写入
        for d in "$OUTPUTS_DIR" "$MODELS_DIR"; do
            if [ -d "$d" ] && [ "$(stat -c %u "$d")" = "0" ] && [ -z "$(ls -A "$d")" ]; then
                chown "$RUN_UID:$RUN_GID" "$d"
            fi
        done
        echo "Running as uid=$RUN_UID gid=$RUN_GID (owner of the mounted directories)"
        exec setpriv --reuid="$RUN_UID" --regid="$RUN_GID" --clear-groups \
            env HOME=/tmp "$0" "$@"
    fi
fi

# Automatically download models on first run if not mounted or empty
if [ ! -d "$MODELS_DIR/SoulX-Singer" ] || [ ! -d "$MODELS_DIR/SoulX-Singer-Preprocess" ]; then
    echo "=========================================================="
    echo " Pretrained models not found. Downloading automatically..."
    echo "=========================================================="
    python /app/download_models.py
fi

exec "$@"
