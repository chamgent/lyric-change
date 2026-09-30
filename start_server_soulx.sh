#!/bin/bash
# 后台启动 WebUI。需先激活依赖环境（conda/venv），或用 PYTHON=/path/to/python 指定解释器；
# 端口默认 7860，可用 GRADIO_SERVER_PORT 指定。
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python}"
PID_FILE="$SCRIPT_DIR/logs/webui.pid"
LOG_FILE="$SCRIPT_DIR/logs/webui.log"
mkdir -p "$SCRIPT_DIR/logs"

if ! command -v "$PYTHON" >/dev/null 2>&1; then
    echo "找不到 Python 解释器 '$PYTHON'：请先激活环境，或设置 PYTHON=/path/to/python" >&2
    exit 1
fi

# 只停止本脚本上次启动的进程（按 pid 文件），不误杀 Docker 容器或其他项目的 app.py
if [ -f "$PID_FILE" ]; then
    OLD_PID="$(cat "$PID_FILE")"
    if kill -0 "$OLD_PID" 2>/dev/null && grep -q "app.py" "/proc/$OLD_PID/cmdline" 2>/dev/null; then
        echo "停止旧进程 PID $OLD_PID"
        kill "$OLD_PID"
        for _ in $(seq 1 20); do
            kill -0 "$OLD_PID" 2>/dev/null || break
            sleep 0.5
        done
    fi
    rm -f "$PID_FILE"
fi

PORT="${GRADIO_SERVER_PORT:-7860}"
# 端口已被其他程序占用时直接报错；否则下面的就绪检测会把别人的服务误判为启动成功
if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
    echo "端口 $PORT 已被其他程序占用，可用 GRADIO_SERVER_PORT=<端口> 换一个端口" >&2
    exit 1
fi

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR/SoulX-Singer:$PYTHONPATH"
export GRADIO_SERVER_PORT="$PORT"

nohup "$PYTHON" app.py > "$LOG_FILE" 2>&1 < /dev/null &
PID=$!
echo "$PID" > "$PID_FILE"

# 等待端口就绪；进程提前退出则打印日志并返回失败
for _ in $(seq 1 120); do
    if ! kill -0 "$PID" 2>/dev/null; then
        echo "WebUI 启动失败，日志末尾：" >&2
        tail -n 30 "$LOG_FILE" >&2
        rm -f "$PID_FILE"
        exit 1
    fi
    if (exec 3<>"/dev/tcp/127.0.0.1/$PORT") 2>/dev/null; then
        echo "SoulX WebUI started with PID $PID -> http://localhost:$PORT (日志: $LOG_FILE)"
        exit 0
    fi
    sleep 1
done
echo "WebUI 进程 $PID 仍在运行，但 120s 内端口 $PORT 未就绪，请查看 $LOG_FILE" >&2
exit 1
