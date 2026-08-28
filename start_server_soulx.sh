#!/bin/bash
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# stop old process
pkill -f 'python app.py' 2>/dev/null || true
sleep 1

export PYTHONUNBUFFERED=1
export PYTHONPATH="$SCRIPT_DIR/SoulX-Singer:$PYTHONPATH"

mkdir -p "$SCRIPT_DIR/logs"
nohup python app.py > "$SCRIPT_DIR/logs/webui.log" 2>&1 < /dev/null &
echo "SoulX WebUI started with PID $!"
