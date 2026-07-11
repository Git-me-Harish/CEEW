#!/usr/bin/env bash
# Start the PashuMitra YOLO Detector Python service
# Usage: bash start.sh
#
# Place version4.pt at ./models/version4.pt before starting.

set -e
cd "$(dirname "$0")"

PORT="${PORT:-8501}"
LOG_FILE="./service.log"
PID_FILE="./service.pid"

# Check if already running
if [ -f "$PID_FILE" ] && kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
  echo "Service already running with PID $(cat $PID_FILE)"
  exit 0
fi

# Use the venv python (which has ultralytics installed)
PYTHON="/home/z/.venv/bin/python"
if [ ! -x "$PYTHON" ]; then
  PYTHON="python3"
fi

echo "Starting PashuMitra YOLO Detector on port $PORT..."
echo "Python: $PYTHON"
echo "Model path: $(pwd)/models/version4.pt"
echo "Model exists: $([ -f ./models/version4.pt ] && echo 'YES' || echo 'NO — service will start in degraded mode')"

# Start in background, log to file
nohup "$PYTHON" -m uvicorn main:app --host 0.0.0.0 --port "$PORT" > "$LOG_FILE" 2>&1 &
echo $! > "$PID_FILE"

sleep 3
if kill -0 "$(cat $PID_FILE)" 2>/dev/null; then
  echo "Service started. PID: $(cat $PID_FILE)"
  echo "Health check: curl http://localhost:$PORT/health"
  echo "Logs: tail -f $LOG_FILE"
else
  echo "Service failed to start. Check $LOG_FILE"
  cat "$LOG_FILE" | tail -20
  rm -f "$PID_FILE"
  exit 1
fi
