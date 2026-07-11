# PashuMitra YOLO Detector — Setup Guide

This Python micro-service loads your trained YOLO model (`version4.pt`) and exposes
it as an HTTP endpoint that the Next.js app calls for breed detection.

## Architecture

```
Next.js /api/classify
       │
       ├── 1. POST http://localhost:8501/detect  (YOLO — primary)
       │      ↓ returns { primary, allDetections, annotatedImage }
       │
       ├── 2. ZAI Vision LLM call (VLM — secondary refinement)
       │      ↓ receives YOLO's hint, validates independently
       │
       └── 3. Consensus scoring
              ↓ combines both, boosts confidence on agreement
              ↓ returns hybrid result to UI
```

## Step 1: Place your model

Drop your `version4.pt` file at:

```
/home/z/my-project/python-services/yolo-detector/models/version4.pt
```

The service auto-detects the file on next restart.

## Step 2: Restart the service

```bash
# Stop existing service (if running)
cd /home/z/my-project/python-services/yolo-detector
[ -f service.pid ] && kill $(cat service.pid) 2>/dev/null
rm -f service.pid

# Start fresh
bash start.sh
```

You should see:
```
Model exists: YES
Service started. PID: XXXX
```

## Step 3: Verify

```bash
curl http://localhost:8501/health
```

Expected response (model loaded):
```json
{
  "status": "ok",
  "model_loaded": true,
  "classes": ["Gir", "Sahiwal", "Murrah Buffalo", ...],
  "error": null
}
```

## Step 4: Test in the UI

1. Open the app at the preview URL
2. Go to **Breed Classifier** tab
3. The "Hybrid Detection Pipeline" banner should now show:
   - YOLO Model (Primary): **Online** (green dot)
   - VLM Refinement (Secondary): Online (green dot)
4. Upload a cattle/buffalo photo and click "Run Hybrid Detection"
5. The result panel shows:
   - Primary breed with consensus confidence
   - YOLO confidence + detected class
   - VLM confidence + identified breed
   - YOLO annotated image (with bounding box)
   - VLM-observed visual characteristics

## Endpoints

| Method | Path         | Description                                  |
|--------|--------------|----------------------------------------------|
| GET    | `/`          | Service info & model status                  |
| GET    | `/health`    | Health check with model status & class list  |
| POST   | `/detect`    | Run YOLO detection on uploaded image         |

## Configuration

Environment variables (set in start.sh or shell):

| Variable        | Default                  | Description                  |
|-----------------|--------------------------|------------------------------|
| `PORT`          | `8501`                   | HTTP port for the service    |
| `YOLO_MODEL_PATH` | `./models/version4.pt` | Path to the YOLO model file  |

## Troubleshooting

**"Model file not found"**: Place `version4.pt` at `models/version4.pt` and restart.

**"Failed to load YOLO model"**: Check `service.log` for details. Common causes:
- Corrupt model file (re-download from your training output)
- Ultralytics version mismatch (the service uses ultralytics 8.4.92)
- Missing dependencies (run `/home/z/.venv/bin/pip3 install ultralytics`)

**Service unreachable from Next.js**: Verify it's running:
```bash
curl http://localhost:8501/health
```
If not running, start it: `bash /home/z/my-project/python-services/yolo-detector/start.sh`

**Next.js falls back to VLM-only**: This is by design. If YOLO is unavailable,
the classifier gracefully degrades to VLM-only mode and shows the status in the UI.

## Dependencies

The service uses the `/home/z/.venv/` Python virtualenv which has:
- ultralytics 8.4.92
- torch 2.13.0+cpu (CPU-only build, no CUDA needed)
- fastapi + uvicorn
- PIL (Pillow)

To reinstall if needed:
```bash
/home/z/.venv/bin/pip3 install -r requirements.txt
/home/z/.venv/bin/pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
