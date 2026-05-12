# Industrial Defect Detection System

Real-time surface defect detection for manufacturing quality control — built with YOLOv9, FastAPI, and Next.js.

## What it does

Automatically detects surface defects on magnetic tile components used in electric motors and industrial machinery. Replaces manual visual inspection with a computer vision system that runs in ~38ms per image.

**Input:** Photo of a magnetic tile surface

**Output:** Detected defects with bounding boxes, class labels, and confidence scores

## Defect Classes

| Class | Description | Test mAP50 |
|---|---|---|
| Free | No defect — clean surface | 99.3% |
| Crack | Fine surface cracks from stress or cooling | 98.0% |
| Uneven | Inconsistent surface coating or texture | 97.4% |
| Blowhole | Circular holes from trapped gas during casting | 97.3% |
| Fray | Worn or frayed edges from surface degradation | 90.2% |
| Break | Physical fractures through the material | 77.9% |

## Model Performance

| Metric | Score |
|---|---|
| mAP50 | **93.35%** |
| mAP50-95 | 93.35% |
| Precision | 85.30% |
| Recall | 89.87% |
| Inference speed | ~38ms per image |

Trained with YOLOv9c — 358 layers, 25.3M parameters, 102 GFLOPs

## Architecture

```
Image upload (drag-drop)
      ↓
FastAPI backend
      ↓
YOLOv9 inference (PyTorch)
      ↓
Bounding box drawing (OpenCV)
      ↓
Base64 annotated image + detections JSON
      ↓
Next.js frontend
      ↓
Annotated image + detection table + confidence bars
```

## Data Pipeline

### Problem — Severe class imbalance

The raw dataset had a 29.8x imbalance ratio:

```
Free     : 1,904 images  ← dominant class
Blowhole :   230 images
Uneven   :   206 images
Break    :   170 images
Crack    :   114 images
Fray     :    64 images  ← minority class
```

### Solution — Augmentation strategy

```
Original images  : 2,688
After augmentation: 4,404
Imbalance ratio  : 29.8x → 3.8x

Augmentations applied to minority classes:
  - Horizontal flip
  - Vertical flip
  - Random brightness (0.7x–1.3x)
  - Random contrast (0.7x–1.3x)
  - Random rotation (±15°)
  - Gaussian blur (20% probability)

All images resized to 640×640 with letterbox padding
```

### Train / Val / Test split

| Split | Images |
|---|---|
| Train | 3,082 |
| Val | 880 |
| Test | 442 |

## Training

Trained on Google Colab T4 GPU:

```
Model          : YOLOv9c (pretrained on COCO)
Epochs         : 50
Batch size     : 16
Image size     : 640×640
Optimizer      : AdamW (lr=0.001)
FP16           : enabled
Early stopping : patience=15
Training time  : ~2.5 hours
```

## Tech Stack

| Layer | Technology |
|---|---|
| Detection model | YOLOv9c (Ultralytics) |
| Training | PyTorch + Google Colab T4 |
| Backend | FastAPI + uvicorn |
| Image processing | OpenCV + PIL |
| Frontend | Next.js 14 + TypeScript + Tailwind |
| Experiment tracking | MLflow |
| Data augmentation | PIL + NumPy |

## Project Structure

```
defect_detection/
├── backend/
│   └── app/
│       └── main.py              ← FastAPI + YOLOv9 inference
├── frontend/
│   └── defect-ui/
│       └── app/
│           └── page.tsx         ← Next.js drag-drop UI
├── notebooks/
│   └── phase1_data_preparation.ipynb
├── data/
│   ├── raw/                     ← not in git
│   ├── processed/               ← not in git
│   └── stats/
├── models/                      ← not in git (51.6MB)
│   └── best.pt
├── mlruns/                      ← not in git
└── scripts/
```

## Setup

### 1 — Create conda environment

```bash
conda create -n ner_env python=3.10 -y
conda activate ner_env
pip install ultralytics fastapi uvicorn opencv-python-headless Pillow
```

### 2 — Download model weights

Train using Colab notebook or download `best.pt` and place at:
```
models/best.pt
```

### 3 — Start MLflow

```bash
cd G:\defect_detection
mlflow server --host 127.0.0.1 --port 5000 \
  --backend-store-uri sqlite:///mlruns/mlflow.db \
  --default-artifact-root ./mlruns/artifacts
```

### 4 — Start FastAPI backend

```bash
cd backend/app
uvicorn main:app --reload --host 127.0.0.1 --port 8003
```

API docs → http://127.0.0.1:8003/docs

### 5 — Start Next.js frontend

```bash
cd frontend/defect-ui
npm install
npm run dev -- --port 3002
```

UI → http://localhost:3002

## API Reference

### GET /health

```json
{
  "status": "ok",
  "model": "YOLOv9c",
  "map50": "93.35%",
  "classes": ["Blowhole", "Break", "Crack", "Fray", "Free", "Uneven"],
  "version": "1.0.0"
}
```

### POST /detect

Upload an image file with optional confidence threshold.

```bash
curl -X POST "http://127.0.0.1:8003/detect?confidence=0.25" \
  -F "file=@surface_image.jpg"
```

Response:
```json
{
  "detections": [
    {
      "class_name": "Blowhole",
      "confidence": 97.8,
      "bbox": [0.001, 0.0002, 0.999, 1.0],
      "color": "#E74C3C"
    }
  ],
  "detection_count": 1,
  "defect_count": 1,
  "image_base64": "...",
  "width": 491,
  "height": 376
}
```

## Dataset

**Source:** Magnetic Tile Defect Dataset

```
G:\defect_detection\data\raw\Magnetic-tile-defect-datasets.-master\
├── MT_Blowhole\Imgs\  (230 images)
├── MT_Break\Imgs\     (170 images)
├── MT_Crack\Imgs\     (114 images)
├── MT_Fray\Imgs\      (64 images)
├── MT_Free\Imgs\      (1904 images)
└── MT_Uneven\Imgs\    (206 images)
```

Images vary in size from 121×289 to 592×247 pixels — all resized to 640×640 during preprocessing.

## MLflow Tracking

All experiments tracked with MLflow:

- Phase 1: Dataset statistics (image counts, imbalance ratio, augmentation results)
- Phase 2: Training metrics (loss curves, mAP per epoch, per-class performance)

View at: http://127.0.0.1:5000

## Real-world Applications

This system can be adapted for:

- Steel surface inspection
- PCB (printed circuit board) defect detection
- Pharmaceutical tablet inspection
- Textile defect detection
- Solar panel crack detection
- Automotive paint inspection

## Author

Hammad Aslam — Data Engineer + ML Engineer

Specializations: Machine Learning, Deep Learning, Computer Vision, NLP, PyTorch