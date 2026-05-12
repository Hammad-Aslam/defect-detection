from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from pathlib import Path
from ultralytics import YOLO
import numpy as np
import cv2
import base64
import io
from PIL import Image

# ── Load model ────────────────────────────────────────────
MODEL_PATH = Path(r'G:\defect_detection\models\best.pt')
model      = YOLO(str(MODEL_PATH))

# Class info
CLASSES = ['Blowhole', 'Break', 'Crack', 'Fray', 'Free', 'Uneven']
COLORS  = {
    'Blowhole': '#E74C3C',
    'Break'   : '#E67E22',
    'Crack'   : '#F39C12',
    'Fray'    : '#9B59B6',
    'Free'    : '#27AE60',
    'Uneven'  : '#2980B9',
}

app = FastAPI(
    title='Defect Detection API',
    version='1.0.0',
    description='YOLOv9 Industrial Defect Detection — mAP50: 93.35%'
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_methods=['*'],
    allow_headers=['*'],
)


class Detection(BaseModel):
    class_name : str
    confidence : float
    bbox       : list[float]   # [x1, y1, x2, y2] normalized 0-1
    color      : str


class DetectionResponse(BaseModel):
    detections      : list[Detection]
    detection_count : int
    defect_count    : int
    image_base64    : str
    width           : int
    height          : int


def draw_detections(img: np.ndarray, detections: list[Detection]) -> np.ndarray:
    """Draw bounding boxes on image."""
    img_draw = img.copy()
    h, w     = img_draw.shape[:2]

    for det in detections:
        if det.class_name == 'Free':
            continue

        # Denormalize bbox
        x1 = int(det.bbox[0] * w)
        y1 = int(det.bbox[1] * h)
        x2 = int(det.bbox[2] * w)
        y2 = int(det.bbox[3] * h)

        # Parse color
        color_hex = det.color.lstrip('#')
        r, g, b   = tuple(int(color_hex[i:i+2], 16) for i in (0, 2, 4))
        color_bgr = (b, g, r)

        # Draw box
        cv2.rectangle(img_draw, (x1, y1), (x2, y2), color_bgr, 2)

        # Draw label
        label    = f'{det.class_name} {det.confidence:.0f}%'
        font     = cv2.FONT_HERSHEY_SIMPLEX
        scale    = 0.6
        thick    = 2
        (tw, th), _ = cv2.getTextSize(label, font, scale, thick)

        cv2.rectangle(img_draw, (x1, y1-th-8), (x1+tw+4, y1), color_bgr, -1)
        cv2.putText(img_draw, label, (x1+2, y1-4),
                    font, scale, (255,255,255), thick)

    return img_draw


@app.get('/health')
def health():
    return {
        'status'   : 'ok',
        'model'    : 'YOLOv9c',
        'map50'    : '93.35%',
        'classes'  : CLASSES,
        'version'  : '1.0.0',
    }


@app.post('/detect', response_model=DetectionResponse)
async def detect(
    file      : UploadFile = File(...),
    confidence: float = 0.25,
):
    # Read image
    contents = await file.read()
    nparr    = np.frombuffer(contents, np.uint8)
    img      = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if img is None:
        return JSONResponse(status_code=400, content={'error': 'Invalid image'})

    h, w = img.shape[:2]

    # Run inference
    results     = model(img, conf=confidence, verbose=False)
    detections  = []

    for result in results:
        boxes = result.boxes
        if boxes is None:
            continue
        for box in boxes:
            cls_id     = int(box.cls[0])
            cls_name   = CLASSES[cls_id]
            conf       = round(float(box.conf[0]) * 100, 1)
            x1,y1,x2,y2 = box.xyxy[0].tolist()

            detections.append(Detection(
                class_name = cls_name,
                confidence = conf,
                bbox       = [x1/w, y1/h, x2/w, y2/h],
                color      = COLORS[cls_name],
            ))

    # Draw boxes on image
    img_annotated = draw_detections(img, detections)

    # Convert to base64
    _, buffer    = cv2.imencode('.jpg', img_annotated, [cv2.IMWRITE_JPEG_QUALITY, 90])
    img_base64   = base64.b64encode(buffer).decode('utf-8')

    defect_count = sum(1 for d in detections if d.class_name != 'Free')

    return DetectionResponse(
        detections      = detections,
        detection_count = len(detections),
        defect_count    = defect_count,
        image_base64    = img_base64,
        width           = w,
        height          = h,
    )