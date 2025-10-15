"""
SafeOrbit - FastAPI Backend for Object Detection
Serves YOLOv8 model predictions with bounding boxes
Enhanced with domain adaptation for real-world images
"""

from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional
import cv2
import numpy as np
from ultralytics import YOLO
import base64
from io import BytesIO
from PIL import Image
import logging
from pathlib import Path
import sys

# Add scripts to path for domain adaptation
sys.path.append(str(Path(__file__).parent / 'scripts'))
from inference_tta import TTAPredictor

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="SafeOrbit Object Detection API",
    description="Real-time object detection for space station safety",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins for development
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model variable
model = None
predictor = None  # TTA predictor
MODEL_PATH = Path(r"E:\safeorbit\ok-computer\optimization_runs\strategy_4_extended_training_20251015_101328\weights\best.pt")

class DetectionRequest(BaseModel):
    image: str  # Base64 encoded image
    confidence: Optional[float] = 0.25  # Increased from 0.4 for better recall
    use_tta: Optional[bool] = True  # Enable TTA by default for real images

class BoundingBox(BaseModel):
    x: float
    y: float
    width: float
    height: float

class Detection(BaseModel):
    name: str
    confidence: float
    bbox: BoundingBox

class DetectionResponse(BaseModel):
    objects: List[Detection]
    inference_time: float
    image_size: List[int]

class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: Optional[str] = None

def load_model():
    """Load YOLO model on startup"""
    global model, predictor
    try:
        # Load the trained YOLOv8m model
        if MODEL_PATH.exists():
            logger.info(f"Loading model from {MODEL_PATH}")
            model = YOLO(str(MODEL_PATH))
            
            # Initialize TTA predictor for enhanced real-world inference
            predictor = TTAPredictor(
                model_path=str(MODEL_PATH),
                conf_threshold=0.25,
                iou_threshold=0.45,
                use_tta=True,
                img_size=640
            )
            logger.info("✅ YOLOv8m model loaded successfully with TTA support")
            return
        
        # Fallback to YOLOv8n pretrained if custom model not found
        logger.warning("Custom model not found, loading YOLOv8n pretrained model")
        model = YOLO("yolov8n.pt")
        predictor = None
        logger.info("✅ YOLOv8n pretrained model loaded")
        
    except Exception as e:
        logger.error(f"❌ Failed to load model: {e}")
        model = None
        predictor = None

@app.on_event("startup")
async def startup_event():
    """Initialize model on startup"""
    load_model()

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint"""
    return {
        "message": "SafeOrbit Object Detection API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect (POST)",
            "detect_with_image": "/detect-image (POST)"
        }
    }

@app.get("/health", response_model=HealthResponse, tags=["Health"])
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy" if model is not None else "unhealthy",
        "model_loaded": model is not None,
        "model_path": str(MODEL_PATH) if MODEL_PATH.exists() else None
    }

def decode_base64_image(base64_string: str) -> np.ndarray:
    """Decode base64 string to OpenCV image"""
    try:
        # Remove header if present (data:image/jpeg;base64,...)
        if "," in base64_string:
            base64_string = base64_string.split(",")[1]
        
        # Decode base64
        image_bytes = base64.b64decode(base64_string)
        
        # Convert to PIL Image
        image = Image.open(BytesIO(image_bytes))
        
        # Convert to OpenCV format (BGR)
        image_np = np.array(image)
        if len(image_np.shape) == 2:  # Grayscale
            image_np = cv2.cvtColor(image_np, cv2.COLOR_GRAY2BGR)
        elif image_np.shape[2] == 4:  # RGBA
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGBA2BGR)
        else:  # RGB
            image_np = cv2.cvtColor(image_np, cv2.COLOR_RGB2BGR)
        
        return image_np
    
    except Exception as e:
        logger.error(f"Error decoding image: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid image data: {str(e)}")

def draw_detections(image: np.ndarray, detections: List[Detection]) -> np.ndarray:
    """Draw bounding boxes and labels on image"""
    img_height, img_width = image.shape[:2]
    
    for det in detections:
        # Convert normalized coordinates to pixel coordinates
        x = int(det.bbox.x * img_width)
        y = int(det.bbox.y * img_height)
        w = int(det.bbox.width * img_width)
        h = int(det.bbox.height * img_height)
        
        # Draw bounding box
        color = (66, 133, 244)  # Google Blue (BGR)
        thickness = 3
        cv2.rectangle(image, (x, y), (x + w, y + h), color, thickness)
        
        # Draw corner accents
        corner_length = 20
        corner_thickness = 4
        
        # Top-left
        cv2.line(image, (x, y), (x + corner_length, y), color, corner_thickness)
        cv2.line(image, (x, y), (x, y + corner_length), color, corner_thickness)
        
        # Top-right
        cv2.line(image, (x + w, y), (x + w - corner_length, y), color, corner_thickness)
        cv2.line(image, (x + w, y), (x + w, y + corner_length), color, corner_thickness)
        
        # Bottom-left
        cv2.line(image, (x, y + h), (x + corner_length, y + h), color, corner_thickness)
        cv2.line(image, (x, y + h), (x, y + h - corner_length), color, corner_thickness)
        
        # Bottom-right
        cv2.line(image, (x + w, y + h), (x + w - corner_length, y + h), color, corner_thickness)
        cv2.line(image, (x + w, y + h), (x + w, y + h - corner_length), color, corner_thickness)
        
        # Prepare label text
        label = f"{det.name} {det.confidence*100:.0f}%"
        
        # Calculate label size
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, font_thickness)
        
        # Draw label background
        label_y = max(y - 10, text_height + 10)
        cv2.rectangle(
            image,
            (x, label_y - text_height - 10),
            (x + text_width + 10, label_y),
            color,
            -1  # Filled
        )
        
        # Draw label text
        cv2.putText(
            image,
            label,
            (x + 5, label_y - 5),
            font,
            font_scale,
            (255, 255, 255),  # White text
            font_thickness
        )
    
    return image

@app.post("/detect", response_model=DetectionResponse, tags=["Detection"])
async def detect_objects(request: DetectionRequest):
    """
    Detect objects in base64 encoded image
    Returns detection metadata (no image)
    Enhanced with TTA for real-world images
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        import time
        start_time = time.time()
        
        # Decode image
        image = decode_base64_image(request.image)
        img_height, img_width = image.shape[:2]
        
        # Use TTA predictor if enabled and available
        if request.use_tta and predictor is not None:
            # Use enhanced TTA predictor for better real-world accuracy
            detections_list = predictor.predict_single(image, preprocess=True)
            
            # Convert to Detection format
            detections = []
            for det in detections_list:
                x1, y1, x2, y2 = det['bbox']
                x_norm = float(x1 / img_width)
                y_norm = float(y1 / img_height)
                w_norm = float((x2 - x1) / img_width)
                h_norm = float((y2 - y1) / img_height)
                
                detections.append(Detection(
                    name=det['class_name'],
                    confidence=det['confidence'],
                    bbox=BoundingBox(
                        x=x_norm,
                        y=y_norm,
                        width=w_norm,
                        height=h_norm
                    )
                ))
        else:
            # Standard inference (fallback)
            results = model(image, conf=request.confidence, verbose=False)
            result = results[0]
            
            # Extract detections
            detections = []
            for box in result.boxes:
                # Get box coordinates (xyxy format)
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                
                # Convert to normalized xywh format
                x_norm = float(x1 / img_width)
                y_norm = float(y1 / img_height)
                w_norm = float((x2 - x1) / img_width)
                h_norm = float((y2 - y1) / img_height)
                
                # Get class name and confidence
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                name = result.names[cls]
                
                detections.append(Detection(
                    name=name,
                    confidence=conf,
                    bbox=BoundingBox(
                        x=x_norm,
                        y=y_norm,
                        width=w_norm,
                        height=h_norm
                    )
                ))
        
        # Calculate inference time
        inference_time = (time.time() - start_time) * 1000  # milliseconds
        
        return DetectionResponse(
            objects=detections,
            inference_time=inference_time,
            image_size=[img_width, img_height]
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/detect-image", tags=["Detection"])
async def detect_objects_with_image(request: DetectionRequest):
    """
    Detect objects in base64 encoded image
    Returns image with bounding boxes drawn
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Decode image
        image = decode_base64_image(request.image)
        
        # Run inference
        results = model(image, conf=request.confidence, verbose=False)
        result = results[0]
        
        # Extract detections
        detections = []
        img_height, img_width = image.shape[:2]
        
        for box in result.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to normalized xywh format
            x_norm = float(x1 / img_width)
            y_norm = float(y1 / img_height)
            w_norm = float((x2 - x1) / img_width)
            h_norm = float((y2 - y1) / img_height)
            
            # Get class name and confidence
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls]
            
            detections.append(Detection(
                name=name,
                confidence=conf,
                bbox=BoundingBox(
                    x=x_norm,
                    y=y_norm,
                    width=w_norm,
                    height=h_norm
                )
            ))
        
        # Draw detections on image
        annotated_image = draw_detections(image.copy(), detections)
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_image)
        
        # Return image as stream
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Inference-Time": str(result.speed['inference']),
                "X-Detections-Count": str(len(detections))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.post("/upload-detect", tags=["Detection"])
async def detect_from_upload(file: UploadFile = File(...), confidence: float = 0.4):
    """
    Detect objects from uploaded image file
    Returns image with bounding boxes drawn
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    try:
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        # Run inference
        results = model(image, conf=confidence, verbose=False)
        result = results[0]
        
        # Extract detections
        detections = []
        img_height, img_width = image.shape[:2]
        
        for box in result.boxes:
            # Get box coordinates (xyxy format)
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            # Convert to normalized xywh format
            x_norm = float(x1 / img_width)
            y_norm = float(y1 / img_height)
            w_norm = float((x2 - x1) / img_width)
            h_norm = float((y2 - y1) / img_height)
            
            # Get class name and confidence
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            name = result.names[cls]
            
            detections.append(Detection(
                name=name,
                confidence=conf,
                bbox=BoundingBox(
                    x=x_norm,
                    y=y_norm,
                    width=w_norm,
                    height=h_norm
                )
            ))
        
        # Draw detections on image
        annotated_image = draw_detections(image.copy(), detections)
        
        # Encode image to JPEG
        _, buffer = cv2.imencode('.jpg', annotated_image)
        
        # Return image as stream
        return StreamingResponse(
            BytesIO(buffer.tobytes()),
            media_type="image/jpeg",
            headers={
                "X-Inference-Time": str(result.speed['inference']),
                "X-Detections-Count": str(len(detections))
            }
        )
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Detection error: {e}")
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
