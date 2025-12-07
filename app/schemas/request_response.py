from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Union
from datetime import datetime
from enum import Enum

class ModelName(str, Enum):
    """Tên model hỗ trợ"""
    YOLOV12N = "yolov12n"
    YOLOV12S = "yolov12s"
    YOLOV12M = "yolov12m"

class DetectionRequest(BaseModel):
    """Request for detection API"""
    model_name: Optional[ModelName] = Field(
        default=ModelName.YOLOV12N,
        description="Model name to use for detection"
    )
    confidence_threshold: Optional[float] = Field(
        default=0.25,
        ge=0.0,
        le=1.0,
        description="Confidence threshold for detections"
    )
    iou_threshold: Optional[float] = Field(
        default=0.45,
        ge=0.0,
        le=1.0,
        description="IoU threshold for NMS"
    )

    @field_validator('confidence_threshold', 'iou_threshold')
    @classmethod
    def validate_thresholds(cls, v):
        if v is None:
            return v
        if v < 0 or v > 1:
            raise ValueError(f"Threshold must be between 0 and 1")
        return v
    
class BoundingBox(BaseModel):
    """Bounding box"""
    x1: float = Field(..., description="Top-left x coordinate")
    y1: float = Field(..., description="Top-left y coordinate")
    x2: float = Field(..., description="Bottom-right x coordinate")
    y2: float = Field(..., description="Bottom-right y coordinate")
    width: float = Field(..., description="Width of bounding box")
    height: float = Field(..., description="Height of bounding box")
    area: float = Field(..., description="Area of bounding box")
    
class ClassInfo(BaseModel):
    """Class Information"""
    id: int = Field(..., description="Class ID")
    name: str = Field(..., description="Class name")
    score: float = Field(..., description="Class score")

class DetectionItem(BaseModel):
    """Một detection item"""
    bbox: BoundingBox
    confidence: float = Field(..., ge=0.0, le=1.0)
    class_info: ClassInfo = Field(..., alias="class")

class ImageInfo(BaseModel):
    """Image information"""
    image_id: Optional[str] = Field(None, description="External ID / S3 key / DB id of image")
    image_name: Optional[str] = Field(None, description="Root image's name")
    original_size: List[int] = Field(..., description="[height, width]")
    processed_size: List[int] = Field(..., description="[height, width]")
    scale_factor: float = Field(..., description="Scale factor")

class DetectionResponse(BaseModel):
    """Response for detection API"""
    success: bool = Field(..., description="Success flag")
    model: str = Field(..., description="Model name used")
    detections: List[DetectionItem] = Field(..., description="List of detections")
    num_detections: int = Field(..., description="Number of detections")
    image_info: ImageInfo = Field(..., description="Image information")
    inference_time: Optional[float] = Field(None, description="Inference time in seconds")
    thresholds: Dict[str, float] = Field(..., description="Thresholds used")
    timestamp: datetime = Field(default_factory=datetime.now)

class ModelInfo(BaseModel):
    """Model information"""
    name: str
    is_default: bool
    input_shape: List[Union[int, str, None]]
    classes: Dict[int, str]

class HealthResponse(BaseModel):
    """Response for health check"""
    status: str
    models_loaded: List[str]
    total_models: int
    timestamp: datetime = Field(default_factory=datetime.now)

class ErrorResponse(BaseModel):
    """Response for Error"""
    error: str
    detail: Optional[str] = None
    timestamp: datetime = Field(default_factory=datetime.now)