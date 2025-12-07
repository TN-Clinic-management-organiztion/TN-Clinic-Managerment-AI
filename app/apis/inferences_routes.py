from fastapi import APIRouter, UploadFile, File, HTTPException, Depends, Form
from fastapi.responses import JSONResponse
import logging
from typing import List

from app.services.inference_service import InferenceService
from app.schemas.request_response import (
    DetectionResponse,
    DetectionRequest,
    HealthResponse,
    ErrorResponse,
    ModelInfo,
    ImageInfo
)

logger = logging.getLogger(__name__)

router = APIRouter()

# Initialize service
inference_service = InferenceService()

@router.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    models = inference_service.get_available_models()
    model_names = [m['name'] for m in models]
    
    return HealthResponse(
        status="healthy",
        models_loaded=model_names,
        total_models=len(model_names)
    )

@router.get("/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of available models"""
    return inference_service.get_available_models()

@router.post("/detect/image", response_model=DetectionResponse)
async def detect_from_image(
    file: UploadFile = File(...),
    params: DetectionRequest = Depends()
):
    """
    Receive image and response detect results
    
    - **file**: medical image (jpg, png, tiff)
    - **model_name**: model name (yolov12n, yolov12s, yolov12m)
    - **confidence_threshold**: confidence (0-1)
    - **iou_threshold**: IoU for NMS (0-1)
    """
    try:
        # Read file
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file")
        
        # Check file type
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        
        # Perform inference
        result = inference_service.predict_from_bytes(
            image_bytes=contents,
            model_name=params.model_name.value if params.model_name else None,
            confidence=params.confidence_threshold,
            iou=params.iou_threshold
        )
        
        # convert to response schema
        response = DetectionResponse(
            success=True,
            model=result['model'],
            detections=result['detections'],
            num_detections=result['num_detections'],
            image_info=result['image_info'],
            inference_time=result.get('performance')['total_ms'],
            thresholds=result['thresholds']
        )
        
        return response
        
    except Exception as e:
        logger.error(f"Detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/detect/batch", response_model=List[DetectionResponse])
async def detect_batch(
    files: List[UploadFile] = File(...),
    image_ids: List[str] = Form(...),
    params: DetectionRequest = Depends()
):
    """
    Receive many images and response detection batch
    
    - **files**: List of images
    - **params**: parameter inference
    """
    print(f"files: {files}")
    print(f"len files: {len(files)}")
    print(f"image_ids: {image_ids}")
    print(f"len image_ids: {len(image_ids)}")

    if len(files) != len(image_ids):
        raise HTTPException(
            status_code=400,
            detail="Number of files and image_ids must match"
        )

    try:
        results = []
        
        for file, image_id in zip(files, image_ids):
            contents = await file.read()
            
            if not contents:
                continue
            
            # perform inference
            result = inference_service.predict_from_bytes(
                image_bytes=contents,
                model_name=params.model_name.value if params.model_name else None,
                confidence=params.confidence_threshold,
                iou=params.iou_threshold
            )

            img_meta = result['image_info']

            image_info = ImageInfo(
                image_id=image_id,                       # ID logic (S3 key, DB id,…)
                image_name=file.filename,                # file's name upload
                original_size=list(img_meta['original_size']),
                processed_size=list(img_meta['processed_size']),
                scale_factor=float(img_meta['scale_factor']),
            )
            
            # convert to response schema
            response = DetectionResponse(
                success=True,
                model=result['model'],
                detections=result['detections'],
                num_detections=result['num_detections'],
                image_info=image_info,
                inference_time=result.get('performance')['total_ms'],
                thresholds=result['thresholds']
            )
            
            results.append(response)
        
        return results
        
    except Exception as e:
        logger.error(f"Batch detection error: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/model/{model_name}/info")
async def get_model_info(model_name: str):
    """Get detail information of model"""
    try:
        info = inference_service.get_model_info(model_name)
        return info
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.error(f"Error getting model info: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))