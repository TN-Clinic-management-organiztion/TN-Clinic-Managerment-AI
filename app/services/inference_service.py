import logging
from typing import Dict, Optional, Tuple, List
from pathlib import Path
import cv2
import numpy as np

from app.core.config import settings
from app.models.yolo_model import YOLOV12Model
from app.utils.image_processor import MedicalImageProcessor

logger = logging.getLogger(__name__)

class InferenceService:
  """
  Service for performing inference
  """

  def __init__(self, model_dir: Optional[str] = None):
    self.model_dir = Path(model_dir or settings.MODEL_DIR)
    self.models: Dict[str, YOLOV12Model] = {}
    self.default_model_name = settings.DEFAULT_MODEL

    # Load models from the model directory
    self._load_models()

  def _load_models(self):
    """ Load all ONNX models from the model directory"""    
    for model_name in settings.MODEL_NAMES:
      model_path = self.model_dir / f"{model_name}.onnx"
      if model_path.exists():
        try:
          self.models[model_name] = YOLOV12Model(model_path=str(model_path), model_name=model_name)
          logger.info(f"Loaded model: {model_name}")
        except Exception as e:
          logger.error(f"Failed to load model {model_name}: {e}")

  def predict(self,
              image: np.ndarray,
              model_name: Optional[str] = None,
              confidence: Optional[float] = None,
              iou: Optional[float] = None) -> Dict:
    """
    Perform prediction using specified model

    Args:
      image (np.ndarray): Input image
      model_name (Optional[str]): Name of the model to use (default is default_model_name)
      confidence (Optional[float]): Confidence threshold
      iot (Optional[float]): IoU threshold

    Returns:
      Dict: Prediction results
    """

    model_name = model_name or self.default_model_name
    if model_name not in self.models:
      raise ValueError(f"Model {model_name} not found. Available: {list(self.models.keys())}")
    
    model = self.models[model_name]

    return model.predict(
      image=image,
      confidence_threshold=confidence,
      iou_threshold=iou
    )
  
  def predict_from_path(self, 
                         image_path: str,
                         model_name: Optional[str] = None,
                         **kwargs) -> Dict:
    """
    Predict from image file path
    """
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Cannot read image: {image_path}")
    
    return self.predict(image, model_name, **kwargs)
  
  def predict_from_bytes(self,
                          image_bytes: bytes,
                          model_name: Optional[str] = None,
                          **kwargs) -> Dict:
    """
    Predict from image bytes
    """
    nparr = np.frombuffer(image_bytes, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        raise ValueError("Cannot decode image bytes")
    
    return self.predict(image, model_name, **kwargs)
  
  def batch_predict(self,
                  images: List[np.ndarray],
                  model_name: Optional[str] = None,
                  **kwargs) -> List[Dict]:
    """Batch prediction (tuần tự)"""
    results = []
    model = self.get_model(model_name)
    
    for i, image in enumerate(images):
      try:
        result = model.predict(image, **kwargs)
        results.append(result)
      except Exception as e:
        logger.error(f"Error processing image {i}: {e}")
        results.append({
          'error': str(e),
          'image_index': i,
          'success': False
        })
    
    return results
  
  def get_available_models(self) -> List[Dict]:
    """Get list of available models"""
    models_info = []
    for name, model in self.models.items():
      info = model.get_model_info()
      models_info.append({
        'name': name,
        'is_default': name == self.default_model_name,
        'input_shape': info['input']['shape'],
        'classes': info['classes']
      })
    return models_info

  def get_model_info(self, model_name: str) -> Dict:
    """Get information about a specific model"""
    if model_name not in self.models:
      raise ValueError(f"Model {model_name} not found.")
    
    return self.models[model_name].get_model_info()