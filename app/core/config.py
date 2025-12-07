import os
from typing import Dict, List, Tuple
from pydantic_settings import BaseSettings, SettingsConfigDict
from pathlib import Path

class Settings(BaseSettings):
  # API setting
  PROJECT_NAME: str = "TN Clinic Management AI"
  APP_VERSION: str = "1.0.0"
  API_PREFIX: str = "/api/v1"
  DEBUG: bool = False
  ONNX_PATH: str = "onnx/yolov12n/yolov12n_uit_multi_imaging_medical/yolov12n_uit_multi_imaging_medical_best.onnx"

  # Model settings
  MODEL_DIR: str = "app/onnx/models/"
  DEFAULT_MODEL: str = "yolov12n"
  MODEL_NAMES: List[str] = ["yolov12n", "yolov12s", "yolov12m"]

  # Inference settings
  CONFIDENCE_THRESHOLD: float = 0.25
  IOU_THRESHOLD: float = 0.45
  STRIDE: int = 32
  PADDING_COLOR: Tuple[int, int, int] = (114, 114, 114)

  # Class dataset
  CLASS_NAMES: Dict[int, str] = {
    0: 'nodule',
    1: 'liver tumor',
    2: 'Brain tumor',
    3: 'Glioma',
    4: 'Meningioma',
    5: 'Pituitary',
    6: 'prostate cancer',
    7: 'Lung Opacity',
    8: 'Tuberculosis'
  }

  IMAGE_INPUT_FORMAT: str = "RGB"

  # LOGGING
  LOG_LEVEL: str = "INFO"

  model_config = SettingsConfigDict(
      env_file=".env",
      env_file_encoding="utf-8",
      case_sensitive=True,
  )

settings = Settings()