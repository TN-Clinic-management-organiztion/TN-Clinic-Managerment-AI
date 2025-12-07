import onnxruntime as ort
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from app.core.config import settings
from app.utils.image_processor import MedicalImageProcessor
from app.utils.post_processing import DetectionPostProcessor
import time

logger = logging.getLogger(__name__)

class YOLOV12Model:
    """YOLO Model wrapper for ONNX inference."""

    def __init__(self, model_path: str, model_name: str = None):
        """
        Initialize YOLOModel with ONNX model.

        Args:
            model_path (str): Path to the ONNX model file.
            model_name (str, optional): Name of the model variant.
        """

        self.model_path = Path(model_path)
        self.model_name = model_name or self.model_path.stem

        # Check if model file exists
        if not self.model_path.exists():
            raise FileNotFoundError(f"Model file not found: {self.model_path}")
    
        # Initialize ONNX Runtime session
        try:
            # Priority: GPU -> CPU
            available_providers = ort.get_available_providers()
            logger.info(f"Available ONNX providers: {available_providers}")

            providers = ['CPUExecutionProvider']
            if 'CUDAExecutionProvider' in available_providers:
                providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("Using GPU acceleration")
            elif 'TensorrtExecutionProvider' in available_providers:
                providers = ['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']
                logger.info("Using TensorRT acceleration")
            else:
                logger.info("Using CPU only")
            
            session_options = ort.SessionOptions()
            session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            session_options.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL

            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=session_options,
                providers=providers
            )

        except Exception as e:
            logger.error(f"Cannot load the ONNX model: {e}")
            raise

        # Get input and output
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Initialize image processor and post-processor
        self.image_processor = MedicalImageProcessor()
        self.post_processor = DetectionPostProcessor()

        logger.info(f"Load model: {self.model_name}")
        logger.info(f"Input: {self.input_name}, Output: {self.output_name}")

    def _extract_model_metadata(self):
        """Trích xuất metadata từ model ONNX"""
        try:
            # Kiểm tra metadata trong model
            model_metadata = self.session.get_modelmeta()
            logger.info(f"Model metadata: {model_metadata.description}")
            
            # Log các custom metadata (nếu có)
            for prop in dir(model_metadata):
                if not prop.startswith('_'):
                    value = getattr(model_metadata, prop, None)
                    if value and prop not in ['graph_name', 'domain', 'description']:
                        logger.debug(f"  {prop}: {value}")
                        
        except Exception as e:
            logger.debug(f"Could not extract model metadata: {e}")

    def predict(self,
                image: np.ndarray,
                confidence_threshold: Optional[float] = None,
                iou_threshold: Optional[float] = None
                ) -> Dict:
        """
        Predict objects in the image.

        Args:
            image (np.ndarray): Input image.
            confidence_threshold (Optional[float]): Confidence threshold for detections.
            iou_threshold (Optional[float]): IOU threshold for NMS

        Returns:
            Dict: Prediction results including bounding boxes, class IDs, and confidence scores.
        """
        start_time = time.time()

        # Update thresholds if provided
        if confidence_threshold is not None:
            self.post_processor.confidence_threshold = confidence_threshold
        if iou_threshold is not None:
            self.post_processor.iou_threshold = iou_threshold

        # Preprocess image
        preprocess_start = time.time()
        input_tensor, meta_info = self.image_processor.smart_resize(image=image)
        normalized_tensor = self.image_processor.normalize_image(image=input_tensor)
        preprocess_time = time.time() - preprocess_start

        # Run inference
        inference_start = time.time()
        outputs = self.session.run(
            [self.output_name],
            {self.input_name: normalized_tensor}
        )
        inference_time = time.time() - inference_start

        # Post-process outputs
        postprocess_start = time.time()
        raw_detections = self.post_processor.parse_yolo_output(predictions=outputs[0], meta_info=meta_info)
        formatted_detections = self.post_processor.format_for_response(detections=raw_detections)
        postprocess_time = time.time() - postprocess_start

        # Create response
        total_time = time.time() - start_time
        response = {
            'model': self.model_name,
            'detections': formatted_detections,
            'num_detections': len(formatted_detections),
            'image_info': {
                'original_size': meta_info['original_size'],
                'processed_size': meta_info['padded_size'],
                'scale_factor': meta_info['scale'],
                'padding': meta_info['padding'],
                'ratios': meta_info['ratio']
            },
            'performance': {
                'preprocess_ms': round(preprocess_time * 1000, 2),
                'inference_ms': round(inference_time * 1000, 2),
                'postprocess_ms': round(postprocess_time * 1000, 2),
                'total_ms': round(total_time * 1000, 2),
                'fps': round(1 / total_time, 2) if total_time > 0 else 0
            },
            'thresholds': {
                'confidence': self.post_processor.confidence_threshold,
                'iou': self.post_processor.iou_threshold
            },
            'timestamp': time.time()
        }

        return response
    
    def predict_from_path(self,
                          image_path: str,
                          **kwargs) -> Dict:
        """
        Predict objects from image file path.
        """

        import cv2

        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        return self.predict(image=image, **kwargs)

    def predict_from_bytes(self,
                          image_bytes: bytes,
                          confidence_threshold: Optional[float] = None,
                          iou_threshold: Optional[float] = None) -> Dict:
        """
        Predict objects from image bytes.
        """

        import cv2
        import numpy as np

        # Decode image bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise ValueError("Cannot decode image from bytes.")
        
        return self.predict(image, confidence_threshold, iou_threshold)

    def get_model_info(self) -> Dict:
        """Get model information."""
        input_info = self.session.get_inputs()[0]
        output_info = self.session.get_outputs()[0]

        num_features = output_info.shape[1]
        num_classes = num_features - 4  # YOLOv12: 4 bbox + classes

        return {
            'name': self.model_name,
            'type': 'YOLOv12',
            'path': str(self.model_path),
            'input': {
                'name': input_info.name,
                'shape': input_info.shape,
                'type': str(input_info.type)
            },
            'output': {
                'name': output_info.name,
                'shape': output_info.shape,
                'type': str(output_info.type)
            },
            'providers': self.session.get_providers(),
            'classes': settings.CLASS_NAMES
        }