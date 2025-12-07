import numpy as np
from typing import List, Dict, Tuple
from app.core.config import settings
import logging

logger = logging.getLogger(__name__)

class DetectionPostProcessor:
  """Post-process YOLO model outputs."""

  def __init__(self, 
               confidence_threshold: float = None, 
               iou_threshold: float = None, 
               class_names: Dict[int, str] = None):
    self.confidence_threshold = confidence_threshold or settings.CONFIDENCE_THRESHOLD
    self.iou_threshold = iou_threshold or settings.IOU_THRESHOLD
    self.class_names = class_names or settings.CLASS_NAMES
    logger.info(f"PostProcessor initialized for YOLOv12 format (no separate objectness)")

  def parse_yolo_output(self,
                        predictions: np.ndarray,
                        meta_info: Dict) -> List[Dict]:
    """
    Parse output từ YOLO model

    Args:
      predictions: Output từ model shape (1, 13, num_boxes)
      meta_info: Thông tin từ image processor

    Returns:
      List[Dict]: Danh sách các phát hiện với bounding boxes, class và confidence
    """

    detections = []

    # predictions shape: (batch_size, num_features, num_boxes)
    # YOLOv12: num_features = 4 + num_classes
    # 9 classes: 4 + 9 = 13 features
    num_boxes = predictions.shape[2]
    num_features = predictions.shape[1]
    num_classes_from_model = num_features - 4
    # validate class
    if num_classes_from_model != len(self.class_names):
      logger.warning(
          f"Model has {num_classes_from_model} classes but config expects {len(self.class_names)}. "
          f"Using {min(num_classes_from_model, len(self.class_names))} classes."
    )

    # Scale info
    original_h, original_w = meta_info['original_size']
    padded_h, padded_w = meta_info['padded_size']
    pad_left, pad_top = meta_info['padding']
    scale_w, scale_h = meta_info['ratio']



    for i in range(num_boxes):
      box_features = predictions[0, :, i]
      # YOLOv12 format: [x_center, y_center, width, height, class1, ..., classN]
      x_center, y_center, width, height = box_features[0:4]

      # Get class scores (start index 4)
      class_scores = box_features[4:4 + num_classes_from_model]

      # YOLOv12: confidence = max(class_scores)
      confidence = float(np.max(class_scores))
      class_id = int(np.argmax(class_scores))
      class_score = float(class_scores[class_id])

      # Filter by confidence threshold
      if confidence < self.confidence_threshold:
          continue

      # Convert từ relative coordinates (0-1) sang absolute coordinates
      # Tọa độ từ model là relative đến ảnh đã padded
      # 1. xywh (center) -> xyxy trong hệ padded
      x1_abs = x_center - width / 2.0
      y1_abs = y_center - height / 2.0
      x2_abs = x_center + width / 2.0
      y2_abs = y_center + height / 2.0

      # 2. Trừ padding để về hệ "resized image" (không có viền)
      x1_abs -= pad_left
      x2_abs -= pad_left
      y1_abs -= pad_top
      y2_abs -= pad_top

      # 3. Scale về kích thước ảnh gốc
      # meta_info['ratio'] = (resized_w / original_w, resized_h / original_h)
      x1_abs = x1_abs / scale_w
      x2_abs = x2_abs / scale_w
      y1_abs = y1_abs / scale_h
      y2_abs = y2_abs / scale_h

      # Clip to image boundaries
      x1_abs = max(0, min(x1_abs, original_w))
      y1_abs = max(0, min(y1_abs, original_h))
      x2_abs = max(0, min(x2_abs, original_w))
      y2_abs = max(0, min(y2_abs, original_h))

      # Width and height of box
      box_width = x2_abs - x1_abs
      box_height = y2_abs - y1_abs

      # If box is too small, skip
      if box_width < 2 or box_height < 2:
        continue

      # Map class name
      if class_id < len(self.class_names):
          class_name = self.class_names[class_id]
      else:
          class_name = f"class_{class_id}"
          logger.warning(f"Class ID {class_id} out of range, using default name")

      # Append into detections
      detection = {
          'bbox': [float(x1_abs), float(y1_abs), float(x2_abs), float(y2_abs)],
          'confidence': float(confidence),
          'class_id': class_id,
          'class_name': class_name,
          'class_score': float(class_score),
          'area': float(box_width * box_height),
          'width': float(box_width),
          'height': float(box_height)
      }

      detections.append(detection)

    logger.debug(f"Found {len(detections)} detections before NMS")

    # Apply NMS
    if detections and self.iou_threshold > 0:
        detections = self.non_max_suppression(detections)
        logger.debug(f"Found {len(detections)} detections after NMS")

    return detections

  def non_max_suppression(self, detections: List[Dict]) -> List[Dict]:
    """
    Apply Non-Maximum Suppression (NMS) to filter overlapping boxes

    Args:
      detections: List of detection dicts

    Returns:
      List[Dict]: Filtered detections after NMS
    """

    if not detections:
      return []
    
    # Sort detections by confidence
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    selected = []

    while detections:
      # Pick the detection with highest confidence
      best = detections.pop(0)
      selected.append(best)

      # Calculate IoU with remaining boxes
      remaining = []
      for det in detections:
        iou = self.calculate_iou(best['bbox'], det['bbox'])
        if iou < self.iou_threshold:
          remaining.append(det)

      detections = remaining

    return selected
  
  def calculate_iou(self, box1: List[float], box2: List[float]) -> float:
    """
    Calculate Intersection over Union (IoU) between two boxes

    Args:
      box1: [x1, y1, x2, y2]
      box2: [x1, y1, x2, y2]

    Returns:
      float: IoU value
    """

    # Calculate intersection
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = max(box1[2], box2[2])
    y2 = max(box1[3], box2[3])

    if x2 <= x1 or y2 <= y1:
      return 0.0
    
    intersection = (x2 - x1) * (y2 - y1)
    
    # Calculate areas for each box
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

    # Calculate union
    union = area1 + area2 - intersection

    return intersection / union if union > 0 else 0.0
  
  def format_for_response(self, detections: List[Dict]) -> List[Dict]:
    """
    Format detections for API response
    """

    formatted = []
    for det in detections:
      formatted.append({
        'bbox': {
          'x1': det['bbox'][0],
          'y1': det['bbox'][1],
          'x2': det['bbox'][2],
          'y2': det['bbox'][3],
          'width': det['width'],
          'height': det['height'],
          'area': det['area']
        },
        'confidence': det['confidence'],
        'class': {
          'id': det['class_id'],
          'name': det['class_name'],
          'score': det['class_score']
        }
      })
    return formatted