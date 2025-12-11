import time
from ultralytics import YOLO
from PIL import Image
import io
from app.core.config import settings
from app.schemas.request_response import BoundingBox, Prediction, PredictionResponse

class YoloService:
  def __init__(self):
    # Load model from config
    print(f"Đang tải model từ {settings.MODEL_PATH}...")
    self.model = YOLO(settings.MODEL_PATH)
    print("Model đã được tải thành công.")

  def predict(self, image_bytes: bytes, filename: str) -> PredictionResponse:
    start_time = time.time()
    # 1. Load image
    image = Image.open(io.BytesIO(image_bytes))
    # 2. Predicty
    results = self.model.predict(image, conf=settings.CONFIDENCE_THRESHOLD)[0]

    # 3. Parse results
    detections = []
    for box in results.boxes:
      coords = box.xyxy.tolist()[0]
      detections.append(Prediction(
        class_name = results.names[int(box.cls[0])],
        class_id = int(box.cls[0]),
        confidence=round(float(box.conf[0]), 4),
        bounding_box = BoundingBox(
          x_min=coords[0],
          y_min=coords[1],
          x_max=coords[2],
          y_max=coords[3]
        )
      ))
    process_time = time.time() - start_time

    return PredictionResponse(
      filename=filename,
      image_height=image.height,
      image_width=image.width,
      process_time=process_time,
      detections=detections
    )

YoloServices = YoloService()