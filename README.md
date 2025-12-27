# TN Clinic Management AI - Inference Service

## Giới thiệu

Đây là một AI service được xây dựng bằng FastAPI để phát hiện các bệnh lý trong hình ảnh y tế. Service nhận hình ảnh y tế (CT scan, MRI, X-ray) và trả về kết quả phát hiện dưới dạng bounding boxes.

Service này được thiết kế để backend NestJS gọi đến, cung cấp các API endpoint để xử lý hình ảnh y tế và trả về kết quả phát hiện.

Service sử dụng mô hình YOLOv12 được chuyển đổi sang định dạng ONNX để tối ưu hiệu suất. Hiện tại hỗ trợ 3 mô hình: yolov12n (nano), yolov12s (small), và yolov12m (medium).

## Các bệnh lý được phát hiện

Service có thể phát hiện 9 loại bệnh lý y tế:
- Nodule
- Liver tumor
- Brain tumor
- Glioma
- Meningioma
- Pituitary
- Prostate cancer
- Lung Opacity
- Tuberculosis

## Cấu trúc thư mục

```
TN-Clinic-Managerment-AI/
├── app/
│   ├── apis/                    # Định nghĩa các API routes
│   │   ├── __init__.py
│   │   └── inferences_routes.py
│   ├── core/                    # Cấu hình chính
│   │   └── config.py
│   ├── models/                  # Wrapper cho YOLO model
│   │   └── yolo_model.py
│   ├── onnx/
│   │   └── models/              # Chứa các file ONNX models
│   │       ├── yolov12n.onnx
│   │       ├── yolov12s.onnx
│   │       └── yolov12m.onnx
│   ├── schemas/                 # Định nghĩa request/response
│   │   └── request_response.py
│   ├── services/                # Business logic
│   │   ├── inference_service.py
│   │   └── yolo_services.py
│   ├── utils/                   # Các hàm tiện ích
│   │   ├── image_processor.py
│   │   └── post_processing.py
│   ├── tests/                   # File test
│   └── main.py                  # Entry point của ứng dụng
├── Dockerfile
├── requirements.txt
├── pyproject.toml
└── README.md
```

## Cách chạy

### Chạy với Docker (Khuyến nghị)

Đây là cách đơn giản nhất để chạy service, đảm bảo môi trường nhất quán.

**Bước 1: Build Docker image**

Từ thư mục gốc của project, chạy lệnh:

```bash
docker build -t tn-clinic-ai .
```

**Bước 2: Kiểm tra image đã được tạo**

```bash
docker images
```

**Bước 3: Chạy container**

```bash
docker run -d -p 8000:8000 --name tn-clinic-ai-container tn-clinic-ai
```

**Bước 4: Kiểm tra service đã chạy**

```bash
# Xem container đang chạy
docker ps

# Xem logs của container
docker logs tn-clinic-ai-container

# Test API health check
curl http://localhost:8000/api/v1/health
```

**Dừng và xóa container:**

```bash
# Dừng container
docker stop tn-clinic-ai-container

# Xóa container
docker rm tn-clinic-ai-container

# Hoặc dừng và xóa cùng lúc
docker rm -f tn-clinic-ai-container
```

### Chạy ở chế độ Development


**Bước 1: Tạo virtual environment**

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux/Mac
python3 -m venv venv
source venv/bin/activate
```

**Bước 2: Cài đặt dependencies**

```bash
pip install -r requirements.txt
```

**Bước 3: Kiểm tra models**

Đảm bảo các file ONNX models đã có trong thư mục `app/onnx/models/`:
- `yolov12n.onnx`
- `yolov12s.onnx`
- `yolov12m.onnx`


**Bước 4: Chạy ứng dụng**

```bash
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

Flag `--reload` cho phép tự động reload khi code thay đổi, rất tiện cho development.

**Bước 5: Test service**

Mở trình duyệt hoặc dùng curl:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Xem danh sách models
curl http://localhost:8000/api/v1/models
```

## API Endpoints

Base URL: `http://localhost:8000/api/v1`

### 1. Health Check

Kiểm tra service có đang chạy và các models đã được load chưa.

```
GET /api/v1/health
```

Response:
```json
{
  "status": "healthy",
  "models_loaded": ["yolov12n", "yolov12s", "yolov12m"],
  "total_models": 3,
  "timestamp": "2024-01-01T00:00:00"
}
```

### 2. Danh sách Models

Lấy danh sách các models có sẵn.

```
GET /api/v1/models
```

### 3. Thông tin Model

Lấy thông tin chi tiết về một model cụ thể.

```
GET /api/v1/model/{model_name}/info
```

Thay `{model_name}` bằng `yolov12n`, `yolov12s`, hoặc `yolov12m`.

### 4. Phát hiện từ Hình ảnh (Upload File)

Endpoint chính để phát hiện bệnh lý từ hình ảnh.

```
POST /api/v1/detect/image
```

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `file`: File hình ảnh (jpg, png, tiff)
  - `model_name`: (Tùy chọn) Tên model, mặc định là `yolov12n`
  - `confidence_threshold`: (Tùy chọn) Ngưỡng confidence (0.0 - 1.0), mặc định 0.25
  - `iou_threshold`: (Tùy chọn) Ngưỡng IoU (0.0 - 1.0), mặc định 0.45

**Response:**
```json
{
  "success": true,
  "model": "yolov12n",
  "detections": [
    {
      "bbox": {
        "x1": 100.5,
        "y1": 200.3,
        "x2": 300.7,
        "y2": 400.9,
        "width": 200.2,
        "height": 200.6,
        "area": 40120.12
      },
      "confidence": 0.95,
      "class": {
        "id": 2,
        "name": "Brain tumor",
        "score": 0.95
      }
    }
  ],
  "num_detections": 1,
  "image_info": {
    "image_id": null,
    "image_name": "test_image.png",
    "original_size": [512, 512],
    "processed_size": [640, 640],
    "scale_factor": 1.25
  },
  "inference_time": 45.23,
  "thresholds": {
    "confidence": 0.25,
    "iou": 0.45
  },
  "timestamp": "2024-01-01T00:00:00"
}
```

**Ví dụ với cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@path/to/image.png" \
  -F "model_name=yolov12n" \
  -F "confidence_threshold=0.3"
```

**Ví dụ với Python:**
```python
import requests

url = "http://localhost:8000/api/v1/detect/image"
files = {"file": open("image.png", "rb")}
data = {
    "model_name": "yolov12n",
    "confidence_threshold": 0.3,
    "iou_threshold": 0.5
}

response = requests.post(url, files=files, data=data)
print(response.json())
```

### 5. Phát hiện Batch (Nhiều hình ảnh)

Xử lý nhiều hình ảnh cùng lúc.

```
POST /api/v1/detect/batch
```

**Request:**
- Content-Type: `multipart/form-data`
- Body:
  - `files`: Mảng các file hình ảnh
  - `image_ids`: Mảng các ID tương ứng với từng hình ảnh
  - `model_name`: (Tùy chọn) Tên model
  - `confidence_threshold`: (Tùy chọn)
  - `iou_threshold`: (Tùy chọn)

**Ví dụ với cURL:**
```bash
curl -X POST "http://localhost:8000/api/v1/detect/batch" \
  -F "files=@image1.png" \
  -F "files=@image2.png" \
  -F "image_ids=img001" \
  -F "image_ids=img002" \
  -F "model_name=yolov12n"
```

### 6. Phát hiện từ URL

Phát hiện từ hình ảnh được lưu trữ trên URL.

```
POST /api/v1/detect/url
```

**Request:**
```json
{
  "image_url": "https://example.com/image.png",
  "model_name": "yolov12n",
  "confidence_threshold": 0.25,
  "iou_threshold": 0.45
}
```

## Cấu hình

Service sử dụng Pydantic Settings để quản lý cấu hình. Bạn có thể tùy chỉnh thông qua file `.env` hoặc environment variables.

Tạo file `.env` ở thư mục gốc (tùy chọn):

```env
# Model Settings
MODEL_DIR=app/onnx/models/
DEFAULT_MODEL=yolov12n

# Inference Settings
CONFIDENCE_THRESHOLD=0.25
IOU_THRESHOLD=0.45

# Logging
LOG_LEVEL=INFO
```

Các tham số chính:
- `DEFAULT_MODEL`: Model mặc định (`yolov12n`, `yolov12s`, `yolov12m`)
- `CONFIDENCE_THRESHOLD`: Ngưỡng confidence mặc định (0.0 - 1.0)
- `IOU_THRESHOLD`: Ngưỡng IoU cho NMS (0.0 - 1.0)
- `MODEL_DIR`: Đường dẫn đến thư mục chứa ONNX models

## Xử lý lỗi thường gặp

### Model không tìm thấy

Nếu gặp lỗi "Model not found", kiểm tra:
- Các file `.onnx` đã có trong `app/onnx/models/` chưa
- Tên file đúng format: `{model_name}.onnx` (ví dụ: `yolov12n.onnx`)
- Đường dẫn `MODEL_DIR` trong config đúng chưa

### Port đã được sử dụng

Nếu port 8000 đã được sử dụng:
- Thay đổi port: `uvicorn app.main:app --port 8001`
- Hoặc dừng process đang sử dụng port 8000

### Lỗi khi decode image

Nếu gặp lỗi "Cannot decode image bytes":
- Kiểm tra file hình ảnh có hợp lệ không
- Đảm bảo format được hỗ trợ (jpg, png, tiff)
- Kiểm tra file không bị corrupt

### Docker container không start

Xem logs để biết lỗi:
```bash
docker logs tn-clinic-ai-container
```

Kiểm tra container status:
```bash
docker ps -a
```

Restart container:
```bash
docker restart tn-clinic-ai-container
```

## Testing

Test với cURL:

```bash
# Health check
curl http://localhost:8000/api/v1/health

# List models
curl http://localhost:8000/api/v1/models

# Detect từ file
curl -X POST "http://localhost:8000/api/v1/detect/image" \
  -F "file=@app/tests/pictures/test_rsna_000592.png" \
  -F "model_name=yolov12n"
```

Chạy test Python:

```bash
uv run python app/tests/test_onnx_yolo.py
```
