import cv2
import numpy as np
import math
from typing import Tuple, Dict, Optional
import logging
from app.core.config import settings

logger = logging.getLogger(__name__)

class MedicalImageProcessor:
    """Process medical images for YOLOv12 dynamic input inference."""
    
    def __init__(self, stride: int = None, padding_color: Tuple[int, int, int] = None):
        self.stride = stride or settings.STRIDE
        self.padding_color = padding_color or settings.PADDING_COLOR
        self.input_format = getattr(settings, 'IMAGE_INPUT_FORMAT', 'RGB')
        logger.debug(f"Processor initialized with stride={self.stride}")

    def validate_image(self, image: np.ndarray) -> bool:
        """Kiểm tra ảnh hợp lệ"""
        if image is None:
            logger.error("Image is None")
            return False
        
        if not isinstance(image, np.ndarray):
            logger.error(f"Image is not numpy array: {type(image)}")
            return False
        
        if len(image.shape) not in [2, 3]:
            logger.error(f"Invalid image dimensions: {image.shape}")
            return False
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            logger.error(f"Invalid number of channels: {image.shape}")
            return False
        
        return True
    
    def convert_to_rgb(self, image: np.ndarray) -> np.ndarray:
      """
      Convert to RGB format for YOLO model.
      Medical images: grayscale -> RGB, BGR -> RGB, RGBA -> RGB
      """
      if len(image.shape) == 2:                # gray
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
      elif image.shape[2] == 4:                # BGRA
        image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
      elif image.shape[2] == 1:
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

      # rồi mới:
      if self.input_format.upper() == 'RGB':
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
      return image
    
    def calculate_multiple_of_stride(self, dimension: int) -> int:
        """Làm tròn LÊN đến bội số gần nhất của stride"""
        return int(math.ceil(dimension / self.stride) * self.stride)
    
    def smart_resize(self, image: np.ndarray, max_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Resize image với dynamic size cho YOLOv12:
        - Giữ nguyên tỷ lệ gốc
        - Scale cả chiều cao và rộng LÊN bội số gần nhất của 32
        - Chỉ padding tối thiểu nếu cần
        - KHÔNG ép thành hình vuông
        
        Ví dụ: 632x954 -> 640x960 (làm tròn lên bội của 32)
        
        Args:
            image: Input image
            max_size: (max_height, max_width) - giới hạn tối đa
            
        Returns:
            processed_image: Image đã xử lý
            meta_info: Thông tin để scale bounding boxes
        """
        if not self.validate_image(image):
            raise ValueError("Invalid image format.")
        
        # Convert to RGB
        image_rgb = self.convert_to_rgb(image)
        original_h, original_w = image_rgb.shape[:2]
        
        logger.debug(f"Original size: {original_w}x{original_h}")
        
        # Bước 1: Tính scale để fit trong max_size (nếu có)
        scale = 1.0
        if max_size:
            max_h, max_w = max_size
            scale = min(max_h / original_h, max_w / original_w)
            logger.debug(f"Scaling to fit max_size {max_w}x{max_h}: scale={scale:.3f}")
        
        # Bước 2: Tính kích thước sau scale (giữ tỷ lệ)
        scaled_h = original_h * scale
        scaled_w = original_w * scale
        
        # Bước 3: Làm tròn LÊN đến bội số của stride
        # Đây là điểm QUAN TRỌNG: làm tròn lên, không làm tròn xuống
        target_h = self.calculate_multiple_of_stride(scaled_h)
        target_w = self.calculate_multiple_of_stride(scaled_w)
        
        # Bước 4: Tính scale thực tế (sau khi làm tròn)
        # Scale cả hai chiều bằng nhau để giữ tỷ lệ
        scale_h = target_h / original_h
        scale_w = target_w / original_w
        
        # Dùng scale nhỏ hơn để đảm bảo fit
        actual_scale = min(scale_h, scale_w)
        
        # Bước 5: Resize với scale thực tế
        resized_h = int(original_h * actual_scale)
        resized_w = int(original_w * actual_scale)
        
        # Đảm bảo resized dimensions là bội của stride
        resized_h = self.calculate_multiple_of_stride(resized_h)
        resized_w = self.calculate_multiple_of_stride(resized_w)
        
        # Resize image
        resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
        # Bước 6: Tính toán padding (nếu cần)
        # Mục tiêu: target_h x target_w (đã là bội của stride)
        pad_h = target_h - resized_h
        pad_w = target_w - resized_w
        
        # Bước 7: Apply padding nếu cần
        if pad_h > 0 or pad_w > 0:
            # Padding đều các phía
            pad_top = pad_h // 2
            pad_bottom = pad_h - pad_top
            pad_left = pad_w // 2
            pad_right = pad_w - pad_left
            
            padded = cv2.copyMakeBorder(
                resized,
                pad_top, pad_bottom,
                pad_left, pad_right,
                cv2.BORDER_CONSTANT,
                value=self.padding_color
            )
            final_h, final_w = padded.shape[:2]
            logger.debug(f"Padding: L={pad_left}, R={pad_right}, T={pad_top}, B={pad_bottom}")
        else:
            padded = resized
            final_h, final_w = resized_h, resized_w
        
        # Tạo meta info
        meta_info = {
            'original_size': (original_h, original_w),  # (height, width)
            'resized_size': (resized_h, resized_w),     # Size after resize
            'padded_size': (final_h, final_w),          # Final size (with padding)
            'padding': (pad_left if 'pad_left' in locals() else 0, 
                       pad_top if 'pad_top' in locals() else 0),
            'scale': actual_scale,
            'ratio': (resized_w / original_w, resized_h / original_h),  # width_ratio, height_ratio
            'stride': self.stride
        }
        
        logger.debug(f"Resize: {original_w}x{original_h} → {resized_w}x{resized_h} → {final_w}x{final_h}")
        
        return padded, meta_info
    
    def normalize_image(self, image: np.ndarray) -> np.ndarray:
        """
        Normalize image for YOLO model input
        Convert: uint8 [0, 255] -> float32 [0, 1]
        Reshape: HWC -> CHW -> BCHW
        """
        # Convert to float32 and normalize
        normalized = image.astype(np.float32) / 255.0
        
        # Transpose: HWC -> CHW
        transposed = np.transpose(normalized, (2, 0, 1))
        
        # Add batch dimension: CHW -> BCHW
        batched = np.expand_dims(transposed, axis=0)
        
        return batched
    
    def process_for_inference(self, image_path: str, max_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process image from file path
        
        Args:
            image_path: Path to image file
            max_size: Optional maximum size (height, width)
            
        Returns:
            tensor: Normalized tensor for inference
            meta_info: Scaling information
        """
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Cannot read image: {image_path}")
        
        # Smart resize (dynamic, keep aspect ratio)
        processed, meta_info = self.smart_resize(image, max_size)
        
        # Normalize
        tensor = self.normalize_image(processed)
        
        return tensor, meta_info
    
    def process_bytes_image(self, image_bytes: bytes, max_size: Optional[Tuple[int, int]] = None) -> Tuple[np.ndarray, Dict]:
        """
        Process image from bytes
        
        Args:
            image_bytes: Image bytes
            max_size: Optional maximum size
            
        Returns:
            tensor: Normalized tensor
            meta_info: Scaling information
        """
        # Decode bytes
        nparr = np.frombuffer(image_bytes, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if image is None:
            raise ValueError("Cannot decode image from bytes.")
        
        # Smart resize
        processed, meta_info = self.smart_resize(image, max_size)
        
        # Normalize
        tensor = self.normalize_image(processed)
        
        return tensor, meta_info
    
    def resize_to_stride_multiple(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Đơn giản: resize ảnh về bội số của stride, giữ tỷ lệ
        Không padding, không giới hạn max_size
        
        Ví dụ: 632x954 -> 640x960
        """
        if not self.validate_image(image):
            raise ValueError("Invalid image format.")
        
        # Convert to RGB
        image_rgb = self.convert_to_rgb(image)
        original_h, original_w = image_rgb.shape[:2]
        
        # Tính kích thước mới (làm tròn lên)
        new_h = self.calculate_multiple_of_stride(original_h)
        new_w = self.calculate_multiple_of_stride(original_w)
        
        # Tính scale (giữ tỷ lệ)
        scale_h = new_h / original_h
        scale_w = new_w / original_w
        scale = min(scale_h, scale_w)  # Dùng scale nhỏ hơn
        
        # Resize
        resized_h = int(original_h * scale)
        resized_w = int(original_w * scale)
        
        # Đảm bảo là bội của stride
        resized_h = self.calculate_multiple_of_stride(resized_h)
        resized_w = self.calculate_multiple_of_stride(resized_w)
        
        resized = cv2.resize(image_rgb, (resized_w, resized_h), interpolation=cv2.INTER_LINEAR)
        
        # Meta info
        meta_info = {
            'original_size': (original_h, original_w),
            'resized_size': (resized_h, resized_w),
            'padded_size': (resized_h, resized_w),  # No padding
            'padding': (0, 0, 0, 0),  # No padding
            'scale': scale,
            'ratio': (resized_w / original_w, resized_h / original_h),
            'stride': self.stride
        }
        
        logger.debug(f"Resize to stride multiple: {original_w}x{original_h} -> {resized_w}x{resized_h}")
        
        return resized, meta_info