import sys
import ast
import json
import onnx


def parse_names_value(raw: str):
    """
    Thử parse value của metadata 'names' thành dict:
    - Ưu tiên ast.literal_eval (chuỗi kiểu {0: 'person', 1: 'car', ...})
    - Nếu fail thì thử parse JSON
    """
    raw = raw.strip()
    # Nếu là string kiểu python dict
    try:
        value = ast.literal_eval(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    # Nếu là JSON
    try:
        value = json.loads(raw)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    return None


def list_model_classes(model_path: str):
    print(f"Loading ONNX model: {model_path}")
    onnx_model = onnx.load(model_path)

    # ---- 1. In ra toàn bộ metadata để debug ----
    meta = {p.key: p.value for p in onnx_model.metadata_props}
    if meta:
        print("\n=== Metadata properties ===")
        for k, v in meta.items():
            short_v = v if len(v) < 200 else v[:200] + "... [truncated]"
            print(f"- {k}: {short_v}")
    else:
        print("\nKhông có metadata_props trong ONNX (model không lưu thêm info).")

    # ---- 2. Thử lấy names từ metadata ----
    # Ultralytics YOLO thường dùng key 'names'
    candidate_keys = ["names", "class_names", "classes"]

    names_dict = None
    used_key = None

    for key in candidate_keys:
        if key in meta:
            parsed = parse_names_value(meta[key])
            if isinstance(parsed, dict):
                names_dict = parsed
                used_key = key
                break

    if names_dict is not None:
        print(f"\n=== Classes found from metadata key='{used_key}' ===")
        # Sort theo index cho chắc
        try:
            items = sorted(names_dict.items(), key=lambda x: int(x[0]))
        except Exception:
            # Nếu key là string không convert được sang int thì cứ sort theo key
            items = sorted(names_dict.items(), key=lambda x: x[0])

        for k, v in items:
            print(f"{k}: {v}")
        print(f"\n→ Total classes: {len(items)}")
        return

    print("\n❌ Không tìm được danh sách class trong metadata ('names' / 'class_names' / 'classes').")

    # ---- 3. Fallback: chỉ đoán số lượng class từ shape output (không có tên) ----
    print("\n>>> Fallback: đoán số lượng class từ output shape (YOLOv8/YOLOv12 style: 4 + num_classes)")

    if not onnx_model.graph.output:
        print("Model không có output nào trong graph.")
        return

    for output in onnx_model.graph.output:
        shape = []
        for dim in output.type.tensor_type.shape.dim:
            if dim.HasField("dim_value"):
                shape.append(dim.dim_value)
            else:
                shape.append("?")

        print(f"Output: {output.name}")
        print(f"  Shape: {shape}")

        if len(shape) == 3 and isinstance(shape[1], int):
            num_classes = shape[1] - 4  # YOLOv8/12: 4 bbox + num_classes
            print(f"  → Inferred num_classes (4 + num_classes): {num_classes}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python list_onnx_classes.py path/to/model.onnx")
        sys.exit(1)

    model_path = sys.argv[1]
    list_model_classes(model_path)


    # uv run --with onnx --with onnxruntime python .\app\tests\test_onnx_class.py .\app\onnx\models\yolov12n.onnx

