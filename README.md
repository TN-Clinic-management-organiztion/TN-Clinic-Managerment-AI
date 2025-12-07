uv run uvicorn app.main:app --reload
uv run python app/tests/test_yolo.py
uv pip install ultralytics==8.3.63