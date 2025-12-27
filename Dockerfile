FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Cài dependency hệ thống cho OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements trước để cache
COPY requirements.txt .

# Cài python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY . .

# Expose port FastAPI
EXPOSE 8000

# Chạy app (KHÔNG reload)
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
