import os
from pathlib import Path

# ĐIỀN ĐƯỜNG DẪN FOLDER CỦA BẠN VÀO ĐÂY
FOLDER_PATH = r"E:\School\HK7\document\KhoaLuan\Dataset\final-dataset-no-normal\images\train"  # Thay đổi đường dẫn này

def find_largest_image_file(folder_path):
    image_extensions = {'.jpg', '.jpeg', '.png', '.gif', '.bmp', 
                       '.tiff', '.webp', '.raw', '.heic', '.heif'}
    
    largest_file = None
    largest_size = 0
    
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            file_extension = Path(file).suffix.lower()
            if file_extension in image_extensions:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                if file_size > largest_size:
                    largest_size = file_size
                    largest_file = file_path
    
    return largest_file, largest_size

def format_file_size(size_bytes):
    if size_bytes < 1024:
        return f"{size_bytes} bytes"
    elif size_bytes < 1024 * 1024:
        return f"{size_bytes / 1024:.2f} KB"
    elif size_bytes < 1024 * 1024 * 1024:
        return f"{size_bytes / (1024 * 1024):.2f} MB"
    else:
        return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"

def main():
    folder_path = FOLDER_PATH
    
    if not os.path.exists(folder_path):
        print(f"Thư mục không tồn tại: {folder_path}")
        return
    
    print(f"🔍 Đang tìm kiếm trong: {folder_path}")
    
    largest_file, largest_size = find_largest_image_file(folder_path)
    
    if largest_file:
        print("\n" + "="*50)
        print(f"📷 File ảnh nặng nhất:")
        print(f"   📄 Tên: {os.path.basename(largest_file)}")
        print(f"   📁 Đường dẫn: {largest_file}")
        print(f"   💾 Dung lượng: {format_file_size(largest_size)}")
        print(f"      ({largest_size:,} bytes)")
        print("="*50)
    else:
        print("❌ Không tìm thấy file ảnh trong thư mục!")

if __name__ == "__main__":
    main()