"""
YOLO11 Training Script
Train model để phát hiện người có/không có mũ bảo hiểm
"""

from ultralytics import YOLO
import torch

if __name__ == '__main__':
    # Kiểm tra CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
        print(f"CUDA version: {torch.version.cuda}")

    # Load model YOLO11 (có thể chọn: yolov11n, yolov11s, yolov11m, yolov11l, yolov11x)
    # n = nano (nhỏ nhất, nhanh nhất)
    # s = small
    # m = medium
    # l = large
    # x = xlarge (lớn nhất, chính xác nhất)
    model = YOLO('yolo11n.pt')  # Bắt đầu với model nhỏ, có thể đổi thành yolov11s.pt, yolov11m.pt, etc.

    # Train model
    results = model.train(
        data='dataset/data.yaml',  # Đường dẫn đến file cấu hình dataset
        epochs=100,                 # Số epoch (có thể tăng nếu cần)
        imgsz=640,                  # Kích thước ảnh input
        batch=16,                   # Batch size (giảm nếu GPU hết bộ nhớ)
        device=0 if torch.cuda.is_available() else 'cpu',  # Sử dụng GPU nếu có
        workers=8,                  # Số worker để load data
        project='runs/detect',      # Thư mục lưu kết quả
        name='helmet-detection',    # Tên thư mục experiment
        exist_ok=True,              # Ghi đè nếu thư mục đã tồn tại
        patience=50,                # Early stopping patience
        save=True,                  # Lưu checkpoint
        save_period=10,             # Lưu checkpoint mỗi 10 epoch
        val=True,                   # Validate trong quá trình train
        plots=True,                 # Tạo plots
        verbose=True,               # Hiển thị chi tiết
    )

    print("Training completed!")
    print(f"Best model saved at: {model.trainer.best}")

