# YOLO11 Helmet Detection Training

Dự án huấn luyện mô hình YOLO11 để phát hiện người có/không có mũ bảo hiểm.

## Yêu cầu

- Python 3.8+
- PyTorch với CUDA
- Ultralytics YOLO11

## Cài đặt

```bash
pip install ultralytics
```

## Cấu trúc Dataset

```
dataset/
├── data.yaml
├── train/
│   ├── images/
│   └── labels/
├── valid/
│   ├── images/
│   └── labels/
└── test/
    ├── images/
    └── labels/
```

## Huấn luyện

Chạy script training:

```bash
python train.py
```

## Tùy chỉnh tham số

Mở file `train.py` và chỉnh sửa các tham số:

- `model = YOLO('yolo11n.pt')`: Chọn model size (n/s/m/l/x)
- `epochs=100`: Số epoch
- `batch=16`: Batch size (giảm nếu GPU hết bộ nhớ)
- `imgsz=640`: Kích thước ảnh input

## Kết quả

Sau khi training, kết quả sẽ được lưu tại:
- `runs/detect/helmet-detection/weights/best.pt` - Model tốt nhất
- `runs/detect/helmet-detection/weights/last.pt` - Model cuối cùng

## Test ảnh

Sau khi training xong, bạn có thể test model trên ảnh bằng script `track.py` ở chế độ detection:

### Cách 1: Chế độ tương tác (Menu)

```bash
python track.py
```

Chọn option "2. Detect ảnh" và làm theo hướng dẫn.

### Cách 2: Dòng lệnh

```bash
# Detect một ảnh
python track.py --mode image --source image.jpg

# Detect thư mục ảnh
python track.py --mode image --source folder/

# Với các tùy chọn
python track.py --mode image --source image.jpg --conf 0.3 --show
```

### Sử dụng trong code Python

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('runs/detect/helmet-detection/weights/best.pt')

# Predict trên ảnh
results = model.predict('path/to/image.jpg', conf=0.25, save=True)

# Hiển thị kết quả
for result in results:
    for box in result.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        class_name = model.names[cls]
        print(f"{class_name}: {conf:.2%}")
```

## Kết quả test

Ảnh kết quả sẽ được lưu tại:
- `runs/detect/helmet-detect/` - Khi detect ảnh đơn hoặc thư mục

## Tracking và Detection

Sau khi training, bạn có thể sử dụng script `track.py` để:
- **Track objects trong video** (theo dõi người có/không có mũ bảo hiểm qua các frame với tracking ID)
- **Detect objects trong ảnh** (phát hiện người có/không có mũ bảo hiểm trong ảnh)

### Cách 1: Chế độ tương tác (Menu)

```bash
python track.py
```

Script sẽ hiển thị menu để bạn chọn:
- Track video hoặc Detect ảnh
- Nhập đường dẫn video/ảnh
- Tùy chỉnh các tham số

### Cách 2: Dòng lệnh

#### Track video

```bash
# Track video file
python track.py --mode video --source video.mp4

# Track webcam (0 = camera đầu tiên)
python track.py --mode video --source 0

# Với các tùy chọn
python track.py --mode video --source video.mp4 --conf 0.3 --tracker bytetrack.yaml --show
```

#### Detect ảnh

```bash
# Detect một ảnh
python track.py --mode image --source image.jpg

# Detect thư mục ảnh
python track.py --mode image --source folder/

# Với các tùy chọn
python track.py --mode image --source image.jpg --conf 0.3 --show
```

### Tham số

- `--mode`: `video` (tracking) hoặc `image` (detection)
- `--source`: Đường dẫn video/ảnh hoặc webcam (0, 1, 2...)
- `--model`: Đường dẫn model (mặc định: `runs/detect/helmet-detection/weights/best.pt`)
- `--conf`: Confidence threshold (mặc định: 0.25)
- `--tracker`: Loại tracker cho video - `bytetrack.yaml` hoặc `botsort.yaml` (mặc định: `bytetrack.yaml`)
- `--show`: Hiển thị kết quả real-time
- `--output`: Thư mục lưu kết quả (mặc định: `runs/track` cho video, `runs/detect` cho ảnh)
- `--args`: Đường dẫn đến `args.yaml` để load cấu hình từ training (mặc định: tự động tìm tại `runs/detect/helmet-detection/args.yaml`)
- `--no-args`: Không sử dụng cấu hình từ `args.yaml`

**Lưu ý**: Script sẽ tự động load cấu hình từ `args.yaml` (nếu có) để sử dụng các tham số như `imgsz`, `iou`, `device` từ quá trình training.

### Kết quả

- **Video tracking**: Kết quả được lưu tại `runs/track/helmet-tracking/`
  - Video đã track: `runs/track/helmet-tracking/video.mp4`
  - Tracking data: `runs/track/helmet-tracking/tracks/` (file .txt với tracking IDs)
  
- **Image detection**: Kết quả được lưu tại `runs/detect/helmet-detect/`
  - Ảnh đã detect: `runs/detect/helmet-detect/`
  - Detection data: `runs/detect/helmet-detect/labels/` (file .txt với bounding boxes)

### Ví dụ sử dụng trong code Python

```python
from ultralytics import YOLO

# Load model đã train
model = YOLO('runs/detect/helmet-detection/weights/best.pt')

# Track video
results = model.track(
    source='video.mp4',
    conf=0.25,
    tracker='bytetrack.yaml',
    save=True
)

# Detect ảnh
results = model.predict(
    source='image.jpg',
    conf=0.25,
    save=True
)

# Xử lý kết quả tracking
for result in results:
    if result.boxes.id is not None:  # Có tracking ID
        for box, track_id in zip(result.boxes, result.boxes.id):
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            track_id = int(track_id)
            class_name = model.names[cls]
            print(f"ID: {track_id}, {class_name}: {conf:.2%}")
```

