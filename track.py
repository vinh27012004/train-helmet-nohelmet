"""
YOLO11 Tracking Script
Chương trình tracking người có/không có mũ bảo hiểm từ video hoặc ảnh
"""

import os
import sys
from pathlib import Path
from ultralytics import YOLO
import torch
import cv2
import argparse
import yaml

# Đường dẫn model mặc định
DEFAULT_MODEL_PATH = 'runs/detect/helmet-detection/weights/best.pt'
# Đường dẫn args.yaml mặc định (nếu có)
DEFAULT_ARGS_PATH = 'runs/detect/helmet-detection/args.yaml'


def detect_source_type(source):
    """
    Tự động phát hiện loại nguồn (video, image, webcam, hoặc thư mục)
    
    Args:
        source: Đường dẫn file/thư mục hoặc webcam ID
    
    Returns:
        str: 'video', 'image', 'webcam', 'directory', hoặc None
    """
    # Kiểm tra webcam
    if isinstance(source, str) and source.isdigit():
        return 'webcam'
    if isinstance(source, int):
        return 'webcam'
    
    # Kiểm tra file tồn tại
    if not os.path.exists(source):
        return None
    
    # Kiểm tra thư mục
    if os.path.isdir(source):
        # Kiểm tra xem thư mục có chứa ảnh hay video
        extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp']
        video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm']
        
        files = os.listdir(source)
        has_images = any(f.lower().endswith(tuple(extensions)) for f in files)
        has_videos = any(f.lower().endswith(tuple(video_extensions)) for f in files)
        
        if has_videos:
            return 'directory_video'
        elif has_images:
            return 'directory_image'
        else:
            return 'directory'
    
    # Kiểm tra extension
    source_lower = source.lower()
    
    # Video extensions
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v']
    if any(source_lower.endswith(ext) for ext in video_extensions):
        return 'video'
    
    # Image extensions
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', '.gif']
    if any(source_lower.endswith(ext) for ext in image_extensions):
        return 'image'
    
    return None


def load_args_config(args_path=None):
    """
    Load cấu hình từ args.yaml nếu có
    
    Args:
        args_path: Đường dẫn đến args.yaml (None để tự động tìm)
    
    Returns:
        dict: Dictionary chứa các tham số từ args.yaml hoặc None
    """
    if args_path is None:
        args_path = DEFAULT_ARGS_PATH
    
    if not os.path.exists(args_path):
        return None
    
    try:
        with open(args_path, 'r', encoding='utf-8') as f:
            config = yaml.safe_load(f)
        print(f"✅ Đã load cấu hình từ: {args_path}")
        return config
    except Exception as e:
        print(f"⚠️  Không thể load args.yaml: {e}")
        return None


def track_video_with_pause(model, video_path, output_dir, conf, tracker, imgsz, iou, device, is_webcam):
    """
    Track video với khả năng pause/resume khi hiển thị
    
    Controls:
    - Space hoặc 'p': Pause/Resume
    - 'q' hoặc ESC: Thoát
    - 's': Step (chỉ khi đang pause, chuyển sang frame tiếp theo)
    """
    
    print("\n⌨️  Điều khiển:")
    print("   - Phím SPACE hoặc 'p': Tạm dừng/Tiếp tục")
    print("   - Phím 's': Bước tiếp (chỉ khi đang pause)")
    print("   - Phím 'q' hoặc ESC: Thoát")
    
    # Mở video
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        print(f"❌ Không thể mở video: {video_path}")
        return None
    
    # Lấy thông tin video
    fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else 0
    
    # Tạo VideoWriter để lưu video
    output_path = os.path.join(output_dir, 'helmet-tracking')
    os.makedirs(output_path, exist_ok=True)
    
    if not is_webcam:
        # Lấy tên file từ video_path (có thể là string hoặc Path)
        video_name = os.path.basename(str(video_path))
        output_video_path = os.path.join(output_path, video_name)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
    else:
        out = None
        output_video_path = None
    
    # Biến trạng thái
    paused = False
    step_mode = False
    frame_count = 0
    total_helmet = 0
    total_no_helmet = 0
    unique_tracks = set()
    
    print(f"\n▶️  Đang phát video... (Nhấn SPACE để pause/resume)")
    
    try:
        while True:
            if not paused or step_mode:
                ret, frame = cap.read()
                if not ret:
                    if is_webcam:
                        continue  # Webcam có thể không có frame ngay
                    break
                
                frame_count += 1
                step_mode = False
                
                # Track frame
                results = model.track(
                    source=frame,
                    conf=conf,
                    tracker=tracker,
                    save=False,
                    verbose=False,
                    imgsz=imgsz,
                    iou=iou,
                    device=device
                )
                
                # Vẽ kết quả lên frame
                annotated_frame = results[0].plot()
                
                # Thống kê
                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        cls = int(box.cls[0])
                        class_name = model.names[cls]
                        
                        # Lấy tracking ID nếu có
                        if hasattr(box, 'id') and box.id is not None:
                            track_id = int(box.id[0])
                            unique_tracks.add(track_id)
                        
                        if 'with_helmet' in class_name:
                            total_helmet += 1
                        elif 'without_helmet' in class_name:
                            total_no_helmet += 1
                
                # Thêm thông tin lên frame
                info_text = f"Frame: {frame_count}"
                if not is_webcam:
                    info_text += f" / {total_frames}"
                if paused:
                    info_text += " [PAUSED]"
                
                cv2.putText(annotated_frame, info_text, (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Lưu frame
                if out:
                    out.write(annotated_frame)
                
                # Hiển thị
                cv2.imshow('Helmet Tracking - Press SPACE to pause, Q to quit', annotated_frame)
            
            # Xử lý phím
            key = cv2.waitKey(1 if not paused else 0) & 0xFF
            
            if key == ord('q') or key == 27:  # 'q' hoặc ESC
                print("\n⏹️  Đã dừng video")
                break
            elif key == ord(' ') or key == ord('p'):  # Space hoặc 'p'
                paused = not paused
                if paused:
                    print("⏸️  Đã tạm dừng (Nhấn SPACE để tiếp tục, 's' để bước tiếp, 'q' để thoát)")
                else:
                    print("▶️  Đã tiếp tục")
            elif key == ord('s') and paused:  # Step (chỉ khi pause)
                step_mode = True
                print("⏭️  Bước tiếp...")
    
    except KeyboardInterrupt:
        print("\n⏹️  Đã dừng video (Ctrl+C)")
    finally:
        cap.release()
        if out:
            out.release()
        cv2.destroyAllWindows()
        
        # Thống kê
        print(f"\n{'='*60}")
        print("THỐNG KÊ KẾT QUẢ")
        print(f"{'='*60}")
        print(f"📊 Tổng số frame đã xử lý: {frame_count}")
        print(f"👤 Tổng số object được phát hiện:")
        print(f"   - Có mũ bảo hiểm: {total_helmet}")
        print(f"   - Không có mũ bảo hiểm: {total_no_helmet}")
        print(f"   - Tổng cộng: {total_helmet + total_no_helmet}")
        if unique_tracks:
            print(f"🆔 Số lượng object unique được track: {len(unique_tracks)}")
        
        print(f"\n✅ Hoàn thành! Kết quả đã được lưu tại: {output_path}")
        if not is_webcam and output_video_path and os.path.exists(output_video_path):
            print(f"📹 Video đã được lưu tại: {output_video_path}")
    
    return None


def track_video(model_path, video_path, output_dir='runs/track', conf=0.25, tracker='bytetrack.yaml', show=False, args_config=None):
    """
    Track objects trong video
    
    Args:
        model_path: Đường dẫn đến model đã train (.pt)
        video_path: Đường dẫn đến video file hoặc webcam (0, 1, 2...)
        output_dir: Thư mục lưu kết quả
        conf: Confidence threshold
        tracker: Loại tracker (bytetrack.yaml, botsort.yaml)
        show: Hiển thị video real-time
    """
    print(f"\n{'='*60}")
    print("TRACKING VIDEO")
    print(f"{'='*60}")
    
    # Load model
    print(f"Đang load model từ: {model_path}")
    try:
        model = YOLO(model_path)
        print("✅ Model đã được load thành công!")
    except Exception as e:
        print(f"❌ Lỗi khi load model: {e}")
        return None
    
    # Kiểm tra video path
    is_webcam = False
    if isinstance(video_path, str) and video_path.isdigit():
        video_path = int(video_path)
        is_webcam = True
        print(f"📹 Đang sử dụng webcam: {video_path}")
    else:
        if not os.path.exists(video_path):
            print(f"❌ Không tìm thấy video: {video_path}")
            return None
        print(f"📹 Đang xử lý video: {video_path}")
    
    # Áp dụng cấu hình từ args.yaml nếu có
    imgsz = 640
    iou = 0.7
    device = 0 if torch.cuda.is_available() else 'cpu'
    
    if args_config:
        # Lấy các tham số từ args.yaml
        if args_config.get('imgsz'):
            imgsz = args_config['imgsz']
        if args_config.get('iou'):
            iou = args_config['iou']
        if args_config.get('device'):
            device = args_config['device']
        if args_config.get('conf') is not None and conf == 0.25:  # Chỉ dùng nếu chưa được chỉ định
            conf = args_config['conf'] if args_config['conf'] else 0.25
        # Lấy tracker từ args.yaml nếu có (nhưng vẫn ưu tiên tham số truyền vào)
        if args_config.get('tracker') and tracker == 'bytetrack.yaml':
            tracker_from_args = args_config['tracker']
            if tracker_from_args in ['bytetrack.yaml', 'botsort.yaml']:
                tracker = tracker_from_args
    
    # Thông tin xử lý
    print(f"\n⚙️  Cấu hình:")
    print(f"   - Confidence threshold: {conf}")
    print(f"   - Tracker: {tracker}")
    print(f"   - Image size: {imgsz}")
    print(f"   - IOU threshold: {iou}")
    print(f"   - Device: {device}")
    print(f"   - Hiển thị real-time: {'Có' if show else 'Không'}")
    if args_config:
        print(f"   - 📋 Đã áp dụng cấu hình từ args.yaml")
    print(f"\n🔄 Đang xử lý video...")
    
    # Track video
    try:
        # Nếu show=True, sử dụng xử lý thủ công để có thể pause
        if show:
            return track_video_with_pause(
                model=model,
                video_path=video_path,
                output_dir=output_dir,
                conf=conf,
                tracker=tracker,
                imgsz=imgsz,
                iou=iou,
                device=device,
                is_webcam=is_webcam
            )
        
        # Nếu không show, sử dụng cách cũ (nhanh hơn)
        results = model.track(
            source=video_path,
            conf=conf,
            tracker=tracker,
            save=True,
            project=output_dir,
            name='helmet-tracking',
            exist_ok=True,
            show=False,
            save_txt=True,  # Lưu tracking results dạng text
            save_conf=True,
            line_width=2,
            verbose=True,
            imgsz=imgsz,  # Kích thước ảnh xử lý từ args.yaml
            iou=iou,  # IOU threshold từ args.yaml
            device=device  # Device từ args.yaml
        )
        
        # Thống kê kết quả
        print(f"\n{'='*60}")
        print("THỐNG KÊ KẾT QUẢ")
        print(f"{'='*60}")
        
        total_frames = 0
        total_helmet = 0
        total_no_helmet = 0
        unique_tracks = set()
        
        for result in results:
            total_frames += 1
            if result.boxes is not None and len(result.boxes) > 0:
                for box in result.boxes:
                    cls = int(box.cls[0])
                    class_name = model.names[cls]
                    
                    # Lấy tracking ID nếu có
                    if hasattr(box, 'id') and box.id is not None:
                        track_id = int(box.id[0])
                        unique_tracks.add(track_id)
                    
                    if 'with_helmet' in class_name:
                        total_helmet += 1
                    elif 'without_helmet' in class_name:
                        total_no_helmet += 1
        
        print(f"📊 Tổng số frame đã xử lý: {total_frames}")
        print(f"👤 Tổng số object được phát hiện:")
        print(f"   - Có mũ bảo hiểm: {total_helmet}")
        print(f"   - Không có mũ bảo hiểm: {total_no_helmet}")
        print(f"   - Tổng cộng: {total_helmet + total_no_helmet}")
        if unique_tracks:
            print(f"🆔 Số lượng object unique được track: {len(unique_tracks)}")
        
        print(f"\n✅ Hoàn thành! Kết quả đã được lưu tại: {output_dir}/helmet-tracking")
        if not is_webcam:
            output_video = os.path.join(output_dir, 'helmet-tracking', os.path.basename(video_path))
            if os.path.exists(output_video):
                print(f"📹 Video đã được lưu tại: {output_video}")
        
        return results
        
    except Exception as e:
        print(f"❌ Lỗi khi xử lý video: {e}")
        import traceback
        traceback.print_exc()
        return None


def detect_image(model_path, image_path, output_dir='runs/detect', conf=0.25, show=False):
    """
    Detect objects trong ảnh (không có tracking vì chỉ là ảnh đơn)
    
    Args:
        model_path: Đường dẫn đến model đã train (.pt)
        image_path: Đường dẫn đến ảnh hoặc thư mục ảnh
        output_dir: Thư mục lưu kết quả
        conf: Confidence threshold
        show: Hiển thị ảnh
    """
    print(f"\n{'='*60}")
    print("DETECT ẢNH")
    print(f"{'='*60}")
    
    # Load model
    print(f"Đang load model từ: {model_path}")
    model = YOLO(model_path)
    
    # Kiểm tra image path
    if not os.path.exists(image_path):
        print(f"❌ Không tìm thấy ảnh/thư mục: {image_path}")
        return
    
    # Detect ảnh
    results = model.predict(
        source=image_path,
        conf=conf,
        save=True,
        project=output_dir,
        name='helmet-detect',
        exist_ok=True,
        show=show,
        save_txt=True,
        save_conf=True,
        line_width=2,
        verbose=True
    )
    
    # Hiển thị thống kê
    print(f"\n{'='*60}")
    print("THỐNG KÊ KẾT QUẢ")
    print(f"{'='*60}")
    
    total_helmet = 0
    total_no_helmet = 0
    
    for result in results:
        for box in result.boxes:
            cls = int(box.cls[0])
            class_name = model.names[cls]
            if 'with_helmet' in class_name:
                total_helmet += 1
            elif 'without_helmet' in class_name:
                total_no_helmet += 1
    
    print(f"Tổng số người có mũ bảo hiểm: {total_helmet}")
    print(f"Tổng số người không có mũ bảo hiểm: {total_no_helmet}")
    print(f"Tổng số người: {total_helmet + total_no_helmet}")
    
    print(f"\n✅ Hoàn thành! Kết quả đã được lưu tại: {output_dir}/helmet-detect")
    return results


def main():
    parser = argparse.ArgumentParser(
        description='Tracking và Detection mũ bảo hiểm với YOLO11 - Tự động phát hiện video/ảnh',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ví dụ sử dụng:
  # Tự động phát hiện và xử lý (khuyến nghị)
  python track.py --source video.mp4
  python track.py --source image.jpg
  python track.py --source 0  # webcam
  python track.py --source folder/  # thư mục ảnh/video
  
  # Chỉ định mode cụ thể (nếu cần)
  python track.py --mode video --source video.mp4
  python track.py --mode image --source image.jpg
  
  # Với các tùy chọn khác
  python track.py --source video.mp4 --conf 0.5 --show
  python track.py --source image.jpg --conf 0.3
        """
    )
    
    parser.add_argument('--mode', type=str, choices=['video', 'image'], default=None,
                       help='Chế độ: video (tracking) hoặc image (detection). Nếu không chỉ định, sẽ tự động phát hiện từ --source')
    parser.add_argument('--source', type=str, required=True,
                       help='Đường dẫn video/ảnh, thư mục, hoặc webcam (0, 1, 2...)')
    parser.add_argument('--model', type=str, default=DEFAULT_MODEL_PATH,
                       help=f'Đường dẫn đến model đã train (mặc định: {DEFAULT_MODEL_PATH})')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold (mặc định: 0.25)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                       choices=['bytetrack.yaml', 'botsort.yaml'],
                       help='Loại tracker (mặc định: bytetrack.yaml) - chỉ dùng cho video')
    parser.add_argument('--show', action='store_true',
                       help='Hiển thị kết quả real-time')
    parser.add_argument('--output', type=str, default=None,
                       help='Thư mục lưu kết quả (mặc định: runs/track cho video, runs/detect cho ảnh)')
    parser.add_argument('--args', type=str, default=None,
                       help='Đường dẫn đến args.yaml để load cấu hình (mặc định: tự động tìm)')
    parser.add_argument('--no-args', action='store_true',
                       help='Không sử dụng cấu hình từ args.yaml')
    
    args = parser.parse_args()
    
    # Load args.yaml nếu có
    args_config = None
    if not args.no_args:
        args_config = load_args_config(args.args)
    
    # Kiểm tra model
    if not os.path.exists(args.model):
        print(f"❌ Không tìm thấy model tại: {args.model}")
        print(f"Vui lòng kiểm tra đường dẫn hoặc train model trước!")
        sys.exit(1)
    
    # Kiểm tra CUDA
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    
    # Tự động phát hiện mode nếu không được chỉ định
    if args.mode is None:
        source_type = detect_source_type(args.source)
        
        if source_type is None:
            print(f"❌ Không thể phát hiện loại file từ: {args.source}")
            print(f"Vui lòng chỉ định --mode video hoặc --mode image")
            sys.exit(1)
        
        if source_type in ['video', 'webcam', 'directory_video']:
            args.mode = 'video'
            print(f"✅ Tự động phát hiện: VIDEO")
        elif source_type in ['image', 'directory_image']:
            args.mode = 'image'
            print(f"✅ Tự động phát hiện: ẢNH")
        else:
            print(f"⚠️  Không thể xác định loại file, mặc định sử dụng mode: image")
            args.mode = 'image'
    else:
        print(f"📋 Sử dụng mode đã chỉ định: {args.mode.upper()}")
    
    # Xử lý theo mode
    if args.mode == 'video':
        output_dir = args.output if args.output else 'runs/track'
        track_video(
            model_path=args.model,
            video_path=args.source,
            output_dir=output_dir,
            conf=args.conf,
            tracker=args.tracker,
            show=args.show,
            args_config=args_config
        )
    elif args.mode == 'image':
        output_dir = args.output if args.output else 'runs/detect'
        detect_image(
            model_path=args.model,
            image_path=args.source,
            output_dir=output_dir,
            conf=args.conf,
            show=args.show
        )


def interactive_mode():
    """Chế độ tương tác với menu - Tự động phát hiện video/ảnh"""
    print(f"\n{'='*60}")
    print("CHƯƠNG TRÌNH TRACKING MŨ BẢO HIỂM")
    print("Tự động phát hiện video/ảnh và tracking")
    print(f"{'='*60}\n")
    
    # Sử dụng model mặc định
    model_path = DEFAULT_MODEL_PATH
    print(f"📦 Sử dụng model: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"❌ Không tìm thấy model tại: {model_path}")
        print(f"Vui lòng kiểm tra đường dẫn hoặc train model trước!")
        return
    
    # Tự động load args.yaml nếu có
    args_config = load_args_config()
    if args_config:
        print(f"📋 Đã tìm thấy args.yaml, sẽ sử dụng cấu hình từ file này")
        use_args = input("   Sử dụng cấu hình từ args.yaml? (y/n) [y]: ").strip().lower()
        if use_args and use_args != 'y':
            args_config = None
    else:
        args_config = None
    
    # Nhập source
    print("\n📁 Nhập đường dẫn file hoặc chọn nguồn:")
    print("   - Video file: video.mp4, video.avi, ...")
    print("   - Ảnh file: image.jpg, image.png, ...")
    print("   - Thư mục: folder/ (tự động phát hiện ảnh/video)")
    print("   - Webcam: 0, 1, 2, ...")
    source = input("\nĐường dẫn hoặc webcam ID: ").strip()
    
    if not source:
        print("❌ Vui lòng nhập đường dẫn hoặc webcam ID!")
        return
    
    # Loại bỏ dấu ngoặc kép nếu có
    source = source.strip('"').strip("'")
    
    # Tự động phát hiện loại source
    source_type = detect_source_type(source)
    
    if source_type is None:
        print(f"❌ Không thể phát hiện loại file từ: {source}")
        print("Vui lòng kiểm tra lại đường dẫn!")
        return
    
    # Hiển thị loại đã phát hiện
    type_names = {
        'video': '📹 Video file',
        'image': '🖼️ Ảnh file',
        'webcam': '📹 Webcam',
        'directory_video': '📁 Thư mục chứa video',
        'directory_image': '📁 Thư mục chứa ảnh',
        'directory': '📁 Thư mục'
    }
    print(f"\n✅ Đã phát hiện: {type_names.get(source_type, source_type)}")
    
    # Confidence threshold
    print("\n⚙️  Confidence threshold (0.0 - 1.0):")
    print("   - Giá trị thấp (0.1-0.3): Phát hiện nhiều hơn, có thể có lỗi")
    print("   - Giá trị cao (0.5-0.9): Chính xác hơn, có thể bỏ sót")
    conf_input = input("   Nhập giá trị [0.25]: ").strip()
    try:
        conf = float(conf_input) if conf_input else 0.25
        if conf < 0 or conf > 1:
            print("⚠️  Giá trị không hợp lệ, sử dụng mặc định 0.25")
            conf = 0.25
    except ValueError:
        print("⚠️  Giá trị không hợp lệ, sử dụng mặc định 0.25")
        conf = 0.25
    
    # Xử lý theo loại source
    if source_type in ['video', 'webcam', 'directory_video']:
        # Tracker - chỉ cho video
        print("\n⚙️  Tracker (thuật toán theo dõi object):")
        print("   1. bytetrack.yaml - Nhanh, hiệu quả (khuyến nghị)")
        print("   2. botsort.yaml - Chính xác hơn, tốt cho object di chuyển nhanh")
        tracker_choice = input("   Lựa chọn (1/2) [1]: ").strip()
        if tracker_choice == '2':
            tracker = 'botsort.yaml'
        else:
            tracker = 'bytetrack.yaml'
        
        show_input = input("Hiển thị video real-time? (y/n) [y]: ").strip().lower()
        show = show_input == 'y' if show_input else True
        
        track_video(model_path, source, conf=conf, tracker=tracker, show=show, args_config=args_config)
    
    elif source_type in ['image', 'directory_image']:
        show = input("Hiển thị ảnh? (y/n) [n]: ").strip().lower() == 'y'
        detect_image(model_path, source, conf=conf, show=show)
    
    else:
        # Thư mục không xác định, hỏi người dùng
        print("\n⚠️  Không thể xác định loại file trong thư mục")
        print("Chọn chế độ xử lý:")
        print("1. Xử lý như ảnh (detection)")
        print("2. Xử lý như video (tracking)")
        choice = input("Lựa chọn (1/2) [1]: ").strip()
        
        if choice == '2':
            print("\n⚙️  Tracker:")
            print("   1. bytetrack.yaml - Nhanh, hiệu quả (khuyến nghị)")
            print("   2. botsort.yaml - Chính xác hơn")
            tracker_choice = input("   Lựa chọn (1/2) [1]: ").strip()
            tracker = 'botsort.yaml' if tracker_choice == '2' else 'bytetrack.yaml'
            show = input("Hiển thị real-time? (y/n) [y]: ").strip().lower() == 'y'
            track_video(model_path, source, conf=conf, tracker=tracker, show=show, args_config=args_config)
        else:
            show = input("Hiển thị ảnh? (y/n) [n]: ").strip().lower() == 'y'
            detect_image(model_path, source, conf=conf, show=show)


if __name__ == '__main__':
    # Nếu không có tham số dòng lệnh, chạy chế độ tương tác
    if len(sys.argv) == 1:
        interactive_mode()
    else:
        main()

