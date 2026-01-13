import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import cv2
import threading
from ultralytics import YOLO
import os
import time  # Import thêm time để kiểm soát FPS

# Cấu hình giao diện
ctk.set_appearance_mode("Dark")
ctk.set_default_color_theme("blue")

class HelmetDetectionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        # --- CẤU HÌNH CỬA SỔ ---
        self.title("Hệ Thống Nhận Diện Mũ Bảo Hiểm - YOLO11")
        self.geometry("1000x600")
        
        # Load Model
        model_path = 'runs/detect/helmet-detection/weights/best.pt'
        if not os.path.exists(model_path):
            print(f"Không tìm thấy model tại {model_path}. Đang dùng yolov8n.pt tạm thời.")
            model_path = 'yolov8n.pt'
            
        self.model = YOLO(model_path)
        print("Đã load model thành công!")

        # Biến quản lý
        self.stop_event = threading.Event()
        self.thread = None
        self.current_image_ref = None # QUAN TRỌNG: Biến giữ ảnh để không bị xóa (Garbage Collection)

        # --- LAYOUT ---
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # === CỘT TRÁI (MENU) ===
        self.sidebar_frame = ctk.CTkFrame(self, width=200, corner_radius=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_rowconfigure(5, weight=1)

        self.logo_label = ctk.CTkLabel(self.sidebar_frame, text="HELMET DETECT", 
                                     font=ctk.CTkFont(size=20, weight="bold"))
        self.logo_label.grid(row=0, column=0, padx=20, pady=(20, 10))

        self.btn_image = ctk.CTkButton(self.sidebar_frame, text="🖼️ Chọn Ảnh", command=self.select_image)
        self.btn_image.grid(row=1, column=0, padx=20, pady=10)

        self.btn_video = ctk.CTkButton(self.sidebar_frame, text="📹 Chọn Video", command=self.select_video)
        self.btn_video.grid(row=2, column=0, padx=20, pady=10)

        self.btn_webcam = ctk.CTkButton(self.sidebar_frame, text="📷 Webcam", 
                                     fg_color="green", hover_color="darkgreen",
                                     command=self.select_webcam)
        self.btn_webcam.grid(row=3, column=0, padx=20, pady=10)
        
        self.btn_stop = ctk.CTkButton(self.sidebar_frame, text="⏹️ Dừng Xử Lý", 
                                    fg_color="red", hover_color="darkred",
                                    command=self.stop_processing)
        self.btn_stop.grid(row=4, column=0, padx=20, pady=10)

        # === CỘT PHẢI (HIỂN THỊ) ===
        self.display_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.display_frame.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        
        self.display_label = ctk.CTkLabel(self.display_frame, text="Vui lòng chọn file hoặc Webcam...", 
                                        font=("Arial", 16))
        self.display_label.pack(expand=True, fill="both")

        self.protocol("WM_DELETE_WINDOW", self.on_closing)

    def stop_processing(self):
        """Dừng luồng xử lý"""
        self.stop_event.set()
        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=0.5)
        self.display_label.configure(image=None, text="Đã dừng xử lý.")

    def start_video_thread(self, cap):
        self.stop_processing()
        self.stop_event.clear()
        
        self.thread = threading.Thread(target=self.process_video_loop, args=(cap,))
        self.thread.daemon = True
        self.thread.start()

    def select_image(self):
        self.stop_processing()
        file_path = filedialog.askopenfilename(filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
        if not file_path: return

        # Dự đoán
        results = self.model.predict(source=file_path, conf=0.25)
        result_img_bgr = results[0].plot() 
        self.display_image_on_gui(result_img_bgr)

    def select_video(self):
        file_path = filedialog.askopenfilename(filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if not file_path: return
        
        cap = cv2.VideoCapture(file_path)
        self.start_video_thread(cap)

    def select_webcam(self):
        self.stop_processing()
        # Thử kết nối camera với DirectShow (Fix lỗi màn hình đen/nhấp nháy ban đầu)
        found_cam = False
        for cam_index in [0, 1]:
            print(f"🔄 Đang thử Webcam {cam_index}...")
            cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
            if cap.isOpened():
                ret, frame = cap.read()
                if ret:
                    print(f"Webcam {cam_index} OK")
                    self.start_video_thread(cap)
                    found_cam = True
                    break
                else:
                    cap.release()
        
        if not found_cam:
            self.display_label.configure(text="Không tìm thấy Webcam!")

    def process_video_loop(self, cap):
        """Luồng xử lý video"""
        # Giới hạn FPS để tránh quá tải GUI (khoảng 30 FPS)
        target_fps = 30
        frame_time = 1.0 / target_fps

        while not self.stop_event.is_set() and cap.isOpened():
            start_time = time.time()
            
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # 1. Xử lý AI
                results = self.model.track(frame, persist=True, conf=0.25, verbose=False)
                annotated_frame = results[0].plot()
                
                # 2. Convert màu
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                
                # 3. QUAN TRỌNG: Đẩy việc vẽ lên GUI sang luồng chính
                # Dùng self.after(0, ...) để tránh xung đột luồng gây nhấp nháy
                self.after(0, self.update_gui_image, rgb_frame)
                
            except Exception as e:
                print(f"Lỗi: {e}")
                break

            # 4. Kiểm soát tốc độ (Sleep nhẹ để GUI kịp thở)
            elapsed_time = time.time() - start_time
            if elapsed_time < frame_time:
                time.sleep(frame_time - elapsed_time)

        cap.release()

    def update_gui_image(self, image_array):
        """Hàm này sẽ được Luồng Chính (Main Thread) gọi"""
        try:
            display_w = self.display_frame.winfo_width()
            display_h = self.display_frame.winfo_height()
            
            if display_w < 10 or display_h < 10: return

            pil_image = Image.fromarray(image_array)
            
            # Tính toán resize
            img_w, img_h = pil_image.size
            ratio = min(display_w/img_w, display_h/img_h)
            new_w = int(img_w * ratio)
            new_h = int(img_h * ratio)
            
            # Tạo CTkImage
            ctk_image = ctk.CTkImage(light_image=pil_image, 
                                   dark_image=pil_image, 
                                   size=(new_w, new_h))
            
            # Cập nhật Label
            self.display_label.configure(image=ctk_image, text="")
            
            # QUAN TRỌNG: Gán vào biến class để tránh bị Garbage Collector xóa ngay lập tức
            self.current_image_ref = ctk_image 
            
        except Exception:
            pass

    def display_image_on_gui(self, cv2_image):
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
        self.update_gui_image(rgb_image)

    def on_closing(self):
        self.stop_processing()
        self.destroy()
        print("Đã thoát.")

if __name__ == "__main__":
    app = HelmetDetectionApp()
    app.mainloop()