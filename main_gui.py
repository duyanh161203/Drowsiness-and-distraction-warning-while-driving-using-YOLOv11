import tkinter as tk
from tkinter import ttk, filedialog
import cv2
import pygame
import threading
from PIL import Image, ImageTk
from predict_image import ImagePredictor
from detect_webcam import DriverMonitoring

class DriverMonitoringGUI:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("Driver Behavior Monitoring System")
        self.root.geometry("1200x800")
        
        # Khởi tạo pygame cho âm thanh
        pygame.mixer.init()
        self.alert_sound = pygame.mixer.Sound(r"D:\DATN\random-alarm-319318.mp3")
        
        # Khởi tạo các predictor với model path
        MODEL_PATH = r"D:\DATN\driver_monitoring\20250602_131046\weights\best.pt"
        self.image_predictor = ImagePredictor(
            model_path=MODEL_PATH,
            conf_threshold=0.5,
            face_conf_threshold=0.7
        )
        self.webcam_monitor = DriverMonitoring(
            model_path=MODEL_PATH,
            conf_threshold=0.5,
            face_conf_threshold=0.7
        )
        
        # Biến điều khiển
        self.is_webcam_running = False
        self.webcam_thread = None
        
        self.create_widgets()
        
    def create_widgets(self):
        # Frame chính
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.grid(row=0, column=0, sticky="nsew")
        
        # Tiêu đề
        title = ttk.Label(
            main_frame, 
            text="Driver Behavior Monitoring System",
            font=("Arial", 20, "bold")
        )
        title.grid(row=0, column=0, columnspan=2, pady=20)
        
        # Buttons
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=1, column=0, columnspan=2, pady=20)
        
        ttk.Button(
            btn_frame,
            text="Image Detection",
            command=self.start_image_detection,
            width=20
        ).grid(row=0, column=0, padx=10)
        
        self.webcam_btn = ttk.Button(
            btn_frame,
            text="Start Webcam",
            command=self.toggle_webcam,
            width=20
        )
        self.webcam_btn.grid(row=0, column=1, padx=10)
        
        # Display area
        self.display_label = ttk.Label(main_frame)
        self.display_label.grid(row=2, column=0, columnspan=2, pady=10)
        
        # Status area
        self.status_label = ttk.Label(
            main_frame,
            text="Ready",
            font=("Arial", 12)
        )
        self.status_label.grid(row=3, column=0, columnspan=2, pady=10)
        
        # Config grid
        self.root.grid_rowconfigure(0, weight=1)
        self.root.grid_columnconfigure(0, weight=1)
        
    def play_alert(self, behavior):
        """Phát âm thanh cảnh báo cho hành vi nguy hiểm"""
        dangerous_behaviors = ['Drinking', 'Drowsy', 'UsePhone', 'DistractedDriving']
        if behavior in dangerous_behaviors:
            self.alert_sound.play()
            
    def start_image_detection(self):
        """Xử lý detection từ ảnh"""
        file_path = filedialog.askopenfilename(
            title="Select Image",
            filetypes=[
                ("Image files", "*.jpg *.jpeg *.png *.bmp"),
                ("All files", "*.*")
            ]
        )
        
        if file_path:
            # Dự đoán
            behavior, confidence = self.image_predictor.predict(file_path)
            
            # Hiển thị ảnh và kết quả
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (800, 600))
            
            # Chuyển sang PhotoImage
            img = Image.fromarray(img)
            img = ImageTk.PhotoImage(img)
            
            self.display_label.configure(image=img)
            self.display_label.image = img
            
            # Cập nhật status
            status_text = f"Behavior: {behavior}\nConfidence: {confidence:.2f}"
            self.status_label.configure(text=status_text)
            
            # Phát cảnh báo nếu cần
            self.play_alert(behavior)
            
    def toggle_webcam(self):
        """Bật/tắt webcam detection"""
        if not self.is_webcam_running:
            self.webcam_btn.configure(text="Stop Webcam")
            self.is_webcam_running = True
            self.webcam_thread = threading.Thread(target=self.run_webcam)
            self.webcam_thread.start()
        else:
            self.webcam_btn.configure(text="Start Webcam")
            self.is_webcam_running = False
            
    def run_webcam(self):
        """Chạy webcam detection trong thread riêng"""
        cap = self.webcam_monitor.initialize_camera()
        if cap is None:
            return
        
        while self.is_webcam_running:
            success, frame = cap.read()
            if not success:
                break
                
            processed_frame = self.webcam_monitor.preprocess_frame(frame)
            if processed_frame is not None:
                results = self.webcam_monitor.model(processed_frame)
                
                current_status = "No behavior detected"
                if len(results) > 0:
                    result = results[0]
                    if len(result.boxes) > 0:
                        boxes = result.boxes
                        # Lọc boxes theo confidence threshold
                        valid_boxes = [box for box in boxes if float(box.conf[0]) > self.webcam_monitor.CONF_THRESHOLD]
                        valid_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
                        
                        if valid_boxes:
                            best_box = valid_boxes[0]
                            cls = int(best_box.cls[0])
                            conf = float(best_box.conf[0])
                            
                            behavior = self.webcam_monitor.class_names[cls]
                            # Cập nhật UI với confidence
                            status_text = f"Detected: {behavior}\nConfidence: {conf:.2f}"
                            self.root.after(0, self.status_label.configure, 
                                          {"text": status_text})
                            
                            # Phát cảnh báo nếu cần
                            self.play_alert(behavior)
                
                # Vẽ frame
                self.webcam_monitor.draw_overlay(frame, current_status, fps=30)
                
                # Hiển thị frame đã vẽ
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, (800, 600))
                img = Image.fromarray(frame)
                img = ImageTk.PhotoImage(img)
                
                self.display_label.configure(image=img)
                self.display_label.image = img
                
        cap.release()
        
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = DriverMonitoringGUI()
    app.run()