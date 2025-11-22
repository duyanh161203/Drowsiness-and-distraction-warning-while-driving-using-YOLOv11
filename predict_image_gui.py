import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import cv2
from ultralytics import YOLO

class_names = ["SafeDriving", "UsePhone", "Drinking", "Drowsy", "DistractedDriving"]
model_path = r"D:\DATN\driver_monitoring\20250621_123613\weights\best.pt"

class ImagePredictorGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("YOLO Image Prediction")
        self.root.geometry("900x700")
        self.model = YOLO(model_path)
        self.img = None
        self.img_path = None

        self.btn_select = tk.Button(root, text="Chọn ảnh", command=self.select_image, font=("Arial", 14))
        self.btn_select.pack(pady=10)

        self.canvas = tk.Label(root)
        self.canvas.pack(pady=10)

        self.btn_predict = tk.Button(root, text="Dự đoán", command=self.predict, font=("Arial", 14))
        self.btn_predict.pack(pady=10)

        self.result_label = tk.Label(root, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

    def select_image(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp"), ("All files", "*.*")]
        )
        if file_path:
            self.img_path = file_path
            img = cv2.imread(file_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (600, 400))
            self.img = img
            im_pil = Image.fromarray(img)
            imgtk = ImageTk.PhotoImage(im_pil)
            self.canvas.configure(image=imgtk)
            self.canvas.image = imgtk
            self.result_label.config(text="")

    def predict(self):
        if self.img_path is None:
            messagebox.showwarning("Cảnh báo", "Vui lòng chọn ảnh trước!")
            return
        results = self.model(self.img_path)
        img = self.img.copy()
        pred_texts = []
        for r in results:
            for box in r.boxes:
                cls_id = int(box.cls[0])
                score = float(box.conf[0])
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                class_name = class_names[cls_id]
                pred_texts.append(f"{class_name}: {score:.2f}")
                cv2.rectangle(img, (x1, y1), (x2, y2), (0,255,0), 2)
                cv2.putText(img, f"{class_name} {score:.2f}", (x1, y1-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
        im_pil = Image.fromarray(img)
        imgtk = ImageTk.PhotoImage(im_pil)
        self.canvas.configure(image=imgtk)
        self.canvas.image = imgtk
        if pred_texts:
            self.result_label.config(text="; ".join(pred_texts))
        else:
            self.result_label.config(text="Không phát hiện hành vi nào.")

if __name__ == "__main__":
    root = tk.Tk()
    app = ImagePredictorGUI(root)
    root.mainloop()