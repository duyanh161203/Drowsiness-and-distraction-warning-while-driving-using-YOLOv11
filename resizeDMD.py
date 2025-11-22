import os
import cv2

def resize_images_in_folder(input_folder, output_folder, size=(640, 640)):
    os.makedirs(output_folder, exist_ok=True)
    for class_name in os.listdir(input_folder):
        class_dir = os.path.join(input_folder, class_name)
        if not os.path.isdir(class_dir):
            continue
        out_class_dir = os.path.join(output_folder, class_name)
        os.makedirs(out_class_dir, exist_ok=True)
        for img_name in os.listdir(class_dir):
            if not img_name.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                continue
            img_path = os.path.join(class_dir, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"Không đọc được ảnh: {img_path}")
                continue
            resized = cv2.resize(img, size)
            out_path = os.path.join(out_class_dir, img_name)
            cv2.imwrite(out_path, resized)
    print("Đã resize xong tất cả ảnh.")

if __name__ == "__main__":
    input_folder = r"d:\DATN\DMD\grouped_images"
    output_folder = r"d:\DATN\DMD\grouped_images_resized"
    resize_images_in_folder(input_folder, output_folder)