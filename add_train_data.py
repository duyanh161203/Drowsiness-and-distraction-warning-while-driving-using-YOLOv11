import os
import shutil

class_mapping = {
    'DistractedDriving': 0,
    'UsePhone': 1,
    'Drinking': 2,
    'Drowsy': 3,
    'SafeDriving': 4,
}

input_dir = r'D:\DATN\PreprocessedData'
train_images_dir = r'd:\DATN\Driver Mentoring.v8i.yolov11\train\images'
train_labels_dir = r'd:\DATN\Driver Mentoring.v8i.yolov11\train\labels'

os.makedirs(train_images_dir, exist_ok=True)
os.makedirs(train_labels_dir, exist_ok=True)

for class_name, class_id in class_mapping.items():
    class_folder = os.path.join(input_dir, class_name)
    if not os.path.isdir(class_folder):
        continue
    for img_name in os.listdir(class_folder):
        if not img_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue
        src_img = os.path.join(class_folder, img_name)
        dst_img = os.path.join(train_images_dir, img_name)
        shutil.copy2(src_img, dst_img)
        label_name = os.path.splitext(img_name)[0] + '.txt'
        dst_label = os.path.join(train_labels_dir, label_name)
        with open(dst_label, 'w') as f:
            f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

print("Đã gán nhãn")