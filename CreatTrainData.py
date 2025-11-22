import os
import shutil
import yaml

class_mapping = {
    'SafeDriving': 0,
    'UsePhone': 1,
    'Drinking': 2,
    'Drowsy': 3,
    'DistractedDriving': 4
}

def create_train_only_yolo_dataset(input_base_dir, output_base_dir):
    """
    Gán nhãn và chỉ cho dữ liệu vào tập train (không chia val/test)
    """
    train_images_dir = os.path.join(output_base_dir, 'train', 'images')
    train_labels_dir = os.path.join(output_base_dir, 'train', 'labels')
    os.makedirs(train_images_dir, exist_ok=True)
    os.makedirs(train_labels_dir, exist_ok=True)

    for class_name, class_id in class_mapping.items():
        class_dir = os.path.join(input_base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue

        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        for img_name in images:
            # Copy image
            src_path = os.path.join(class_dir, img_name)
            dst_path = os.path.join(train_images_dir, img_name)
            shutil.copy2(src_path, dst_path)
            # Create label
            label_path = os.path.join(train_labels_dir, img_name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

    # Tạo file data.yaml chỉ cho train
    data = {
        'train': './train/images',
        'val': './val/images',   # Để trống hoặc trỏ tới val thực tế nếu có
        'test': './test/images', # Để trống hoặc trỏ tới test thực tế nếu có
        'nc': len(class_mapping),
        'names': list(class_mapping.keys())
    }
    with open(os.path.join(output_base_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)

    print("Đã gán nhãn và copy toàn bộ dữ liệu vào tập train.")

if __name__ == "__main__":
    input_dir = r"D:\DATN\NewPreprocessedDatav2"
    output_dir = r"D:\DATN\NewYOLODatasetv2"
    create_train_only_yolo_dataset(input_dir, output_dir)