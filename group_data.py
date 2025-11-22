import os
import shutil
import yaml

def load_class_names(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    # Hỗ trợ cả dạng dict và list
    if isinstance(data, dict) and 'names' in data:
        return data['names']
    elif isinstance(data, list):
        return data
    else:
        raise ValueError("Không tìm thấy danh sách tên lớp trong file yaml.")

def group_images_by_label(dataset_dir, output_dir, yaml_path):
    class_names = load_class_names(yaml_path)
    class_map = {str(i): name for i, name in enumerate(class_names)}

    os.makedirs(output_dir, exist_ok=True)
    for name in class_names:
        os.makedirs(os.path.join(output_dir, name), exist_ok=True)

    for split in ['train', 'valid', 'test']:
        images_dir = os.path.join(dataset_dir, split, 'images')
        labels_dir = os.path.join(dataset_dir, split, 'labels')
        if not os.path.exists(images_dir) or not os.path.exists(labels_dir):
            continue

        for label_file in os.listdir(labels_dir):
            if not label_file.endswith('.txt'):
                continue
            label_path = os.path.join(labels_dir, label_file)
            with open(label_path, 'r') as f:
                lines = f.readlines()
            # Lấy tất cả class_id trong file label (nếu ảnh có nhiều nhãn, sẽ copy vào nhiều folder)
            class_ids = set([line.split()[0] for line in lines if line.strip()])
            image_name = label_file.replace('.txt', '.jpg')
            image_path = os.path.join(images_dir, image_name)
            if not os.path.exists(image_path):
                # Thử với .png
                image_name = label_file.replace('.txt', '.png')
                image_path = os.path.join(images_dir, image_name)
                if not os.path.exists(image_path):
                    continue
            for class_id in class_ids:
                class_name = class_map.get(class_id)
                if class_name:
                    dst = os.path.join(output_dir, class_name, image_name)
                    shutil.copy2(image_path, dst)

if __name__ == "__main__":
    dataset_dir = r"D:\DATN\Driver Mentoring.v8i.yolov11"
    output_dir = r"D:\DATN\Driver Mentoring.v8i.yolov11\grouped_images"
    yaml_path = os.path.join(dataset_dir, "data.yaml")
    group_images_by_label(dataset_dir, output_dir, yaml_path)
    print("Đã gộp xong ảnh theo nhãn vào các folder.")