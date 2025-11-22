import os
import shutil
from sklearn.model_selection import train_test_split
import yaml

class_mapping = {
    'SafeDriving': 0,
    'UsePhone': 1,
    'Drinking': 2,
    'Drowsy': 3,
    'DistractedDriving': 4
}

def create_yolo_dataset(input_base_dir, output_base_dir, train_ratio=0.7, val_ratio=0.2):
    """
    Organize images into YOLO format and create labels
    
    Args:
        input_base_dir: Directory containing class folders
        output_base_dir: Directory to create YOLO dataset
        train_ratio: Ratio of training data
        val_ratio: Ratio of validation data (remaining will be test)
    """
    # Create directory structure
    dirs = {
        'train': os.path.join(output_base_dir, 'train'),
        'val': os.path.join(output_base_dir, 'val'),
        'test': os.path.join(output_base_dir, 'test')
    }
    
    for dir_path in dirs.values():
        os.makedirs(os.path.join(dir_path, 'images'), exist_ok=True)
        os.makedirs(os.path.join(dir_path, 'labels'), exist_ok=True)

    # Process each class
    for class_name in class_mapping.keys():
        class_dir = os.path.join(input_base_dir, class_name)
        if not os.path.exists(class_dir):
            print(f"Warning: Directory {class_dir} not found")
            continue

        # Get all images (including flipped versions)
        images = [f for f in os.listdir(class_dir) if f.endswith('.jpg')]
        
        # Split into train, val, and test sets
        train_images, temp_images = train_test_split(images, train_size=train_ratio, random_state=42)
        val_ratio_adjusted = val_ratio / (1 - train_ratio)
        val_images, test_images = train_test_split(temp_images, train_size=val_ratio_adjusted, random_state=42)

        # Process each split
        for split_name, split_images in [('train', train_images), ('val', val_images), ('test', test_images)]:
            for img_name in split_images:
                # Copy image
                src_path = os.path.join(class_dir, img_name)
                dst_path = os.path.join(dirs[split_name], 'images', img_name)
                shutil.copy2(src_path, dst_path)
                
                # Create and save label
                label_path = os.path.join(dirs[split_name], 'labels', img_name.replace('.jpg', '.txt'))
                create_yolo_label(label_path, class_mapping[class_name])

    # Create data.yaml file
    create_data_yaml(output_base_dir, class_mapping)
    print("Dataset organization completed!")

def create_yolo_label(label_path, class_id):
    """Create YOLO format label file"""
    # For face detection, we'll use the entire image as the bounding box
    with open(label_path, 'w') as f:
        # Format: <class> <x_center> <y_center> <width> <height>
        # Using normalized coordinates (0-1)
        f.write(f"{class_id} 0.5 0.5 1.0 1.0\n")

def create_data_yaml(output_dir, class_mapping):
    """Create YOLO data.yaml file"""
    data = {
        'train': './train/images',
        'val': './val/images',
        'test': './test/images',
        'nc': len(class_mapping),
        'names': list(class_mapping.keys())
    }
    
    with open(os.path.join(output_dir, 'data.yaml'), 'w') as f:
        yaml.dump(data, f, sort_keys=False)

if __name__ == "__main__":
    input_dir = r"D:\DATN\NewPreprocessedDatav2"
    output_dir = r"D:\DATN\NewYOLODatasetv2"
    
    create_yolo_dataset(input_dir, output_dir)