import torch
from ultralytics import YOLO
import os
from datetime import datetime

def train_yolov11(
    data_yaml_path,
    epochs=100,
    batch_size=8,
    imgsz=512,
    device='0',
    project_name='driver_monitoring',
):
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"Using GPU: {gpu_name} with {gpu_memory:.1f}GB memory")
        torch.backends.cudnn.benchmark = True
    else:
        print("WARNING: CUDA not available, using CPU!")
        device = 'cpu'

    model = YOLO('D:\DATN\yolo11n.pt')

    args = {
        'data': data_yaml_path,
        'epochs': epochs,
        'batch': 8,
        'imgsz': 512,
        'device': device,
        'project': project_name,
        'name': datetime.now().strftime("%Y%m%d_%H%M%S"),
        'exist_ok': True,
        'pretrained': True,
        'optimizer': 'AdamW',
        'lr0': 0.005,          
        'lrf': 0.02,            
        'weight_decay': 0.005,
        'warmup_epochs': 3,
        'warmup_momentum': 0.8,
        'cos_lr': True,
        'patience': 20,
        'save': True,
        'save_period': 10,
        'cache': False,        
        'multi_scale': False,  
        'workers': 4,          
        'amp': True,

        # augmentation
        'degrees': 15.0,
        'translate': 0.15,
        'scale': 0.4,
        'shear': 3.0,
        'perspective': 0.002,
        'flipud': 0.3,
        'fliplr': 0.5,
        'mosaic': 0.5,
        'mixup': 0.2,
        'copy_paste': 0.2,
        'dropout': 0.1,
        'hsv_h': 0.02,
        'hsv_s': 0.8,
        'hsv_v': 0.5,
        'erasing': 0.3,
        'rect': False,
        'auto_augment': True,
    }

    try:
        print("Starting training...")
        results = model.train(**args)
        print("Training completed successfully!")

        print("Evaluating model...")
        metrics = model.val()
        print(f"Validation metrics: {metrics}")

        return results, metrics

    except Exception as e:
        print(f"Error during training: {str(e)}")
        return None, None

def main():
    data_yaml = r"D:\DATN\Driver Mentoring.v8i.yolov11\data.yaml"
    device = '0' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {'GPU' if device == '0' else 'CPU'}")
    params = {
        'data_yaml_path': data_yaml,
        'epochs': 100,
        'batch_size': 8,
        'imgsz': 512,
        'device': device,
        'project_name': 'driver_monitoring'
    }
    results, metrics = train_yolov11(**params)

    if results is not None:
        print("Training completed successfully!")
        print(f"Final metrics: {metrics}")
    else:
        print("Training failed!")

if __name__ == "__main__":
    main()