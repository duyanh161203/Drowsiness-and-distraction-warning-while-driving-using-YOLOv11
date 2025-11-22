import cv2
import os
import numpy as np
from retinaface import RetinaFace
from albumentations import (
    Compose, RandomBrightnessContrast,
    GaussianBlur, RandomGamma, ISONoise
)

def apply_augmentations(image):
    transform = Compose([
        RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        GaussianBlur(blur_limit=(3, 7), p=0.3),
        RandomGamma(gamma_limit=(80, 120), p=0.3),
        ISONoise(p=0.3),
    ])
    augmented = transform(image=image)
    return augmented['image']

def extend_bbox(x1, y1, x2, y2, img_shape, extend_ratio=0.3):
    height, width = img_shape[:2]
    
    h = y2 - y1
    w = x2 - x1
    
    y_extend = h * extend_ratio
    x_extend = w * extend_ratio
    
    new_x1 = max(0, x1 - x_extend)
    new_y1 = max(0, y1 - y_extend/2)  
    new_x2 = min(width, x2 + x_extend)
    new_y2 = min(height, y2 + y_extend*2)  
    
    return int(new_x1), int(new_y1), int(new_x2), int(new_y2)

def preprocess_faces(input_folder, output_folder, target_size=(640, 640), use_grayscale=True, use_augmentation=True):

    if not os.path.exists(output_folder):
        print(f"Error: Output folder {output_folder} does not exist!")
        return

    img_counter = 0 

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            image_path = os.path.join(input_folder, filename)
            img = cv2.imread(image_path)
            if img is None:
                print(f"Could not read image: {image_path}")
                continue

            faces = RetinaFace.detect_faces(img)
            if not isinstance(faces, dict):
                print(f"No faces detected in: {filename}")
                continue

            for idx, (face_key, face_data) in enumerate(faces.items()):
                facial_area = face_data['facial_area']
                confidence = face_data['score']
                if confidence < 0.95:
                    continue

                x1, y1, x2, y2 = facial_area
                x1, y1, x2, y2 = extend_bbox(x1, y1, x2, y2, img.shape)
                face_img = img[y1:y2, x1:x2]
                face_img = cv2.resize(face_img, target_size)

                if use_augmentation:
                    aug_img = apply_augmentations(face_img)
                else:
                    aug_img = face_img.copy()

                img_counter += 1
                if use_grayscale and img_counter % 2 == 0:
                    gray_img = cv2.cvtColor(aug_img, cv2.COLOR_BGR2GRAY)
                    gray_img = cv2.cvtColor(gray_img, cv2.COLOR_GRAY2BGR)
                    gray_path = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_face_{idx+1}_gray.jpg"
                    )
                    cv2.imwrite(gray_path, gray_img)
                    print(f"Processed face {idx+1} from {filename} (gray only)")
                else:
                    color_path = os.path.join(
                        output_folder, f"{os.path.splitext(filename)[0]}_face_{idx+1}_color.jpg"
                    )
                    cv2.imwrite(color_path, aug_img)
                    print(f"Processed face {idx+1} from {filename} (color only)")

def main():
    """Main function to run the preprocessing"""
    input_folder = r"D:\DATN\Data\Drinking"
    output_folder = r"D:\DATN\PreprocessedData\Drinking"
    if not os.path.exists(output_folder):
        print(f"Error: Output folder {output_folder} does not exist!")
        return
    
    preprocess_faces(
        input_folder=input_folder,
        output_folder=output_folder,
        target_size=(640, 640),
        use_grayscale=True,
        use_augmentation=True
    )
    
    print("Preprocessing completed!")

if __name__ == "__main__":
    main()