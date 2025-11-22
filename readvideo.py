import cv2
import os

def extract_frames(video_path, output_dir):
    # Tạo thư mục lưu frame nếu chưa có
    os.makedirs(output_dir, exist_ok=True)

    # Mở video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Không thể mở video:", video_path)
        return

    frame_count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break  # Đã đọc hết video

        # Lưu frame ra file ảnh
        frame_filename = os.path.join(output_dir, f"frame_{frame_count:05d}.jpg")
        cv2.imwrite(frame_filename, frame)
        frame_count += 1

    cap.release()
    print(f"Đã cắt {frame_count} frame vào thư mục {output_dir}")

if __name__ == "__main__":
    video_path = "D:\DATN\DataDATN\IMG_6812.MOV"         
    output_dir = "D:\DATN\Data\Drinking2"            
    extract_frames(video_path, output_dir)