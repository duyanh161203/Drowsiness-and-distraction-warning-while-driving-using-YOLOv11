import cv2
import torch
import numpy as np
from ultralytics import YOLO
from collections import deque
import time
import mediapipe as mp
from playsound import playsound
import threading

class DriverMonitoring:
    def __init__(self, model_path, conf_threshold=0.5, face_conf_threshold=0.7):
        self.CAMERA_WIDTH = 640
        self.CAMERA_HEIGHT = 480 
        self.CAMERA_FPS = 30
        self.CONF_THRESHOLD = conf_threshold
        self.FACE_CONF_THRESHOLD = face_conf_threshold
        self.model = YOLO(model_path)
        self.model.model.eval()
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=0.1
        )
        
        self.class_names = {
            0: 'Distracted',
            1: 'UsePhone',
            2: 'Drinking',
            3: 'Drowsy',
            4: 'Smoking'
        }

        self.colors = {
            'Distracted': (0, 0, 255),   
            'UsePhone': (128, 0, 128),    
            'Drinking': (255, 165, 0),    
            'Drowsy': (255, 0, 255),       
            'Smoking': (0, 255, 0),       
        }

        self.BUFFER_SIZE = 10
        self.prediction_buffer = deque(maxlen=self.BUFFER_SIZE)
        self.alert_playing = False  

    def initialize_camera(self):
        cap = cv2.VideoCapture(0)
        
        if not cap.isOpened():
            print("Error: Cannot connect to webcam!")
            return None
            
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.CAMERA_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.CAMERA_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, self.CAMERA_FPS)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        actual_fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Camera initialized:")
        print(f"- Resolution: {actual_width}x{actual_height}")
        print(f"- FPS: {actual_fps}")
        
        return cap

    def preprocess_frame(self, frame):
        if frame is None:
            return None
            
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_detector.process(rgb_frame)
        
        if results.detections:
            detection = results.detections[0]
            bbox = detection.location_data.relative_bounding_box
            
            h, w = frame.shape[:2]
            x = int(bbox.xmin * w)
            y = int(bbox.ymin * h)
            width = int(bbox.width * w)
            height = int(bbox.height * h)
            
            y_extend = int(height * 0.6)
            x_extend = int(width * 0.3)
            
            face_frame = frame[
                max(0, y-y_extend):min(h, y+height+y_extend),
                max(0, x-x_extend):min(w, x+width+x_extend)
            ]
            
            if face_frame.size == 0:
                return None
                
            face_frame = cv2.resize(face_frame, (640, 640))
            
            return face_frame
            
        return None

    def get_smooth_prediction(self, current_pred):
        self.prediction_buffer.append(current_pred)
        if len(self.prediction_buffer) > 0:
            pred_counts = {}
            for pred in self.prediction_buffer:
                pred_counts[pred] = pred_counts.get(pred, 0) + 1
            return max(pred_counts, key=pred_counts.get)
        return current_pred

    def draw_overlay(self, frame, current_status, fps, box_info=None):
        h, w = frame.shape[:2]
        status_height = 60
        cv2.rectangle(frame, (0, h-status_height), (w, h), (50, 50, 50), -1)
        status_color = self.colors.get(current_status, (0, 255, 0))
        cv2.putText(frame, f'Behavior: {current_status}', 
                    (10, h-35), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    status_color, 2, cv2.LINE_AA)
        cv2.putText(frame, f'FPS: {fps:.1f}', 
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                    (0, 255, 0), 2, cv2.LINE_AA)
        if box_info:
            x1, y1, x2, y2, conf, cls = box_info
            color = self.colors.get(self.class_names[cls], (0, 255, 0))
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f'{self.class_names[cls]} {conf:.2f}'
            t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)[0]
            c2 = int(x1) + t_size[0], int(y1) - t_size[1] - 3
            cv2.rectangle(frame, (int(x1), int(y1)), c2, color, -1)
            cv2.putText(frame, label, (int(x1), int(y1)-2), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1, cv2.LINE_AA)

    def play_alert_sound(self, sound_path):
        if not self.alert_playing:
            self.alert_playing = True
            threading.Thread(target=playsound, args=(sound_path,), daemon=True).start()

    def stop_alert_sound(self):
        self.alert_playing = False

    def run(self, alert_sound_path=None):
        cap = self.initialize_camera()
        if cap is None:
            return
            
        print("Press 'q' to quit")
        
        try:
            while cap.isOpened():
                success, frame = cap.read()
                if not success:
                    break
                
                start_time = time.time()
                processed_frame = self.preprocess_frame(frame)
                
                if processed_frame is not None:
                    device = 'cuda' if torch.cuda.is_available() else 'cpu'
                    with torch.amp.autocast(device_type=device):
                        results = self.model(
                            processed_frame, 
                            conf=self.CONF_THRESHOLD,
                            augment=True
                        )
                    
                    fps = 1.0 / (time.time() - start_time)
                    
                    current_status = "No behavior detected"
                    box_info = None
                    
                    if len(results) > 0:
                        result = results[0]
                        if len(result.boxes) > 0:
                            boxes = result.boxes
                            valid_boxes = [box for box in boxes if float(box.conf[0]) > self.CONF_THRESHOLD]
                            valid_boxes.sort(key=lambda x: float(x.conf[0]), reverse=True)
                            
                            if valid_boxes:
                                best_box = valid_boxes[0]
                                cls = int(best_box.cls[0])
                                conf = float(best_box.conf[0])
                                
                                current_status = self.class_names[cls]
                                current_status = self.get_smooth_prediction(current_status)
                                
                                x1, y1, x2, y2 = best_box.xyxy[0]
                                box_info = (x1, y1, x2, y2, conf, cls)

                                if alert_sound_path and current_status != "No behavior detected":
                                    self.play_alert_sound(alert_sound_path)
                                else:
                                    self.stop_alert_sound()
                            else:
                                self.stop_alert_sound()
                        else:
                            self.stop_alert_sound()
                    else:
                        self.stop_alert_sound()
                    
                    self.draw_overlay(frame, current_status, fps, box_info)
                    cv2.imshow('Driver Monitoring', frame)
                else:
                    cv2.putText(frame, "No face detected", 
                              (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                              (0, 0, 255), 2, cv2.LINE_AA)
                    cv2.imshow('Driver Monitoring', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
        finally:
            cap.release()
            cv2.destroyAllWindows()

if __name__ == "__main__":
    monitor = DriverMonitoring(
        r'D:\DATN\driver_monitoring\20250703_200824\weights\best.pt',
        conf_threshold=0.38
    )
    alert_mp3 = r"D:\DATN\random-alarm-319318.mp3"
    monitor.run(alert_sound_path=alert_mp3)