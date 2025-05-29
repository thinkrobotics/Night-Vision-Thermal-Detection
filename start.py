import cv2
import numpy as np
from ultralytics import YOLO
import argparse
from datetime import datetime
import time

class ThermalYOLODetector:
    def __init__(self, model_path, threshold=0.5):
        self.model = YOLO(model_path)
        self.threshold = threshold
        self.class_colors = {
            0: (0, 255, 0),   # Person - Green
            1: (255, 0, 0),   # Vehicle - Blue
            2: (0, 0, 255),    # Animal - Red
            3: (255, 255, 0),  # Drone - Cyan
            4: (255, 0, 255),  # Other - Magenta
        }
        
    def preprocess_thermal_image(self, image):
        if len(image.shape) == 3:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            
        normalized = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        pseudo_color = cv2.applyColorMap(normalized, cv2.COLORMAP_HOT)
        
        return pseudo_color
    
    def detect(self, image):
        processed_img = self.preprocess_thermal_image(image)
        
        results = self.model(processed_img, conf=self.threshold)
        
        detections = []
        annotated_img = processed_img.copy()
        
        for result in results:
            for box in result.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls_id = int(box.cls[0])
                cls_name = self.model.names[cls_id]
                
                detections.append({
                    'class': cls_name,
                    'confidence': conf,
                    'bbox': (x1, y1, x2, y2),
                    'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                })
                
                color = self.class_colors.get(cls_id, (255, 255, 255))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color, 2)
                
                label = f"{cls_name}: {conf:.2f}"
                cv2.putText(annotated_img, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return detections, annotated_img

def open_camera_source(source, is_thermal=False):
  
    if isinstance(source, str) and source.startswith('rtsp://'):
        cap = cv2.VideoCapture(source, cv2.CAP_FFMPEG)
    else:
      try:
            source = int(source) if source.isdigit() else source
            cap = cv2.VideoCapture(source)
        except:
            cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video source: {source}")
    
    if is_thermal:
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y','1','6',' '))
        cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    else:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    
    cap.set(cv2.CAP_PROP_FPS, 30)
    return cap

def main():
    parser = argparse.ArgumentParser(description='Thermal Night Vision YOLO Detector')
    parser.add_argument('--model', type=str, required=True, help='Path to YOLO model weights')
    parser.add_argument('--source', type=str, default='0', 
                       help='Video source (0 for webcam, path to video file, or RTSP URL)')
    parser.add_argument('--threshold', type=float, default=0.5, 
                       help='Detection confidence threshold (0-1)')
    parser.add_argument('--thermal', action='store_true',
                       help='Flag if the source is a thermal camera')
    parser.add_argument('--save', type=str, default=None,
                       help='Path to save output video (optional)')
    args = parser.parse_args()
    
    detector = ThermalYOLODetector(args.model, args.threshold)
    
    cap = None
    out = None
    try:
        cap = open_camera_source(args.source, args.thermal)
        
        if args.save:
            frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)
            if fps <= 0:
                fps = 30 
            
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(args.save, fourcc, fps, (frame_width, frame_height))
        
        frame_count = 0
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to grab frame")
                break
                
            detections, annotated_frame = detector.detect(frame)
            
            frame_count += 1
            if frame_count % 10 == 0:
                fps = frame_count / (time.time() - start_time)
                cv2.putText(annotated_frame, f"FPS: {fps:.2f}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Thermal YOLO Detection', annotated_frame)
            
            if out is not None:
                out.write(annotated_frame)
            
            for detection in detections:
                print(f"[{detection['timestamp']}] Detected {detection['class']} "
                      f"with confidence {detection['confidence']:.2f} at {detection['bbox']}")
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
                
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        if cap is not None:
            cap.release()
        if out is not None:
            out.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
