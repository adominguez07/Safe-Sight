import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from camera import Camera
from visualize import visualize

BASE_DIR = Path(__file__).resolve().parent.parent #path C:/Projects/Safe-Sight
MODEL_FILE = BASE_DIR / 'models' / 'blaze_face_short_range.tflite'
VIDEO_FILE = BASE_DIR / 'Vision' / 'img' / 'video.mp4'
CAMERA = Camera()

def landmark_test():
    CAMERA.start()
    running_mode = mp.tasks.vision.RunningMode.IMAGE
    base_options = mp.tasks.BaseOptions(model_asset_path = str(MODEL_FILE))
    landmark_options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options = base_options,
        running_mode = running_mode #Sets running mode to a Image as I am taking frames of the live_stream
    )
    landmark_detector = mp.tasks.vision.FaceLandmarker(
        base_options = landmark_options,
    )
    landmark_model_path = mp.tasks.vision.FaceLandmarker.create_from_model_path(model_path = str(MODEL_FILE))
    while True:
        frame = CAMERA.read_frame()
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)
        
        results = landmark_detector.detect(mp_frame)
        annotated_frame = visualize(rgb_frame, results)

        cv2.imshow("Safe-Sight", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    CAMERA.release()

if __name__ == '__main__':
    landmark_test()