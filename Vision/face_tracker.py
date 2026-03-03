import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from camera import camera_feed
from camera import Camera
from visualize import visualize


BASE_DIR = Path(__file__).resolve().parent #path C:/Projects/Safe-Sight
MODEL_FILE = BASE_DIR / 'models' / 'blaze_face_short_range.tflite'
CAMERA = Camera()


def stream_test():
    CAMERA.start()      #This starts the camera and prepares for a camera feed
    
    base_op = mp.tasks.BaseOptions(model_asset_path = str(MODEL_FILE))
    face_options = mp.tasks.vision.FaceDetectorOptions(base_options = base_op)
    detector = mp.tasks.vision.FaceDetector.create_from_options(face_options)

    while True:
        frame = CAMERA.read_frame()
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  #coverts frame from BGR to RGB
        
        mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)   #Converts the frame into something mp can work witjh
        
        results = detector.detect(mp_frame)
        annotated_stream = visualize(frame ,results)

        cv2.imshow("Safe_Sight", annotated_stream)
        if cv2.waitKey(1) & 0xFF == 'q':
            break
    
    #close camera
    CAMERA.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    stream_test()
