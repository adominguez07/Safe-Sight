import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from Vision.camera import Camera
from Vision.visualize import visualize


BASE_DIR = Path(__file__).resolve().parent #path C:/Projects/Safe-Sight
MODEL_FILE = BASE_DIR / 'models' / 'blaze_face_short_range.tflite'
VIDEO_FILE = BASE_DIR / 'Vision' / 'img' / 'video.mp4'
CAMERA = Camera()

def main():
  #CAMERA.start()
  vid = cv2.VideoCapture(str(VIDEO_FILE))
  if vid is None:
    raise FileNotFoundError("Error: Could not read video file")
  
  base_options = mp.tasks.BaseOptions(model_asset_path = str(MODEL_FILE))
  options = mp.tasks.vision.FaceDetectorOptions(base_options=base_options)
  detector = mp.tasks.vision.FaceDetector.create_from_options(options)

  while True:
    ret, frame = vid.read()
    if not ret:
      break
    mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = frame)
    detection_results = detector.detect(mp_frame)
    annotated_video = visualize(frame, detection_results)
    cv2.imshow("Safe-Sight",annotated_video)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break

  vid.release()
  cv2.destroyAllWindows()

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

    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
    
  #close camera
  CAMERA.release()
  cv2.destroyAllWindows()


if __name__ == "__main__":
  stream_test()


#if __name__ == '__main__':
 # main()