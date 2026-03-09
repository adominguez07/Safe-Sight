import cv2
import mediapipe as mp
import numpy as np
from pathlib import Path
from Vision.camera import Camera
from Vision.visualize import visualize
import time

BASE_DIR = Path(__file__).resolve().parent #path C:/Projects/Safe-Sight
MODEL_FILE = BASE_DIR / 'models' / 'blaze_face_short_range.tflite' 
TASK_MODEL_FILE = BASE_DIR / 'models' / 'face_landmarker.task'
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

def stream_test_with_frames():
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

def stream_test_with_video():
  
  #Setup for options

  Base_Options= mp.tasks.BaseOptions
  FaceLandmarker= mp.tasks.vision.FaceLandmarker
  FaceLandmarkerOptions = mp.tasks.vision.FaceLandmarkerOptions
  VisionRunningMode = mp.tasks.vision.RunningMode

  options = FaceLandmarkerOptions(
    base_options = Base_Options(model_asset_path = str(TASK_MODEL_FILE)),
    running_mode = VisionRunningMode.VIDEO
  )

  #Initialize the detector
  with FaceLandmarker.create_from_options(options) as landmarker:
    CAMERA.start()
    while True:
      frame = CAMERA.read_frame() 

      rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) #converts OpenCV BGR to RGB
      mp_frame = mp.Image(image_format = mp.ImageFormat.SRGB, data = rgb_frame)   #creates the mp frame using a RGB format and recieves data from rgb_frame

      frame_timestamp_ms = int(time.time() * 1000) #Timestamp for the video detection in ms

      detection_result = landmarker.detect_for_video( #detection for video with args from github repo
        image = mp_frame,
        timestamp_ms =  frame_timestamp_ms
      )

      if detection_result.face_landmarks:
        pass

      cv2.imshow("Safe-Sight", frame)
      if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    CAMERA.release()


if __name__ == "__main__":
  # stream_test_with_frames()
  stream_test_with_video()

#if __name__ == '__main__':
 # main()