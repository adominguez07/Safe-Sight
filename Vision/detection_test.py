from typing import Tuple, Union
import math
from pathlib import Path
import cv2
import numpy as np
from visualize import visualize

MARGIN = 10  # pixels
ROW_SIZE = 10  # pixels
FONT_SIZE = 1
FONT_THICKNESS = 1
TEXT_COLOR = (255, 0, 0)  # red

def main():
  BASE_DIR = Path(__file__).resolve().parent.parent  #makes BASE_DIR == C:/Projects/Safe-Sight/Vision
  IMAGE_FILE = BASE_DIR /'Vision' / 'img' / 'image.jpg'
  VIDEO_FILE = BASE_DIR / 'Vision' / 'img' / 'video.mp4'
  MODEL_FILE = BASE_DIR /'models' / 'blaze_face_short_range.tflite'

  img = cv2.imread(str(IMAGE_FILE))
  if img is None:
    raise FileNotFoundError(f'Could not read image file: {IMAGE_FILE}')


  #step one import modules
  import mediapipe as mp
  from mediapipe.tasks import python
  from mediapipe.tasks.python import vision

  # STEP 2: Create an FaceDetector object.
  if not MODEL_FILE.exists():
    raise FileNotFoundError(
        f'Could not find model file: {MODEL_FILE}. Put {MODEL_FILE.name} in the Vision folder.')

  base_options = python.BaseOptions(model_asset_path=str(MODEL_FILE))
  options = vision.FaceDetectorOptions(base_options=base_options)
  detector = vision.FaceDetector.create_from_options(options)

  # STEP 3: Load the input image.
  image = mp.Image.create_from_file(str(IMAGE_FILE))

  # STEP 4: Detect faces in the input image.
  detection_result = detector.detect(image)

  # STEP 5: Process the detection result. In this case, visualize it.
  image_copy = np.copy(image.numpy_view())
  annotated_image = visualize(image_copy, detection_result)
  rgb_annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
  cv2.imshow("Safe-Sight", rgb_annotated_image)
  cv2.waitKey(0)
  cv2.destroyAllWindows()

if __name__ == "__main__":
  main()
