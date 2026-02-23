import cv2
import mediapipe as mp
import numpy as np

class Camera():
  def __init__(self, camera_index = 0):
    self.camera_index = camera_index
    self.cap = None

  def start(self):
    self.cap = cv2.VideoCapture(self.camera_index)

    if not self.cap.isOpened():
      self.cap = None
      raise RuntimeError("Error: could not open camera index.")
  
  def read_frame(self):
    if self.cap is None:
      raise RuntimeError("Camera was not started.")
    
    suc, frame = self.cap.read()

    if suc is None:
      raise RuntimeError("Camera failed to grab from")

    return frame
  
  def release(self):
    if self.cap is not None:
      self.cap.release()
      self.cap = None
    cv2.destroyAllWindows()

def camera_feed():
  camera = Camera()
  camera.start()
  while True:
    frame = camera.read_frame()
    cv2.imshow("Safe-Sight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  camera.release()