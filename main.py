import cv2
import mediapipe as mp
import numpy as np
from Vision.camera import Camera

def main():
  
  camera = Camera()
  camera.start()
  while True:
    frame = camera.read_frame()
    cv2.imshow("Safe-Sight", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
      break
  camera.release()

if __name__ == '__main__':
  main()