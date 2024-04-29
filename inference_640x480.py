#!/usr/bin/env python
import os
import cv2
import imageio
from PIL import Image
import PIL.ImageOps as ImageOps
import numpy as np
import tensorflow as tf
from button_recognition import ButtonRecognizer
import pyrealsense2 as rs 

DRAW = True
# DRAW = False

def warm_up(model):
  assert isinstance(model, ButtonRecognizer)
  image = imageio.imread('./test_panels/1.jpg')
  model.predict(image)

if __name__ == '__main__':
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  pipeline.start(config)

  recognizer = ButtonRecognizer(use_optimized=True)
  warm_up(recognizer)
  try:
    while True:
      frames = pipeline.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()
      if not depth_frame or not color_frame:
        continue
      color_image = np.asanyarray(color_frame.get_data())
      # perform button recognition
      t0 = cv2.getTickCount()
      recognizer.predict(color_image, depth_frame, draw=DRAW)
      t1 = cv2.getTickCount()
      time = (t1 - t0) / cv2.getTickFrequency()
      fps = 1.0 / time
      print('FPS :', fps)
      if DRAW:
          cv2.imshow('Button Recognition', color_image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  finally:
    pipeline.stop()
    cv2.destroyAllWindows()
