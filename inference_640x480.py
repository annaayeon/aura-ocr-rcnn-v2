#!/usr/bin/env python
from __future__ import print_function
import os
import cv2
import imageio
from PIL import Image
import PIL.ImageOps as ImageOps
import numpy as np
import tensorflow as tf
from button_recognition import ButtonRecognizer

DRAW = True
# DRAW = False

def get_image_name_list(target_path):
  assert os.path.exists(target_path)
  image_name_list = []
  file_set = os.walk(target_path)
  for root, dirs, files in file_set:
    for image_name in files:
      image_name_list.append(image_name.split('.')[0])
  return image_name_list

def warm_up(model):
  assert isinstance(model, ButtonRecognizer)
  image = imageio.imread('./test_panels/1.jpg')
  model.predict(image)

if __name__ == '__main__':
  data_dir = './test_panels'
  data_list = get_image_name_list(data_dir)
  recognizer = ButtonRecognizer(use_optimized=True)
  warm_up(recognizer)
  overall_time = 0
  for data in data_list:
    img_path = os.path.join(data_dir, data+'.jpg')
    with open(img_path, 'rb') as f:
      image = Image.open(f)
      # resize to 640x480 with ratio kept
      image = image.resize((640, 480))
      delta_w, delta_h = 640 - image.size[0], 480 - image.size[1]
      padding = (delta_w // 2, delta_h // 2, delta_w - (delta_w // 2), delta_h - (delta_h // 2))
      new_im = ImageOps.expand(image, padding)
      img_np = np.copy(np.asarray(new_im))
      # perform button recognition
      t0 = cv2.getTickCount()
      recognizer.predict(img_np, draw=DRAW)
      t1 = cv2.getTickCount()
      time = (t1 - t0) / cv2.getTickFrequency()
      overall_time += time
      print('Time elapsed: {}'.format(time)) 
      if DRAW:
          image = Image.fromarray(img_np)
          image.show()



    average_time = overall_time / len(data_list)
    print('Average_used: {}'.format(average_time))
    # recognizer.clear_session()