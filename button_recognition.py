#!/usr/bin/env python
import os
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from utils.ops import native_crop_and_resize
from utils import visualization_utils as vis_util
from tensorflow.python.compiler.tensorrt import trt_convert as trt

tf.compat.v1.disable_eager_execution()

charset = {'0': 0,  '1': 1,  '2': 2,  '3': 3,  '4': 4,  '5': 5,
           '6': 6,  '7': 7,  '8': 8,  '9': 9,  'A': 10, 'B': 11,
           'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
           'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
           'O': 24, 'P': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29,
           'V': 30, 'X': 31, 'Z': 32, '<': 33, '>': 34, '(': 35,
           ')': 36, '$': 37, '#': 38, '^': 39, 's': 40, '-': 41,
           '*': 42, '%': 43, '?': 44, '!': 45, '+': 46} # <nul> = +

class ButtonRecognizer:
  def __init__(self, rcnn_path= None, ocr_path=None,
               use_trt=False, precision='FP16', use_optimized=False,
               use_tx2=False):
    self.ocr_graph_path = ocr_path
    self.rcnn_graph_path = rcnn_path
    self.use_trt = use_trt
    self.precision=precision  #'INT8, FP16, FP32'
    self.use_optimized = use_optimized
    self.use_tx2 = use_tx2 # if use tx2, gpu memory is limited to 3GB
    self.session = None

    self.ocr_input = None
    self.ocr_output = None
    self.rcnn_input = None
    self.rcnn_output = None

    self.class_num = 1
    self.image_size = [480, 640]
    self.recognition_size = [180, 180]
    self.category_index = {1: {'id': 1, 'name': 'button'}}
    self.idx_lbl = {}
    for key in charset.keys():
      self.idx_lbl[charset[key]] = key
    self.load_and_merge_graphs()
    print('Button recognizer initialized!')

  def __enter__(self):
      return self

  def __exit__(self, exc_type, exc_value, traceback):
      self.clear_session()

  def optimize_rcnn(self, input_graph_def):
      trt_graph = tf.experimental.tensorrt.Converter(
          input_graph_def,
          precision_mode=self.precision)
      return trt_graph.convert()

  def optimize_ocr(self, input_graph_def):
      trt_graph = tf.experimental.tensorrt.Converter(
          input_graph_def,
          precision_mode=self.precision)
      return trt_graph.convert()

  def load_and_merge_graphs(self):
    # check graph paths
    if self.ocr_graph_path is None:
      self.ocr_graph_path = './frozen_model/ocr_graph.pb'
    if self.rcnn_graph_path is None:
      self.rcnn_graph_path = './frozen_model/detection_graph_640x480.pb'
    if self.use_optimized:
      self.ocr_graph_path.replace('.pb', '_optimized.pb')
      self.rcnn_graph_path.replace('.pb', '_optimized.pb')
    assert os.path.exists(self.ocr_graph_path) and os.path.exists(self.rcnn_graph_path)

    # merge the frozen graphs
    ocr_rcnn_graph = tf.Graph()
    with ocr_rcnn_graph.as_default():

      # load button detection graph definition
      with tf.io.gfile.GFile(self.rcnn_graph_path, 'rb') as fid:
        detection_graph_def = tf.compat.v1.GraphDef()
        serialized_graph = fid.read()
        detection_graph_def.ParseFromString(serialized_graph)
        # for node in detection_graph_def.node:
        #   print node.name
        if self.use_trt:
          detection_graph_def = self.optimize_rcnn(detection_graph_def)
        tf.import_graph_def(detection_graph_def, name='detection')

      # load character recognition graph definition
      with tf.io.gfile.GFile(self.ocr_graph_path, 'rb') as fid:
        recognition_graph_def = tf.compat.v1.GraphDef()
        serialized_graph = fid.read()
        recognition_graph_def.ParseFromString(serialized_graph)
        if self.use_trt:
          recognition_graph_def = self.optimize_ocr(recognition_graph_def)
        tf.import_graph_def(recognition_graph_def, name='recognition')

      # retrive detection tensors
      rcnn_input = ocr_rcnn_graph.get_tensor_by_name('detection/image_tensor:0')
      rcnn_boxes = ocr_rcnn_graph.get_tensor_by_name('detection/detection_boxes:0')
      rcnn_scores = ocr_rcnn_graph.get_tensor_by_name('detection/detection_scores:0')
      rcnn_number = ocr_rcnn_graph.get_tensor_by_name('detection/num_detections:0')

      # crop and resize valida boxes (only valid when rcnn input has an known shape)
      rcnn_number = tf.cast(rcnn_number, tf.int32)
      valid_boxes = tf.slice(rcnn_boxes, [0, 0, 0], [1, rcnn_number[0], 4])
      ocr_boxes = native_crop_and_resize(rcnn_input, valid_boxes, self.recognition_size)

      # retrive recognition tensors
      ocr_input = ocr_rcnn_graph.get_tensor_by_name('recognition/ocr_input:0')
      ocr_chars = ocr_rcnn_graph.get_tensor_by_name('recognition/predicted_chars:0')
      ocr_beliefs = ocr_rcnn_graph.get_tensor_by_name('recognition/predicted_scores:0')

      self.rcnn_input = rcnn_input
      self.rcnn_output = [rcnn_boxes, rcnn_scores, rcnn_number, ocr_boxes]
      self.ocr_input = ocr_input
      self.ocr_output = [ocr_chars, ocr_beliefs]
    if self.use_tx2:
      gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=3.0/8.0)
      self.session = tf.compat.v1.Session(graph=ocr_rcnn_graph, config=tf.ConfigProto(gpu_options=gpu_options))
    else:
      self.session = tf.compat.v1.Session(graph=ocr_rcnn_graph)

  def clear_session(self):
    if self.session is not None:
      self.session.close()

  def decode_text(self, codes, scores):
    score_ave = 0
    text = ''
    for char, score in zip(codes, scores):
      if not self.idx_lbl[char] == '+':
        score_ave += score
        text += self.idx_lbl[char]
    score_ave /= len(text)
    return text, score_ave

  def predict(self, image_np, depth_frame, draw=False):
    # input data
    assert image_np.shape == (480, 640, 3)
    img_in = np.expand_dims(image_np, axis=0)

    # output data
    recognition_list = []

    # perform detection and recognition
    boxes, scores, number, ocr_boxes = self.session.run(self.rcnn_output, feed_dict={self.rcnn_input:img_in})
    boxes, scores, number = [np.squeeze(x) for x in [boxes, scores, number]]

    for i in range(number):
        if scores[i] < 0.5: continue
        center_x = (boxes[i][1] + boxes[i][3]) * 0.5 * self.image_size[1]
        center_y = (boxes[i][0] + boxes[i][2]) * 0.5 * self.image_size[0]
        depth = depth_frame.get_distance(int(center_x), int(center_y))
        if ocr_boxes.any():
            chars, beliefs = self.session.run(self.ocr_output, feed_dict={self.ocr_input: ocr_boxes[:,i]})
            chars, beliefs = [np.squeeze(x) for x in [chars, beliefs]]
            text, belief = self.decode_text(chars, beliefs)
        else:
            text, belief = '', 0.0
        recognition_list.append([boxes[i], scores[i], text, belief, [center_x, center_y, depth]])
    m_depth = depth_frame.get_distance(320, 240)

    if draw:
      classes = [1]*len(boxes)
      self.draw_detection_result(image_np, boxes, classes, scores, self.category_index)
      self.draw_recognition_result(image_np, recognition_list)

    return recognition_list, m_depth

  @staticmethod
  def draw_detection_result(image_np, boxes, classes, scores, category, predict_chars=None):
    vis_util.visualize_boxes_and_labels_on_image_array(
      image_np,
      np.squeeze(boxes),
      np.squeeze(classes).astype(np.int32),
      np.squeeze(scores),
      category,
      max_boxes_to_draw=100,
      use_normalized_coordinates=True,
      line_thickness=5,
      predict_chars=predict_chars
    )

  def draw_recognition_result(self, image_np, recognitions):
    for item in recognitions:
      # crop button patches
      y_min = int(item[0][0] * self.image_size[0])
      x_min = int(item[0][1] * self.image_size[1])
      y_max = int(item[0][2] * self.image_size[0])
      x_max = int(item[0][3] * self.image_size[1])
      button_patch = image_np[y_min: y_max, x_min: x_max]
      # generate image layer for drawing
      img_pil = Image.fromarray(button_patch)
      img_show = ImageDraw.Draw(img_pil)
      # draw at a proper location
      x_center = (x_max-x_min) / 2.0
      y_center = (y_max-y_min) / 2.0
      font_size = min(x_center, y_center)*1.1
      text_center = int(x_center-0.5*font_size), int(y_center-0.5*font_size)
      try:
          font = ImageFont.truetype('/Library/Fonts/Arial.ttf', int(font_size))  # 변경 가능한 폰트 경로
      except IOError:
          font = ImageFont.load_default() 
      img_show.text(text_center, text=item[2], font=font, fill=(255, 0, 255))
      # img_pil.show()
      image_np[y_min: y_max, x_min: x_max] = np.array(img_pil)


if __name__ == '__main__':
    recognizer = ButtonRecognizer(use_optimized=True)
    image = imageio.imread('./test_panels/1.jpg')
    recognition_list, m_depth = recognizer.predict(image, True)
    image = Image.fromarray(image)
    image.show()
    recognizer.clear_session()
