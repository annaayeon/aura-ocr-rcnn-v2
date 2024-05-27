#!/usr/bin/env python
import os
import imageio
import numpy as np
import tensorflow as tf
from PIL import Image, ImageDraw, ImageFont
from utils.ops import native_crop_and_resize
from utils import visualization_utils as vis_util

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
tf.compat.v1.disable_eager_execution()
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1" # CPU 사용

charset = {'0': 0,  '1': 1,  '2': 2,  '3': 3,  '4': 4,  '5': 5,
           '6': 6,  '7': 7,  '8': 8,  '9': 9,  'A': 10, 'B': 11,
           'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17,
           'I': 18, 'J': 19, 'K': 20, 'L': 21, 'M': 22, 'N': 23,
           'O': 24, 'P': 25, 'R': 26, 'S': 27, 'T': 28, 'U': 29,
           'V': 30, 'X': 31, 'Z': 32, '<': 33, '>': 34, '(': 35,
           ')': 36, '$': 37, '#': 38, '^': 39, 's': 40, '-': 41,
           '*': 42, '%': 43, '?': 44, '!': 45, '+': 46} # <nul> = +

class ButtonRecognizer:
  def __init__(self, rcnn_path=None, ocr_path=None, precision='FP16', use_optimized=False, batch_size=1):
    self.ocr_graph_path = ocr_path
    self.rcnn_graph_path = rcnn_path
    self.precision = precision  #'INT8, FP16, FP32'
    self.use_optimized = use_optimized
    self.batch_size = batch_size
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
      self.ocr_graph_path = self.ocr_graph_path.replace('.pb', '_optimized.pb')
      self.rcnn_graph_path = self.rcnn_graph_path.replace('.pb', '_optimized.pb')
    assert os.path.exists(self.ocr_graph_path) and os.path.exists(self.rcnn_graph_path)

    # merge the frozen graphs
    ocr_rcnn_graph = tf.Graph()
    with ocr_rcnn_graph.as_default():

      # load button detection graph definition
      with tf.io.gfile.GFile(self.rcnn_graph_path, 'rb') as fid:
        detection_graph_def = tf.compat.v1.GraphDef()
        serialized_graph = fid.read()
        detection_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(detection_graph_def, name='detection')

      # load character recognition graph definition
      with tf.io.gfile.GFile(self.ocr_graph_path, 'rb') as fid:
        recognition_graph_def = tf.compat.v1.GraphDef()
        serialized_graph = fid.read()
        recognition_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(recognition_graph_def, name='recognition')

      # retrive detection tensors
      rcnn_input = ocr_rcnn_graph.get_tensor_by_name('detection/image_tensor:0')
      rcnn_boxes = ocr_rcnn_graph.get_tensor_by_name('detection/detection_boxes:0')
      rcnn_scores = ocr_rcnn_graph.get_tensor_by_name('detection/detection_scores:0')
      rcnn_number = ocr_rcnn_graph.get_tensor_by_name('detection/num_detections:0')

      # crop and resize valid boxes (only valid when rcnn input has an known shape)
      rcnn_number = tf.cast(rcnn_number, tf.int32)
      valid_boxes = tf.slice(rcnn_boxes, [0, 0, 0], [self.batch_size, rcnn_number[0], 4])

      ocr_boxes = native_crop_and_resize(rcnn_input, valid_boxes, self.recognition_size)

      # retrive recognition tensors
      ocr_input = ocr_rcnn_graph.get_tensor_by_name('recognition/ocr_input:0')
      ocr_chars = ocr_rcnn_graph.get_tensor_by_name('recognition/predicted_chars:0')
      ocr_beliefs = ocr_rcnn_graph.get_tensor_by_name('recognition/predicted_scores:0')

      self.rcnn_input = rcnn_input
      self.rcnn_output = [rcnn_boxes, rcnn_scores, rcnn_number, ocr_boxes]
      self.ocr_input = ocr_input
      self.ocr_output = [ocr_chars, ocr_beliefs]

      config = tf.compat.v1.ConfigProto()
      config.gpu_options.allow_growth = True
      self.session = tf.compat.v1.Session(graph=ocr_rcnn_graph, config=config)

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

  def predict(self, images_np, draw=False):
    # input data
    if images_np.ndim == 3:
        images_np = np.expand_dims(images_np, axis=0)
    assert images_np.shape == (self.batch_size, 480, 640, 3), f"Expected shape ({self.batch_size}, 480, 640, 3), got {images_np.shape}"

    img_in = images_np

    # output data
    recognition_list = []

    # perform detection and recognition
    boxes, scores, number, ocr_boxes = self.session.run(self.rcnn_output, feed_dict={self.rcnn_input: img_in})
    for b in range(self.batch_size):
        batch_boxes = boxes[b]
        batch_scores = scores[b]
        batch_number = number[b]
        for i in range(int(batch_number)):
            if batch_scores[i] < 0.5:
                continue
            center_x = int((batch_boxes[i][1] + batch_boxes[i][3]) * 0.5 * self.image_size[1])
            center_y = int((batch_boxes[i][0] + batch_boxes[i][2]) * 0.5 * self.image_size[0])
            if ocr_boxes.any():
                chars, beliefs = self.session.run(self.ocr_output, feed_dict={self.ocr_input: ocr_boxes[:, i]})
                chars, beliefs = [np.squeeze(x) for x in [chars, beliefs]]
                text, belief = self.decode_text(chars, beliefs)
            else:
                text, belief = '', 0.0
            recognition_list.append([batch_boxes[i], batch_scores[i], text, belief, [center_x, center_y]])

    if draw:
      for b in range(self.batch_size):
          batch_image = images_np[b]
          batch_boxes = boxes[b]
          batch_scores = scores[b]
          classes = [1] * len(batch_boxes)
          self.draw_detection_result(batch_image, batch_boxes, classes, batch_scores, self.category_index)
          self.draw_recognition_result(batch_image, recognition_list)

    return recognition_list

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
      font_size = min(x_center, y_center) * 1.1
      text_center = int(x_center - 0.5 * font_size), int(y_center - 0.5 * font_size)
      try:
          font = ImageFont.truetype('/Library/Fonts/Arial.ttf', int(font_size))  # 변경 가능한 폰트 경로
      except IOError:
          font = ImageFont.load_default() 
      img_show.text(text_center, text=item[2], font=font, fill=(255, 0, 255))
      image_np[y_min: y_max, x_min: x_max] = np.array(img_pil)


if __name__ == '__main__':
    recognizer = ButtonRecognizer(use_optimized=False, batch_size=4)
    images = np.array([imageio.imread('./test_panels/{}.jpg'.format(i+1)) for i in range(4)])
    recognition_list = recognizer.predict(images, True)
    for img in images:
        image = Image.fromarray(img)
        image.show()
    recognizer.clear_session()
