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
import rospy
import tf2_ros
from geometry_msgs.msg import TransformStamped

DRAW = True
# DRAW = False

class BoxTFPublisher:
    def __init__(self):
        rospy.init_node('bounding_box_tf_publisher')
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.frame_id = 'camera_frame' 

    def publish_transforms(self, boxes_xyd):
        # x 값에 따라 오름차순 정렬
        sorted_boxes = sorted(boxes_xyd, key=lambda box: box[0])
        
        for i, box in enumerate(sorted_boxes):
            center_x, center_y, depth = box
            self.send_transform(center_x, center_y, depth, i)

    def send_transform(self, x, y, d, box_id):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.frame_id
        t.child_frame_id = f"box_{box_id+1}"
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = d
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)

if __name__ == '__main__':
  pipeline = rs.pipeline()
  config = rs.config()
  config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
  config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
  pipeline.start(config)

  recognizer = ButtonRecognizer(use_optimized=True)
  box_publisher = BoxTFPublisher()
  rate = rospy.Rate(10)

  try:
    while not rospy.is_shutdown():
      frames = pipeline.wait_for_frames()
      depth_frame = frames.get_depth_frame()
      color_frame = frames.get_color_frame()
      if not depth_frame or not color_frame:
        continue
      color_image = np.asanyarray(color_frame.get_data())
      # perform button recognition
      t0 = cv2.getTickCount()
      recognition_list, boxes_xyd = recognizer.predict(color_image, depth_frame, draw=DRAW)
      # recognizer.predict(color_image, depth_frame, draw=DRAW)
      t1 = cv2.getTickCount()
      time = (t1 - t0) / cv2.getTickFrequency()
      fps = 1.0 / time
      print('FPS :', fps)
      box_publisher.publish_transforms(boxes_xyd)
      if DRAW:
          cv2.imshow('Button Recognition', color_image)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
  except rospy.ROSInterruptException:
      pass
  finally:
    pipeline.stop()
    recognizer.clear_session()
    cv2.destroyAllWindows()
