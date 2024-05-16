#!/usr/bin/env python
import cv2
import numpy as np
import tensorflow as tf
from button_recognition import ButtonRecognizer
import pyrealsense2 as rs 
import rospy
import tf2_ros
import math
from geometry_msgs.msg import TransformStamped

DRAW = True
# DRAW = False

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30):
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.intrinsics = depth_profile.get_intrinsics()
        self.pointcloud = rs.pointcloud()
        self.colorizer = rs.colorizer()

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None
        color_image = np.asanyarray(color_frame.get_data())
        return depth_frame, color_frame, color_image

    def create_pointcloud(self, depth_frame, color_frame):
        points = self.pointcloud.calculate(depth_frame)
        self.pointcloud.map_to(color_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        depth_colormap = np.asanyarray(self.colorizer.colorize(depth_frame).get_data())
        return verts, texcoords, depth_colormap

    def stop(self):
        self.pipeline.stop()

class BoxTFPublisher:
    def __init__(self):
        rospy.init_node('bounding_box_tf_publisher')
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.frame_id = 'camera_frame' 

    def publish_transforms(self, recognition_list, m_depth):
        for recognition in recognition_list:
           text = recognition[2]
           center_x, center_y, depth = recognition[4]
           tf_y, tf_z = self.calculate_transform(m_depth, center_x, center_y, depth)
           self.send_transform(m_depth, tf_y, tf_z, text)

    def calculate_transform(self, m_depth, center_x, center_y, depth):
      mx = 320
      my = 240
      
      pixel_y = center_x - mx
      if pixel_y < 0:
        pixel_y = -pixel_y
      pixel_z = center_y - my
      if pixel_z < 0:
         pixel_z = -pixel_z
      
      if depth**2 < m_depth**2:
        yz_distance = math.sqrt(m_depth**2 - depth**2)
      else:
        yz_distance = math.sqrt(depth**2 - m_depth**2)  # yz 평면 상 거리
      yz_pixel = math.sqrt(pixel_y**2 + pixel_z**2) # yz 평면 상 픽셀 거리
      scale_yz = yz_distance / yz_pixel if yz_pixel != 0 else 0

      tf_y = pixel_y * scale_yz
      tf_z = pixel_z * scale_yz

      return tf_y, tf_z

    def send_transform(self, x, y, d, text):
        t = TransformStamped()
        t.header.stamp = rospy.Time.now()
        t.header.frame_id = self.frame_id
        t.child_frame_id = 'button_'+text
        t.transform.translation.x = x
        t.transform.translation.y = y
        t.transform.translation.z = d
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 0.0
        self.broadcaster.sendTransform(t)


if __name__ == '__main__':
  camera = RealSenseCamera()
  recognizer = ButtonRecognizer(use_optimized=True)
  box_publisher = BoxTFPublisher()
  rate = rospy.Rate(10)

  try:
    while not rospy.is_shutdown():
      depth_frame, color_frame, color_image = camera.get_frames()
      if depth_frame is None or color_frame is None or color_image is None:
          continue
      verts, texcoords, depth_colormap = camera.create_pointcloud(depth_frame, color_frame) 

      # perform button recognition
      t0 = cv2.getTickCount()
      recognition_list, m_depth = recognizer.predict(color_image, depth_frame, draw=DRAW)
      # recognizer.predict(color_image, depth_frame, draw=DRAW)
      t1 = cv2.getTickCount()
      time = (t1 - t0) / cv2.getTickFrequency()
      fps = 1.0 / time
      print('FPS :', fps)

      box_publisher.publish_transforms(recognition_list, m_depth)

      if DRAW:
          cv2.imshow('Button Recognition', color_image)
          cv2.imshow('Point Cloud', depth_colormap)
          if cv2.waitKey(1) & 0xFF == ord('q'):
              break
          
  except rospy.ROSInterruptException:
      pass
  finally:
    camera.stop()
    recognizer.clear_session()
    cv2.destroyAllWindows()
