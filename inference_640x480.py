#!/usr/bin/env python
import rospy
import tf2_ros
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import pyrealsense2 as rs
import struct
from button_recognition import ButtonRecognizer
import math

DRAW = True
depth_DRAW = False

class RealSenseCamera:
    def __init__(self, width=640, height=480, fps=30, pub_pc=False):
        '''
        camera instrinsics : [ 640x480  p[319.582 234.765]  f[388.996 388.996]  Brown Conrady [0 0 0 0 0] ] 
        '''
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        self.profile = self.pipeline.start(self.config)
        self.aligned_stream = rs.align(rs.stream.color)
        self.depth_profile = rs.video_stream_profile(self.profile.get_stream(rs.stream.depth))
        self.intrinsics = self.depth_profile.get_intrinsics()
        self.pointcloud = rs.pointcloud()
        self.colorizer = rs.colorizer()
        self.pub_pc = pub_pc
        if self.pub_pc:   
            self.pointcloud_pub = rospy.Publisher('pointcloud', PointCloud2, queue_size=10)
            self.frame_id = 'camera_frame'

    def get_frames(self):
        frames = self.pipeline.wait_for_frames()
        frames = self.aligned_stream.process(frames)
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()
        if not depth_frame or not color_frame:
            return None, None, None
        color_image = np.asanyarray(color_frame.get_data())
        depth_image = np.asanyarray(depth_frame.get_data())
        return depth_frame, depth_image, color_frame, color_image

    def stop(self):
        self.pipeline.stop()

    def publish_pointcloud(self, depth_frame, color_frame):
        points = self.pointcloud.calculate(depth_frame)
        self.pointcloud.map_to(color_frame)

        # Get vertices and texture coordinates
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        colors = np.asanyarray(color_frame.get_data())

        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id

        h, w, _ = colors.shape

        # Convert texcoords to pixel indices
        texcoords[:, 0] = np.clip(texcoords[:, 0] * w + 0.5, 0, w - 1)
        texcoords[:, 1] = np.clip(texcoords[:, 1] * h + 0.5, 0, h - 1)
        texcoords = texcoords.astype(np.int32)

        # Prepare point cloud data
        points_list = []
        for i in range(len(verts)):
            x, y, z = verts[i]
            u, v = texcoords[i]
            r, g, b = colors[v, u]
            rgb = struct.unpack('I', struct.pack('BBBB', b, g, r, 0))[0]
            points_list.append([x, y, z, rgb])
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]

        pc2_msg = pc2.create_cloud(header, fields, points_list)
        self.pointcloud_pub.publish(pc2_msg)

class BoxTFPublisher:
    def __init__(self):
        rospy.init_node('bounding_box_tf_publisher')
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.frame_id = 'camera_frame' 

    def publish_transforms(self, recognition_list, depth_frame, intrinsics):
        for recognition in recognition_list:
            text = recognition[2]
            pixel = recognition[4]
            on = recognition[5]
            point = self.get_3d_coordinates(depth_frame, pixel, intrinsics) # realsense
            # point = self.calculate_transform(depth_frame, pixel)  # ay
            self.send_transform(point, text, on)
            
    def get_3d_coordinates(self, depth_frame, pixel, intrinsics, window_size=3):
        u, v = pixel
        depth_values = []
        offset = window_size // 2

        # Collect depth values in the surrounding window
        for du in range(-offset, offset + 1):
            for dv in range(-offset, offset + 1):
                depth = depth_frame.get_distance(u + du, v + dv)
                if depth > 0:
                    depth_values.append(depth)
        if not depth_values:
            return [0.0, 0.0, 0.0]

        avg_depth = sum(depth_values) / len(depth_values)
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], avg_depth)
        tf_point = self.transform_point(point)
        return tf_point

    def transform_point(self, point):
        '''
        3D 좌표 (X, Y, Z) = (오른쪽+, 아래+, 앞+)
        tf 좌표 (X, Y, Z) = (앞, 왼쪽+, 위)
        '''
        return [point[2], -point[0], -point[1]]


    def calculate_transform(self, depth_frame, pixel):
        center_x, center_y = pixel
        m_depth = depth_frame.get_distance(320, 240)
        depth = depth_frame.get_distance(center_x, center_y)
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

        point = [depth, tf_y, tf_z]

        return point

    def send_transform(self, point, text, on):
        light = 'ON' if on else 'OFF'
        t = TransformStamped()  
        t.header.stamp = rospy.Time.now()                             
        t.header.frame_id = self.frame_id                 
        t.child_frame_id = 'button_' + text + '_' + light
        t.transform.translation.x = point[0]
        t.transform.translation.y = point[1] * 0.9
        t.transform.translation.z = point[2]
        t.transform.rotation.x = 0.0
        t.transform.rotation.y = 0.0
        t.transform.rotation.z = 0.0
        t.transform.rotation.w = 1.0
        self.broadcaster.sendTransform(t)

if __name__ == '__main__':
    camera = RealSenseCamera(pub_pc=False)
    recognizer = ButtonRecognizer(use_optimized=False)
    box_publisher = BoxTFPublisher()
    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            depth_frame, depth_image, color_frame, color_image = camera.get_frames()
            if depth_frame is None or depth_image is None or color_image is None:
                continue

            # Colorize depth image for visualization
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # perform button recognition
            t0 = cv2.getTickCount()
            recognition_list = recognizer.predict(color_image, draw=DRAW)
            t1 = cv2.getTickCount()
            time = (t1 - t0) / cv2.getTickFrequency()
            fps = 1.0 / time
            print('recognition FPS :', fps)

            box_publisher.publish_transforms(recognition_list, depth_frame, camera.intrinsics)
            if camera.pub_pc:
                camera.publish_pointcloud(depth_frame, color_frame)

            if DRAW:
                cv2.imshow('Button Recognition', color_image)
                if depth_DRAW:
                    cv2.imshow('Depth Frame', depth_colormap)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except rospy.ROSInterruptException:
        pass
    finally:
        camera.stop()
        recognizer.clear_session()
        cv2.destroyAllWindows()
      