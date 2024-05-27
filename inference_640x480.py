#!/usr/bin/env python
import rospy
import tf2_ros
import cv2
import numpy as np
from sensor_msgs.msg import PointCloud2, PointField
from geometry_msgs.msg import TransformStamped
import sensor_msgs.point_cloud2 as pc2
import pyrealsense2 as rs
from button_recognition import ButtonRecognizer

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
            return None, None, None
        color_image = np.asanyarray(color_frame.get_data())
        return depth_frame, color_frame, color_image

    def create_pointcloud(self, depth_frame, color_frame):
        points = self.pointcloud.calculate(depth_frame)
        self.pointcloud.map_to(color_frame)
        verts = np.asanyarray(points.get_vertices()).view(np.float32).reshape(-1, 3)
        texcoords = np.asanyarray(points.get_texture_coordinates()).view(np.float32).reshape(-1, 2)
        colors = np.asanyarray(color_frame.get_data()).reshape(-1, 3)
        return verts, texcoords, colors

    def stop(self):
        self.pipeline.stop()

class BoxTFPublisher:
    def __init__(self):
        rospy.init_node('bounding_box_tf_publisher')
        self.broadcaster = tf2_ros.TransformBroadcaster()
        self.pointcloud_pub = rospy.Publisher('pointcloud', PointCloud2, queue_size=10)
        self.frame_id = 'camera_frame' 

    def publish_transforms(self, recognition_list, depth_frame, intrinsics):
        for recognition in recognition_list:
            text = recognition[2]
            pixel = recognition[4]
            point = self.get_3d_coordinates(depth_frame, pixel, intrinsics)
            self.send_transform(point, text)
            
    def get_3d_coordinates(self, depth_frame, pixel, intrinsics):
        u, v = pixel
        depth = depth_frame.get_distance(u, v)
        if depth == 0:
            return [0.0, 0.0, 0.0]
        point = rs.rs2_deproject_pixel_to_point(intrinsics, [u, v], depth)
        tf_point = [point[2], -point[0], -point[1]]

        return tf_point

    def send_transform(self, point, text):
        if 1 > point[2] > 0 :
            t = TransformStamped()  
            t.header.stamp = rospy.Time.now()                             
            t.header.frame_id = self.frame_id                 
            t.child_frame_id = 'button_' + text
            t.transform.translation.x = point[0]
            t.transform.translation.y = point[1]
            t.transform.translation.z = point[2]
            t.transform.rotation.x = 0.0
            t.transform.rotation.y = 0.0
            t.transform.rotation.z = 0.0
            t.transform.rotation.w = 1.0
            self.broadcaster.sendTransform(t)
        
    def publish_pointcloud(self, verts, colors):
        header = rospy.Header()
        header.stamp = rospy.Time.now()
        header.frame_id = self.frame_id
        points = []
        for i in range(len(verts)):
            x, y, z = verts[i]
            r, g, b = colors[i]
            rgb = (int(r) << 16) | (int(g) << 8) | int(b)
            points.append([x, y, z, rgb])
        
        fields = [
            PointField('x', 0, PointField.FLOAT32, 1),
            PointField('y', 4, PointField.FLOAT32, 1),
            PointField('z', 8, PointField.FLOAT32, 1),
            PointField('rgb', 12, PointField.UINT32, 1)
        ]

        pc2_msg = pc2.create_cloud(header, fields, points)
        self.pointcloud_pub.publish(pc2_msg)

if __name__ == '__main__':
    camera = RealSenseCamera()
    recognizer = ButtonRecognizer(use_optimized=False)
    box_publisher = BoxTFPublisher()
    rate = rospy.Rate(10)

    try:
        while not rospy.is_shutdown():
            depth_frame, color_frame, color_image = camera.get_frames()
            if depth_frame is None or color_frame is None or color_image is None:
                continue

            # perform button recognition
            t0 = cv2.getTickCount()
            recognition_list = recognizer.predict(color_image, draw=DRAW)
            t1 = cv2.getTickCount()
            time = (t1 - t0) / cv2.getTickFrequency()
            fps = 1.0 / time
            print('FPS :', fps)

            box_publisher.publish_transforms(recognition_list, depth_frame, camera.intrinsics)

            verts, texcoords, colors = camera.create_pointcloud(depth_frame, color_frame)
            box_publisher.publish_pointcloud(verts, colors)

            if DRAW:
                cv2.imshow('Button Recognition', color_image)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                
    except rospy.ROSInterruptException:
        pass
    finally:
        camera.stop()
        recognizer.clear_session()
        cv2.destroyAllWindows()
