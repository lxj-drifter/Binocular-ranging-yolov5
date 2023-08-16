#!/usr/bin/env python
# -*- coding: utf-8 -*-

import cv2
import torch
import rospy
import numpy as np
import random
import pyrealsense2 as rs

from std_msgs.msg import Header
from sensor_msgs.msg import Image
from yolov5_ros_msgs.msg import BoundingBox, BoundingBoxes


class Yolo_Dect:
    def __init__(self):

        # load parameters
        yolov5_path = rospy.get_param('/yolov5_path', '')

        weight_path = rospy.get_param('~weight_path', '')
        image_topic = rospy.get_param(
            '~image_topic', '/camera/color/image_raw')
        pub_topic = rospy.get_param('~pub_topic', '/yolov5/BoundingBoxes')
        self.camera_frame = rospy.get_param('~camera_frame', '')
        # pipeline = rs.pipeline()
        # try:
        #     while True:
        #         frames = pipeline.wait_for_frames()
        #         depth_frame = frames.get_depth_frame()
        #         color_frame = frames.get_color_frame()
        #         if not depth_frame or color_frame:
        #             continue
        #         self.color_image = np.asanyarray(color_frame.get_data())
        #         self.depth_image = np.asanyarray(depth_frame.get_data())
        #         if key & 0xFF == ord('q') or key == 27:
        #             break
        # finally:
        #     pipeline.stop()
        conf = rospy.get_param('~conf', '0.5')

        # load local repository(YoloV5:v6.0)
        self.model = torch.hub.load(yolov5_path, 'custom',
                                    path=weight_path, source='local')

        # which device will be used
        if (rospy.get_param('/use_cpu', 'true')):
            self.model.cpu()
        else:
            self.model.cuda()

        self.model.conf = conf
        self.color_image = Image()
        self.depth_image = Image()
        self.getImageStatus = False

        # Load class color
        self.classes_colors = {}

        # image subscribe
        self.color_sub = rospy.Subscriber(image_topic, Image, self.image_callback,
                                          queue_size=1, buff_size=52428800)

        # output publishers
        self.position_pub = rospy.Publisher(
            pub_topic,  BoundingBoxes, queue_size=1)

        self.image_pub = rospy.Publisher(
            '/yolov5/detection_image',  Image, queue_size=1)

        # if no image messages
        # while (not self.getImageStatus) :
        #     rospy.loginfo("waiting for image.")
        #     rospy.sleep(2)

    def image_callback(self, image):
        self.boundingBoxes = BoundingBoxes()
        self.boundingBoxes.header = image.header
        self.boundingBoxes.image_header = image.header
        self.getImageStatus = True
        # self.color_image = np.frombuffer(image.data, dtype=np.uint8).reshape(
        #     image.height, image.width, -1)
        # self.color_image = cv2.cvtColor(self.color_image, cv2.COLOR_BGR2RGB)
 

    def get_mid_pos(self, frame, box, depth_data, randnum):
        distance_list = []
        mid_pos = [(box[0] + box[2])//2, (box[1]+box[3])//2]
        min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))
        for i in range(randnum):
            bias = random.randint(-min_val//4, min_val//4)
            dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            cv2.circle(frame, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
            if dist:
                distance_list.append(dist)
        distance_list = np.array(distance_list)
        distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4]
        return np.mean(distance_list)

    # def dectshow(self, org_img, boxs, depth_data, height, width):
    def dectshow(self, org_img, boxs, depth_data):
        img = org_img.copy()

        # count = 0
        # for i in boxs:
        #     count += 1

        for box in boxs:
            # boundingBox = BoundingBox()
            # boundingBox.probability =np.float64(box[4])
            # boundingBox.xmin = np.int64(box[0])
            # boundingBox.ymin = np.int64(box[1])
            # boundingBox.xmax = np.int64(box[2])
            # boundingBox.ymax = np.int64(box[3])
            # boundingBox.num = np.int16(count)
            # boundingBox.Class = box[-1]

            # if box[-1] in self.classes_colors.keys():
            #     color = self.classes_colors[box[-1]]
            # else:
            #     color = np.random.randint(0, 183, 3)
            #     self.classes_colors[box[-1]] = color

            cv2.rectangle(img, (int(box[0]), int(box[1])),
                          (int(box[2]), int(box[3])), (0, 255, 0), 2)
            # distance_list = []
            # mid_pos = [(box[0] + box[2])//2, (box[1]+box[3])//2]
            # min_val = min(abs(box[2] - box[0]), abs(box[3] - box[1]))
            # randnum = 24
            # for i in range(randnum):
            #     bias = random.randint(-min_val//4, min_val//4)
            #     dist = depth_data[int(mid_pos[1] + bias), int(mid_pos[0] + bias)]
            #     cv2.circle(org_img, (int(mid_pos[0] + bias), int(mid_pos[1] + bias)), 4, (255,0,0), -1)
            #     if dist:
            #         distance_list.append(dist)
            # distance_list = np.array(distance_list)
            # distance_list = np.sort(distance_list)[randnum//2-randnum//4:randnum//2+randnum//4]
            # dist = np.mean(distance_list)
            dist = self.get_mid_pos(org_img, box, depth_data, 24)
            # if box[1] < 20:
            #     text_pos_y = box[1] + 30
            # else:
            #     text_pos_y = box[1] - 10
                
            cv2.putText(img, box[-1] + str(dist / 1000)[:4] + 'm',
                        (int(box[0]), int(box[1])), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)


        #     self.boundingBoxes.bounding_boxes.append(boundingBox)
        #     self.position_pub.publish(self.boundingBoxes)
        # self.publish_image(img, height, width)
        cv2.imshow('YOLOv5', img)

    def publish_image(self, imgdata, height, width):
        image_temp = Image()
        header = Header(stamp=rospy.Time.now())
        header.frame_id = self.camera_frame
        image_temp.height = height
        image_temp.width = width
        image_temp.encoding = 'bgr8'
        image_temp.data = np.array(imgdata).tobytes()
        image_temp.header = header
        image_temp.step = width * 3
        self.image_pub.publish(image_temp)


def main():
    rospy.init_node('yolov5_ros', anonymous=True)
    yolo_dect = Yolo_Dect()
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 60)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 60)
    # print("i am right")
    pipeline.start(config)
    try:
        while True:
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            results = yolo_dect.model(color_image)
            # xmin    ymin    xmax   ymax  confidence  class    name

            boxs = results.pandas().xyxy[0].values
            # self.dectshow(self.color_image, boxs, image.height, image.width)
            yolo_dect.dectshow(color_image, boxs, depth_image)
            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # Stack both images horizontally
            images = np.hstack((color_image, depth_colormap))
            # Show images
            cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
            cv2.imshow('RealSense', images)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break
    finally:
        pipeline.stop()
    rospy.spin()


if __name__ == "__main__":
    
    main()
