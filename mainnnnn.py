# !/usr/bin/env python3
import rclpy
import numpy as np
import cv2
from rclpy.node import Node

from sensor_msgs.msg import Image, CompressedImage
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import Header

def detect(image):
    # 이미지를 그레이스케일로 변환
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 블러 처리
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Adaptive Thresholding 적용
    adaptive_thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                            cv2.THRESH_BINARY_INV, 17, 9)

    # 컨투어 검출
    contours, _ = cv2.findContours(adaptive_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_count=0
    blue_count=0
    green_count=0

    # 검출된 컨투어 중 모니터로 보이는 것을 찾기
    for contour in contours:
        # 컨투어의 최소 외접 사각형을 구하기 (기울어진 경우를 대비)
        rect = cv2.minAreaRect(contour)
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 검출된 영역이 특정 크기 이상일 경우
        if rect[1][0] > 50 and rect[1][1] > 50:
            # 모니터로 추정되는 영역에 사각형 그리기
            cv2.drawContours(image, [box], 0, (0, 255, 0), 2)
            im=cv2.drawContours(image, [box], 0, (0, 255, 0), 2)

            # 사각형 영역을 자르기
            width, height = int(rect[1][0]), int(rect[1][1])
            src_pts = box.astype("float32")
            dst_pts = np.array([[0, height-1],
                                [0, 0],
                                [width-1, 0],
                                [width-1, height-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            warped = cv2.warpPerspective(image, M, (width, height))

            # 자른 영역을 HSV 색상 공간으로 변환
            hsv = cv2.cvtColor(warped, cv2.COLOR_BGR2HSV)

            # 각 색상의 범위 설정
            lower_red = np.array([0, 51, 51])
            upper_red = np.array([10, 255, 255])
            lower_green = np.array([45, 51, 51])
            upper_green = np.array([70, 255, 255])
            lower_blue = np.array([105, 51, 51])
            upper_blue = np.array([130, 255, 255])

            # 각 색상의 픽셀 계산
            red_mask = cv2.inRange(hsv, lower_red, upper_red)
            green_mask = cv2.inRange(hsv, lower_green, upper_green)
            blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)

            red_count += cv2.countNonZero(red_mask)
            green_count += cv2.countNonZero(green_mask)
            blue_count += cv2.countNonZero(blue_mask)
    # 가장 많은 색상 출력
    cv2.imshow('image',im)

    if red_count > green_count and red_count > blue_count:
        print('가장 많은 색상은 Red입니다.')
        return 'R'
    elif green_count > red_count and green_count > blue_count:
        print('가장 많은 색상은 Green입니다.')
        return 'G'
    else:
        print('가장 많은 색상은 Blue입니다.')
        return 'B'

class DetermineColor(Node):
    def __init__(self):
        super().__init__('color_detector')
        self.image_sub = self.create_subscription(Image, '/camera/color/image_raw', self.callback, 10)
        self.color_pub = self.create_publisher(Header, '/rotate_cmd', 10)
        self.bridge = CvBridge()
        #self.count = 0

    def callback(self, data):
        try:
            image = self.bridge.imgmsg_to_cv2(data, 'bgr8')
            msg = Header()
            msg = data.header
            msg.frame_id = '0'  # default: STOP
            c=detect(image)
            if c=='B':
                msg.frame_id='+1'
            elif c=='R':
                msg.frame_id='-1'
            else:
                msg.frame_id='0'
            self.color_pub.publish(msg)
        except CvBridgeError as e:
            self.get_logger().error('Failed to convert image: %s' % e)


if __name__ == '__main__':
    rclpy.init()
    detector = DetermineColor()
    rclpy.spin(detector)
    detector.destroy_node()
    rclpy.shutdown()