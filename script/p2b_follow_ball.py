#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image, LaserScan
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class BallFollower:
    def __init__(self):
        self.bridge = CvBridge()

        # Publishers
        self.cmd_pub   = rospy.Publisher('/cmd_vel', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('/image_converter/output_video', Image, queue_size=1)

        # Subscribers
        rospy.Subscriber('/camera/rgb/image_raw', Image,     self.rgb_callback)
        rospy.Subscriber('/scan',                 LaserScan, self.scan_callback)

        self.ball_col    = None   # horizontal pixel position of ball
        self.ball_size   = None   # radius in pixels
        self.front_dist  = None   # distance from laser scan (meters)

        rospy.loginfo("Ball Follower Started!")

    # ── Laser scan: get distance directly in front ──
    def scan_callback(self, msg):
        # Front of robot = index 0 in laser scan
        # Take average of a small window around front to be more stable
        ranges = np.array(msg.ranges)
        num    = len(ranges)

        # Look at center 30 degrees (15 each side)
        window = 20
        front_ranges = np.concatenate([
            ranges[:window],           # first 20 readings
            ranges[num - window:]      # last 20 readings
        ])

        # Filter out inf and nan values
        valid = front_ranges[np.isfinite(front_ranges) & (front_ranges > 0.01)]

        if len(valid) > 0:
            self.front_dist = float(np.min(valid))  # closest object in front
        else:
            self.front_dist = None

    # ── RGB: Professor's logic to find most-red pixel ──
    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
            cv_image = cv2.cvtColor(cv_image, cv2.COLOR_RGB2BGR)
        except Exception as e:
            rospy.logerr("RGB error: %s", str(e))
            return

        rows, cols = cv_image.shape[:2]

        # ── Professor's formula: find most-red pixel ──
        b = cv_image[:, :, 0].astype(np.int32)
        g = cv_image[:, :, 1].astype(np.int32)
        r = cv_image[:, :, 2].astype(np.int32)
        dis = (r - 255)**2 + g**2 + b**2

        #np.argmin(dis) — finds the index of the minimum value in the flattened array
        #np.unravel_index(np.argmin(dis), dis.shape) — converts flat index back to (row, col)-> min row and col
        min_idx   = np.unravel_index(np.argmin(dis), dis.shape)
        best_row  = min_idx[0]
        best_col  = min_idx[1]

        # ── Also use contour to get ball size ──
        #HSV: H=color, S=saturation, V=brightness
        #cv2.COLOR_BGR2HSV: Conversion code: BGR → HSV 
        hsv      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        #red range 1:catches H: 0-10
        lower_r1 = np.array([0,   50, 50])
        upper_r1 = np.array([10, 255, 255])
        #red range 2:catches H: 155-180
        lower_r2 = np.array([155, 50, 50])
        upper_r2 = np.array([180, 255, 255])
        #A pixel is red if it matches range1 OR range2
        mask     = cv2.bitwise_or(
                       cv2.inRange(hsv, lower_r1, upper_r1),
                       cv2.inRange(hsv, lower_r2, upper_r2))

        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c              = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            if radius > 5:
                self.ball_col  = int(x)
                self.ball_size = radius
                # Draw green circle around ball
                cv2.circle(cv_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
            else:
                self.ball_col  = None
                self.ball_size = None
        else:
            self.ball_col  = None
            self.ball_size = None

        # Draw professor's red dot on most-red pixel
        cv2.circle(cv_image, (best_col, best_row), 10, (0, 0, 255), 2)

        # Draw distance info on image
        dist_text = "{:.2f}m".format(self.front_dist) if self.front_dist else "N/A"
        cv2.putText(cv_image, "dist: " + dist_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(cv_image, "size: {:.1f}px".format(self.ball_size if self.ball_size else 0),
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        # Publish output image
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

    # ── Control robot ──
    def follow(self):
        rate      = rospy.Rate(10)
        img_width = 640

        while not rospy.is_shutdown():
            twist = Twist()

            if self.ball_col is not None:
                # Horizontal error from image center
                error = self.ball_col - img_width // 2

                # Rotate to center ball
                twist.angular.z = -float(error) / 500.0

                # Use laser scan for distance control
                if self.front_dist is not None:
                    if self.front_dist > 1.1:
                        twist.linear.x = 0.2    # too far → move forward
                    elif self.front_dist < 0.9:
                        twist.linear.x = -0.2   # too close → move back
                    else:
                        twist.linear.x = 0.0    # ~1 meter → stop

                    rospy.loginfo("Ball col: %d | laser dist: %.2fm | error: %d",
                                  self.ball_col, self.front_dist, error)
                else:
                    # No laser reading → move forward slowly
                    twist.linear.x = 0.1
                    rospy.loginfo("Ball col: %d | no laser | error: %d",
                                  self.ball_col, error)
            else:
                # No ball detected → stop
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("No ball detected, stopping...")

            self.cmd_pub.publish(twist)
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('ball_follower')
    follower = BallFollower()
    follower.follow()