#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class BallFollower:
    def __init__(self):
        self.bridge = CvBridge()

        # ── Publishers ──
        self.cmd_pub   = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('/image_converter/output_video', Image, queue_size=1)

        # ── Subscribers ( rgb + depth) ──
        rospy.Subscriber('/usb_cam/image_raw',      Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        self.ball_col    = None   # horizontal pixel of ball center
        self.ball_row    = None   # vertical pixel of ball center
        self.ball_size   = None   # radius in pixels
        self.front_dist  = None   # distance in meters from depth camera
        self.depth_image = None   # latest depth frame

        rospy.loginfo("Ball Follower Started!")

    # ────────────────────────────────────────────
    # DEPTH CALLBACK — Astra depth camera
    # ────────────────────────────────────────────
    def depth_callback(self, msg):
        try:
            # 32FC1 = 32-bit float, 1 channel (distance in meters per pixel)
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
        except Exception as e:
            rospy.logerr("Depth error: %s", str(e))

    # ────────────────────────────────────────────
    # RGB CALLBACK — usb_cam
    # ────────────────────────────────────────────
    def rgb_callback(self, msg):
        try:
            # usb_cam publishes bgr8 — no conversion needed
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("RGB error: %s", str(e))
            return

        
        # Find most-red pixel using color distance from pure red (255,0,0)
        b = cv_image[:, :, 0].astype(np.int32)
        g = cv_image[:, :, 1].astype(np.int32)
        r = cv_image[:, :, 2].astype(np.int32)

        
        # dis = (r-255)^2 + g^2 + b^2
        dis      = (r - 255)**2 + g**2 + b**2
        min_idx  = np.unravel_index(np.argmin(dis), dis.shape)
        best_row = min_idx[0]   # y position
        best_col = min_idx[1]   # x position

        # ── Step 2: HSV masking to find ball contour ──
        hsv      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)

        # Red wraps around HSV — need two ranges
        lower_r1 = np.array([0,   100, 100])   # H:0-10   red
        upper_r1 = np.array([10,  255, 255])
        lower_r2 = np.array([160, 100, 100])   # H:160-180 red
        upper_r2 = np.array([180, 255, 255])

        mask = cv2.bitwise_or(
            cv2.inRange(hsv, lower_r1, upper_r1),
            cv2.inRange(hsv, lower_r2, upper_r2))

        # ── Step 3: Find contours in mask ──
        #return: list of all outlines found & hierarchy info(discard) 
        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            # Pick largest contour = most likely the ball
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 10:  # ignore tiny noise blobs
                self.ball_col  = int(x)
                self.ball_row  = int(y)
                self.ball_size = radius

                # ── Step 4: Get depth at exact ball pixel ──
                if self.depth_image is not None:
                    dist = self.depth_image[int(y), int(x)]
                    if np.isfinite(dist) and dist > 0.1:
                        self.front_dist = float(dist)
                    else:
                        self.front_dist = None

                # Draw green circle around detected ball
                cv2.circle(cv_image,
                           (int(x), int(y)), int(radius),
                           (0, 255, 0), 2)
            else:
                # Ball too small — reset
                self.ball_col  = None
                self.ball_row  = None
                self.ball_size = None
                self.front_dist = None
        else:
            # No red found — reset
            self.ball_col  = None
            self.ball_row  = None
            self.ball_size = None
            self.front_dist = None

        # ── Step 5: Draw professor's red dot ──
        cv2.circle(cv_image,
                   (best_col, best_row), 10,
                   (0, 0, 255), 2)

        # ── Step 6: Draw info text on image ──
        dist_text = "{:.2f}m".format(self.front_dist) if self.front_dist else "N/A"
        size_text = "{:.1f}px".format(self.ball_size) if self.ball_size else "N/A"

        cv2.putText(cv_image, "dist: " + dist_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)
        cv2.putText(cv_image, "size: " + size_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # ── Step 7: Publish output image ──
        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

    # ────────────────────────────────────────────
    # FOLLOW LOOP — control robot movement
    # ────────────────────────────────────────────
    def follow(self):
        rate      = rospy.Rate(10)
        img_width = 640   # camera image width in pixels

        while not rospy.is_shutdown():
            twist = Twist()

            if self.ball_col is not None:
                # How far left/right is ball from image center?
                error = self.ball_col - img_width // 2

                # Rotate toward ball center
                # negative = turn right, positive = turn left
                twist.angular.z = -float(error) / 1000.0

                # Move forward/back based on depth distance
                if self.front_dist is not None:
                    if self.front_dist > 1.1:
                        twist.linear.x =  0.1   # too far  → move forward
                    elif self.front_dist < 0.9:
                        twist.linear.x = -0.1   # too close → move back
                    else:
                        twist.linear.x =  0.0   # ~1 meter  → stop

                    rospy.loginfo(
                        "Ball col: %d | dist: %.2fm | error: %d",
                        self.ball_col, self.front_dist, error)
                else:
                    # Ball visible but no depth reading
                    # Move slowly forward until depth works
                    twist.linear.x = 0.05
                    rospy.loginfo(
                        "Ball col: %d | no depth | error: %d",
                        self.ball_col, error)
            else:
                # No ball detected — stop and wait
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                rospy.loginfo("No ball detected, stopping...")

            self.cmd_pub.publish(twist)
            rate.sleep()

# ────────────────────────────────────────────
# MAIN
# ────────────────────────────────────────────
if __name__ == '__main__':
    rospy.init_node('ball_follower')
    follower = BallFollower()
    follower.follow()