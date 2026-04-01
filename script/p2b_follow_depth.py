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

        # ── Subscribers ──
        rospy.Subscriber('/usb_cam/image_raw',      Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        # RGB detection results
        self.rgb_ball_col   = None   # ball x pixel from RGB
        self.rgb_ball_row   = None   # ball y pixel from RGB
        self.rgb_ball_size  = None   # ball radius from RGB
        self.rgb_ball_dist  = None   # distance from depth at RGB ball position

        # Depth detection results
        self.depth_ball_col  = None  # ball x pixel from depth
        self.depth_ball_row  = None  # ball y pixel from depth
        self.depth_ball_dist = None  # distance from depth detection

        # Raw camera frames
        self.depth_image = None      # latest depth frame
        self.rgb_image   = None      # latest rgb frame

        # Tracking mode for logging
        self.mode = "NONE"

        rospy.loginfo("Ball Follower Started! (Bonus Mode: Depth First)")

    # ────────────────────────────────────────────
    # DEPTH CALLBACK — store latest depth frame
    # ────────────────────────────────────────────
    def depth_callback(self, msg):
        try:
            self.depth_image = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            # Try to detect ball from depth every frame
            self.detect_ball_from_depth()
        except Exception as e:
            rospy.logerr("Depth error: %s", str(e))

    # ────────────────────────────────────────────
    # BONUS: Detect ball using DEPTH only
    # No RGB needed — works with any color ball!
    # ────────────────────────────────────────────
    def detect_ball_from_depth(self):
        if self.depth_image is None:
            return

        # Step 1: Normalize depth to 0-255 for circle detection
        depth_norm = cv2.normalize(self.depth_image, None, 0, 255, cv2.NORM_MINMAX)
        depth_8u = np.uint8(depth_norm)

        # Step 2: Blur to reduce noise
        blurred = cv2.GaussianBlur(depth_8u, (9, 9), 2)

        # Step 3: Hough circle detection — finds circular shapes
        circles = cv2.HoughCircles(
            blurred,
            cv2.HOUGH_GRADIENT,
            dp=1,           # inverse ratio of resolution
            minDist=50,     # minimum distance between circles
            param1=50,      # edge detection threshold
            param2=30,      # circle detection threshold (lower=more circles)
            minRadius=10,   # minimum ball radius in pixels
            maxRadius=150)  # maximum ball radius in pixels

        if circles is not None:
            # Pick the largest circle (most likely the ball)
            circles = np.round(circles[0, :]).astype("int")
            largest = max(circles, key=lambda c: c[2])  # c[2] = radius
            x, y, r = largest

            # Get actual distance at circle center
            dist = self.depth_image[y, x]

            if np.isfinite(dist) and dist > 0.1:
                self.depth_ball_col  = x
                self.depth_ball_row  = y
                self.depth_ball_dist = float(dist)
            else:
                self.depth_ball_col  = None
                self.depth_ball_row  = None
                self.depth_ball_dist = None
        else:
            self.depth_ball_col  = None
            self.depth_ball_row  = None
            self.depth_ball_dist = None

    # ────────────────────────────────────────────
    # RGB CALLBACK — store frame + detect red ball
    # ────────────────────────────────────────────
    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("RGB error: %s", str(e))
            return

        self.rgb_image = cv_image.copy()

        # ── Professor's formula: find most-red pixel ──
        b = cv_image[:, :, 0].astype(np.int32)
        g = cv_image[:, :, 1].astype(np.int32)
        r = cv_image[:, :, 2].astype(np.int32)
        dis      = (r - 255)**2 + g**2 + b**2
        min_idx  = np.unravel_index(np.argmin(dis), dis.shape)
        best_row = min_idx[0]
        best_col = min_idx[1]

        # ── HSV masking for red ball ──
        hsv      = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        lower_r1 = np.array([0,   100, 100])
        upper_r1 = np.array([10,  255, 255])
        lower_r2 = np.array([160, 100, 100])
        upper_r2 = np.array([180, 255, 255])
        mask     = cv2.bitwise_or(
                       cv2.inRange(hsv, lower_r1, upper_r1),
                       cv2.inRange(hsv, lower_r2, upper_r2))

        contours, _ = cv2.findContours(
            mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)

            if radius > 10:
                self.rgb_ball_col  = int(x)
                self.rgb_ball_row  = int(y)
                self.rgb_ball_size = radius

                # Get depth at RGB ball position
                if self.depth_image is not None:
                    dist = self.depth_image[int(y), int(x)]
                    if np.isfinite(dist) and dist > 0.1:
                        self.rgb_ball_dist = float(dist)
                    else:
                        self.rgb_ball_dist = None

                # Draw green circle — RGB detected ball
                cv2.circle(cv_image,
                           (int(x), int(y)), int(radius),
                           (0, 255, 0), 2)
            else:
                self.rgb_ball_col  = None
                self.rgb_ball_row  = None
                self.rgb_ball_dist = None
        else:
            self.rgb_ball_col  = None
            self.rgb_ball_row  = None
            self.rgb_ball_dist = None

        # ── Draw depth detection result ──
        if self.depth_ball_col is not None:
            # Blue circle = depth detected ball
            cv2.circle(cv_image,
                       (self.depth_ball_col, self.depth_ball_row), 20,
                       (255, 0, 0), 2)

        # ── Draw professor's red dot ──
        cv2.circle(cv_image,
                   (best_col, best_row), 10,
                   (0, 0, 255), 2)

        # ── Draw mode and distance info ──
        mode_color = (255, 0, 0) if self.mode == "DEPTH" else (0, 255, 0)
        cv2.putText(cv_image, "Mode: " + self.mode,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, mode_color, 2)

        if self.mode == "DEPTH" and self.depth_ball_dist:
            dist_text = "{:.2f}m".format(self.depth_ball_dist)
        elif self.mode == "RGB" and self.rgb_ball_dist:
            dist_text = "{:.2f}m".format(self.rgb_ball_dist)
        else:
            dist_text = "N/A"

        cv2.putText(cv_image, "dist: " + dist_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (255, 255, 0), 2)

        # ── Publish output image ──
        self.image_pub.publish(
            self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

    # ────────────────────────────────────────────
    # FOLLOW LOOP — depth first, RGB fallback
    # ────────────────────────────────────────────
    def follow(self):
        rate      = rospy.Rate(10)
        img_width = 640

        while not rospy.is_shutdown():
            twist = Twist()

            # ════════════════════════════════
            # BONUS: Try DEPTH detection first
            # Works with ANY color ball!
            # Cover RGB camera → still works!
            # ════════════════════════════════
            if self.depth_ball_col is not None and \
               self.depth_ball_dist is not None:

                self.mode = "DEPTH"
                error = self.depth_ball_col - img_width // 2

                # Rotate toward ball
                twist.angular.z = -float(error) / 1000.0

                # Maintain 1 meter distance
                if self.depth_ball_dist > 1.1:
                    twist.linear.x =  0.1
                elif self.depth_ball_dist < 0.9:
                    twist.linear.x = -0.1
                else:
                    twist.linear.x =  0.0

                rospy.loginfo(
                    "[DEPTH] dist: %.2fm | error: %d",
                    self.depth_ball_dist, error)

            # ════════════════════════════════
            # PRIMARY: Fallback to RGB
            # Finds RED ball specifically
            # ════════════════════════════════
            elif self.rgb_ball_col is not None:

                self.mode = "RGB"
                error = self.rgb_ball_col - img_width // 2

                # Rotate toward red ball
                twist.angular.z = -float(error) / 1000.0

                # Maintain 1 meter distance
                if self.rgb_ball_dist is not None:
                    if self.rgb_ball_dist > 1.1:
                        twist.linear.x =  0.1
                    elif self.rgb_ball_dist < 0.9:
                        twist.linear.x = -0.1
                    else:
                        twist.linear.x =  0.0
                else:
                    # No depth at RGB ball → move slowly
                    twist.linear.x = 0.05

                rospy.loginfo(
                    "[RGB] dist: %.2fm | error: %d",
                    self.rgb_ball_dist or 0, error)

            # ════════════════════════════════
            # No ball found — stop
            # ════════════════════════════════
            else:
                self.mode = "NONE"
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
