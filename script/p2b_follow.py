#!/usr/bin/env python3

import rospy
import cv2
import numpy as np
import threading
from sensor_msgs.msg import Image
from geometry_msgs.msg import Twist
from cv_bridge import CvBridge

class BallFollower:
    def __init__(self):
        self.bridge = CvBridge()
        self.lock   = threading.Lock()

        self.cmd_pub   = rospy.Publisher('/cmd_vel_mux/input/teleop', Twist, queue_size=1)
        self.image_pub = rospy.Publisher('/image_converter/output_video', Image, queue_size=1)

        # Astra camera topics
        rospy.Subscriber('/camera/rgb/image_raw',   Image, self.rgb_callback)
        rospy.Subscriber('/camera/depth/image_raw', Image, self.depth_callback)

        self.ball_col    = None
        self.ball_row    = None
        self.ball_size   = None
        self.front_dist  = None
        self.depth_image = None
        self.img_width   = 640    # default, updated from real image

        self.target_dist = 1.0    # 1 meter per assignment
        self.tolerance   = 0.05
        self.RED_THRESH  = 3000   # max dis² to consider scene "has red"

        rospy.loginfo("Ball Follower Started!")

    # ─────────────────────────────────────────
    def depth_callback(self, msg):
        try:
            depth_raw = self.bridge.imgmsg_to_cv2(msg, '16UC1')
            depth = depth_raw.astype(np.float32) / 1000.0
        except:
            try:
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            except Exception as e:
                rospy.logerr("Depth error: %s", str(e))
                return
        with self.lock:
            self.depth_image = depth

    # ─────────────────────────────────────────
    def rgb_callback(self, msg):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            rospy.logerr("RGB error: %s", str(e))
            return

        img_h, img_w = cv_image.shape[:2]

        # store real image width for follow()
        with self.lock:
            self.img_width = img_w

        # ── Step 1: find most-red pixel (professor's method) ──
        b = cv_image[:, :, 0].astype(np.int32)
        g = cv_image[:, :, 1].astype(np.int32)
        r = cv_image[:, :, 2].astype(np.int32)

        dis      = (r - 255)**2 + g**2 + b**2
        min_idx  = np.unravel_index(np.argmin(dis), dis.shape)
        best_row = min_idx[0]
        best_col = min_idx[1]
        min_dis  = dis[best_row, best_col]

        # draw red dot at most-red pixel
        cv2.circle(cv_image, (best_col, best_row), 10, (0, 0, 255), 2)

        # ── Step 2: gate — if no red in scene, skip detection ──
        if min_dis > self.RED_THRESH:
            rospy.loginfo_throttle(1.0, "No red in scene (min_dis=%d)", min_dis)
            with self.lock:
                self.ball_col   = None
                self.ball_row   = None
                self.ball_size  = None
                self.front_dist = None
            self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))
            return

        # ── Step 3: HSV mask for red ──
        hsv  = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
        mask = cv2.bitwise_or(
            cv2.inRange(hsv, np.array([0,   100, 100]), np.array([10,  255, 255])),
            cv2.inRange(hsv, np.array([160, 100, 100]), np.array([180, 255, 255]))
        )

        # ── Step 4: contours + circularity check ──
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        ball_detected = False

        if contours:
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            area      = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)

            # correct circularity: 1.0 = perfect circle, < 1.0 otherwise
            circularity = (4 * 3.14159 * area / (perimeter * perimeter)
                           if perimeter > 0 else 0)

            if radius > 35 and circularity > 0.6:
                ball_detected = True

                # ── Step 5: depth lookup with resolution scaling ──
                with self.lock:
                    depth_snapshot = self.depth_image

                front_dist = None
                if depth_snapshot is not None:
                    dh, dw  = depth_snapshot.shape[:2]
                    scale_x = dw / img_w
                    scale_y = dh / img_h
                    dx = int(x * scale_x)
                    dy = int(y * scale_y)

                    y1 = max(0,  dy - 2)
                    y2 = min(dh, dy + 2)
                    x1 = max(0,  dx - 2)
                    x2 = min(dw, dx + 2)

                    region = depth_snapshot[y1:y2, x1:x2]
                    valid  = region[
                        (region > 0.1) & (region < 5.0) & np.isfinite(region)
                    ]
                    if len(valid) > 0:
                        front_dist = float(np.mean(valid))

                with self.lock:
                    self.ball_col   = int(x)
                    self.ball_row   = int(y)
                    self.ball_size  = radius
                    self.front_dist = front_dist

                # draw green circle around detected ball
                cv2.circle(cv_image, (int(x), int(y)), int(radius), (0, 255, 0), 2)
                cv2.circle(cv_image, (int(x), int(y)), 4, (0, 255, 0), -1)

        if not ball_detected:
            with self.lock:
                self.ball_col   = None
                self.ball_row   = None
                self.ball_size  = None
                self.front_dist = None

        # ── Draw HUD ──
        with self.lock:
            front_dist = self.front_dist
            ball_size  = self.ball_size

        dist_text = "{:.2f}m".format(front_dist) if front_dist is not None else "N/A"
        size_text = "{:.1f}px".format(ball_size)  if ball_size  is not None else "N/A"

        cv2.putText(cv_image, "dist:   " + dist_text,
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(cv_image, "size:   " + size_text,
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(cv_image, "target: {:.1f}m".format(self.target_dist),
                    (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, 'bgr8'))

    # ─────────────────────────────────────────
    def follow(self):
        rate = rospy.Rate(10)

        while not rospy.is_shutdown():
            twist = Twist()

            with self.lock:
                ball_col   = self.ball_col
                front_dist = self.front_dist
                img_width  = self.img_width

            if ball_col is not None:
                error           = ball_col - img_width // 2
                twist.angular.z = -float(error) / 300.0

                if front_dist is not None:
                    dist_error = front_dist - self.target_dist

                    if dist_error > self.tolerance:
                        twist.linear.x = min(0.2, dist_error * 0.4)
                    elif dist_error < -self.tolerance:
                        twist.linear.x = max(-0.2, dist_error * 0.4)
                    else:
                        twist.linear.x = 0.0

                    rospy.loginfo_throttle(0.5,
                        "dist: %.2fm | error: %.2fm | col: %d",
                        front_dist, dist_error, ball_col)
                else:
                    twist.linear.x = 0.05
                    rospy.loginfo_throttle(1.0, "Ball col: %d | no depth", ball_col)
            else:
                twist.linear.x  = 0.0
                twist.angular.z = 0.0
                rospy.loginfo_throttle(1.0, "No ball detected, stopping...")

            self.cmd_pub.publish(twist)
            rate.sleep()


if __name__ == '__main__':
    rospy.init_node('ball_follower')
    follower = BallFollower()
    follower.follow()
