#!/usr/bin/env python3
import rospy
import actionlib
import math
from move_base_msgs.msg import MoveBaseAction, MoveBaseGoal
from geometry_msgs.msg import Quaternion
import tf.transformations

def make_goal(x, y, z=0.0):
    #MoveBaseGoal(): prebuild structure: Header: frame and timestamp; pose position: x,y,z,w
    goal = MoveBaseGoal()
    goal.target_pose.header.frame_id = "map"
    goal.target_pose.header.stamp = rospy.Time.now()
    goal.target_pose.pose.position.x = x
    goal.target_pose.pose.position.y = y
    degree = math.radians(z)
    #use quaternions for computer instead of euler(human way)
    #z = sin(theta/2) ; w = cos(theta/2)
    q = tf.transformations.quaternion_from_euler(0, 0, degree)
    #unpack q list by *
    goal.target_pose.pose.orientation = Quaternion(*q)
    return goal

def navigate_to(client, x, y, label, z=0.0):
    rospy.loginfo(f"Navigating to {label}: ({x}, {y})")
    goal = make_goal(x, y, z)
    client.send_goal(goal)
    client.wait_for_result()
    state = client.get_state()
    if state == actionlib.GoalStatus.SUCCEEDED:
        rospy.loginfo(f"Reached {label}!")
    else:
        rospy.logwarn(f"Failed to reach {label}, state: {state}")

def main():
    rospy.init_node('p2a_navi')
    #'move_base' is the ROS node, MoveBaseAction is the type of message
    client = actionlib.SimpleActionClient('move_base', MoveBaseAction)
    rospy.loginfo("Waiting for move_base action server...")
    client.wait_for_server()
    rospy.loginfo("Connected to move_base!")

    # ---- DEFINE  L1, L2, L3 HERE ----
    #L1 is the starter point; L2 is the middle point; L3 is the end; then return to L1
    L1 = (-2.09, -0.501)
    L2 = (1.63, 1.34)
    L3 = (0.425, 1.7)
    # ---------------------------------------

    navigate_to(client, L1[0], L1[1], "L1 (start)")
    rospy.sleep(1.0)
    navigate_to(client, L2[0], L2[1], "L2")
    rospy.sleep(1.0)
    navigate_to(client, L3[0], L3[1], "L3")
    rospy.sleep(1.0)
    navigate_to(client, L1[0], L1[1], "L1 (return)")
    rospy.loginfo("Mission complete: L1 → L2 → L3 → L1")

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        rospy.loginfo("Navigation interrupted.")