import rospy
from geometry_msgs.msg import Twist, Pose, PoseStamped
from std_msgs.msg import Float32,Float32MultiArray
from tf.transformations import quaternion_from_euler
import math
import time
import json
from std_msgs.msg import String
from state import *

# Initialize publishers (one-time initialization)
cmd_vel_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
end_position_pub = rospy.Publisher('/set_target_panda_hand_world', Pose, queue_size=10)
gripper_angle_pub = rospy.Publisher('/set_finger_width', Float32, queue_size=10)
nav_goal_pub = rospy.Publisher("/move_base_simple/goal", PoseStamped, queue_size=1, latch=True)
pub = rospy.Publisher('/pick', Float32MultiArray, queue_size=10)
rospy.init_node("test") 

def move_forward(distance_m, speed_mps=0.2):
    """Move the robot forward by the specified distance (meters)."""
    duration = distance_m / speed_mps
    _send_vel(linear=speed_mps, angular=0.0, duration=duration)

def move_backward(distance_m, speed_mps=0.2):
    """Move the robot backward by the specified distance (meters)."""
    duration = distance_m / speed_mps
    _send_vel(linear=-speed_mps, angular=0.0, duration=duration)

def turn_left(angle_deg, angular_speed_dps=30):
    """Rotate the robot left by the specified angle (degrees)."""
    angular_speed_rps = math.radians(angular_speed_dps)
    angle_rad = math.radians(angle_deg)
    duration = angle_rad / angular_speed_rps
    _send_vel(linear=0.0, angular=angular_speed_rps, duration=duration)

def turn_right(angle_deg, angular_speed_dps=30):
    """Rotate the robot right by the specified angle (degrees)."""
    angular_speed_rps = math.radians(angular_speed_dps)
    angle_rad = math.radians(angle_deg)
    duration = angle_rad / angular_speed_rps
    _send_vel(linear=0.0, angular=-angular_speed_rps, duration=duration)

def _send_vel(linear, angular, duration):
    """Internal helper: publish velocity and maintain it for a duration."""
    vel_msg = Twist()
    vel_msg.linear.x = linear
    vel_msg.angular.z = angular

    rospy.loginfo(f"Sending velocity: linear={linear:.2f} m/s, angular={angular:.2f} rad/s, duration={duration:.2f}s")

    rate = rospy.Rate(10)  # 10 Hz
    start_time = rospy.Time.now().to_sec()
    while rospy.Time.now().to_sec() - start_time < duration:
        cmd_vel_pub.publish(vel_msg)
        rate.sleep()

    # Stop
    stop_msg = Twist()
    cmd_vel_pub.publish(stop_msg)
    rospy.loginfo("Stopped motion")

def send_nav_goal(x, y, yaw_deg):
    """
    Publish a navigation goal.
    :param x: Target x position (map frame)
    :param y: Target y position (map frame)
    :param yaw_deg: Target yaw in degrees
    """
    q = quaternion_from_euler(0, 0, math.radians(yaw_deg))

    goal = PoseStamped()
    goal.header.frame_id = "map"
    goal.header.stamp = rospy.Time.now()
    goal.pose.position.x = x
    goal.pose.position.y = y
    goal.pose.orientation.x, goal.pose.orientation.y = q[0], q[1]
    goal.pose.orientation.z, goal.pose.orientation.w = q[2], q[3]

    nav_goal_pub.publish(goal)
    rospy.loginfo(f"Published nav goal → x={x:.2f}, y={y:.2f}, yaw={yaw_deg}°")

def set_end_position(x, y, z, qx, qy, qz, qw):
    """
    Set end-effector position and orientation using a quaternion.
    :param x: Target x (meters)
    :param y: Target y (meters)
    :param z: Target z (meters)
    :param qx: Quaternion x
    :param qy: Quaternion y
    :param qz: Quaternion z
    :param qw: Quaternion w
    """
    if None in (qx, qy, qz, qw):
        rospy.logwarn("Orientation contains None, setting all quaternion components to 0.")
        qx, qy, qz, qw = 0.0, 0.0, 0.0, 0.0

    pose_msg = Pose()
    pose_msg.position.x = x
    pose_msg.position.y = y
    pose_msg.position.z = z
    pose_msg.orientation.x = qx
    pose_msg.orientation.y = qy
    pose_msg.orientation.z = qz
    pose_msg.orientation.w = qw

    end_position_pub.publish(pose_msg)
    rospy.loginfo(f"Published end-effector target: x={x}, y={y}, z={z}, qx={qx}, qy={qy}, qz={qz}, qw={qw}")

def set_gripper_angles(angle_deg):
    """
    Set gripper angle.
    :param angle_deg: Target gripper angle (degrees)
    """
    angle_msg = Float32()
    angle_msg.data = angle_deg

    gripper_angle_pub.publish(angle_msg)
    rospy.loginfo(f"Published gripper target angle: {angle_deg}°")

def get_operable_objs(topic_name="/environment_objects", timeout=5.0):
    """
    Subscribe to the environment_objects topic once, parse JSON, and return a dict.
    :param topic_name: Topic name
    :param timeout: Wait timeout (seconds)
    :return: dict or None if timed out
    """
    rospy.loginfo(f"[get_operable_objects] Waiting for {topic_name} ...")
    try:
        msg = rospy.wait_for_message(topic_name, String, timeout=timeout)
        objects_dict = json.loads(msg.data)
        rospy.loginfo(f"[get_operable_objects] Received {len(objects_dict)} objects")
        return objects_dict

    except rospy.ROSException:
        rospy.logwarn(f"[get_operable_objects] Timeout waiting for {topic_name} ({timeout} s)")
        return None
    except json.JSONDecodeError as e:
        rospy.logerr(f"[get_operable_objects] JSON parse failed: {e}")
        return None

def grasp(object,topic_name="/environment_objects", timeout=5.0):
    if object == "cube":
        #task2的
        # set_end_position(0.10, 0.05, 0.6, 0.706, 0.706, 0.0, 0.0)
        # rospy.sleep(10)
        # set_gripper_angles(0.051)
        # rospy.sleep(1)

        #task3的
        joint_angles = [-1.269877, 1.163136, 1.866899, -1.804529, -2.407888, 2.485351, -1.090143]
        msg = Float32MultiArray(data=joint_angles)
        pub.publish(msg)
        rospy.sleep(4)
    # else:
    #     list = {"cube": (0,0,0)}
    #     t = list[object]

def approach(object,topic_name="/environment_objects", timeout=5.0):
    return


def get_obj_position(object,topic_name="/environment_objects", timeout=5.0):
    rospy.loginfo(f"[get_obj_position] Waiting for {topic_name} ...")
    try:
        msg = rospy.wait_for_message(topic_name, String, timeout=timeout)
        objects_dict = json.loads(msg.data)
        rospy.loginfo(f"[get_obj_position] Received {len(objects_dict)} objects")
        return objects_dict[object]

    except rospy.ROSException:
        rospy.logwarn(f"[get_obj_position] Timeout waiting for {topic_name} ({timeout} s)")
        return None
    except json.JSONDecodeError as e:
        rospy.logerr(f"[get_obj_position] JSON parse failed: {e}")
        return None
    
def check_object_relation(obj1, obj2, relation, topic_name="/environment_objects", timeout=5.0):
    """
    检查 obj1 与 obj2 的空间关系
    relation: "above" 或 "below"
    基于 Z 坐标（第三个值）比较
    """
    pos1 = get_obj_position(obj1, topic_name, timeout)
    pos2 = get_obj_position(obj2, topic_name, timeout)

    if pos1 is None or pos2 is None:
        rospy.logwarn(f"[check_object_relation] Cannot get positions for {obj1} or {obj2}")
        return False

    try:
        z1 = pos1[2]
        z2 = pos2[2]
    except (IndexError, TypeError) as e:
        rospy.logerr(f"[check_object_relation] Invalid position data: {e}")
        return False

    if relation == "above":
        return z1 > z2
    elif relation == "below":
        return z1 < z2
    else:
        rospy.logwarn(f"[check_object_relation] Unknown relation: {relation}")
        return False
 
class CompileError(Exception):
    """Custom compile error"""
    pass