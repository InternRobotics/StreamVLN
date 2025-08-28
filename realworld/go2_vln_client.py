import rclpy
import sys
import threading
import PIL.Image as PIL_Image
import io
import json
import requests
import time
import numpy as np

from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped
from geometry_msgs.msg import Twist
from std_msgs.msg import Bool
from cv_bridge import CvBridge
from rclpy.node import Node

# unitree related
from unitree_go.msg import SportModeState
from unitree_api.msg import Request
from unitree_api.msg import RequestHeader

# user-specific
from pid_controller import *
# global variable
policy_init = True
pid = PID_controller(Kp_trans=3.0, Kd_trans=0.5, Kp_yaw=3.0, Kd_yaw=0.5, max_v=1.0, max_w=1.2)
manager = None

rgb_rw_lock = ReadWriteLock()
depth_rw_lock = ReadWriteLock()
odom_rw_lock = ReadWriteLock()
    
def eval_vln(image, depth, camera_pose, instruction, url='http://localhost:5801/eval_vln'):
    global policy_init
    image = PIL_Image.fromarray(image)
    # image = image.resize((384, 384))
    image_bytes = io.BytesIO()
    image.save(image_bytes, format='jpeg')
    image_bytes.seek(0)
    
    data = {"reset":policy_init}
    json_data = json.dumps(data)
    policy_init = False
    
    files = {'image': ('rgb_image', image_bytes, 'image/jpg')}
    start = time.time()
    response = requests.post(url, files=files,data={'json': json_data} , timeout=150)
    print(f"total time(delay + policy): {time.time() - start}")
    print(response.text)
    
    action = json.loads(response.text)['action']
    return action

def control_thread():
    while True:
        # odom_rw_lock.acquire_read()
        homo_odom = manager.homo_odom.copy() if manager.homo_odom is not None else None
        vel = manager.vel.copy() if manager.vel is not None else None 
        homo_goal = manager.homo_goal.copy() if manager.homo_goal is not None else None
        # odom_rw_lock.release_read()
        e_p, e_r = 0.0, 0.0
        if homo_odom is not None and vel is not None and homo_goal is not None:
            v, w, e_p, e_r = pid.solve(homo_odom, homo_goal, vel)
            manager.move(v, 0, w)
        if abs(e_p) < 0.1 and abs(e_r) < 0.1:
            manager.trigger_replan()
        time.sleep(0.1)


def planning_thread():
    while True:
        if not manager.should_plan:
            continue
        
        print(f"planning_thread running")
        rgb_rw_lock.acquire_read()
        rgb_image = manager.rgb_image
        rgb_rw_lock.release_read()

        odom_rw_lock.acquire_read()
        request_cnt = manager.request_cnt
        odom_rw_lock.release_read()
        if rgb_image is None:
            time.sleep(0.1)
            continue

        actions = eval_vln(rgb_image, None, None, None)
        print(f"111")
        odom_rw_lock.acquire_write()
        manager.should_plan = False
        manager.request_cnt += 1
        manager.incremental_change_goal(actions)
        odom_rw_lock.release_write()
        time.sleep(0.1)


class Go2VlnManager(Node):
    def __init__(self):
        super().__init__('go2_manager')

        # subsucriber
        self.rgb_sub = self.create_subscription(Image, "/camera/camera/color/image_raw", self.rgb_callback, 1)
        # self.depth_sub = self.create_subscription(Image, "/camera/camera/aligned_depth_to_color/image_raw",
        #                                           self.depth_callback, 1)
        self.odom_sub = self.create_subscription(SportModeState, "/sportmodestate", self.odom_callback, 10)

        # publisher
        self.control_pub = self.create_publisher(Request, '/api/sport/request', 5)

        # class member variable
        self.cv_bridge = CvBridge()
        self.rgb_image = None
        self.depth_image = None
        self.homo_goal = None
        self.homo_odom = None
        self.vel = None
        
        self.request_cnt = 0
        self.odom_cnt = 0
        
        self.should_plan = False
        self.last_plan_time = 0.0
    def rgb_callback(self, msg):
        rgb_rw_lock.acquire_write()
        raw_image = self.cv_bridge.imgmsg_to_cv2(msg, 'bgr8')[:, :, :]
        self.rgb_image = raw_image
        rgb_rw_lock.release_write()

    def depth_callback(self, msg):
        depth_rw_lock.acquire_write()
        if self.rgb_image is None:
            depth_rw_lock.release_write()
            return
        # print("Received depth image")
        raw_depth = self.cv_bridge.imgmsg_to_cv2(msg, '16UC1')
        raw_depth[np.isnan(raw_depth)] = 0
        raw_depth[np.isinf(raw_depth)] = 0
        self.depth_image = raw_depth / 1000.0
        self.depth_image -= 0.0
        self.depth_image[np.where(self.depth_image < 0)] = 0
        depth_rw_lock.release_write()

    def odom_callback(self, msg):
        DOWNSAMPLE_RATIO = 5
        self.odom_cnt += 1
        if self.odom_cnt % DOWNSAMPLE_RATIO != 0:
            return
        odom_rw_lock.acquire_write()
        R0 = np.array([[np.cos(msg.imu_state.rpy[2]), -np.sin(msg.imu_state.rpy[2])],
                       [np.sin(msg.imu_state.rpy[2]), np.cos(msg.imu_state.rpy[2])]])
        self.homo_odom = np.eye(4)
        self.homo_odom[:2, :2] = R0
        self.homo_odom[:2, 3] = [msg.position[0], msg.position[1]]
        self.vel = [msg.velocity[0], msg.yaw_speed]
        
        if self.odom_cnt == DOWNSAMPLE_RATIO:
            # fisrst odom
            self.homo_goal = self.homo_odom.copy()
        odom_rw_lock.release_write()

    def trigger_replan(self):
        self.should_plan = True

    def incremental_change_goal(self, actions):
        if self.homo_goal is None:
            raise ValueError("Please initialize homo_goal before change it!")
        homo_goal = self.homo_goal
        
        for each_action in actions:
            if each_action == 0:
                pass
            elif each_action == 1:
                yaw = math.atan2(homo_goal[1, 0], homo_goal[0, 0])
                homo_goal[0, 3] += 0.25 * np.cos(yaw)
                homo_goal[1, 3] += 0.25 * np.sin(yaw)
            elif each_action == 2:
                angle = math.radians(15)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])
            elif each_action == 3:
                angle = -math.radians(15.0)
                rotation_matrix = np.array([
                    [math.cos(angle), -math.sin(angle), 0],
                    [math.sin(angle),  math.cos(angle), 0],
                    [0,                0,               1]
                ])
                homo_goal[:3, :3] = np.dot(rotation_matrix, homo_goal[:3, :3])  
        self.homo_goal = homo_goal
        
    def move(self, vx, vy, vyaw):
        SPORT_API_ID_MOVE = 1008
        p = {"x": vx, "y": vy, "z": vyaw}
        parameter = json.dumps(p)
        header = RequestHeader()
        header.identity._api_id = SPORT_API_ID_MOVE
        header.identity.id = time.monotonic_ns()
        input_dict = {'parameter': parameter, 'header': header}
        request = Request(**input_dict)

        self.control_pub.publish(request)


if __name__ == '__main__':

    control_thread_instance = threading.Thread(target=control_thread)
    planning_thread_instance = threading.Thread(target=planning_thread)

    rclpy.init()

    try:
        manager = Go2VlnManager()

        control_thread_instance.start()
        planning_thread_instance.start()

        rclpy.spin(manager)
    except KeyboardInterrupt:
        pass
    finally:
        manager.destroy_node()
        rclpy.shutdown()