import math
import numpy as np

class PID_controller:
    def __init__(self, Kp_trans=1.0, Kd_trans=0.1, Kp_yaw=1.0, Kd_yaw=1.0, max_v=1.0, max_w=1.2):
        self.Kp_trans = Kp_trans
        self.Kd_trans = Kd_trans
        self.Kp_yaw = Kp_yaw
        self.Kd_yaw = Kd_yaw
        self.max_v = max_v
        self.max_w = max_w
    
    def solve(self, odom, target, vel=np.zeros(2)):
        translation_error, yaw_error = self.calculate_errors(odom, target)
        v, w = self.pd_step(translation_error, yaw_error, vel[0], vel[1])
        return v, w, translation_error, yaw_error
    
    def pd_step(self, translation_error, yaw_error, linear_vel, angular_vel):
        translation_error = max(-1.0, min(1.0, translation_error))
        yaw_error = max(-1.0, min(1.0, yaw_error))

        linear_velocity = self.Kp_trans * translation_error - self.Kd_trans * linear_vel
        angular_velocity = self.Kp_yaw * yaw_error - self.Kd_yaw * angular_vel

        linear_velocity = max(-self.max_v, min(self.max_v, linear_velocity))
        angular_velocity = max(-self.max_w, min(self.max_w, angular_velocity))
                
        return linear_velocity, angular_velocity
    
    def calculate_errors(self, odom, target):
        
        dx =  target[0, 3] - odom[0, 3]
        dy =  target[1, 3] - odom[1, 3]

        odom_yaw = math.atan2(odom[1, 0], odom[0, 0])
        target_yaw = math.atan2(target[1, 0], target[0, 0])
        
        translation_error = dx * np.cos(odom_yaw) + dy * np.sin(odom_yaw)    

        yaw_error = target_yaw - odom_yaw
        yaw_error = (yaw_error + math.pi) % (2 * math.pi) - math.pi
        
        return translation_error, yaw_error