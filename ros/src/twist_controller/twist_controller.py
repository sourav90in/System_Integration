
from yaw_controller import YawController
from pid import PID
from lowpass import LowPassFilter
import rospy

GAS_DENSITY = 2.858
ONE_MPH = 0.44704


class Controller(object):
    def __init__(self, vehicle_mass,fuel_capacity,brake_deadband,decel_limit,accel_limit,
                 wheel_radius,wheel_base,steer_ratio,max_lat_accel,max_steer_angle):
        
        self.yaw_controller = YawController(wheel_base,steer_ratio,0.1,max_lat_accel,max_steer_angle)
        self.pid_controller = PID(0.3,0.1,0.1,0,0.3)
        self.lpf = LowPassFilter(0.5,0.02)
        self.last_time = rospy.get_time()
        self.decel_limit = decel_limit
        self.veh_mass = vehicle_mass
        self.wheel_rad = wheel_radius

    def control(self,current_linear_velocity,twist_linear_velocity,twist_angular_velocity,is_dbw_enabled ):
        if is_dbw_enabled == False:
            self.pid_controller.reset()
            #TODO - double check the time base update
            self.last_time = rospy.get_time()
            return 0.,0.,0.
        
        rospy.loginfo("Velocity values are: %f %f %f", twist_linear_velocity,twist_angular_velocity,current_linear_velocity)
        current_linear_velocity = self.lpf.filt(current_linear_velocity)
        steering_angle = self.yaw_controller.get_steering(twist_linear_velocity,
                                                          twist_angular_velocity,
                                                          current_linear_velocity)
        vel_error = twist_linear_velocity - current_linear_velocity
        cur_time = rospy.get_time()
        time_delta = cur_time - self.last_time
        self.last_time = cur_time
        
        throttle = self.pid_controller.step(vel_error,time_delta)
        brake = 0
        
        if twist_linear_velocity == 0.0 and current_linear_velocity < 0.1:
            throttle = 0.0
            brake = 700.0
        elif throttle < 0.1 and vel_error < 0.0:
            throttle = 0.0
            decel = max(vel_error,self.decel_limit)
            brake = abs(decel)*self.veh_mass*self.wheel_rad       
        
        # Return throttle, brake, steer
        rospy.loginfo('Throttle:%f, Brake:%f, Steering: %f', throttle, brake, steering_angle)
        return throttle, brake, steering_angle
