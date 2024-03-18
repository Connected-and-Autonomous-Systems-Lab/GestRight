import sys
import time
from colorama import Fore
import numpy as np
from math import pi
import hl2ss
from pynput import keyboard
import pandas as pd

import rospy
from geometry_msgs.msg import Twist

import pickle, os


def dump_si(si):
    #global df
    out_fields["Frame"] = frame_num
    out_fields["Timestamp"] = data.timestamp
    out_fields["UTC Time"] = time.time_ns()
    # print("dumped: " + str(out_fields["Timestamp"]))
    

    dump_path = file_path + "/" + "dumps/"
    
    if not os.path.exists(dump_path):
        os.mkdir(dump_path)

    file_to_dump = open(dump_path+ "/" + str(out_fields["Timestamp"]), 'wb')
    pickle.dump(si, file_to_dump)

    file_to_dump.close()
    print("dumped: " + str(out_fields["Timestamp"]))

def save_si(si):
    global df
    out_fields["Frame"] = frame_num
    out_fields["Timestamp"] = data.timestamp
    out_fields["UTC Time"] = time.time_ns()

    if (si.is_valid_head_pose()):
        head_pose = si.get_head_pose()
        print(Fore.BLUE, "\tHead data captured")
        
        out_fields["Head Position.x"] = head_pose.position[0]
        out_fields["Head Position.y"] = head_pose.position[1]
        out_fields["Head Position.z"] = head_pose.position[2]
        
        out_fields["Head Forward.x"] = head_pose.forward[0]
        out_fields["Head Forward.y"] = head_pose.forward[1]
        out_fields["Head Forward.z"] = head_pose.forward[2]

        out_fields["Head Up.x"] = head_pose.up[0]
        out_fields["Head Up.y"] = head_pose.up[1]
        out_fields["Head Up.z"] = head_pose.up[2]

    else:
        print(Fore.BLUE, "\tNo head data")

        out_fields["Head Position.x"] = None
        out_fields["Head Position.y"] = None
        out_fields["Head Position.z"] = None
        
        out_fields["Head Forward.x"] = None
        out_fields["Head Forward.y"] = None
        out_fields["Head Forward.z"] = None

        out_fields["Head Up.x"] = None
        out_fields["Head Up.y"] = None
        out_fields["Head Up.z"] = None



    if (si.is_valid_eye_ray()):
        eye_ray = si.get_eye_ray()
        print(Fore.BLUE, f'\tEye data captured')

        out_fields["Eye Origin.x"] = eye_ray.origin[0]
        out_fields["Eye Origin.y"] = eye_ray.origin[1]
        out_fields["Eye Origin.z"] = eye_ray.origin[2]

        out_fields["Eye Direction.x"] = eye_ray.direction[0]
        out_fields["Eye Direction.y"] = eye_ray.direction[1]
        out_fields["Eye Direction.z"] = eye_ray.direction[2]

        out_fields["Eye Distance"] = eye_ray.distance

    else:
        print(Fore.BLUE, '\tNo eye tracking data')

        out_fields["Eye Origin.x"] = None
        out_fields["Eye Origin.y"] = None
        out_fields["Eye Origin.z"] = None

        out_fields["Eye Direction.x"] = None
        out_fields["Eye Direction.y"] = None
        out_fields["Eye Direction.z"] = None

    # See
    # https://learn.microsoft.com/en-us/uwp/api/windows.perception.people.jointpose?view=winrt-22621
    # for hand data details

    if (si.is_valid_hand_left()):
        hand_left = si.get_hand_left()

        print(Fore.BLUE, "\tLeft hand data captured")
        
        write_joints(out_fields, "Left", "Palm" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.Palm))
        write_joints(out_fields, "Left", "Wrist" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.Wrist))
        write_joints(out_fields, "Left", "ThumbMetacarpal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbMetacarpal))
        write_joints(out_fields, "Left", "ThumbProximal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbProximal))
        write_joints(out_fields, "Left", "ThumbDistal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbDistal))
        write_joints(out_fields, "Left", "ThumbTip" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip))
        write_joints(out_fields, "Left", "IndexMetacarpal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal))
        write_joints(out_fields, "Left", "IndexProximal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal))
        write_joints(out_fields, "Left", "IndexIntermediate" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate))
        write_joints(out_fields, "Left", "IndexDistal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal))
        write_joints(out_fields, "Left", "IndexTip" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip))
        write_joints(out_fields, "Left", "MiddleMetacarpal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleMetacarpal))
        write_joints(out_fields, "Left", "MiddleProximal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleProximal))
        write_joints(out_fields, "Left", "MiddleIntermediate" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleIntermediate))
        write_joints(out_fields, "Left", "MiddleDistal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleDistal))
        write_joints(out_fields, "Left", "MiddleTip" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleTip))
        write_joints(out_fields, "Left", "RingMetacarpal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.RingMetacarpal))
        write_joints(out_fields, "Left", "RingProximal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.RingProximal))
        write_joints(out_fields, "Left", "RingIntermediate" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.RingIntermediate))
        write_joints(out_fields, "Left", "RingDistal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.RingDistal))
        write_joints(out_fields, "Left", "RingTip" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.RingTip))
        write_joints(out_fields, "Left", "LittleMetacarpal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleMetacarpal))
        write_joints(out_fields, "Left", "LittleProximal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleProximal))
        write_joints(out_fields, "Left", "LittleIntermediate" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleIntermediate))
        write_joints(out_fields, "Left", "LittleDistal" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleDistal))
        write_joints(out_fields, "Left", "LittleTip" , hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleTip))

    else:
        print(Fore.BLUE, '\tNo left hand data')

        for joint in joint_fields:
            write_null_joints(out_fields, chirality="Left", joint_str=joint)

    
    if (si.is_valid_hand_right()):
        hand_right = si.get_hand_right()

        pose = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist)
        print(Fore.BLUE, "\tRight hand data captured")
        
        write_joints(out_fields, "Right", "Palm" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Palm))
        write_joints(out_fields, "Right", "Wrist" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.Wrist))
        write_joints(out_fields, "Right", "ThumbMetacarpal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbMetacarpal))
        write_joints(out_fields, "Right", "ThumbProximal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbProximal))
        write_joints(out_fields, "Right", "ThumbDistal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbDistal))
        write_joints(out_fields, "Right", "ThumbTip" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip))
        write_joints(out_fields, "Right", "IndexMetacarpal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal))
        write_joints(out_fields, "Right", "IndexProximal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal))
        write_joints(out_fields, "Right", "IndexIntermediate" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexIntermediate))
        write_joints(out_fields, "Right", "IndexDistal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal))
        write_joints(out_fields, "Right", "IndexTip" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip))
        write_joints(out_fields, "Right", "MiddleMetacarpal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleMetacarpal))
        write_joints(out_fields, "Right", "MiddleProximal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleProximal))
        write_joints(out_fields, "Right", "MiddleIntermediate" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleIntermediate))
        write_joints(out_fields, "Right", "MiddleDistal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleDistal))
        write_joints(out_fields, "Right", "MiddleTip" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleTip))
        write_joints(out_fields, "Right", "RingMetacarpal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingMetacarpal))
        write_joints(out_fields, "Right", "RingProximal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingProximal))
        write_joints(out_fields, "Right", "RingIntermediate" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingIntermediate))
        write_joints(out_fields, "Right", "RingDistal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingDistal))
        write_joints(out_fields, "Right", "RingTip" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingTip))
        write_joints(out_fields, "Right", "LittleMetacarpal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleMetacarpal))
        write_joints(out_fields, "Right", "LittleProximal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleProximal))
        write_joints(out_fields, "Right", "LittleIntermediate" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleIntermediate))
        write_joints(out_fields, "Right", "LittleDistal" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleDistal))
        write_joints(out_fields, "Right", "LittleTip" , hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleTip))

    else:
        print(Fore.BLUE, '\tNo right hand data')

        for joint in joint_fields:
            write_null_joints(out_fields, chirality="Right", joint_str=joint)

def write_null_joints(out_fields, chirality, joint_str):
    out_fields[f"{chirality} {joint_str} Positionx"] = None
    out_fields[f"{chirality} {joint_str} Positiony"] = None
    out_fields[f"{chirality} {joint_str} Positionz"] = None

    out_fields[f"{chirality} {joint_str} Orientation.x"] = None
    out_fields[f"{chirality} {joint_str} Orientation.y"] = None
    out_fields[f"{chirality} {joint_str} Orientation.z"] = None
    out_fields[f"{chirality} {joint_str} Orientation.w"] = None

    out_fields[f"{chirality} {joint_str} Radius"] = None
    out_fields[f"{chirality} {joint_str} Accuracy"] = None

def write_joints(out_fields, chirality, joint_str, joint_data):
    out_fields[f"{chirality} {joint_str} Positionx"] = joint_data.position[0]
    out_fields[f"{chirality} {joint_str} Positiony"] = joint_data.position[1]
    out_fields[f"{chirality} {joint_str} Positionz"] = joint_data.position[2]

    out_fields[f"{chirality} {joint_str} Orientation.x"] = joint_data.orientation[0]
    out_fields[f"{chirality} {joint_str} Orientation.y"] = joint_data.orientation[1]
    out_fields[f"{chirality} {joint_str} Orientation.z"] = joint_data.orientation[2]
    out_fields[f"{chirality} {joint_str} Orientation.w"] = joint_data.orientation[3]

    out_fields[f"{chirality} {joint_str} Radius"] = joint_data.radius
    out_fields[f"{chirality} {joint_str} Accuracy"] = joint_data.accuracy


def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

def closing(e):
    print(Fore.BLUE, "Saving SI...")
    df.to_csv(f"{file_path}/hand_info.csv")
    print(Fore.BLUE, "SI Saved.")

    print(e)
    listener.join()
    client.close()
    hl2ss.stop_subsystem_pv(host, port)
    quit()

def is_open(index, tolerance=0.1):
    point_dist = np.linalg.norm(index[0] - index[3]) 
    
    summed_dist = 0
    for i in range(len(index)-1):
        summed_dist += np.linalg.norm(index[i] - index[i+1])

    return ((point_dist * (1+tolerance) > summed_dist) and (point_dist * (1-tolerance) < summed_dist))

def is_touching(toucher, touchee, proportion=.15, scale = 1.5):
    if (toucher is None or touchee is None): return False
    dist = toucher - touchee
    return np.linalg.norm(dist[:3]) < (proportion * scale)

def proj_xz(v):
    projv = np.array([v[0], v[2]])
    projv = projv/np.linalg.norm(projv)

    return projv

def angle_between(v1, v2):
    projv1 = proj_xz(v1)
    projv2 = proj_xz(v2)

    angle = np.arccos(np.dot(projv1, projv2))
    if np.cross(projv1, projv2) < 0:
        angle *= -1
    return angle


class movement :
    DEGREE_TO_RADIAN = 0.0174532925
    def __init__(self):
        rospy.init_node('move_robot_node', anonymous=False)
        self.pub_move = rospy.Publisher("/spot/cmd_vel",Twist,queue_size=1)
        self.move = Twist()

    def move_forward(self):        
        self.move.linear.x=2
        self.move.angular.z=0.0
        self.pub_move.publish(self.move)

    def move_backward(self):      
        self.move.linear.x=-2
        self.move.angular.z=0.0
        self.pub_move.publish(self.move)

    def strafe_right(self):        
        self.move.linear.y=2
        self.move.angular.z=0.0
        self.pub_move.publish(self.move)

    def strafe_left(self):        
        self.move.linear.y=-2
        self.move.angular.z=0.0
        self.pub_move.publish(self.move)

    def turn_left(self):
        self.move.angular.z = 50 * self.DEGREE_TO_RADIAN
        self.pub_move.publish(self.move)

    def turn_right(self):
        self.move.angular.z = -50 * self.DEGREE_TO_RADIAN
        self.pub_move.publish(self.move)

    def stop(self):
        self.stop_move()
        self.stop_strafe()
        self.stop_rotate()
        
    def stop_move(self):
        self.move.linear.x=0
        self.pub_move.publish(self.move)

    def stop_strafe(self):
        self.move.linear.y=0
        self.pub_move.publish(self.move)

    def stop_rotate(self):
        self.move.angular.x = 0
        self.move.angular.y = 0
        self.move.angular.z = 0
        self.pub_move.publish(self.move)

class fist_based():
    def __init__(this):
        this.r_index = None
        this.l_index = None

    def update_info(this, si) -> None:
        if (si.is_valid_hand_right()):
            this.l_index = None

            hand_right = si.get_hand_right()
            this.r_index = [hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position,
                            hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position,
                            hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal).position,
                            hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal).position]
            
        else: 
            this.r_index = None
            
            if (si.is_valid_hand_left()):
                hand_left = si.get_hand_left()
                this.l_index = [hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position,
                                hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position,
                                hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexProximal).position,
                                hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexMetacarpal).position]
            else:
                this.l_index = None

    
    def handle_skips(this, si) -> bool:
        return False
    
    def handle_linear(this) -> str:
        if (this.r_index is not None) and (is_open(this.r_index)):
            return "forward"
        if (this.l_index is not None) and (is_open(this.l_index)):
            return "backward"
        return "no-op"
        
    def handle_strafe(this) -> str:
        return "no-op"

    def handle_turn(this) -> str:
        if (this.r_index is not None) and (not is_open(this.r_index)):
            return "right"
        if (this.l_index is not None) and (not is_open(this.l_index)):
            return "left"
        return "no-op"

class touch_based():
    def update_info(this, si) -> None:
        if (si.is_valid_hand_right()):
            hand_right              = si.get_hand_right()
            this.r_index            = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
            this.r_index_knuckle    = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position
            this.r_middle           = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.MiddleTip).position
            this.r_ring             = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.RingTip).position
            this.r_thumb            = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position
            this.r_pinky            = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.LittleTip).position

            this.r_knuckle_len      = np.linalg.norm(this.r_index - this.r_index_knuckle)
        else:
            this.r_index            = None
            this.r_index_knuckle    = None
            this.r_middle           = None
            this.r_ring             = None
            this.r_thumb            = None
            this.r_pinky            = None

            this.r_knuckle_len      = 0


        if (si.is_valid_hand_left()):
            hand_left               = si.get_hand_left()
            this.l_index            = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
            this.l_index_knuckle    = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position
            this.l_thumb            = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position
            this.l_pinky            = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleTip).position

            this.l_knuckle_len      = np.linalg.norm(this.l_index - this.l_index_knuckle)
        else:
            this.l_index            = None
            this.l_index_knuckle    = None
            this.l_thumb            = None
            this.l_pinky            = None

            this.l_knuckle_len      = 0


    def handle_skips(this, si) -> bool:
        return False


    def handle_linear(this) -> str:
        if is_touching(this.r_index, this.r_thumb, this.r_knuckle_len):
            return "forward"
        if is_touching(this.r_pinky, this.r_thumb, this.r_knuckle_len, scale=2):
            return "backward"
        return "no-op"
    
    def handle_strafe(this) -> str:
        if is_touching(this.r_middle, this.r_thumb, this.r_knuckle_len, scale=2):
            return "left"
        if is_touching(this.r_ring, this.r_thumb, this.r_knuckle_len, scale=2):
            return "right"
        return "no-op"
    
    def handle_turn(this) -> str:
        if is_touching(this.l_pinky, this.l_thumb, this.l_knuckle_len, scale=2):
            return "left"
        if is_touching(this.l_index, this.l_thumb, this.l_knuckle_len):
            return "right"
        return "no-op"
    
class wheel_based():
    def __init__(this):
        this.linear_neutral_y = None
        this.head_pos =  None
        this.head_fwd =  None
        this.r_hand =    None
        this.r_index =   None
        this.r_thumb =   None
        this.l_hand =    None
        this.l_index =   None
        this.l_thumb =   None
        this.r_knuckle_len = 0


    def update_info(this, si) -> bool:
        if (si.is_valid_head_pose()): 
            this.head_pos = si.get_head_pose().position
            this.head_fwd = si.get_head_pose().forward

        if (si.is_valid_hand_right()): 
            this.r_hand     = si.get_hand_right()
            this.r_index    = this.r_hand.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
            this.r_thumb    = this.r_hand.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position

            this.r_index_knuckle    = this.r_hand.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position
            this.r_knuckle_len      = np.linalg.norm(this.r_index - this.r_index_knuckle)
            
        if (si.is_valid_hand_left()): 
            this.l_hand     = si.get_hand_left()
            this.l_index    = this.l_hand.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
            this.l_thumb    = this.l_hand.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position
        
        #crunch numbers

        if ((this.head_pos is None) or (this.r_hand is None) or (this.l_hand is None)): return

        this.lr_center = ((this.r_index + this.l_index) / 2) - this.head_pos
        this.fwrd_to_north_angle = angle_between(np.array([-1,0,0]), this.head_fwd)
        this.hand_to_north_angle = angle_between(np.array([-1,0,0]), this.lr_center)
        this.strafe_angle = this.hand_to_north_angle - this.fwrd_to_north_angle

        if this.linear_neutral_y is None: 
            print(Fore.BLUE, "Enter neutral position...")
            if is_touching(this.r_index, this.r_thumb, this.r_knuckle_len): 
                this.linear_neutral_y = this.lr_center[1]

    def handle_skips(this, si) -> bool:
        if ((this.head_pos is None) or (this.r_hand is None) or (this.l_hand is None)): 
            return True

        if ((not si.is_valid_hand_right()) and (not si.is_valid_hand_left())): 
            return True
        
        if this.linear_neutral_y is None:
            return True

        return False


    def handle_linear(this, tolerance=0.10) -> str:
        y_delta = this.lr_center[1] - this.linear_neutral_y
        if abs(y_delta) < tolerance:
            return "no-op"
        elif y_delta < 0:
            return "backward"
        else:
            return "forward"

    def handle_strafe(this, tolerance=0.35) -> str:
        #tolerance is in radians, default is about 20 degreees
        print(this.strafe_angle)
        if abs(this.strafe_angle) < tolerance:
            return "no-op"
        elif this.strafe_angle < 0:
            return "left"
        else:
            return "right"
        
    def handle_turn(this, tolerance=0.05) -> str:
        #tolerance is in meters, how much higher one hand has to be
        delta_y = this.l_index[1] - this.r_index[1]
        
        if abs(delta_y) < tolerance:
            return "no-op"
        elif delta_y < 0:
            return "left"
        else:
            return "right"
        
class nothing_based():
    def update_info(this, si):
        pass
    def handle_linear(this, si):
        return "no-op"
    def handle_strafe(this, si):
        return "no-op"
    def handle_turn(this, si):
        return "no-op"
    def handle_skips(this, si):
        return False

if __name__ == "__main__":
    global enable

    # HoloLens address
    with open("hl_address.txt", "r") as hl_addr:
        host = hl_addr.read().split()[0]

    # Port
    port = hl2ss.StreamPort.SPATIAL_INPUT

    file_path = sys.argv[1:][0]

    #------------------------------------------------------------------------------

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    client = hl2ss.rx_si(host, port, hl2ss.ChunkSize.SPATIAL_INPUT)
    print(Fore.BLUE, "Connecting...")
    client.open()
    print(Fore.BLUE, "Connected")
    enable = True


    head_eye_fields = ["Head Position", "Head Forward", "Head Up",
                    "Eye Origin",    "Eye Direction"]

    joint_fields = ["Palm", "Wrist",
                    "ThumbMetacarpal",  "ThumbProximal",    "ThumbDistal",          "ThumbTip",
                    "IndexMetacarpal",  "IndexProximal",    "IndexIntermediate",    "IndexDistal",  "IndexTip",
                    "MiddleMetacarpal", "MiddleProximal",   "MiddleIntermediate",   "MiddleDistal", "MiddleTip",
                    "RingMetacarpal",   "RingProximal",     "RingIntermediate",     "RingDistal",   "RingTip",
                    "LittleMetacarpal", "LittleProximal",   "LittleIntermediate",   "LittleDistal", "LittleTip"]

    out_fields = {"Frame":None, "Timestamp":None, "UTC Time":None, 
                  "Scheme":None, "Linear": None, "Strafe": None, "Turn": None}

    for field in head_eye_fields:
        out_fields[f"{field}.x"] = None
        out_fields[f"{field}.y"] = None
        out_fields[f"{field}.z"] = None
    out_fields["Eye Distance"] = None    

    for joint in joint_fields:
        out_fields[f"Left {joint} Positionx"] = None
        out_fields[f"Left {joint} Positionx"] = None
        out_fields[f"Left {joint} Positionx"] = None

        out_fields[f"Left {joint} Orientation.x"] = None
        out_fields[f"Left {joint} Orientation.y"] = None
        out_fields[f"Left {joint} Orientation.z"] = None
        out_fields[f"Left {joint} Orientation.w"] = None

        out_fields[f"Left {joint} Radius"] = None
        out_fields[f"Left {joint} Accuracy"] = None

    for joint in joint_fields:
        out_fields[f"Right {joint} Positionx"] = None
        out_fields[f"Right {joint} Positionx"] = None
        out_fields[f"Right {joint} Positionx"] = None

        out_fields[f"Right {joint} Orientation.x"] = None
        out_fields[f"Right {joint} Orientation.y"] = None
        out_fields[f"Right {joint} Orientation.z"] = None
        out_fields[f"Right {joint} Orientation.w"] = None

        out_fields[f"Right {joint} Radius"] = None
        out_fields[f"Right {joint} Accuracy"] = None

    df = pd.DataFrame()

    mov = movement()
    rate = rospy.Rate(10)
    counter = 0
    r_knuckle_len = 0
    l_knuckle_len = 0

    touch = touch_based()
    wheel = wheel_based()
    fist = fist_based()
    nothing = nothing_based()

    str_scheme = sys.argv[1:][1]
    out_fields["Scheme"] = str_scheme

    if str_scheme == "wheel":
        scheme = wheel
    elif str_scheme == "touch":
        scheme = touch
    elif str_scheme == "fist":
        scheme = fist
    else:
        scheme = nothing


    frames_since_pause = 0
    WAIT_BETWEEN_PAUSE = 20
    is_paused = False
    frame_num = -1

    new_out_fields = {"Schme":0, "Frame":0, "Timestamp":0, "UTC Time": 0, 
                      "Linear":0, "Strafe":0, "Turn":0}

    try:
        while (not rospy.is_shutdown()) and enable:
            data = client.get_next_packet()
            frame_num += 1
            si = hl2ss.unpack_si(data.payload)

            print(Fore.BLUE, f'Frame {frame_num} of SI at timestamp: {data.timestamp}')
            new_out_fields["Scheme"] = str_scheme
            new_out_fields["Frame"] = frame_num
            new_out_fields["Timestamp"] = data.timestamp
            new_out_fields["UTC Time"] = time.time_ns()

            #save_si(si)
            if si is None:
                pass
            else:
                dump_si(si)
            
            """
            if (si.is_valid_hand_right()):
                hand_right          = si.get_hand_right()
                r_index             = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
                r_index_knuckle     = hand_right.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position
                
                r_knuckle_len       = np.linalg.norm(r_index - r_index_knuckle)
            else:
                r_index             = None
                r_index_knuckle     = None
                
                r_knuckle_len       = 0

            if (si.is_valid_hand_left()):
                hand_left           = si.get_hand_left()
                l_index             = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexTip).position
                l_index_knuckle     = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.IndexDistal).position
                l_pinky             = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.LittleTip).position
                l_thumb             = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.ThumbTip).position
                l_middle            = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.MiddleTip).position
            else:
                l_index             = None
                l_pinky             = None
                l_thumb             = None
                l_middle            = None

                
            #check for if pause was hit
            frames_since_pause += 1
            if (is_touching(r_index, l_thumb, r_knuckle_len)) and (frames_since_pause > WAIT_BETWEEN_PAUSE): 
                print(Fore.BLUE, "pausing...")
                is_paused = not is_paused
                frames_since_pause = 0

            if is_paused:
                print(Fore.BLUE, "Paused...")
                if (is_touching(r_index, l_index, r_knuckle_len)) and (frames_since_pause > WAIT_BETWEEN_PAUSE): 
                    print(Fore.BLUE, "_________________________________")
                    print(Fore.BLUE, "TOUCH BASED CONTROL SCHEME ACTIVE")
                    print(Fore.BLUE, "_________________________________")
                    
                    scheme = touch
                    is_paused = not is_paused
                    frames_since_pause = 0
                if (is_touching(r_index, l_pinky, r_knuckle_len)) and (frames_since_pause > WAIT_BETWEEN_PAUSE): 
                    print(Fore.BLUE, "_________________________________")
                    print(Fore.BLUE, "WHEEL BASED CONTROL SCHEME ACTIVE")
                    print(Fore.BLUE, "_________________________________")

                    scheme = wheel
                    is_paused = not is_paused
                    frames_since_pause = 0
                if (is_touching(r_index, l_middle, r_knuckle_len)) and (frames_since_pause > WAIT_BETWEEN_PAUSE): 
                    print(Fore.BLUE, "________________________________")
                    print(Fore.BLUE, "FIST BASED CONTROL SCHEME ACTIVE")
                    print(Fore.BLUE, "________________________________")

                    scheme = fist
                    is_paused = not is_paused
                    frames_since_pause = 0
                continue
            """
            scheme.update_info(si)

            if scheme.handle_skips(si):
                new_out_fields["Linear"] = "no-op"
                new_out_fields["Strafe"] = "no-op"
                new_out_fields["Turn"] = "no-op"

                df = pd.concat([df, pd.DataFrame(new_out_fields, index=[0])], ignore_index=True)
                continue

            #Uses control scheme specific methods of determining commands
            linear_result = scheme.handle_linear()
            strafe_result = scheme.handle_strafe()
            turn_result = scheme.handle_turn()

            new_out_fields["Linear"] = linear_result
            new_out_fields["Strafe"] = strafe_result
            new_out_fields["Turn"] = turn_result

#Handles Linear (Forward/Backward) 
            if linear_result == "forward":
                print(Fore.BLUE, "Forward", end=" | ")
                mov.move_forward()
            elif linear_result == "backward":
                print(Fore.BLUE, "Backwards", end=" | ")
                mov.move_backward()
            else: 
                mov.stop_move()

            #Handles Strafe (Left/Right) 
            if strafe_result == "left":
                print(Fore.BLUE, "Strafe Left", end=" | ")
                mov.strafe_left()
            elif strafe_result == "right":
                print(Fore.BLUE, "Strafe Right", end=" | ")                    
                mov.strafe_right()
            else: 
                mov.stop_strafe()
                
            #Handles Rotate (Left/Right)
            if turn_result == "left":
                print(Fore.BLUE, "Left", end="")
                mov.turn_left()
            elif turn_result == "right":
                print(Fore.BLUE, "Right", end="")
                mov.turn_right()
            else:
                mov.stop_rotate()

            print()
            
            df = pd.concat([df, pd.DataFrame(new_out_fields, index=[0])], ignore_index=True)

    except Exception as e:
        closing(e)       
    except KeyboardInterrupt as e:
        closing(e)
    except:
        closing("Unknown Error")
    else:
        closing("Exit 0")



