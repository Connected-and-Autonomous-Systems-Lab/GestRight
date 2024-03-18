#------------------------------------------------------------------------------
# This script receives spatial input data from the HoloLens, which comprises:
# 1) Head pose, 2) Eye ray, 3) Hand tracking, and prints it. 30 Hz sample rate.
# Press esc to stop.
#------------------------------------------------------------------------------

import sys
import time
from pynput import keyboard
import hl2ss_utilities
import os
from datetime import datetime, timedelta, timezone
import pandas as pd
from colorama import Fore

import hl2ss

# Settings --------------------------------------------------------------------

# HoloLens address
with open("hl_address.txt", "r") as hl_addr:
	host = hl_addr.read().split("\n")[0]

# Port
port = hl2ss.StreamPort.SPATIAL_INPUT

#------------------------------------------------------------------------------


file_path = sys.argv[1:][0]

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

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


listener = keyboard.Listener(on_press=on_press)
listener.start()

print(Fore.BLUE, "Connecting")
client = hl2ss.rx_si(host, port, hl2ss.ChunkSize.SPATIAL_INPUT)
client.open()
print(Fore.BLUE, "Connected")


head_eye_fields = ["Head Position", "Head Forward", "Head Up",
                   "Eye Origin",    "Eye Direction"]

joint_fields = ["Palm", "Wrist",
                "ThumbMetacarpal",  "ThumbProximal",    "ThumbDistal",          "ThumbTip",
                "IndexMetacarpal",  "IndexProximal",    "IndexIntermediate",    "IndexDistal",  "IndexTip",
                "MiddleMetacarpal", "MiddleProximal",   "MiddleIntermediate",   "MiddleDistal", "MiddleTip",
                "RingMetacarpal",   "RingProximal",     "RingIntermediate",     "RingDistal",   "RingTip",
                "LittleMetacarpal", "LittleProximal",   "LittleIntermediate",   "LittleDistal", "LittleTip"]

out_fields = {"Frame":None, "Timestamp":None, "UTC Time":None}

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

frame_num = -1
while (enable):
    data = client.get_next_packet()
    si = hl2ss.unpack_si(data.payload)

    frame_num += 1

    print(Fore.BLUE, f'Frame {frame_num} of SI at timestamp: {data.timestamp}')

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

        pose = hand_left.get_joint_pose(hl2ss.SI_HandJointKind.Wrist)
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
 
 
 
    df = pd.concat([df, pd.DataFrame(out_fields, index=[0])], ignore_index=True)


print(Fore.BLUE, "Saving SI...")
df.to_csv(f"{file_path}/hand_info.csv")
print(Fore.BLUE, "SI Saved.")

client.close()
listener.join()
