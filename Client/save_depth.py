#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in long throw mode
# and plays it. The resolution is 320x288 @ 5 FPS. The stream supports three
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop. 
#------------------------------------------------------------------------------

import sys
from pynput import keyboard
from datetime import datetime, timedelta, timezone
import time
import os
import pandas as pd
from colorama import Fore


import numpy as np
import cv2
import hl2ss_imshow
import hl2ss

# Settings --------------------------------------------------------------------

# HoloLens address
with open("hl_address.txt", "r") as hl_addr:
	host = hl_addr.read().split("\n")[0]

# Port
port = hl2ss.StreamPort.RM_DEPTH_LONGTHROW

# Operating mode
# 0: video
# 1: video + rig pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# PNG filter
png_filter = hl2ss.PngFilterMode.Paeth

#------------------------------------------------------------------------------


file_path = sys.argv[1:][0]
print(file_path)
os.makedirs(f"{file_path}/depth")
os.makedirs(f"{file_path}/ab")


enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

print(Fore.RED, "Connecting...")
client = hl2ss.rx_decoded_rm_depth_longthrow(host, port, hl2ss.ChunkSize.RM_DEPTH_LONGTHROW, mode, png_filter)
client.open()
print(Fore.RED, "Connected")

out_fields = {"Frame": None, "Timestamp": None, "UTC Time": None, 
                "Pose 0,0": None, "Pose 0,1": None, "Pose 0,2": None, "Pose 0,3": None, 
                "Pose 1,0": None, "Pose 1,1": None, "Pose 1,2": None, "Pose 1,3": None, 
                "Pose 2,0": None, "Pose 2,1": None, "Pose 2,2": None, "Pose 2,3": None}

df = pd.DataFrame()

frame_num = -1
while (enable):
    data = client.get_next_packet()
    frame_num += 1

    print(Fore.RED, f'Frame {frame_num} of Longthrow at timestamp: {data.timestamp}')

    out_fields["Frame"] = frame_num
    out_fields["Timestamp"] = data.timestamp
    out_fields["UTC Time"] = time.time_ns()

    out_fields["Pose 0,0"] = data.pose[0][0]
    out_fields["Pose 0,1"] = data.pose[0][1]
    out_fields["Pose 0,2"] = data.pose[0][2]
    out_fields["Pose 0,3"] = data.pose[0][3]
    out_fields["Pose 1,0"] = data.pose[1][0]
    out_fields["Pose 1,1"] = data.pose[1][1]
    out_fields["Pose 1,2"] = data.pose[1][2]
    out_fields["Pose 1,3"] = data.pose[1][3]
    out_fields["Pose 2,0"] = data.pose[2][0]
    out_fields["Pose 2,1"] = data.pose[2][1]
    out_fields["Pose 2,2"] = data.pose[2][2]
    out_fields["Pose 2,3"] = data.pose[2][3]

    df = pd.concat([df, pd.DataFrame([out_fields])], ignore_index=True)

    #cv2.imshow('Depth', data.payload.depth / np.max(data.payload.depth)) # Scaled for visibility
    #cv2.imshow('AB', data.payload.ab / np.max(data.payload.ab)) # Scaled for visibility

    #NOT SCALED
    cv2.imwrite(f"{file_path}/depth/{frame_num}.png", data.payload.depth)
    cv2.imwrite(f"{file_path}/ab/{frame_num}.png", data.payload.ab)

    cv2.waitKey(1)

print(Fore.RED, "Saving Longthrow...")
df.to_csv(f"{file_path}/depth_info.csv")
print(Fore.RED, "Longthrow Saved.")

client.close()
listener.join()
