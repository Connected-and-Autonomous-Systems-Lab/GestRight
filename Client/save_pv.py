#------------------------------------------------------------------------------
# This script receives video from the HoloLens front RGB camera and plays it.
# The camera supports various resolutions and framerates. See
# https://github.com/jdibenes/hl2ss/blob/main/etc/pv_configurations.txt
# for a list of supported formats. The default configuration is 1080p 30 FPS. 
# The stream supports three operating modes: 0) video, 1) video + camera pose, 
# 2) query calibration (single transfer).
# Press esc to stop.
#------------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
import sys
import time
from pynput import keyboard
import os
import pandas as pd
from colorama import Fore

import cv2
import hl2ss_imshow
import hl2ss

# Settings --------------------------------------------------------------------

# HoloLens address
with open("hl_address.txt", "r") as hl_addr:
	host = hl_addr.read().split("\n")[0]

# Port
port = hl2ss.StreamPort.PERSONAL_VIDEO

# Operating mode
# 0: video
# 1: video + camera pose
# 2: query calibration (single transfer)
mode = hl2ss.StreamMode.MODE_1

# Camera parameters
width     = 1920
height    = 1080
framerate = 30

# Video encoding profile
profile = hl2ss.VideoProfile.H265_MAIN

# Encoded stream average bits per second
# Must be > 0
bitrate = hl2ss.get_video_codec_bitrate(width, height, framerate, hl2ss.get_video_codec_default_factor(profile))

# Decoded format
# Options include:
# 'bgr24'
# 'rgb24'
# 'bgra'
# 'rgba'
# 'gray8'
decoded_format = 'bgr24'

#------------------------------------------------------------------------------

hl2ss.start_subsystem_pv(host, port)


file_path = sys.argv[1:][0]
os.makedirs(f"{file_path}/images")

enable = True

def on_press(key):
    global enable
    enable = key != keyboard.Key.esc
    return enable

listener = keyboard.Listener(on_press=on_press)
listener.start()

print(Fore.GREEN, "Connecting")
client = hl2ss.rx_decoded_pv(host, port, hl2ss.ChunkSize.PERSONAL_VIDEO, mode, width, height, framerate, profile, bitrate, decoded_format)
client.open()
print(Fore.GREEN, "Connected")

out_fields = {"Frame": None, "Timestamp": None, "UTC Time": None, 
                "Pose 0,0": None, "Pose 0,1": None, "Pose 0,2": None, "Pose 0,3": None, 
                "Pose 1,0": None, "Pose 1,1": None, "Pose 1,2": None, "Pose 1,3": None, 
                "Pose 2,0": None, "Pose 2,1": None, "Pose 2,2": None, "Pose 2,3": None, 
                "Focal Length 0": None, "Focal Length 1": None,
                "Principle Point 0": None, "Principle Point 1": None}

df = pd.DataFrame()

frame_num = -1
skip_frame = 30
while (enable):
    data = client.get_next_packet()
    frame_num += 1

    print(Fore.GREEN, f'Frame {frame_num} of PV at timestamp: {data.timestamp}')

    out_fields["Frame"] = frame_num
    out_fields["Timestamp"] = data.timestamp
    out_fields["UTC Time"] = time.time_ns()

    try:
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
    except:
        out_fields["Pose 0,0"] = None
        out_fields["Pose 0,1"] = None
        out_fields["Pose 0,2"] = None
        out_fields["Pose 0,3"] = None
        out_fields["Pose 1,0"] = None
        out_fields["Pose 1,1"] = None
        out_fields["Pose 1,2"] = None
        out_fields["Pose 1,3"] = None
        out_fields["Pose 2,0"] = None
        out_fields["Pose 2,1"] = None
        out_fields["Pose 2,2"] = None
        out_fields["Pose 2,3"] = None

    try:     
        out_fields["Focal Length 0"] = data.payload.focal_length[0]
        out_fields["Focal Length 1"] = data.payload.focal_length[1]
    except:
        out_fields["Focal Length 0"] = None
        out_fields["Focal Length 1"] = None

    try:     
        out_fields["Principle Point 0"] = data.payload.principal_point[0]
        out_fields["Principle Point 1"] = data.payload.principal_point[1]
    except:
        out_fields["Principle Point 0"] = None
        out_fields["Principle Point 1"] = None


    df = pd.concat([df, pd.DataFrame([out_fields])], ignore_index=True)

    try:
        cv2.imwrite(f"{file_path}/images/{frame_num}.jpg", data.payload.image)
        cv2.waitKey(1)
    except:
        pass


print(Fore.GREEN, "Saving PV...")
df.to_csv(f"{file_path}/pv_info.csv")
print(Fore.GREEN, "PV Saved.")

client.close()
listener.join()

hl2ss.stop_subsystem_pv(host, port)

