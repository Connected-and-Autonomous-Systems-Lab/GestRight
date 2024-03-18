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
mode = hl2ss.StreamMode.MODE_2

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

print(Fore.GREEN, "Starting...")
hl2ss.start_subsystem_pv(host, port)
print(Fore.GREEN, "Started")

file_path = sys.argv[1:][0]

out_fields = {"UTC Time": None,
	      	  "Radial Distortion 0": None, "Radial Distortion 1": None, "Radial Distortion 2": None,
			  "Tangential Distortion 0": None, "Tangential Distortion 1": None,
			  "Focal Length 0": None, "Focal Length 1": None,
			  "Principle Point 0": None, "Principle Point 1": None,
	      	  "Projection 0,0": None, "Projection 0,1": None, "Projection 0,2": None, "Projection 0,3": None, 
              "Projection 1,0": None, "Projection 1,1": None, "Projection 1,2": None, "Projection 1,3": None, 
              "Projection 2,0": None, "Projection 2,1": None, "Projection 2,2": None, "Projection 2,3": None, 
              "Intrinsics 0,0": None, "Intrinsics 0,1": None, "Intrinsics 0,2": None, "Intrinsics 0,3": None, 
              "Intrinsics 1,0": None, "Intrinsics 1,1": None, "Intrinsics 1,2": None, "Intrinsics 1,3": None, 
              "Intrinsics 2,0": None, "Intrinsics 2,1": None, "Intrinsics 2,2": None, "Intrinsics 2,3": None}

print(Fore.GREEN, "Downloading...")
data = hl2ss.download_calibration_pv(host, port, width, height, framerate)
print(Fore.GREEN, "Downloaded...")


out_fields["UTC Time"] = time.time_ns()

out_fields["Radial Distortion 0"] = data.radial_distortion[0]
out_fields["Radial Distortion 1"] = data.radial_distortion[1]
out_fields["Radial Distortion 2"] = data.radial_distortion[2]

out_fields["Tangential Distortion 0"] = data.tangential_distortion[0]
out_fields["Tangential Distortion 1"] = data.tangential_distortion[1]

out_fields["Focal Length 0"] = data.focal_length[0]
out_fields["Focal Length 1"] = data.focal_length[1]

out_fields["Principle Point 0"] = data.principal_point[0]
out_fields["Principle Point 1"] = data.principal_point[1]


out_fields["Projection 0,0"] = data.projection[0][0]
out_fields["Projection 0,1"] = data.projection[0][1]
out_fields["Projection 0,2"] = data.projection[0][2]
out_fields["Projection 0,3"] = data.projection[0][3]
out_fields["Projection 1,0"] = data.projection[1][0]
out_fields["Projection 1,1"] = data.projection[1][1]
out_fields["Projection 1,2"] = data.projection[1][2]
out_fields["Projection 1,3"] = data.projection[1][3]
out_fields["Projection 2,0"] = data.projection[2][0]
out_fields["Projection 2,1"] = data.projection[2][1]
out_fields["Projection 2,2"] = data.projection[2][2]
out_fields["Projection 2,3"] = data.projection[2][3]

out_fields["Intrinsics 0,0"] = data.intrinsics[0][0]
out_fields["Intrinsics 0,1"] = data.intrinsics[0][1]
out_fields["Intrinsics 0,2"] = data.intrinsics[0][2]
out_fields["Intrinsics 0,3"] = data.intrinsics[0][3]
out_fields["Intrinsics 1,0"] = data.intrinsics[1][0]
out_fields["Intrinsics 1,1"] = data.intrinsics[1][1]
out_fields["Intrinsics 1,2"] = data.intrinsics[1][2]
out_fields["Intrinsics 1,3"] = data.intrinsics[1][3]
out_fields["Intrinsics 2,0"] = data.intrinsics[2][0]
out_fields["Intrinsics 2,1"] = data.intrinsics[2][1]
out_fields["Intrinsics 2,2"] = data.intrinsics[2][2]
out_fields["Intrinsics 2,3"] = data.intrinsics[2][3]

df = pd.Series(out_fields)

print(Fore.GREEN, "Saving Calibration PV...")
df.to_csv(f"{file_path}/pv_calibration.csv")
print(Fore.GREEN, "PV Calibration Saved.")

hl2ss.stop_subsystem_pv(host, port)
