#------------------------------------------------------------------------------
# This script receives video from the HoloLens depth camera in long throw mode
# and plays it. The resolution is 320x288 @ 5 FPS. The stream supports three
# operating modes: 0) video, 1) video + rig pose, 2) query calibration (single 
# transfer). Depth and AB data are scaled for visibility. The ahat and long 
# throw streams cannot be used simultaneously.
# Press esc to stop. 
#------------------------------------------------------------------------------

from datetime import datetime, timedelta, timezone
import sys
import time
import os
import pandas as pd
from colorama import Fore
import numpy as np
import cv2

from pynput import keyboard
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
mode = hl2ss.StreamMode.MODE_2

# PNG filter
png_filter = hl2ss.PngFilterMode.Paeth

#------------------------------------------------------------------------------

file_path = sys.argv[1:][0]

out_fields = {}

print("Approach")
data = hl2ss.download_calibration_rm_depth_longthrow(host, port)
print("huh")

out_fields["UTC Time"] = time.time_ns()

out_fields["UV2XY"] = data.uv2xy       

out_fields["Undistort Map"] = data.undistort_map

out_fields["Scale"] = data.scale

out_fields["Extrinsics 0,0"] = data.extrinsics[0][0]
out_fields["Extrinsics 0,1"] = data.extrinsics[0][1]
out_fields["Extrinsics 0,2"] = data.extrinsics[0][2]
out_fields["Extrinsics 0,3"] = data.extrinsics[0][3]
out_fields["Extrinsics 1,0"] = data.extrinsics[1][0]
out_fields["Extrinsics 1,1"] = data.extrinsics[1][1]
out_fields["Extrinsics 1,2"] = data.extrinsics[1][2]
out_fields["Extrinsics 1,3"] = data.extrinsics[1][3]
out_fields["Extrinsics 2,0"] = data.extrinsics[2][0]
out_fields["Extrinsics 2,1"] = data.extrinsics[2][1]
out_fields["Extrinsics 2,2"] = data.extrinsics[2][2]
out_fields["Extrinsics 2,3"] = data.extrinsics[2][3]

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

print(Fore.RED, "Saving Longthrow Calibration...")
df.to_csv(f"{file_path}/depth_calibration.csv")
print(Fore.RED, "Longthrow Calibration Saved.")

quit()


