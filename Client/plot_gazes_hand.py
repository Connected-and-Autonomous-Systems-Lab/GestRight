#from pynput import keyboard
import multiprocessing as mp
import numpy as np
import cv2
import hl2ss_imshow
import hl2ss
#import hl2ss_lnm
import hl2ss_utilities
#import hl2ss_mp
import hl2ss_3dcv
import struct
import ast
#import hl2ss_sa
import argparse
from pathlib import Path
import matplotlib.pyplot as plt
import os

# Marker properties
radius = 10
head_color  = (  0,   0, 255)
left_color  = (  0, 255,   0)
right_color = (255,   0,   0)
gaze_color  = (255,   0, 255)
thickness = -1

def load_data_v3(csv_path):
    data = np.loadtxt(csv_path, delimiter=',', skiprows=2)

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))
    head_trans_transformed = np.zeros((n_frames, 3))

    left_index_tip = np.zeros((n_frames, 3))
    left_middle_tip = np.zeros((n_frames, 3))
    left_index_tip_available = np.ones(n_frames, dtype=bool)

    right_index_tip = np.zeros((n_frames, 3))
    right_middle_tip = np.zeros((n_frames, 3))
    right_index_tip_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 7))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[1]
        # head
        head_transs[i_frame, :] = frame[7:10]
        
        #left hand
        left_index_tip_available[i_frame] = (frame[113] !=0)
        left_index_tip[i_frame, :] = frame[113:116]
        left_middle_tip[i_frame, :] = frame[158:161]

        #right hand
        right_index_tip_available[i_frame] = (frame[347] !=0)
        right_index_tip[i_frame, :] = frame[347:350]
        right_middle_tip[i_frame, :] = frame[392:395]

        # gaze
        gaze_available[i_frame] = (frame[16] != 0)
        gaze_data[i_frame, :3] = frame[16:19]
        gaze_data[i_frame, 3:6] = frame[19:22]
        gaze_data[i_frame, 6] = frame[22]

        head_trans_transformed[i_frame, :] = head_transs[i_frame, :]
        head_trans_transformed[i_frame, 0] = 0.9965503*head_transs[i_frame, 0] -0.1755028*head_transs[i_frame, 1] -0.0024546*head_transs[i_frame, 2] -0.0625326
        head_trans_transformed[i_frame, 1] = 0*head_transs[i_frame, 0] + 0.8947*head_transs[i_frame, 1] -0*head_transs[i_frame, 2]
        head_trans_transformed[i_frame, 2] = 0*head_transs[i_frame, 0] -0.0974022*head_transs[i_frame, 1] +1.0019543*head_transs[i_frame, 2] + 0.0263197

    return (timestamps, gaze_data, gaze_available, head_trans_transformed, left_index_tip,left_middle_tip, left_index_tip_available, right_index_tip,right_middle_tip, right_index_tip_available)

def load_pv_info(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    # focal_lengths = np.zeros((n_frames, 2))
    # principal_points = np.zeros((n_frames, 2))
    pv2world_rotation = np.zeros((n_frames, 3,3))

    focal_lengths = np.zeros( 2)
    principal_points = np.zeros( 2)
    #zeros = np.zeros((3,4 ))
    focal_lengths[ 0] = 1448.168091
    focal_lengths[1] = 1448.512573
    principal_points[0] = 1080#1920#941.7901001
    principal_points[ 1] = 1920#1080#505.1950378

    # for i_frame, frame in enumerate(lines[3819:3929]):
    for i_frame, frame in enumerate(lines[1:]):
        

        # Row format is
        # timestamp, focal length (2), transform PVtoWorld (4x4)
        print("here")
        frame = frame.split(',')
        frame_timestamps[i_frame] = int(frame[2])
        # focal_lengths[i_frame, 0] = float(frame[16])
        # focal_lengths[i_frame, 1] = float(frame[17])
        # principal_points[i_frame, 0] = float(frame[18])
        # principal_points[i_frame, 1] = float(frame[19])
        # pv2world_transforms[i_frame] = np.array(frame[4:16]).astype(float).reshape((3, 4))
        # zeros = np.array(frame[4:16]).astype(float).reshape((3, 4))
        # pv2world_transforms[i_frame], pv2world_transforms[i_frame][:3] 
        #pv2world_transforms[i_frame][0] = np.array(frame[4:8]).astype(float).reshape((3, 4))
        pv2world_rotation[i_frame][0] = np.array(frame[4:7])
        pv2world_rotation[i_frame][1] = np.array(frame[8:11])
        pv2world_rotation[i_frame][2] = np.array(frame[12:15])
        #pv2world_transforms[i_frame][3,3] = 1

    return (frame_timestamps, focal_lengths, pv2world_rotation,principal_points)

def read_csv_file(folder):
    print("Starting to read csv files!!!")
    (timestamps, gaze_data, gaze_available, head_transs, left_index_tip,left_middle_tip, left_index_tip_available, right_index_tip, right_middle_tip, right_index_tip_available) = load_data_v3(list(folder.glob('data_v3.csv'))[0])
    (frame_timestamps, focal_lengths, pv2world_rotation,principal_points) = load_pv_info(list(folder.glob('pv_info.csv'))[0])
    print("csv files are read!")
    return (frame_timestamps, focal_lengths, pv2world_rotation,principal_points, timestamps, gaze_data, gaze_available, head_transs, left_index_tip,left_middle_tip, left_index_tip_available, right_index_tip, right_middle_tip, right_index_tip_available)

def match_frame(target, all_frames):
    return np.argmin([abs(x - target) for x in all_frames])

def join_Rt(pv2world_rotation,head_transs,index_at_datav3,homogenius_matrix,pvinfo_frame,pv_id):
    n_frames = len(homogenius_matrix)
    homogenius_matrix[pv_id][:3,:3] = pv2world_rotation[pv_id]
    homogenius_matrix[pv_id][:3,3] = head_transs[index_at_datav3]
    homogenius_matrix[pv_id][3,3] = 1
    return homogenius_matrix

def plot_points(folder,frame_timestamps, pv_intrinsics, pv2world_rotation, pv_extrinsics, timestamps, gaze_data, gaze_available, head_transs, left_index_tip, left_middle_tip, left_index_tip_available, right_index_tip, right_middle_tip, right_index_tip_available):
    pv_paths = sorted(list((folder / 'images').glob('*.jpg')))
    #pv_paths=[(folder / 'images')]
    n_frames = len(pv_paths)
    #print("franes: ", str(n_frames))
    homogenius_matrix = np.zeros((n_frames, 4,4))
    for pv_id in range(n_frames):
        pv_path = pv_paths[pv_id]
        
        try:
            pvinfo_frame = str(pv_path.name).replace('.jpg', '')
        except:
            continue
        print(f"\rProcessing frame {pv_id} of {n_frames}", end='', flush=True)
        print("frame num: " + str(pvinfo_frame))
        #timestamp = frame_timestamps[pvinfo_frame] # assume that the first frame is named as 0000.jpg
        timestamp = frame_timestamps[pv_id] 

        subtimestamps = [ts for ts in timestamps if ts <= timestamp]
        index_at_datav3 = match_frame(timestamp, subtimestamps) + 100

        
        print("ts in gaze: " + str(timestamps[index_at_datav3]))
        homogenius_matrix = join_Rt(pv2world_rotation,head_transs,index_at_datav3,homogenius_matrix,pvinfo_frame,pv_id)
        
        load_img_path = str(folder / 'images' / str(str(pvinfo_frame) + '.jpg')).replace('\\', '/')
        image = cv2.imread(load_img_path, cv2.IMREAD_COLOR)
        # image = cv2.resize(image, (760,428))
        world_to_image = hl2ss_3dcv.world_to_reference(homogenius_matrix[pv_id]) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

        # gaze 
        if gaze_available[index_at_datav3]:
            origin=gaze_data[index_at_datav3][0:3]
            direction=gaze_data[index_at_datav3][3:6]
            distance=gaze_data[index_at_datav3][6]

            gaze_point = origin + direction*distance
            gaze_image_point = hl2ss_3dcv.project(gaze_point, world_to_image)

            cv2.circle(image, (int(gaze_image_point[0]),int(gaze_image_point[1])), radius, gaze_color, thickness)
            print(gaze_image_point)

        # Draw Left Hand joints -----------------------------------------------
        if left_index_tip_available[index_at_datav3]:
            left_image_points = hl2ss_3dcv.project(left_index_tip[index_at_datav3], world_to_image)
            left_image_points_middle = hl2ss_3dcv.project(left_middle_tip[index_at_datav3], world_to_image)
            cv2.circle(image,(int(left_image_points[0]),int(left_image_points[1])) , radius, left_color, thickness)
            cv2.circle(image,(int(left_image_points_middle[0]),int(left_image_points_middle[1])) , radius, left_color, thickness)
            print(left_image_points)

        # Draw Right Hand joints ----------------------------------------------
        if right_index_tip_available[index_at_datav3]:
            right_image_points = hl2ss_3dcv.project(right_index_tip[index_at_datav3], world_to_image)
            right_image_points_middle = hl2ss_3dcv.project(right_middle_tip[index_at_datav3], world_to_image)
            cv2.circle(image, (int(right_image_points[0]),int(right_image_points[1])), radius, right_color, thickness)
            cv2.circle(image,(int(right_image_points_middle[0]),int(right_image_points_middle[1])) , radius, right_color, thickness)

        projected_images_folder = folder / 'projected_images_fixed_intrinsic'
        if not projected_images_folder.exists():
            projected_images_folder.mkdir(parents=True)

        # Assuming `pvinfo_frame` and `image` are defined earlier
        filename = 'Projected_Image_' + str(pvinfo_frame) + '.jpg'
        output_path = str(projected_images_folder / filename).replace('\\', '/')
        #image2 = cv2.resize(image, (760, 420))

        cv2.imwrite(output_path, image)

    return pv_paths

def plot_points_vary(folder,frame_timestamps, focal_lengths, pv2world_rotation, principal_points, timestamps, gaze_data, gaze_available, head_transs, left_index_tip, left_middle_tip, left_index_tip_available, right_index_tip, right_middle_tip, right_index_tip_available):
    pv_paths = sorted(list((folder / 'images').glob('*.jpg')))
    n_frames = len(pv_paths)
    homogenius_matrix = np.zeros((n_frames, 4,4))

    # focal_lengths = np.zeros(2)
    # principal_points = np.zeros(2)

    # focal_lengths[0] = 1448.168091
    # focal_lengths[1] = 1448.512573
    # principal_points[0] = 941.7901001
    # principal_points[1] = 505.1950378
    for pv_id in range(n_frames):
        pv_path = pv_paths[pv_id]
        
        try:
            pvinfo_frame = str(pv_path.name).replace('.jpg', '')
        except:
            continue
        print(f"\rProcessing frame {pv_id} of {n_frames}", end='', flush=True)
        # timestamp = frame_timestamps[int(pvinfo_frame)]  # assume that the first frame is named as 0000.jpg
        # index_at_datav3 = match_frame(timestamp, timestamps)

        timestamp = frame_timestamps[pv_id] 

        subtimestamps = [ts for ts in timestamps if ts <= timestamp]
        index_at_datav3 = match_frame(timestamp, subtimestamps) + 100

        # pv_intrinsics = hl2ss.create_pv_intrinsics(focal_lengths, principal_points)
        # pv_intrinsics = hl2ss.create_pv_intrinsics(focal_lengths[pv_id], (1080, 1920))
        
        # pv_extrinsics = np.eye(4, 4, dtype=np.float32)
        # pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)

        homogenius_matrix = join_Rt(pv2world_rotation,head_transs,index_at_datav3,homogenius_matrix,pvinfo_frame,pv_id)
        
        load_img_path = str(folder / 'images' / str(str(pvinfo_frame) + '.jpg')).replace('\\', '/')
        image = cv2.imread(load_img_path, cv2.IMREAD_COLOR)
        #image = cv2.resize(image, (760,428))
        world_to_image = hl2ss_3dcv.world_to_reference(homogenius_matrix[pv_id]) @ hl2ss_3dcv.rignode_to_camera(pv_extrinsics) @ hl2ss_3dcv.camera_to_image(pv_intrinsics)

        #QR
        # try :
        #     QR_x,QR_y = DetectQRcode(image)
        #     cv2.circle(image, (QR_x,QR_y), radius, ORANGE, thickness)
        # except: continue 

        # gaze 
        if gaze_available[index_at_datav3]:
            origin=gaze_data[index_at_datav3][0:3]
            direction=gaze_data[index_at_datav3][3:6]
            distance=gaze_data[index_at_datav3][6]

            gaze_point = origin + direction*distance
            gaze_image_point = hl2ss_3dcv.project(gaze_point, world_to_image)
            if (gaze_image_point[0] >= 0 and gaze_image_point[1]>= 0 and gaze_image_point[0] < image.shape[1] and gaze_image_point[1] < image.shape[0]):
                cv2.circle(image, (int(gaze_image_point[0]),int(gaze_image_point[1])), radius, gaze_color, thickness)

        # Draw Left Hand joints -----------------------------------------------
        if left_index_tip_available[index_at_datav3]:
            left_image_points = hl2ss_3dcv.project(left_index_tip[index_at_datav3], world_to_image)
            left_image_points_middle = hl2ss_3dcv.project(left_middle_tip[index_at_datav3], world_to_image)
            if (left_image_points[0]>= 0 and left_image_points[1] >= 0 and left_image_points[0] < image.shape[1] and left_image_points[1] < image.shape[0]):
                cv2.circle(image,(int(left_image_points[0]),int(left_image_points[1])) , radius, left_color, thickness)
            if (left_image_points_middle[0] >= 0 and left_image_points_middle[1] >= 0 and left_image_points_middle[0] < image.shape[1] and left_image_points_middle[1] < image.shape[0]):
                cv2.circle(image,(int(left_image_points_middle[0]),int(left_image_points_middle[1])) , radius, left_color, thickness)

        # Draw Right Hand joints ----------------------------------------------
        if right_index_tip_available[index_at_datav3]:
            right_image_points = hl2ss_3dcv.project(right_index_tip[index_at_datav3], world_to_image)
            right_image_points_middle = hl2ss_3dcv.project(right_middle_tip[index_at_datav3], world_to_image)
            if (right_image_points[0] >= 0 and right_image_points[1] >= 0 and right_image_points[0] < image.shape[1] and right_image_points[1] < image.shape[0]):
                cv2.circle(image, (int(right_image_points[0]),int(right_image_points[1])), radius, right_color, thickness)
            if (right_image_points_middle[0] >= 0 and right_image_points_middle[1] >= 0 and right_image_points_middle[0] < image.shape[1] and right_image_points_middle[1] < image.shape[0]):
                cv2.circle(image,(int(right_image_points_middle[0]),int(right_image_points_middle[1])) , radius, right_color, thickness)

        projected_images_folder = folder / 'projected_images_varied_intrinsics'
        if not projected_images_folder.exists():
            projected_images_folder.mkdir(parents=True)

        # Assuming `pvinfo_frame` and `image` are defined earlier
        filename = 'Projected_Image_' + str(pvinfo_frame) + '.jpg'
        output_path = str(projected_images_folder / filename).replace('\\', '/')

        cv2.imwrite(output_path, image)

    return pv_paths

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Process recorded data.')
    parser.add_argument("--recording_path", required=False,help="Path to recording folder")

    args = parser.parse_args()
    args.recording_path = "C:/Users/kj373/Documents/00-DATA/Experiment/azim_khan/qr/run_1/08_08_17_48_25"
    folder = Path(args.recording_path)
    (frame_timestamps, focal_lengths, pv2world_rotation, principal_points, timestamps, gaze_data, gaze_available, head_transs, left_index_tip, left_middle_tip, left_index_tip_available, right_index_tip, right_middle_tip, right_index_tip_available)= read_csv_file(folder)
    
    # Update PV intrinsics ------------------------------------------------
    # PV intrinsics may change between frames due to autofocus
    focal_lengths = np.zeros(2)
    principal_points = np.zeros(2)

    focal_lengths[0] = 1448.168091
    focal_lengths[1] = 1448.512573
    principal_points[0] = 960#941.7901001
    principal_points[1] = 540#505.1950378

    # hl2ss.create_pv_intrinsics(focal_lengths[pv_id], principal_points[pv_id])

    pv_intrinsics = hl2ss.create_pv_intrinsics(focal_lengths, principal_points)
    pv_extrinsics = np.eye(4, 4, dtype=np.float32)
    pv_intrinsics, pv_extrinsics = hl2ss_3dcv.pv_fix_calibration(pv_intrinsics, pv_extrinsics)
    print(frame_timestamps)
    pv_paths = plot_points(folder,frame_timestamps, pv_intrinsics,  pv2world_rotation, pv_extrinsics, timestamps, gaze_data, gaze_available, head_transs, left_index_tip,left_middle_tip, left_index_tip_available, right_index_tip,right_middle_tip, right_index_tip_available)
    # Compute world to PV image transformation matrix ---------------------
   

    ##varying
    #pv_paths = plot_points_vary(folder,frame_timestamps, focal_lengths,  pv2world_rotation, principal_points, timestamps, gaze_data, gaze_available, head_transs, left_index_tip,left_middle_tip, left_index_tip_available, right_index_tip,right_middle_tip, right_index_tip_available)
    
    print('')
    print("All frames are projected!!!")



    ##### frame 
    # 505, 941
    # 1080, 1920
    # 