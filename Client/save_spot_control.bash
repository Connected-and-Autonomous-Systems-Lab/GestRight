#!/usr/bin/env bash

name=$1
control=$2
course=$3
if [[ $name == "" ]]; then
    echo "Name must be supplied"
    echo "Format: first_last"
    exit
fi

if [[ $control == "" ]]; then
    echo "Control type must be supplied"
    echo "Format: touch, fist, wheel"
    exit
fi

if [[ $course == "" ]]; then
    echo "Course must be supplied"
    echo "Format: simple, complex"
    exit
fi

filename="study0/$name"
mkdir -p $filename

filename=$filename"/$control"
mkdir -p $filename

filename=$filename"/$course"
mkdir -p $filename

filename=$filename"/$(date +%m_%d_%T)"
echo $filename
mkdir $filename

python3 save_pv_calibration.py $filename
python3 save_depth_calibration.py $filename

python3 save_pv.py $filename &
python3 gesture_control_spot.py $filename $control &
python3 save_depth.py $filename &

exit