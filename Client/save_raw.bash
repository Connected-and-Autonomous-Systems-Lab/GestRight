#!/usr/bin/env bash

name=$1
type=$2
run=$3
if [[ $name == "" ]]; then
    echo "Name must be supplied"
    echo "Format: first_last"
    exit
fi

if [[ $type == "" ]]; then
    echo "Type must be supplied"
    echo "Format: unprompted, prompted, type_a"
    exit
fi

if [[ $run == "" ]]; then
    echo "Run number must be supplied"
    echo "Format: run_1, run_2"
    exit
fi


filename="study0/$name"
mkdir -p $filename

filename=$filename"/$type"
mkdir -p $filename

filename=$filename"/$run"
mkdir -p $filename

filename=$filename"/$(date +%m_%d_%T)"
echo $filename
mkdir $filename

python3 save_pv_calibration.py $filename
python3 save_depth_calibration.py $filename

python3 save_pv.py $filename &
python3 save_si.py $filename &
python3 save_depth.py $filename &

exit