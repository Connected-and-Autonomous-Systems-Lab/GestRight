### Prerequisites ###
* Microsoft Hololens 2 Sensor Streaming
The receiver code is built on top of the HL2SS project found here: https://github.com/jdibenes/hl2ss
The .sln file for the HL2 server app needs to be running on the Hololens device while the Python client establishes websockets for receiving streaming data.

* ROS2 for Boston Dynamics Spot (Follow instructions here: https://www.clearpathrobotics.com/assets/guides/melodic/spot-ros/robot_setup.html)

### Gesture Classification and Navigational Control ###
* The streamed hand tracking data is then classified as navigational commands depending on the three schemes: Fist, Touch and Wheel-based
* Navigation commands are passed as cmd_vel topics through ROS

To run the system:
* Launch the HL2SS app on the Hololens 2 device
* Initialize ROS setup/configs on the host machine (ensure username, password and IP addresses are correct)
./start_driver.bash
./start_viewer.bash

* Run spot control passing <directory_name, gesture_scheme, run_number> as arguments
  ./save_spot_control.bash test_user wheel run_1

* This handles both controlling the robot as well as storing raw streamed data (hand/eye/head tracking, RGB images and depth images from the HoloLens for further analyses)
* Alternatively, run ./save_gesture_control.bash <dir> <scheme> <run> to run the gesture classifier but without the robot being controlled.
