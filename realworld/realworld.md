# Real-world Experiment Guide
We provide the server code of StreamVLN and the client code of unitree Go2 to deploy StreamVLN on real-world robots. We assume an Intel Realsense D400 series camera is mounted on the robot.  

## Prerequisites
1. Install [ROS2](https://docs.ros.org/en/foxy/index.html) on robot. Note that the unitree go2 is usually pre-installed with ROS2 foxy. 

    ```You could also use ROS1 for deployment. In this case you need to modify the topic subscriber and publisher in `go2_vln_client.py`. In our sample code, ROS2 is only used for subscribing to the odometry/image and publishing the desired velocity command to the robot. 

2. Install [realsense-ros](https://github.com/IntelRealSense/realsense-ros) package.

## Run the Real-world Experiment

1. Run the realsense-ros on robot
    ```bash
    ros2 launch realsense2_camera rs_align_depth_launch.py
    ```
    If you install realsense-ros from binary, you may not find rs_align_depth_launch.py. In this case you could also use `rs_camera.launch.py` instead. Please remember to change the topic name in `go2_vln_client.py`.

2. Run the `http_realworld_server.py` on remote server
    ```bash
    cd path/to/StreamVLN/streamvln
    python http_realworld_server.py
    ```

3. Run the `go2_vln_client.py` on robot

   Change the server ip in `go2_vln_client.py` to the remote server ip. And check odometry/image topics are subscribed successfully. Then just run,
   ```bash
   python3 go2_vln_client.py
   ```

If everything goes well, the robot should start moving and the server should print the action sequence.
   
    

