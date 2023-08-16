# Binocular-ranging-yolov5
This project is modified based on yolov5_ros, which you can find in https://github.com/qq44642754a/Yolov5_ros.git
In this project, the ranging function is added
The range was measured using a realsense d435i binocular camera
The pyrealsense2 feature pack requires installation , You can use the command "pip install pyrealsense2"
This project only needs to launch the launch file and will not publish related topics in ros （After we run the yolo_v5.launch file, the d435i camera will be started directly, and there is no need to boot it up）
