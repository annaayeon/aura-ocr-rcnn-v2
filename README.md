# Accurate and Efficient Elevator Button Localization

Welcome to the OCR-RCNN-v2 repository! This project is designed for autonomous elevator manipulation, enabling robots to autonomously operate previously unvisited elevators. This repository contains the perception part of the project, based on our initial publication: [A Novel OCR-RCNN for Elevator Button Recognition](https://ieeexplore.ieee.org/abstract/document/8594071). In this version, we've improved accuracy by 20% and achieved a real-time running speed of ~10 FPS (640x480) on a graphics card (≥ GTX950). We've also tested it on a laptop with a GTX950M (2GB memory), achieving a running speed of ~6 FPS. We're currently working on optimizing the TX2 version for faster performance, which will be released soon along with the dataset and post-processing code.

## Requirements

To run OCR-RCNN-v2, ensure you have the following dependencies:

- **Operating System:** Ubuntu 20.04
- **Python Version:** 3.8
- **Libraries:**
  - TensorFlow 2.3.0
  - Numpy 1.18.5

## Installation

For running on laptops and desktops (x86_64), follow these steps:

1. Install necessary libraries:
    ```sh
    sudo apt install libjpeg-dev libpng-dev libfreetype6-dev libxml2-dev libxslt1-dev
    ```
2. Install Python packages:
    ```sh
    pip install tensorflow==2.3.0
    pip install numpy==1.18.5
    pip install protobuf==3.19.6
    pip install pillow matplotlib lxml imageio --user
    pip install pyrealsense2
    ```

## Inference

To run the inference, follow these steps:

1. Clone the repository:
    ```sh
    git clone https://github.com/annaayeon/ocr-rcnn-v2.git
    ```
2. Navigate to the project directory:
    ```sh
    cd ocr-rcnn-v2
    ```
3. Run the inference script:
    ```sh
    python inference_640x480.py
    ```

## Real-Time Elevator Button Detection

This project utilizes the Intel RealSense Depth Camera D435i for real-time elevator button detection. The camera captures depth information, which is crucial for accurately locating and identifying elevator buttons in three-dimensional space.

### ROS Integration

OCR-RCNN-v2 is integrated with ROS (Robot Operating System) to publish the detected button locations. The detected button coordinates are published as `TransformStamped` messages in the `geometry_msgs.msg` format. This allows seamless communication with other ROS nodes, enabling comprehensive robot control and automation.
