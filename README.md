# Accurate and Efficient Elevator Button Localization

OCR-RCNN-v2 is designed for autonomous elevator manipulation, the goal of which is to enable the robot to autonomously operate elevators that are previously unvisited. This repository contains the perception part of this project.  We published the initial version in paper  [A Novel OCR-RCNN for Elevator Button Recognition](https://ieeexplore.ieee.org/abstract/document/8594071) and this version improves the accuracy by 20% and achieves a real-time running speed ~10FPS (640*480)  on a graphical card (>=GTX950).  We have also tested on a laptop installed with a GTX950M (2G memory). It can achieves a running speed of ~6FPS. We are working on optimizing the TX2 version to make it faster,  which will be soon released with the dataset, as well as the post-processing code. 

### Requirements

1.  Ubuntu == 20.04
2.  TensorFlow == 2.3.0
3.  Numpy == 1.18.5
4.  Python == 3.8
5.  Tensorrt == 4.0 (optional)
6.  2GB GPU (or shared) memory 

### Inference

Before running the code, please first download the [models](https://drive.google.com/file/d/1FVXI-G-EsCrkKbknhHL-9Y1pBshY7JCv/view?usp=sharing) into the code folder. There are five frozen tensorflow models:

1. *detection_graph.pb*: a general detection model that can handle panel images with arbitrary size.
2.  *ocr_graph.pb*: a character recognition model that can handle button images with a size of 180x180.
3. *detection_graph_640x480.pb*: a detection model  fixed-size image as input.
4. *detection_graph_640x480_optimized.pb*: an optimized version of the detection model.
5. *ocr_graph_optimized.pb*:  an optimized version of the recognition model.

For running on laptops and desktops (x86_64), you may need to install some packages :

1. `sudo apt install libjpeg-dev libpng12-dev libfreetype6-dev libxml2-dev libxslt1-dev `
2. `sudo apt install ttf-mscorefonts-installer`
3. `sudo add-apt-repository multiverse`
4. `sudo apt update`
5. `pip install pillow matplotlib lxml imageio --user` 
6. `git clone https://github.com/annaayeon/ocr-rcnn-v2.git`
7. `cd ocr-rcnn-v2`
8. ``mv frozen/ ocr-rcnn-v2/``
9. ``python inference_640x480.py``
