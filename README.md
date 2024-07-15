# Sign Language Detection

This project aims to detect and classify American Sign Language (ASL) hand signs in real-time from a live video feed using AI machine learning techniques. It leverages YOLOv8 for object detection, PyTorch for deep learning capabilities, OpenCV for image and video processing, CVZone for additional computer vision utilities, and ClearML for experiment management and automation.

## Purpose

The main purpose of this project is to explore the intersection of Computer Vision and ASL recognition, as well as creating a template that others interested in Computer Vision can modify and learn from. In this project, I have created a script to easily create your own database with bounding boxes in YOLOv8 format, a script to train a YOLOv8 model on said database, and a script to feed that model frame from live video capture, where it will give a feedback view drawing a box around the hand signs given and labelling them.
## Setup Instructions

### Prerequisites

Make sure you have the following installed:
- PyTorch
- OpenCV
- YOLOv8
- CVZone
- ClearML

### Order of Operations

**Data Collection:**
Use data collection.py to create your own ASL hand sign images for training.

**Model Training:**
Train your YOLOv8 model using model_training.py on the collected data.

**Run the Model:**
Use run_model.py to deploy and run the trained model on a live video feed for real-time ASL hand sign detection.

**License**
This project is licensed under the MIT License - see the LICENSE file for details.

**Contact**
For questions or suggestions regarding the project, feel free to contact me:

Email: jett.b.tirrell@gmail.com
GitHub: github.com/jettbtirrell
