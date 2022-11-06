# Yolo 2D object detection for Real-Time Applications 

## Teaming Information
Project Team 4: Wil Carrasco, Aaron Choi, Benjamin Hochstadt

## Dataset
Datasets under consideration: Coco Dataset, Waymo dataset converted to Coco format, custom dataset to apply transfer learning in Coco format.

## Project Idea
Detecting objects is one of the cornerstones of deep learning and practical computer vision. Our team aims to correctly classify objects and track them as they move about with low latency in each frame or video stream for a real-time application. We propose to use the latest advancements in YOLO (v5/v7) to implement a deep neural network that is capable of correctly identifying multiple objects in a given video frame and/or stream with a relatively high mean average precision and fast inference rate (>20ms). The 
is to train a YOLO model and apply techniques/skills acquired in this course to successfully train and inference different scenes of pedestrians and vehicles. Ultimately, our goal is to be able to process a video stream on an embedded target using our trained model and pipeline. 

## Proposed Solution
### Step 1: Set up development environment on HPC
### Step 2: Data ingestion and testing (split train/test data).
### Step 3: Train model using dataset in Coco format on HPC or local high-end compute device.
### Step 4: Evaluate model and retrain or keep if acceptable.
### Step 5: Run inference on Raspberry Pi using pre-recorded image frames to determine latency performance and limitations.
### Step 6: Run Inference on video stream from Raspberry Pi’s peripheral camera and report latency and performance results.
