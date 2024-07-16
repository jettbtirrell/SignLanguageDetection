"""
This module provides functionality for training a YOLO model using ClearML for experiment 
tracking. It includes functions for logging into ClearML, creating a task, creating a YOLO
model, and training the model on a dataset.
"""

import torch
from ultralytics import YOLO
import clearml
from clearml import Task
import project_directory

def login_clearml():
    """
    Logs in to ClearML using browser authentication.
    """
    clearml.browser_login()

def create_clearml_task():
    """
    Creates a new ClearML task for tracking experiments.
    :return: Task object for the created ClearML task.
    """
    project_name = 'ASL_Recognition'
    task_name = 'Training_3'
    return Task.init(project_name=project_name, task_name=task_name)

def create_model():
    """
    Creates a YOLO model using the YOLOv8s architecture.
    :return: YOLO model object.
    """
    model = YOLO('yolov8s.yaml')
    return model

def train_model(model):
    """
    Trains the given YOLO model on ASL recognition dataset.
    :param model: YOLO model object to be trained.
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.train(
    data='PersonalData/data.yaml',
    epochs=100,
    imgsz=640,
    device=device,
    augment=True,
    mixup=0.1,
    project=f"{project_directory.root}/Data/output",
    plots=True
    )

def main():
    """
    Main function to run the model training pipeline.
    """
    login_clearml()
    task = create_clearml_task()
    model = create_model()
    train_model(model)

if __name__ == "__main__":
    main()
