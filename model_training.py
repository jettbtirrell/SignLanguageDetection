import torch
from ultralytics import YOLO
import clearml
from clearml import Task

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
    task_name = 'Training_2'
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
    data='Data\\data.yaml',
    epochs=20,
    imgsz=640,
    device=device,
    augment=True,
    mixup=0.1,
    project="C:\\Users\\jettb\\SignLanguageDetection\\Data\\output",
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