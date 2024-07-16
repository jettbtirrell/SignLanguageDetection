import torch
from ultralytics import YOLO
import clearml
from clearml import Task

def login_clearml():
    clearml.browser_login()

def create_clearml_task():
    project_name = 'ASL_Recognition'
    task_name = 'Training_2'
    return Task.init(project_name=project_name, task_name=task_name)

def create_model():
    model = YOLO('yolov8s.yaml')
    return model

def train_model(model):
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
    login_clearml()
    task = create_clearml_task()
    model = create_model()
    train_model(model)

if __name__ == "__main__":
    main()