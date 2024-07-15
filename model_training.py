import torch
from ultralytics import YOLO
import clearml
from clearml import Task

clearml.browser_login()
task = Task.init(project_name='ASL_Recognition', task_name='Training_0')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = YOLO('yolov8s.yaml')

if __name__ == '__main__':
    model.train(
    data='YOLOData2\\data.yaml', 
    epochs=100, 
    imgsz=640, 
    device=device,
    augment=True,
    mixup=0.1,
    project="C:\\Users\\jettb\\SignLanguageDetection\\YOLOData2\\output",
    plots=True
    )