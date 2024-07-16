from ultralytics import YOLO
import cv2
import torch

def initialize_camera():
    """
    Initializes the camera for capturing images.
    :return: OpenCV VideoCapture object.
    """
    return cv2.VideoCapture(0)

def initialize_device():
    """
    Initializes a device object for running the model.
    :return: Pytorch Device object.
    """
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
def initialize_categories():
    """
    Initializes the list of gesture categories.
    :return: List of gesture category names.
    """
    return ['Hello', 'I Love You', 'No', 'Okay', 'Please', 'Thank You', 'Yes']

def load_model(model_path):
    """
    Loads the YOLO model from the given path.
    :param model_path: Path to the model weights.
    :return: YOLO model object.
    """
    return YOLO(model_path)

def process_frame(frame, model, device, categories, padding=20, conf_threshold=0.6):
    """
    Processes a single frame using the YOLO model.
    :param frame: Input frame to process.
    :param model: YOLO model object.
    :param device: Pytorch device object.
    :param categories: List of category names.
    :param padding: Padding for bounding boxes.
    :param conf_threshold: Confidence threshold for detections.
    :return: Processed frame with detections.
    """
    results = model(frame, conf=conf_threshold, device=device)
    
    for result in results:
        boxes = result.boxes
        for box in boxes:
            bbox = box.xyxy[0]
            conf = box.conf.item()
            cls = int(box.cls.item())
            class_string = categories[cls]
            
            label = f'Class: {class_string}, Conf: {conf:.2f}'
            
            cv2.rectangle(frame, 
                          (int(bbox[0]) - padding, int(bbox[1]) - padding), 
                          (int(bbox[2]) + padding, int(bbox[3]) + padding), 
                          (255, 0, 0), 2)
            cv2.putText(frame, label, 
                        (int(bbox[0]), int(bbox[1]) - padding - 5), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    return frame

def main():
    model_path = 'C:\\Users\\jettb\\SignLanguageDetection\\YOLOData2\\output\\train\\weights\\best.pt'
    model = load_model(model_path)
    cap = initialize_camera()
    device = initialize_device()
    categories = initialize_categories()
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        processed_frame = process_frame(frame, model, device, categories)
        cv2.imshow('Detected Hands', processed_frame)
        
        key = cv2.waitKey(1)
        
        if key == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()