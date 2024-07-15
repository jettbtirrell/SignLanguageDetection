from ultralytics import YOLO
import cv2
import torch

model = YOLO('C:\\Users\\jettb\\SignLanguageDetection\\YOLOData2\\output\\train\\weights\\best.pt')
cap = cv2.VideoCapture(0)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
class_names = ['Hello','ILoveYou','No','Okay','Please','ThankYou','Yes']
padding = 20
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    
    results = model(frame, conf = 0.6, device=device)

    if results and len(results) > 0:
        boxes = results[0].boxes
        for box in boxes:
            bbox = box.xyxy[0].cpu().numpy()
            conf = box.conf.item()
            cls = int(box.cls.item())
            class_string = class_names[cls]
            
            label = f'Class: {class_string}, Conf: {conf:.2f}'

            cv2.rectangle(frame, (int(bbox[0]) - padding, int(bbox[1]) - padding), (int(bbox[2]) + padding, int(bbox[3]) + padding), (255, 0, 0), 2)
            cv2.putText(frame, label, (int(bbox[0]), int(bbox[1]) - padding - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)


    cv2.imshow('Detected Hands', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()