import cv2
from cvzone.HandTrackingModule import HandDetector
import time

cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands = 1)
counter = 0

folder = "C:/Users/jettb/SignLanguageDetection/YoloData2/valid"
category_int = 0

def convert_bbox_to_yolo_format(bbox,img_width,img_height):
    x_topleft,y_topleft,w,h = bbox
    x_center = x_topleft + (w / 2)
    y_center = y_topleft + (h / 2)
    
    x_center /= img_width
    y_center /= img_height
    bbox_width = w / img_width
    bbox_height = h / img_height
    
    return x_center, y_center, bbox_width, bbox_height

while cap.isOpened() :
    ret, img = cap.read()
    hands, img = detector.findHands(img,draw=False)
    if hands :
        hand = hands[0]
        x,y,w,h = convert_bbox_to_yolo_format(hand['bbox'],img.shape[1],img.shape[0])

    cv2.imshow("Image", img)
    key = cv2.waitKey(1)
    if key == ord('s') :
        counter += 1
        file_name = f'Image_{time.time()}'
        cv2.imwrite(f'{folder}/images/{file_name}.jpg',img)
        label_path = f'{folder}/labels/{file_name}.txt'
        with open(label_path, 'w') as f:
            f.write(f"{category_int} {x} {y} {w} {h}\n")
        print(counter)
    if key == ord('q') :
        break
    if key == ord('n') :
        category_int += 1
        print(category_int)

    