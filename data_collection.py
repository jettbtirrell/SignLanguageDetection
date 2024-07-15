import cv2
from cvzone.HandTrackingModule import HandDetector
import time

def initialize_camera():
    return cv2.VideoCapture(0)

def initialize_hand_detector():
    return HandDetector(maxHands=1)

def convert_bbox_to_yolo_format(bbox,img_width, img_height):
    x_topleft,y_topleft,w,h = bbox
    x_center = x_topleft + (w / 2)
    y_center = y_topleft + (h / 2)
    
    x_center /= img_width
    y_center /= img_height
    bbox_width = w / img_width
    bbox_height = h / img_height
    
    return x_center, y_center, bbox_width, bbox_height

def detect_hands(detector, img):
    return detector.findHands(img, draw=False)

def save_image_and_label(folder, img, bbox, category_int):
    file_name = f'Image_{time.time()}'
    image_path = f'{folder}/images/{file_name}.jpg'
    label_path = f'{folder}/labels/{file_name}.txt'
    label_data = create_label_data(bbox, category_int)
    cv2.imwrite(image_path,img)
    with open(label_path, 'w') as f:
            f.write(label_data)
    return file_name

def create_label_data(bbox, category_int):
    return f"{category_int} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

def main():
    base_folder = "C:/Users/jettb/SignLanguageDetection/YoloData2"
    folders = {
        't': f"{base_folder}/train",
        'v': f"{base_folder}/valid",
        'e': f"{base_folder}/test"
    }
    categories = ['Hello', 'ILoveYou', 'No', 'Okay', 'Please', 'ThankYou', 'Yes']
    current_folder_key = 't'
    category_int = 0
    counter = 0
    
    cap = initialize_camera()
    detector = initialize_hand_detector()
    
    print("Press 't' to save to train folder, 'v' for valid, and 'e' for test")
    print("Press 'n' to go to the next category")
    print(f"Current folder: {folders[current_folder_key]}")
    print(f"Current category: {categories[category_int]}")
    
    while cap.isOpened():
        ret, img = cap.read()
        hands, img = detect_hands(detector, img)
        
        if hands:
            hand = hands[0]
            bbox = convert_bbox_to_yolo_format(hand['bbox'], img.shape[1], img.shape[0])

        cv2.imshow("Image", img)
        key = cv2.waitKey(1)
        
        if key == ord('s'):
            counter += 1
            file_name = save_image_and_label(f"{base_folder}/{folders[current_folder_key]}", img, bbox, category_int)
            print(f"Saved image {counter}: {file_name}")
        elif key == ord('q'):
            break
        elif key == ord('n'):
            category_int += 1
            print(f"New category: {categories[category_int]}")
        elif key == ord('t'):
            current_folder_key = 't'
            print(f"Current folder: {folders[current_folder_key]}")
        elif key == ord('v'):
            current_folder_key = 'v'
            print(f"Current folder: {folders[current_folder_key]}")
        elif key == ord('e'):
            current_folder_key = 'e'
            print(f"Current folder: {folders[current_folder_key]}")
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()