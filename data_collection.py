import cv2
from cvzone.HandTrackingModule import HandDetector
import time

def initialize_folders():
    base_folder = "C:/Users/jettb/SignLanguageDetection/Data"
    return [
        f"{base_folder}/train",
        f"{base_folder}/valid",
        f"{base_folder}/test"
    ]

def initialize_categories():
    return ['Hello', 'ILoveYou', 'No', 'Okay', 'Please', 'ThankYou', 'Yes']

def initialize_camera():
    return cv2.VideoCapture(0)

def initialize_hand_detector():
    return HandDetector(maxHands=1)

def print_instructions():
    print("Press 'f' to toggle between train, valid, and test folders")
    print("Press 'c' to cycle through categories")
    print("Press 's' to save an image")
    print("Press 'q' to quit")
    
def print_current_settings(folders, current_folder_index, categories, category_index):
    print(f"Current folder: {folders[current_folder_index]}")
    print(f"Current category: {categories[category_index]}")

def detect_hands(detector, img):
    return detector.findHands(img, draw=False)

def convert_bbox_to_yolo_format(bbox,img_width, img_height):
    x_topleft,y_topleft,w,h = bbox
    x_center = x_topleft + (w / 2)
    y_center = y_topleft + (h / 2)
    
    x_center /= img_width
    y_center /= img_height
    bbox_width = w / img_width
    bbox_height = h / img_height
    
    return x_center, y_center, bbox_width, bbox_height

def create_file_name():
    return f'Image_{time.time()}'

def create_label_data(bbox, category_index):
    return f"{category_index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

def save_image(folder, img, file_name):
    image_path = f'{folder}/images/{file_name}.jpg'
    cv2.imwrite(image_path,img)

def save_label(folder, label_data, file_name):
    label_path = f'{folder}/labels/{file_name}.txt'
    with open(label_path, 'w') as f:
        f.write(label_data)

def print_image_saved(file_name,counter):
    print(f"Saved image {counter}: {file_name}")

def handle_keypress(key, folders, current_folder_index, categories, category_index, counter, img, bbox):
    if key == ord('s'):
        counter += 1
        file_name = create_file_name()
        label_data = create_label_data(bbox, category_index)
        save_image(folders[current_folder_index],img,file_name)
        save_label(folders[current_folder_index],label_data,file_name)
        print_image_saved(file_name,counter)
        
    elif key == ord('c'):
        category_index = (category_index + 1) % len(categories)
        print(f"New category: {categories[category_index]}")
        
    elif key == ord('f'):
        current_folder_index = (current_folder_index + 1) % len(folders)
        print(f"Current folder: {folders[current_folder_index]}")

    return current_folder_index, category_index, counter

def main():
    folders = initialize_folders()
    categories = initialize_categories()
    current_folder_index = 0
    category_index = 0
    saved_image_counter = 0
    
    cap = initialize_camera()
    detector = initialize_hand_detector()
    
    print_instructions()
    print_current_settings(folders, current_folder_index, categories, category_index)
    
    while cap.isOpened():
        ret, img = cap.read()
        if ret:
            hands, img = detect_hands(detector, img)
            
            bbox = None
            if hands:
                hand = hands[0]
                bbox = convert_bbox_to_yolo_format(hand['bbox'], img.shape[1], img.shape[0])

            cv2.imshow("Image", img)
            key = cv2.waitKey(1)
            
            if key == ord('q'):
                break
            
            current_folder_index, category_index, saved_image_counter = handle_keypress(key, folders, current_folder_index, categories, category_index, saved_image_counter, img, bbox)
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()