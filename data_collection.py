"""
This module provides functionality for collecting hand gesture data using a camera,
detecting hand positions, and saving images with corresponding YOLO format labels
for machine learning purposes.
"""

import time
import cv2
from cvzone.HandTrackingModule import HandDetector


def initialize_folders():
    """
    Initializes the folder structure for storing collected data.
    :return: List of folder paths for train, valid, and test data.
    """
    base_folder = "C:/Users/jettb/SignLanguageDetection/Data"
    return [
        f"{base_folder}/train",
        f"{base_folder}/valid",
        f"{base_folder}/test"
    ]

def initialize_categories():
    """
    Initializes the list of gesture categories.
    :return: List of gesture category names.
    """
    return ['Hello', 'ILoveYou', 'No', 'Okay', 'Please', 'ThankYou', 'Yes']

def initialize_camera():
    """
    Initializes the camera for capturing images.
    :return: OpenCV VideoCapture object.
    """
    return cv2.VideoCapture(0)

def initialize_hand_detector():
    """
    Initializes the hand detector.
    :return: HandDetector object configured for single hand detection.
    """
    return HandDetector(maxHands=1)

def print_instructions():
    """
    Prints instructions for user interaction with the program.
    """
    print("Press 'f' to toggle between train, valid, and test folders")
    print("Press 'c' to cycle through categories")
    print("Press 's' to save an image")
    print("Press 'q' to quit")

def print_current_settings(folders, current_folder_index, categories, category_index):
    """
    Prints the current folder and category settings.
    :param folders: List of folder paths.
    :param current_folder_index: Index of the current folder.
    :param categories: List of category names.
    :param category_index: Index of the current category.
    """
    print(f"Current folder: {folders[current_folder_index]}")
    print(f"Current category: {categories[category_index]}")

def detect_hands(detector, img):
    """
    Detects hands in the given image.
    :param detector: HandDetector object.
    :param img: Image to detect hands in.
    :return: Tuple containing detected hands and the image.
    """
    return detector.findHands(img, draw=False)

def convert_bbox_to_yolo_format(bbox,img_width, img_height):
    """
    Converts bounding box coordinates to YOLO format.
    :param bbox: Bounding box coordinates (x, y, w, h).
    :param img_width: Width of the image.
    :param img_height: Height of the image.
    :return: YOLO format coordinates (x_center, y_center, width, height).
    """
    x_topleft,y_topleft,w,h = bbox
    x_center = x_topleft + (w / 2)
    y_center = y_topleft + (h / 2)

    x_center /= img_width
    y_center /= img_height
    bbox_width = w / img_width
    bbox_height = h / img_height

    return x_center, y_center, bbox_width, bbox_height

def create_file_name():
    """
    Creates a unique file name based on the current timestamp.
    :return: String containing the file name.
    """
    return f'Image_{time.time()}'

def create_label_data(bbox, category_index):
    """
    Creates label data in YOLO format.
    :param bbox: Bounding box coordinates in YOLO format.
    :param category_index: Index of the current category.
    :return: String containing the label data.
    """
    return f"{category_index} {bbox[0]} {bbox[1]} {bbox[2]} {bbox[3]}\n"

def save_image(folder, img, file_name):
    """
    Saves the image to the specified folder.
    :param folder: Path to the folder where the image will be saved.
    :param img: Image to be saved.
    :param file_name: Name of the file.
    """
    image_path = f'{folder}/images/{file_name}.jpg'
    cv2.imwrite(image_path,img)

def save_label(folder, label_data, file_name):
    """
    Saves the label data to the specified folder.
    :param folder: Path to the folder where the label will be saved.
    :param label_data: Label data to be saved.
    :param file_name: Name of the file.
    """
    label_path = f'{folder}/labels/{file_name}.txt'
    with open(label_path, 'w') as f:
        f.write(label_data)

def print_image_saved(file_name,counter):
    """
    Prints a message indicating that an image has been saved.
    :param file_name: Name of the saved file.
    :param counter: Counter of saved images.
    """
    print(f"Saved image {counter}: {file_name}")

def handle_keypress(key, folders, current_folder_index, categories, category_index, counter, img, bbox):
    """
    Handles user key presses and performs corresponding actions.
    :param key: Pressed key code.
    :param folders: List of folder paths.
    :param current_folder_index: Index of the current folder.
    :param categories: List of category names.
    :param category_index: Index of the current category.
    :param counter: Counter of saved images.
    :param img: Current image.
    :param bbox: Bounding box of detected hand.
    :return: Updated current_folder_index, category_index, and counter.
    """
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
    """
    Main function to run the hand gesture data collection module.
    """
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

            current_folder_index, category_index, saved_image_counter = handle_keypress(
                key, folders, current_folder_index, categories, 
                category_index, saved_image_counter, img, bbox)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
