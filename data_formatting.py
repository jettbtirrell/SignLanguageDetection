import os
import shutil
import random
from PIL import Image


def format_data_for_yolo(source_dir, yolo_dir):
    class_names = ['Hello', 'ILoveYou', 'No', 'Okay', 'Please', 'ThankYou', 'Yes']
    for id, class_name in enumerate(class_names):
        class_dir = source_dir + '/' + class_name
        file_names = []
        for file_name in os.listdir(class_dir):
            file_names.append(file_name)

        random.shuffle(file_names)
        train_split = int(0.8 * len(file_names))
        valid_split = int(0.9 * len(file_names))
        
        for i, file_name in enumerate(file_names):
            if i < train_split:
                subset = 'train'
            elif i < valid_split:
                subset = 'valid'
            else:
                subset = 'test'
                
            
            src_image_path = class_dir + "/" + file_name
            new_image_path = os.path.join(yolo_dir, subset, 'images', file_name)
            shutil.copy(src_image_path,new_image_path)
            
            
            x,y,w,h = get_bounding_box(src_image_path)
            label_path = os.path.join(yolo_dir, subset, 'labels', file_name.replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write(f"{id} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n")
            
def get_bounding_box(image_path):
    with Image.open(image_path) as img:
        
        gray = img.convert('L')
        
        
        width, height = gray.size
        
        
        left, top, right, bottom = width, height, 0, 0
        for y in range(height):
            for x in range(width):
                if gray.getpixel((x, y)) < 250:  
                    left = min(left, x)
                    top = min(top, y)
                    right = max(right, x)
                    bottom = max(bottom, y)
        
        
        x_center = (left + right) / 2 / width
        y_center = (top + bottom) / 2 / height
        box_width = (right - left) / width
        box_height = (bottom - top) / height
        
        return x_center, y_center, box_width, box_height


source_directory = "C:/Users/jettb/SignLanguageDetection/Data"
yolo_directory = "C:/Users/jettb/SignLanguageDetection/YOLOData"
format_data_for_yolo(source_directory,yolo_directory)