import os
import cv2
import numpy as np
import argparse
from ultralytics import YOLO
from tqdm import tqdm

def annotate_image(img, bbox_list, label_list, conf_list, color=(0, 255, 0)):
    """Overlay bounding boxes with labels on an image."""
    for bbox, label, conf in zip(bbox_list, label_list, conf_list):
        x_min, y_min, x_max, y_max = map(int, bbox)
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), color, 2)
        label_text = f"{label} {conf:.2f}"
        cv2.putText(img, label_text, (x_min, y_min - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    return img

def extract_region(img, bbox, buffer=20):
    """Extract a region from an image with some padding."""
    x_min, y_min, x_max, y_max = map(int, bbox)
    img_height, img_width = img.shape[:2]
    x_min = max(0, x_min - buffer)
    y_min = max(0, y_min - buffer)
    x_max = min(img_width, x_max + buffer)
    y_max = min(img_height, y_max + buffer)
    return img[y_min:y_max, x_min:x_max]

def map_coords(crop_coords, original_bbox, orig_shape):
    """Map coordinates from cropped image back to the original image."""
    crop_x_min, crop_y_min, crop_x_max, crop_y_max = map(int, crop_coords)
    orig_height, orig_width = orig_shape[:2]
    orig_x_min, orig_y_min, orig_x_max, orig_y_max = map(int, original_bbox)
    
    scale_x = (orig_x_max - orig_x_min) / orig_width
    scale_y = (orig_y_max - orig_y_min) / orig_height
    
    return [
        orig_x_min + int(crop_x_min * scale_x),
        orig_y_min + int(crop_y_min * scale_y),
        orig_x_min + int(crop_x_max * scale_x),
        orig_y_min + int(crop_y_max * scale_y)
    ]

def process_images(input_folder, output_folder, person_model_path, ppe_model_path):
    """Detect persons and PPE in images, then save annotated results."""
    os.makedirs(output_folder, exist_ok=True)
    
    detect_person = YOLO(person_model_path)
    detect_ppe = YOLO(ppe_model_path)
    
    image_files = [f for f in os.listdir(input_folder) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    
    for img_name in tqdm(image_files, desc="Processing images"):
        img_path = os.path.join(input_folder, img_name)
        img = cv2.imread(img_path)
        
        person_detections = detect_person(img)[0]
        person_bboxes, person_labels, person_confs = [], [], []
        
        for det in person_detections.boxes:
            if det.cls == 0:  # Only process persons
                person_bboxes.append(det.xyxy[0].tolist())
                person_labels.append("Person")
                person_confs.append(float(det.conf[0]))
        
        ppe_bboxes, ppe_labels, ppe_confs = [], [], []
        for bbox in person_bboxes:
            person_cropped = extract_region(img, bbox)
            if person_cropped.size == 0:
                continue
            
            ppe_detections = detect_ppe(person_cropped)[0]
            for ppe_det in ppe_detections.boxes:
                transformed_bbox = map_coords(ppe_det.xyxy[0], bbox, person_cropped.shape)
                ppe_bboxes.append(transformed_bbox)
                ppe_labels.append(ppe_detections.names[int(ppe_det.cls[0])])
                ppe_confs.append(float(ppe_det.conf[0]))
        
        img = annotate_image(img, person_bboxes, person_labels, person_confs, (0, 255, 0))
        img = annotate_image(img, ppe_bboxes, ppe_labels, ppe_confs, (0, 0, 255))
        
        cv2.imwrite(os.path.join(output_folder, img_name), img)

def main():
    parser = argparse.ArgumentParser(description='Perform person and PPE detection on images')
    parser.add_argument('--input_folder', required=True, help='Path to the input image directory')
    parser.add_argument('--output_folder', required=True, help='Directory to save processed images')
    parser.add_argument('--person_model', required=True, help='Path to the trained person detection model')
    parser.add_argument('--ppe_model', required=True, help='Path to the trained PPE detection model')
    args = parser.parse_args()
    
    process_images(args.input_folder, args.output_folder, args.person_model, args.ppe_model)

if __name__ == '__main__':
    main()
