import os
import xml.etree.ElementTree as ET
import argparse
from pathlib import Path

def voc_to_yolo(size, box):
    """Convert PascalVOC bounding box format to YOLO format."""
    dw = 1.0 / size[0]
    dh = 1.0 / size[1]
    x_center = (box[0] + box[2]) / 2.0 * dw
    y_center = (box[1] + box[3]) / 2.0 * dh
    width = (box[2] - box[0]) * dw
    height = (box[3] - box[1]) * dh
    return x_center, y_center, width, height

def process_annotation(xml_file, output_file, class_dict):
    """Parse a PascalVOC XML file and convert annotations to YOLO format."""
    tree = ET.parse(xml_file)
    root = tree.getroot()
    img_size = root.find("size")
    img_w = int(img_size.find("width").text)
    img_h = int(img_size.find("height").text)
    
    with open(output_file, "w") as yolo_file:
        for obj in root.findall("object"):
            class_name = obj.find("name").text
            if class_name not in class_dict:
                continue
            class_idx = class_dict[class_name]
            bbox = obj.find("bndbox")
            b_coords = (
                float(bbox.find("xmin").text),
                float(bbox.find("ymin").text),
                float(bbox.find("xmax").text),
                float(bbox.find("ymax").text),
            )
            yolo_coords = voc_to_yolo((img_w, img_h), b_coords)
            yolo_file.write(f"{class_idx} {' '.join(map(str, yolo_coords))}\n")

def convert_annotations(input_dir, output_dir):
    """Convert all PascalVOC annotations in a directory to YOLO format."""
    os.makedirs(output_dir, exist_ok=True)
    
    class_mapping = {}
    with open(os.path.join(input_dir, "classes.txt"), "r") as class_file:
        for idx, cls_name in enumerate(class_file):
            class_mapping[cls_name.strip()] = idx
    
    annotations_path = os.path.join(input_dir, "labels")
    for annotation in os.listdir(annotations_path):
        if annotation.endswith(".xml"):
            xml_filepath = os.path.join(annotations_path, annotation)
            yolo_filepath = os.path.join(output_dir, annotation.replace(".xml", ".txt"))
            process_annotation(xml_filepath, yolo_filepath, class_mapping)
    
    print(f"Conversion completed. YOLO annotations are saved in: {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="PascalVOC to YOLO Format Converter")
    parser.add_argument("input_dir", help="Directory containing PascalVOC annotations")
    parser.add_argument("output_dir", help="Output directory for YOLO annotations")
    args = parser.parse_args()
    
    convert_annotations(args.input_dir, args.output_dir)
    
if __name__ == "__main__":
    main()
