import os
import xml.etree.ElementTree as ET

# Folders
ANNOTATIONS_DIR = "pothole_dataset/annotations"
OUTPUT_DIR = "pothole_dataset/labels"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset info
CLASS_NAME = "pothole"
CLASS_ID = 0  # since only one class

for xml_file in os.listdir(ANNOTATIONS_DIR):
    if not xml_file.endswith(".xml"):
        continue

    xml_path = os.path.join(ANNOTATIONS_DIR, xml_file)
    tree = ET.parse(xml_path)
    root = tree.getroot()

    img_width = int(root.find("size/width").text)
    img_height = int(root.find("size/height").text)

    txt_filename = xml_file.replace(".xml", ".txt")
    txt_path = os.path.join(OUTPUT_DIR, txt_filename)

    with open(txt_path, "w") as f:
        for obj in root.findall("object"):
            name = obj.find("name").text
            if name.lower() != CLASS_NAME:
                continue

            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text)
            ymin = float(bbox.find("ymin").text)
            xmax = float(bbox.find("xmax").text)
            ymax = float(bbox.find("ymax").text)

            # Convert to YOLO normalized format
            x_center = ((xmin + xmax) / 2) / img_width
            y_center = ((ymin + ymax) / 2) / img_height
            width = (xmax - xmin) / img_width
            height = (ymax - ymin) / img_height

            f.write(f"{CLASS_ID} {x_center} {y_center} {width} {height}\n")

print("✅ Conversion complete! YOLO label files saved in:", OUTPUT_DIR)
