import argparse
import csv
import json
import os
import cv2
from glob import glob

def csv_to_coco(csv_dir, image_dir, json_file):
    images = []
    annotations = []
    categories = {
        "text": 1  # Define a category ID for text
    }
    annotation_id = 1
    image_id = 0

    # Get all jpg files in the image directory
    jpg_files = glob(os.path.join(image_dir, '*.jpeg'))
    for jpg_file in jpg_files:
        try:
            image = cv2.imread(jpg_file)
            height, width = image.shape[:2]
            file_name = os.path.basename(jpg_file)
            image_entry = {
                "id": image_id,
                "file_name": file_name,
                "width": width, 
                "height": height
            }

            # Find the corresponding CSV file
            csv_file = os.path.join(csv_dir, file_name.replace('.jpeg', '.csv'))
            if not os.path.exists(csv_file):
                print(f"Warning: {csv_file} does not exist. Skipping this image.")
                continue

            with open(csv_file, newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    annotations.append({
                        "id": annotation_id,
                        "image_id": image_id,
                        "category_id": categories["text"],
                        "bbox": [float(row["x"]), float(row["y"]), float(row["width"]), float(row["height"])],
                        "area": float(row["width"]) * float(row["height"]),
                        "iscrowd": 0,
                        "link_text": row["link_text"],
                        "is_article": row["is_article"] == "True",
                        "all_text": row["all_text"]
                    })
                    annotation_id += 1

            images.append(image_entry)
            image_id += 1
        except Exception as e:
            print(f"Error in translating {jpg_file}: {e}")

    categories_list = [{"id": id, "name": name} for name, id in categories.items()]
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }

    with open(json_file, 'w') as f:
        json.dump(coco_format, f)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Convert CSV annotations to COCO JSON format.')
    parser.add_argument('--csv_dir', type=str, help='Directory containing your CSV files')
    parser.add_argument('--image_dir', type=str, help='Directory containing your JPG images')
    parser.add_argument('--json_file', type=str, help='Output JSON file')

    args = parser.parse_args()

    csv_to_coco(args.csv_dir, args.image_dir, args.json_file)
