import csv
import json
import os
import cv2
from glob import glob
from tqdm import tqdm
from multiprocessing import Pool, Manager
import argparse

def process_image(args):
    jpg_file, csv_dir, image_id = args
    categories = {
        "text": 1  # Define a category ID for text
    }
    annotations = []
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
        csv_file = os.path.join(csv_dir, file_name.replace('.fullpage.jpg', '.html.csv'))
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} does not exist. Skipping this image.")
            return None, None

        with open(csv_file, newline='') as f:
            reader = csv.DictReader(f)
            annotation_id = image_id * 1000  # Ensure unique annotation IDs
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

        return image_entry, annotations
    except Exception as e:
        print(f"Error in translating {jpg_file}: {e}")
        return None, None

def csv_to_coco(csv_dir, image_dir, json_file):
    images = []
    annotations = []
    image_id = 0

    # Get all jpg files in the image directory
    jpg_files = glob(os.path.join(image_dir, '*.jpg'))

    # Create a pool of workers
    with Manager() as manager:
        pool = Pool()
        args = [(jpg_file, csv_dir, image_id + i) for i, jpg_file in enumerate(jpg_files)]
        
        # Process images in parallel
        results = list(tqdm(pool.imap(process_image, args), total=len(jpg_files), desc="CSV2Coco"))
        
        pool.close()
        pool.join()

        for result in results:
            if result[0] is not None and result[1] is not None:
                images.append(result[0])
                annotations.extend(result[1])
                image_id += 1

    categories_list = [{"id": id, "name": name} for name, id in {"text": 1}.items()]
    coco_format = {
        "images": images,
        "annotations": annotations,
        "categories": categories_list
    }

    with open(json_file, 'w') as f:
        json.dump(coco_format, f)
    print("Done: " + str(image_id))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert CSV annotations to COCO format")
    parser.add_argument('--csv_dir', required=True, help="Directory containing your CSV files")
    parser.add_argument('--image_dir', required=True, help="Directory containing your JPG images")
    parser.add_argument('--json_file', required=True, help="Output JSON file")
    
    args = parser.parse_args()
    
    csv_to_coco(args.csv_dir, args.image_dir, args.json_file)
