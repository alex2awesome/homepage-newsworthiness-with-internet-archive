import os
import time
import pandas as pd
import argparse
import random
import logging
from tqdm import tqdm
from internetarchive import search_items, get_item
import asyncio
import sys; sys.path.insert(0, '../scripts/')
import get_bounding_boxes_from_html as bb

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s (%(filename)s:%(lineno)d)')

# Directory to save downloaded files and results
DOWNLOAD_DIR = '../data/data-html'
RESULTS_DIR = '../data/bounding-box-results'
LOG_FILE = 'downloaded_files.txt'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

<<<<<<< HEAD
=======
# Load the list of downloaded files
if os.path.exists(LOG_FILE):
    with open(LOG_FILE, 'r') as f:
        downloaded_files = set(f.read().splitlines())
else:
    downloaded_files = set()


# Function to check if a file is already downloaded
def is_file_downloaded(filename):
    return filename in downloaded_files

# Function to log the downloaded file
def log_downloaded_file(filename):
    downloaded_files.add(filename)
    with open(LOG_FILE, 'a') as f:
        f.write(filename + '\n')

for root, dirs, files in os.walk("../data"):
    for file in files:
        if not is_file_downloaded(file):
            log_downloaded_file(file)
>>>>>>> d2d39bd8e1bbe348ffaf9691f02ba34570493e68

# Function to handle retries for downloading files
def download_with_retry(file, download_dir, retries=3, delay=5):
    for i in range(retries):
        try:
            file.download(download_dir)
            return True
        except Exception as e:
            logging.error(f"Download failed ({i+1}/{retries}): {e}")
            time.sleep(delay)
    return False

async def download_html_files(item_files, subcollection_id, page):
    html_files = list(filter(lambda x: x.name.endswith('html'), item_files))
    random.shuffle(html_files)
    SPECIFIC_DOWNLOAD_HTML_DIR = os.path.join(DOWNLOAD_DIR, os.path.join(subcollection_id, "html"))
    SPECIFIC_RESULTS_DIR = os.path.join(RESULTS_DIR, subcollection_id)
    os.makedirs(SPECIFIC_RESULTS_DIR, exist_ok=True)

    os.makedirs(SPECIFIC_DOWNLOAD_HTML_DIR, exist_ok=True)
    html_file_count = 0
    for html_file in tqdm(html_files):
        # Download HTML file if it doesn't exist
        download_path = os.path.join(SPECIFIC_DOWNLOAD_HTML_DIR, html_file.name)
        if not os.path.exists(download_path):
            if download_with_retry(html_file, download_path):
                html_file_count += 1

                # Apply bounding box algorithm
                browser_fp = f'file://{os.getcwd()}/{download_path}'
                try:
                    bounding_box = await bb.get_bounding_box_one_file(page, file=browser_fp)

                    # Save results
                    result_path = os.path.join(SPECIFIC_RESULTS_DIR, f'{html_file.name}.csv')
                    bounding_box_df = pd.DataFrame.from_dict(bounding_box["bounding_boxes"])
                    bounding_box_df.to_csv(result_path, index=False)
                except Exception as e:
                    logging.error(f"Error processing homepage {html_file}: {e}")


async def download_jpg_files(item_files, subcollection_id):
    jpg_files = list(filter(lambda x: x.name.endswith('fullpage.jpg'), item_files))
    random.shuffle(jpg_files)
    SPECIFIC_DOWNLOAD_JPG_DIR = os.path.join(DOWNLOAD_DIR, os.path.join(subcollection_id, "jpg"))
    os.makedirs(SPECIFIC_DOWNLOAD_JPG_DIR, exist_ok=True)
    for jpg_file in tqdm(jpg_files):
        # Download JPG file if it doesn't exist
        download_path = os.path.join(SPECIFIC_DOWNLOAD_JPG_DIR, jpg_file.name)
        if not os.path.exists(download_path):
            if not download_with_retry(jpg_file, download_path):
                logging.error(f"Failed to download {jpg_file.name} after multiple attempts.")


async def download_json_files(item_files, subcollection_id):
    json_files = list(filter(lambda x: x.name.endswith('hyperlinks.json'), item_files))
    random.shuffle(json_files)
    SPECIFIC_DOWNLOAD_JSON_DIR = os.path.join(DOWNLOAD_DIR, os.path.join(subcollection_id, "hyperlink_json"))
    os.makedirs(SPECIFIC_DOWNLOAD_JSON_DIR, exist_ok=True)
    for jpg_file in tqdm(json_files):
        # Download JPG file if it doesn't exist
        download_path = os.path.join(SPECIFIC_DOWNLOAD_JSON_DIR, jpg_file.name)
        if not os.path.exists(download_path):
            if not download_with_retry(jpg_file, download_path):
                logging.error(f"Failed to download {jpg_file.name} after multiple attempts.")



# Function to process a single collection
async def process_collection(
    collection_id, 
    page,
    start_idx=None,
    end_idx=None,
    download_html=True,
    download_jpg=True,
    download_json=True
):
    search = search_items(f'collection:{collection_id}')
    collection_results = list(search)

    # Shuffle the collections to avoid race conditions
    random.shuffle(collection_results)
    
    # Slice the collections if start_idx and end_idx are provided
    if start_idx is not None and end_idx is not None:
        collection_results = collection_results[start_idx:end_idx]

    html_file_count = 0
    for result in tqdm(collection_results, desc=f'Processing {collection_id}'):
        try:
            subcollection_id = result['identifier']
            logging.info(f'Fetching from: {subcollection_id}')
            item = get_item(identifier=subcollection_id, request_kwargs={"timeout": 300})
            item_files = list(item.get_files())
<<<<<<< HEAD

            if download_html:
                await download_html_files(item_files, subcollection_id, page)

            if download_jpg:
                await download_jpg_files(item_files, subcollection_id)

            if download_json:
                await download_json_files(item_files, subcollection_id)
=======
            
            html_files = list(filter(lambda x: x.name.endswith('html'), item_files))
            jpg_files = list(filter(lambda x: x.name.endswith('fullpage.jpg'), item_files))
            random.shuffle(html_files)
            random.shuffle(jpg_files)
            SPECIFIC_DOWNLOAD_HTML_DIR = os.path.join(DOWNLOAD_DIR, os.path.join(subcollection_id, "html"))
            SPECIFIC_DOWNLOAD_JPG_DIR = os.path.join(DOWNLOAD_DIR, os.path.join(subcollection_id, "jpg"))
            SPECIFIC_RESULTS_DIR = os.path.join(RESULTS_DIR, subcollection_id)
            os.makedirs(SPECIFIC_DOWNLOAD_HTML_DIR, exist_ok=True)
            os.makedirs(SPECIFIC_DOWNLOAD_JPG_DIR, exist_ok=True)
            os.makedirs(SPECIFIC_RESULTS_DIR, exist_ok=True)
            for html_file in tqdm(html_files):
                # Download HTML file if it doesn't exist
                download_path = os.path.join(SPECIFIC_DOWNLOAD_HTML_DIR, html_file.name)
                if not os.path.exists(download_path) and not is_file_downloaded(html_file.name):
                    if download_with_retry(html_file, download_path):
                        html_file_count += 1
                        log_downloaded_file(html_file.name)
                        
                        # Apply bounding box algorithm
                        browser_fp = f'file://{os.getcwd()}/{download_path}'
                        try:
                            bounding_box = await bb.get_bounding_box_one_file(page, file=browser_fp)
                            
                            # Save results
                            result_path = os.path.join(SPECIFIC_RESULTS_DIR, f'{html_file.name}.csv')
                            bounding_box_df = pd.DataFrame.from_dict(bounding_box["bounding_boxes"])
                            bounding_box_df.to_csv(result_path, index=False)
                        except Exception as e:
                            logging.error(f"Error processing homepage {html_file}: {e}")
            
            for jpg_file in tqdm(jpg_files):
                # Download JPG file if it doesn't exist
                download_path = os.path.join(SPECIFIC_DOWNLOAD_JPG_DIR, jpg_file.name)
                if not os.path.exists(download_path) and not is_file_downloaded(jpg_file.name):
                    if download_with_retry(jpg_file, download_path):
                        log_downloaded_file(jpg_file.name)
                    else:
                        logging.error(f"Failed to download {jpg_file.name} after multiple attempts.")
>>>>>>> d2d39bd8e1bbe348ffaf9691f02ba34570493e68

        except Exception as e:
            logging.error(f"Error processing homepage group {result['identifier']}: {e}")
    
    return html_file_count

# Main function to process collections and handle arguments
async def main(args):
    page, browser, playwright = None, None, None
    try:
        # Initialize Playwright browser
        page, browser, playwright = await bb.instantiate_new_page_object(headless=True, block_external_files=True)

        # Process all collections
        all_html_file_count = 0
        collections_to_process = ['news-homepages']  # Add other collections as needed

        for collection_id in collections_to_process:
            html_file_count = await process_collection(collection_id, page, start_idx=args.start_idx, end_idx=args.end_idx)
            all_html_file_count += html_file_count
            logging.info(f"Processed {html_file_count} HTML files from collection {collection_id}")

        logging.info(f"Total HTML files processed: {all_html_file_count}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")
    finally:
        # Close the Playwright browser if it was initialized
        if browser:
            await browser.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process news homepage collections and download the homepage files.')
    parser.add_argument('--start_idx', type=int, default=None, help='Start index for processing collections')
    parser.add_argument('--end_idx', type=int, default=None, help='End index for processing collections')
    parser.add_argument('--download_html', action='store_true')
    parser.add_argument('--download_jpg', action='store_true')
    parser.add_argument('--download_json', action='store_true')
    args = parser.parse_args()

    # Run the async main function
    asyncio.run(main(args))
