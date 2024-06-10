import os
import time
import pandas as pd
from tqdm import tqdm
from internetarchive import search_items, get_item
import asyncio
import sys; sys.path.insert(0, '../scripts/')
import get_bounding_boxes_from_html as bb

# Directory to save downloaded files and results
DOWNLOAD_DIR = '../data/data-html'
RESULTS_DIR = '../data/bounding-box-results'
os.makedirs(DOWNLOAD_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)


# Function to handle retries for downloading files
def download_with_retry(file, download_dir, retries=3, delay=5):
    for i in range(retries):
        try:
            file.download(download_dir)
            return True
        except Exception as e:
            print(f"Download failed ({i+1}/{retries}): {e}")
            time.sleep(delay)
    return False

# Function to process a single collection
async def process_collection(collection_id, page):
    search = search_items(f'collection:{collection_id}')
    collection_results = list(search)
    
    html_file_count = 0
    for result in tqdm(collection_results, desc=f'Processing {collection_id}'):
        try:
            item = get_item(identifier=result['identifier'], request_kwargs={"timeout":30})
            item_files = list(item.get_files())
            html_files = list(filter(lambda x: x.name.endswith('html'), item_files))
            
            for html_file in html_files: 
                # Download HTML file
                try:
                    download_path = os.path.join(DOWNLOAD_DIR, html_file.name)
                    if download_with_retry(html_file, download_path):
                        html_file_count += 1
                        
                        # Apply bounding box algorithm
                        browser_fp = f'file://{os.getcwd()}/{download_path}'
                        bounding_box = await bb.get_bounding_box_one_file(page, file=browser_fp)
                        
                        # Save results
                        result_path = os.path.join(RESULTS_DIR, f'{html_file.name}.csv')
                        bounding_box_df = pd.DataFrame.from_dict(bounding_box["bounding_boxes"])
                        bounding_box_df.to_csv(result_path, index=False)
                except Exception as e:
                    print(f"Error processing homepage {html_file}: {e}")

        except Exception as e:
            print(f"Error processing homepage group {result['identifier']}: {e}")
    
    return html_file_count

async def main():
    # Initialize Playwright browser
    page, browser, playwright = await bb.instantiate_new_page_object(headless=True, block_external_files=True)

    # Process all collections
    all_html_file_count = 0
    collections_to_process = ['news-homepages']  # Add other collections as needed

    for collection_id in collections_to_process:
        html_file_count = await process_collection(collection_id, page)
        all_html_file_count += html_file_count
        print(f"Processed {html_file_count} HTML files from collection {collection_id}")

    print(f"Total HTML files processed: {all_html_file_count}")

    # Close the Playwright browser
    await browser.close()

# Run the async main function
asyncio.run(main())
