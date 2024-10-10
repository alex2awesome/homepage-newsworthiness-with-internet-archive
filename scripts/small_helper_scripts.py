## these are small scripts I wrote/ran in my interpreter to do some basic file manipulation.
# # 
# script to move files into directories
from tqdm.auto import tqdm
import glob
all_files_1 = glob.glob('data/data-html/*.*')
for fp in tqdm(all_files_1):
    b = fp.split('/')[-1]
    f = b.split('-')[0]
    import os
    if not os.path.exists('data/data-html/' + f):
        os.mkdir('data/data-html/' + f)
    os.rename(fp, f'data/data-html/{f}/{b}')



### 
## 
# script to merge all files across different types that have filename as their key 
import glob
import shutil
import os 
import pandas as pd 

# get all lists of files 
data_dir = 'data-html'
data_dir = 'html-bb-jpg-samples'
htmls = glob.glob(f'{data_dir}/*/*.html') + glob.glob(f'{data_dir}/*.html')
jpgs = glob.glob(f'{data_dir}/*/*.jpg') + glob.glob(f'{data_dir}/*.jpg')
bounding_boxes = glob.glob('bounding-box-results/*')
html_s = pd.Series(htmls).to_frame('html_paths')
jpg_s = pd.Series(jpgs).to_frame('jpg_paths')
bounding_box_s = pd.Series(bounding_boxes).to_frame('bb_paths')
# make a unique identifying key
html_s['key'] = html_s['html_paths'].str.split('/').str.get(-1).str.replace('.html', '')
jpg_s['key'] = jpg_s['jpg_paths'].str.split('/').str.get(-1).str.replace('.fullpage.jpg', '')
bounding_box_s['key'] = bounding_box_s['bb_paths'].str.split('/').str.get(-1).str.replace('.html.csv', '')
# merge 
full_df = html_s.merge(jpg_s).merge(bounding_box_s).drop_duplicates()
# copy 
for _, (h, b, j) in full_df.sample(1000)[['html_paths', 'bb_paths', 'jpg_paths']].iterrows():
    shutil.copy(h, os.path.expanduser('~/html-bb-jpg-samples/') + h.split('/')[-1])
    shutil.copy(b, os.path.expanduser('~/html-bb-jpg-samples/') + b.split('/')[-1])
    shutil.copy(j, os.path.expanduser('~/html-bb-jpg-samples/') + j.split('/')[-1])




import glob
from tqdm.auto import tqdm
bbs = glob.glob('bounding-box-results/*.html.csv')
jpgs = glob.glob('data-html/*.fullpage.jpg')
for bb in tqdm(bbs):
   filename = bb.split('/')[1].replace('.html.csv', '')
   jpg_fname = f'data-html/{filename}.fullpage.jpg'
   if jpg_fname in jpgs:
       shutil.move(jpg_fname, 'bounding-box-results/')





## delete all checkpoints:

import pandas as pd
import glob
import os
import shutil

checkpoints = glob.glob('**/*checkpoint*', recursive=True)
c_df = pd.Series(checkpoints).to_frame('full_path')
c_df['filename'] = c_df['full_path'].apply(os.path.basename)
c_df['dirname'] = c_df['full_path'].apply(os.path.dirname)
c_df['checkpoint_num'] = c_df['filename'].str.replace('checkpoint-', '').astype(int, errors='ignore')
c_df = c_df.loc[lambda df: df['checkpoint_num'].apply(lambda x: x.isdigit())].assign(checkpoint_num=lambda df: df['checkpoint_num'].astype(int))
checkpoints_to_delete = c_df.loc[lambda df: df.groupby('dirname')['checkpoint_num'].transform(max) != df['checkpoint_num']]
for f in checkpoints_to_delete['full_path']:
    shutil.rmtree(f)
