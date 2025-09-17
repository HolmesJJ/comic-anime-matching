import os
import pandas as pd

from tqdm import tqdm
from dotenv import load_dotenv


load_dotenv()

COMIC = os.getenv('COMIC')
ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), COMIC, '1')
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), COMIC, '1')

FRAME_RATE = 24


def run(video_id):
    file_name = f'{video_id}.csv'
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, file_name), dtype={'Video ID': str})
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df)):
        block_id = row['Comic Block ID']
        if not pd.isna(block_id):
            comic_id = block_id.split('_')[0]
            page = block_id.split('_')[2]
            image_id = block_id.split('_')[-1]
            image_path = os.path.join(COMIC_DIR, comic_id, f'page_{page}', f'{image_id}.jpg')
            if not os.path.exists(image_path):
                print('\n', row, image_path)


if __name__ == '__main__':
    for i in range(1, 101):
        run(f'{i:03d}')
