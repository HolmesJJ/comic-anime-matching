import os
import cv2
import pandas as pd

from datetime import datetime
from datetime import timedelta
from moviepy.video.io.VideoFileClip import VideoFileClip


ANIME_DIR = 'anime'
COMIC_DIR = 'comic'
OUTPUT_DIR = 'output'
FRAME_RATE = 24


def parse_timestamp(timestamp):
    try:
        time_part, frames = timestamp[:-3], int(timestamp[-2:])
        parsed_time = datetime.strptime(time_part, '%H:%M:%S') + timedelta(milliseconds=(frames * 1000 / FRAME_RATE))
        return parsed_time
    except Exception as e:
        print('Invalid timestamp', e)
        return None


def timestamp_to_seconds(timestamp):
    time_part, frames = timestamp[:-3], int(timestamp[-2:])
    h, m, s = map(int, time_part.split(':'))
    total_seconds = h * 3600 + m * 60 + s + frames / FRAME_RATE
    return total_seconds


def check_timestamps(data_df, tolerance=1e-5):
    data_df['Parsed Start Timestamp'] = data_df['Start Timestamp'].apply(
        lambda x: parse_timestamp(x) if pd.notnull(x) else None)
    data_df['Parsed End Timestamp'] = data_df['End Timestamp'].apply(
        lambda x: parse_timestamp(x) if pd.notnull(x) else None)
    violations = []
    for i in range(len(data_df) - 1):
        current_end = data_df.at[i, 'Parsed End Timestamp']
        next_start = data_df.at[i + 1, 'Parsed Start Timestamp']
        if current_end and next_start:
            expected_next_start = current_end + timedelta(milliseconds=(1000 / FRAME_RATE))
            if abs((next_start - expected_next_start).total_seconds()) > tolerance:
                if not pd.isnull(data_df.at[i + 1, 'Comic Block ID']):
                    violations.append((i, i + 1))
    for curr_idx, next_idx in violations:
        print(f'Violation between rows {curr_idx} and {next_idx}:')
        print(data_df.iloc[[curr_idx, next_idx]][['Comic Block ID', 'Start Timestamp', 'End Timestamp']])


def extract_video_clips(data_df, video_file):
    with VideoFileClip(video_file) as video:
        for _, row in data_df.iterrows():
            if pd.isna(row['Start Timestamp']) or pd.isna(row['End Timestamp']):
                continue
            start_time = timestamp_to_seconds(row['Start Timestamp'])
            end_time = timestamp_to_seconds(row['End Timestamp'])
            video_id = str(row['Video ID'])
            comic_block_id = row['Comic Block ID']
            print(f'{video_id}_{comic_block_id}:', start_time, end_time)
            output_path = os.path.join(OUTPUT_DIR, f'{video_id}_{comic_block_id}.mp4')
            try:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                print(f'Saved clip: {output_path}')
            except Exception as e:
                print(f'Failed to process {video_id}_{comic_block_id}: {e}')


def display_comic_and_video(data_df):
    max_width = 480
    for _, row in data_df.iterrows():
        if pd.isna(row['Start Timestamp']) or pd.isna(row['End Timestamp']):
            continue
        video_id = str(row['Video ID'])
        comic_block_id = row['Comic Block ID']
        parts = comic_block_id.split('_')
        comic_block_path = os.path.join(parts[0], f"page_{parts[1][4:]}", f"{parts[2]}.jpg")
        image_path = os.path.join(COMIC_DIR, comic_block_path)
        video_path = os.path.join(OUTPUT_DIR, f'{video_id}_{comic_block_id}.mp4')
        if not os.path.exists(image_path):
            print(f'Image not found: {image_path}')
            continue
        if not os.path.exists(video_path):
            print(f'Video not found: {video_path}')
            continue
        image = cv2.imread(image_path)
        img_aspect_ratio = image.shape[0] / image.shape[1]
        img_height = int(max_width * img_aspect_ratio)
        image = cv2.resize(image, (max_width, img_height))
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f'Failed to open video: {video_path}')
            continue
        vid_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        vid_aspect_ratio = vid_height / vid_width
        vid_resized_height = int(max_width * vid_aspect_ratio)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            video_frame = cv2.resize(frame, (max_width, vid_resized_height))
            combined_height = max(img_height, vid_resized_height)
            img_padded = cv2.copyMakeBorder(image, 0, combined_height - img_height, 0, 0, cv2.BORDER_CONSTANT,
                                            value=[0, 0, 0])
            video_padded = cv2.copyMakeBorder(video_frame, 0, combined_height - vid_resized_height, 0, 0,
                                              cv2.BORDER_CONSTANT, value=[0, 0, 0])
            combined_frame = cv2.hconcat([img_padded, video_padded])
            cv2.imshow(f'Comic and Video Viewer: {video_id}_{comic_block_id}', combined_frame)
            if cv2.waitKey(int(1000 / 24)) & 0xFF == ord('q'):
                break
        cap.release()
        cv2.destroyAllWindows()


def run():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_file = os.path.join(ANIME_DIR, '001.mp4')
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, '001.csv'), dtype={'Video ID': str})
    print(parse_timestamp('00:03:37:18'))
    check_timestamps(data_df)
    extract_video_clips(data_df, video_file)
    display_comic_and_video(data_df)


if __name__ == '__main__':
    run()
