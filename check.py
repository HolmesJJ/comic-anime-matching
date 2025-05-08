import os
import cv2
import shutil
import argparse
import pandas as pd

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips


ANIME_DIR = os.path.join('anime', 'Slam Dunk')
COMIC_DIR = os.path.join('comic', 'Slam Dunk')
OUTPUT_DIR = os.path.join('output', 'Slam Dunk')
FRAME_RATE = 24


# ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=nb_read_frames -of default=nokey=1:noprint_wrappers=1 001.mp4
def get_total_frames(video_id):
    clip = VideoFileClip(os.path.join(ANIME_DIR, f'{video_id}.mp4'))
    print('fps:', clip.fps)
    print('duration:', clip.duration)
    total_frames = clip.fps * clip.duration
    return total_frames


def parse_timestamp(timestamp):
    try:
        time_part, frames = timestamp[:-3], int(timestamp[-2:])
        parsed_time = datetime.strptime(time_part, '%H:%M:%S') + timedelta(milliseconds=(frames * 1000 / FRAME_RATE))
        return parsed_time
    except Exception as e:
        print('Invalid timestamp', e)
        return None


def is_valid_timestamp(timestamp):
    if timestamp is None or pd.isnull(timestamp):
        return False
    return timestamp.date() != datetime(1900, 1, 1).date()


def timestamp_to_seconds(timestamp):
    time_part, frames = timestamp[:-3], int(timestamp[-2:])
    h, m, s = map(int, time_part.split(':'))
    total_seconds = h * 3600 + m * 60 + s + frames / FRAME_RATE
    return total_seconds


def check_timestamps(data_df, use_keyframe_from_video, tolerance=1e-5):
    data_df['Parsed Start Timestamp'] = data_df['Start Timestamp'].apply(
        lambda x: parse_timestamp(x) if pd.notnull(x) else None)
    data_df['Parsed End Timestamp'] = data_df['End Timestamp'].apply(
        lambda x: parse_timestamp(x) if pd.notnull(x) else None)
    if use_keyframe_from_video:
        data_df['Parsed Key Timestamp'] = data_df['Key Timestamp'].apply(
            lambda x: parse_timestamp(x) if pd.notnull(x) else None)
    violations1 = []
    violations2 = []
    violations3 = []
    for i in range(len(data_df)):
        current_start = data_df.at[i, 'Parsed Start Timestamp']
        current_end = data_df.at[i, 'Parsed End Timestamp']
        next_start = None
        if i < len(data_df) - 1:
            next_start = data_df.at[i + 1, 'Parsed Start Timestamp']
        if is_valid_timestamp(current_start) and is_valid_timestamp(current_end) and current_start >= current_end:
            violations1.append(i)
        if is_valid_timestamp(current_end) and is_valid_timestamp(next_start):
            expected_next_start = current_end + timedelta(milliseconds=(1000 / FRAME_RATE))
            if abs((next_start - expected_next_start).total_seconds()) > tolerance:
                if not use_keyframe_from_video and not pd.isnull(data_df.at[i + 1, 'Comic Block ID']):
                    violations2.append((i, i + 1))
        if use_keyframe_from_video:
            key_ts = data_df.at[i, 'Parsed Key Timestamp']
            if is_valid_timestamp(key_ts) and is_valid_timestamp(current_start) and is_valid_timestamp(current_end):
                if not (current_start <= key_ts <= current_end):
                    violations3.append(i)
    for idx in violations1:
        print(f'Invalid timestamp range in row {idx}: Start >= End')
        print(data_df.iloc[idx][['Start Timestamp', 'End Timestamp']])
    for curr_idx, next_idx in violations2:
        print(f'Violation between rows {curr_idx} and {next_idx}:')
        print(data_df.iloc[[curr_idx, next_idx]][['Start Timestamp', 'End Timestamp']])
    for idx in violations3:
        print(f'Key Timestamp not in range at row {idx}:')
        print(data_df.iloc[idx][['Start Timestamp', 'End Timestamp', 'Key Timestamp']])
    return len(violations1) == 0 and len(violations2) == 0 and len(violations3) == 0


def analyze_comic_blocks(data_df, video_id):
    data_df = data_df.dropna(subset=['Comic Block ID'])
    data_df['Comic ID'] = data_df['Comic Block ID'].str.split('_').str[0]
    data_df['Page'] = data_df['Comic Block ID'].str.split('_').str[2]
    data_df['Image ID'] = data_df['Comic Block ID'].str.split('_').str[-1]
    results = []
    for comic_id, comic_group in data_df.groupby('Comic ID'):
        comic_id_dir = os.path.join(COMIC_DIR, comic_id)
        grouped = comic_group.groupby('Page')['Image ID'].apply(set).to_dict()
        for page, images_in_csv in grouped.items():
            page_folder = os.path.join(comic_id_dir, f'page_{page}')
            all_files = os.listdir(page_folder)
            image_files = set()
            for file_name in all_files:
                image_files.add(os.path.splitext(file_name)[0])
            present_in_csv = images_in_csv & image_files
            missing_in_csv = sorted(image_files - images_in_csv)
            results.append([
                video_id,
                comic_id,
                page,
                len(present_in_csv),
                len(missing_in_csv),
                len(image_files),
                ','.join(missing_in_csv)
            ])
    result_df = pd.DataFrame(results, columns=[
        'Video ID', 'Comic ID', 'Page', 'Images in CSV', 'Images Missing', 'Total Images', 'Missing Image Names'
    ])
    result_df.to_csv(os.path.join(OUTPUT_DIR, f'{video_id}_summary.csv'), index=False, encoding='utf-8')


def extract_video_clips(data_df, video_file, use_keyframe_from_video):
    with VideoFileClip(video_file) as video:
        for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Extracting clips'):
            if pd.isna(row['Start Timestamp']) or pd.isna(row['End Timestamp']):
                continue
            start_time = timestamp_to_seconds(row['Start Timestamp'])
            end_time = timestamp_to_seconds(row['End Timestamp'])
            video_id = str(row['Video ID'])
            if use_keyframe_from_video:
                block_id = f'{idx + 1:03d}'
            else:
                block_id = row['Comic Block ID']
            print(f'{video_id}_{block_id}:', start_time, end_time)
            os.makedirs(os.path.join(OUTPUT_DIR, video_id), exist_ok=True)  # f'{video_id}_updated'
            output_path = os.path.join(OUTPUT_DIR, video_id, f'{block_id}.mp4')  # f'{video_id}_updated'
            try:
                clip = video.subclip(start_time, end_time)
                clip.write_videofile(output_path, codec='libx264', audio_codec='aac')
                print(f'Saved clip: {output_path}')
            except Exception as e:
                print(f'Failed to process {video_id}_{block_id}: {e}')


def extract_frame_from_video(video_file, timestamp_str):
    timestamp_sec = timestamp_to_seconds(timestamp_str)
    with VideoFileClip(video_file) as clip:
        frame = clip.get_frame(timestamp_sec)
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
        return frame


def display_comic_and_video(data_df, video_id, video_file, use_keyframe_from_video, fps=24):
    max_width = 480
    temp_dir = os.path.join(OUTPUT_DIR, 'temp')
    os.makedirs(temp_dir, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_clips = []
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Generating clips'):
        if pd.isna(row['Start Timestamp']) or pd.isna(row['End Timestamp']):
            continue
        if use_keyframe_from_video:
            block_id = f'{idx + 1:03d}'
            video_path = os.path.join(OUTPUT_DIR, video_id, f'{block_id}.mp4')  # f'{video_id}_updated'
            image = extract_frame_from_video(video_file, row['Key Timestamp'])
            output_path = os.path.join(temp_dir, f'{video_id}_{block_id}.mp4')
        else:
            block_id = row['Comic Block ID']
            parts = block_id.split('_')
            comic_block_path_jpg = os.path.join(parts[0], f'page_{parts[2]}', f'{parts[3]}.jpg')
            comic_block_path_png = os.path.join(parts[0], f'page_{parts[2]}', f'{parts[3]}.png')
            image_path_jpg = os.path.join(COMIC_DIR, comic_block_path_jpg)
            image_path_png = os.path.join(COMIC_DIR, comic_block_path_png)
            if not os.path.exists(image_path_jpg) and not os.path.exists(image_path_png):
                print(f'Image not found: {image_path_jpg} or {image_path_png}')
                continue
            video_path = os.path.join(OUTPUT_DIR, video_id, f'{block_id}.mp4')  # f'{video_id}_updated'
            if not os.path.exists(video_path):
                print(f'Video not found: {video_path}')
                continue
            image_path = None
            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
            elif os.path.exists(image_path_png):
                image_path = image_path_png
            image = cv2.imread(image_path)
            output_path = os.path.join(temp_dir, f'{video_id}_{block_id}.mp4')
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
        video_writer = None
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
            # cv2.imshow(f'Comic and Video Viewer: {video_id}_{comic_block_id}', combined_frame)
            # if cv2.waitKey(int(1000 / 24)) & 0xFF == ord('q'):
            #     break
            if video_writer is None:
                output_frame_size = (combined_frame.shape[1], combined_frame.shape[0])
                video_writer = cv2.VideoWriter(output_path, fourcc, fps, output_frame_size)
            video_writer.write(combined_frame)
        cap.release()
        # cv2.destroyAllWindows()
        video_writer.release()
        video_clips.append(output_path)
    final_clips = []
    for clip_path in video_clips:
        clip = VideoFileClip(clip_path)
        resized_clip = resize(clip, height=1080, width=1920)
        final_clips.append(resized_clip)
    final_video = concatenate_videoclips(final_clips, method='compose')
    final_video_path = os.path.join(OUTPUT_DIR, f'{video_id}.mp4')  # _updated
    final_video.write_videofile(final_video_path, fps=fps)
    final_video.close()
    for clip in final_clips:
        clip.close()
    shutil.rmtree(temp_dir)


def generate_comic(data_df, video_id, video_file):
    output_images = []
    temp_image_dir = os.path.join(OUTPUT_DIR, 'temp_images')
    os.makedirs(temp_image_dir, exist_ok=True)
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Extracting frames for PDF'):
        if pd.isna(row['Key Timestamp']):
            continue
        frame = extract_frame_from_video(video_file, row['Key Timestamp'])
        image_path = os.path.join(temp_image_dir, f'{idx+1:03d}.jpg')
        cv2.imwrite(image_path, frame)
        output_images.append(image_path)
    composite_images = []
    for i in range(0, len(output_images), 3):
        images = [Image.open(p).convert('RGB') for p in output_images[i:i + 3]]
        widths = [img.width for img in images]
        heights = [img.height for img in images]
        max_width = max(widths)
        total_height = sum(heights)
        combined = Image.new('RGB', (max_width, total_height), (255, 255, 255))
        y_offset = 0
        for img in images:
            combined.paste(img, (0, y_offset))
            y_offset += img.height + 10
        composite_images.append(combined)
    pdf_path = os.path.join(OUTPUT_DIR, f'{video_id}.pdf')  # _updated
    composite_images[0].save(pdf_path, save_all=True, append_images=composite_images[1:])
    print(f'PDF comic saved at: {pdf_path}')
    shutil.rmtree(temp_image_dir)


def run(video_id, use_keyframe_from_video):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    video_file = os.path.join(ANIME_DIR, f'{video_id}.mp4')
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, f'{video_id}.csv'), dtype={'Video ID': str})  # _updated
    # print(parse_timestamp('00:03:37:18'))
    if check_timestamps(data_df, use_keyframe_from_video):
        if not use_keyframe_from_video:
            analyze_comic_blocks(data_df, video_id)
        extract_video_clips(data_df, video_file, use_keyframe_from_video)
        display_comic_and_video(data_df, video_id, video_file, use_keyframe_from_video)
        if use_keyframe_from_video:
            generate_comic(data_df, video_id, video_file)


def check_missing():
    total_missing = 0
    total_images = 0
    for root, dirs, files in os.walk(OUTPUT_DIR):
        for file in files:
            if file.endswith('_summary.csv'):
                summary_path = os.path.join(root, file)
                summary_df = pd.read_csv(summary_path)
                total_missing += summary_df['Images Missing'].sum()
                total_images += summary_df['Total Images'].sum()
    print(f'TOTAL Images Missing: {total_missing}')
    print(f'TOTAL Images: {total_images}')
    print(f'Missing Rate: {total_missing / total_images}')


if __name__ == '__main__':
    # check_missing()
    # print('Total Frames:', get_total_frames('001'))
    run('071', False)
    # parser = argparse.ArgumentParser(description='Process video and comic IDs.')
    # parser.add_argument('-vid', '--video_id', required=True, help="The ID of the video (e.g., '001')")
    # args = parser.parse_args()
    # run(args.video_id, True)
