import os
import cv2
import glob
import shutil
import argparse
import subprocess
import pandas as pd

from PIL import Image
from tqdm import tqdm
from datetime import datetime
from datetime import timedelta
from dotenv import load_dotenv
from moviepy.video.fx.resize import resize
from moviepy.video.io.VideoFileClip import VideoFileClip
from moviepy.video.compositing.concatenate import concatenate_videoclips


load_dotenv()

COMIC = os.getenv('COMIC')
ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), COMIC, '1')
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), COMIC, '1')

FRAME_RATE = 24


def parse_timestamp(timestamp):
    time_part, frames = timestamp[:-3], int(timestamp[-2:])
    parsed_time = datetime.strptime(time_part, '%H:%M:%S') + timedelta(milliseconds=(frames * 1000 / FRAME_RATE))
    return parsed_time


def is_valid_timestamp(timestamp):
    if timestamp is None or pd.isnull(timestamp):
        return False
    return True


def timestamp_to_frame_index(timestamp):
    h, m, s = map(int, timestamp[:8].split(':'))
    frame = int(timestamp[-2:])
    frame_index = (h * 3600 + m * 60 + s) * FRAME_RATE + frame + 1
    return frame_index


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
                if use_keyframe_from_video:
                    violations2.append((i, i + 1))
                else:
                    if not pd.isnull(data_df.at[i + 1, 'Comic Block ID']):
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
    data_df = data_df.dropna(subset=['Comic Block ID']).copy()
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


def get_total_frames(video_id, use_camera_id):
    folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
    frame_folder = os.path.join(OUTPUT_DIR, folder_name)
    frame_files = glob.glob(os.path.join(frame_folder, 'frame_*.png'))
    return len(frame_files)


def extract_frames(video_id, use_camera_id):
    video_file = os.path.join(ANIME_DIR, f'{video_id}.mp4')
    folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
    frame_folder = os.path.join(OUTPUT_DIR, folder_name)
    os.makedirs(frame_folder, exist_ok=True)
    pattern = os.path.join(frame_folder, f'frame_%06d.png')
    cmd = ['ffmpeg', '-i', video_file, '-vsync', '0', pattern]
    # cmd = ['ffmpeg', '-i', video_file, '-vsync', '0', '-q:v', '30', pattern]
    try:
        print(f'{video_id} extract frames...')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'{video_id} extract frames error: {e}')


def delete_frames(video_id, use_camera_id):
    folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
    frame_folder = os.path.join(OUTPUT_DIR, folder_name)
    pattern = os.path.join(frame_folder, f'frame_*.png')
    files = glob.glob(pattern)
    for file in tqdm(files, desc=f'Deleting frames in {folder_name}'):
        os.remove(file)


def extract_video_clips(video_id, data_df, use_keyframe_from_video, use_camera_id):
    folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
    frame_folder = os.path.join(OUTPUT_DIR, folder_name)
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Extracting clips'):
        if pd.isna(row['Start Timestamp']) or pd.isna(row['End Timestamp']):
            continue
        start_frame = timestamp_to_frame_index(row['Start Timestamp'])
        end_frame = timestamp_to_frame_index(row['End Timestamp'])
        if use_keyframe_from_video:
            block_id = f'{idx + 1:03d}'
        else:
            block_id = row['Comic Block ID']
        print(f'{video_id}_{block_id}:', start_frame, end_frame)
        output_path = os.path.join(frame_folder, f'{block_id}.mp4')
        frame_paths = []
        for i in range(start_frame, end_frame + 1):
            frame_file = os.path.join(frame_folder, f'frame_{i:06d}.png')
            frame_paths.append(frame_file)
        first_frame = cv2.imread(frame_paths[0])
        height, width, _ = first_frame.shape
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, FRAME_RATE, (width, height))
        for path in frame_paths:
            frame = cv2.imread(path)
            out.write(frame)
        out.release()


def extract_frame_from_video(video_id, use_camera_id, timestamp):
    folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
    frame_folder = os.path.join(OUTPUT_DIR, folder_name)
    frame_index = timestamp_to_frame_index(timestamp)
    frame_path = os.path.join(frame_folder, f'frame_{frame_index:06d}.png')
    frame = cv2.imread(frame_path)
    return frame


def safe_resize(clip, max_height=1080, min_height=240, step=120):
    original_w, original_h = clip.size
    for height in range(max_height, min_height - 1, -step):
        scale = height / original_h
        width = int(original_w * scale)
        try:
            return resize(clip, height=height)
        except MemoryError as e:
            print(f'MemoryError at resolution {width}x{height}, trying lower: {e}')
    raise MemoryError("All fallback resolutions failed due to memory constraints.")


def display_comic_and_video(data_df, video_id, use_keyframe_from_video, use_camera_id):
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
            folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
            video_path = os.path.join(OUTPUT_DIR, folder_name, f'{block_id}.mp4')
            image = extract_frame_from_video(video_id, use_camera_id, row['Key Timestamp'])
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
            folder_name = f'{video_id}_updated' if use_camera_id else f'{video_id}'
            video_path = os.path.join(OUTPUT_DIR, folder_name, f'{block_id}.mp4')
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
                video_writer = cv2.VideoWriter(output_path, fourcc, FRAME_RATE, output_frame_size)
            video_writer.write(combined_frame)
        cap.release()
        # cv2.destroyAllWindows()
        video_writer.release()
        del cap
        del video_writer
        video_clips.append(output_path)
    final_clips = []
    for clip_path in video_clips:
        try:
            clip = VideoFileClip(clip_path)
            if clip.duration == 0:
                print(f'Skipping empty clip: {clip_path}')
                continue
            try:
                resized_clip = safe_resize(clip)
                final_clips.append(resized_clip)
            except MemoryError as e:
                print(f"Failed to resize clip {clip_path}: {e}")
        except Exception as e:
            print(f'Error loading clip {clip_path}: {e}')
    final_video = concatenate_videoclips(final_clips, method='compose')
    file_name = f'{video_id}_updated.mp4' if use_camera_id else f'{video_id}.mp4'
    final_video_path = os.path.join(OUTPUT_DIR, file_name)
    final_video.write_videofile(final_video_path, fps=FRAME_RATE)
    final_video.close()
    for clip in final_clips:
        clip.close()
    shutil.rmtree(temp_dir)


def generate_comic(data_df, video_id, use_camera_id):
    output_images = []
    temp_image_dir = os.path.join(OUTPUT_DIR, 'temp_images')
    os.makedirs(temp_image_dir, exist_ok=True)
    for idx, row in tqdm(data_df.iterrows(), total=len(data_df), desc='Extracting frames for PDF'):
        if pd.isna(row['Key Timestamp']):
            continue
        frame = extract_frame_from_video(video_id, use_camera_id, row['Key Timestamp'])
        image_path = os.path.join(temp_image_dir, f'{idx+1:03d}.png')
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
            img.close()
        composite_images.append(combined)
    file_name = f'{video_id}_updated.pdf' if use_camera_id else f'{video_id}.pdf'
    pdf_path = os.path.join(OUTPUT_DIR, file_name)
    composite_images[0].save(pdf_path, save_all=True, append_images=composite_images[1:])
    for img in composite_images:
        img.close()
    print(f'PDF comic saved at: {pdf_path}')
    shutil.rmtree(temp_image_dir)


def run(video_id, use_keyframe_from_video, use_camera_id):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    file_name = f'{video_id}_updated.csv' if use_camera_id else f'{video_id}.csv'
    data_df = pd.read_csv(os.path.join(OUTPUT_DIR, file_name), dtype={'Video ID': str})
    # print(parse_timestamp('00:03:37:18'))
    if check_timestamps(data_df, use_keyframe_from_video):
        if not use_keyframe_from_video:
            analyze_comic_blocks(data_df, video_id)
        extract_frames(video_id, use_camera_id)
        print(f'{video_id} total frames:', get_total_frames(video_id, use_camera_id))
        extract_video_clips(video_id, data_df, use_keyframe_from_video, use_camera_id)
        display_comic_and_video(data_df, video_id, use_keyframe_from_video, use_camera_id)
        if use_keyframe_from_video:
            generate_comic(data_df, video_id, use_camera_id)
        delete_frames(video_id, use_camera_id)


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
    # run('029', True, True)
    for i in range(46, 79):
        print(f'{i:03d}')
        run(f'{i:03d}', False, False)
    # parser = argparse.ArgumentParser(description='Process video and comic IDs.')
    # parser.add_argument('-vid', '--video_id', required=True, help="The ID of the video (e.g., '001')")
    # args = parser.parse_args()
    # run(args.video_id, True)
