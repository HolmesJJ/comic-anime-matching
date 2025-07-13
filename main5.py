import io
import os
import cv2
import glob
import time
import base64
import subprocess
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image as PILImage


load_dotenv()

ANIME = os.getenv('ANIME')
ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), 'Tom and Jerry')
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), 'Tom and Jerry')
GPT_O3_MODEL = os.getenv('GPT_O3_MODEL')
GPT_4O_MODEL = os.getenv('GPT_4O_MODEL')
GPT_KEY = os.getenv('GPT_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# GEMINI_KEY = os.getenv('GEMINI_KEY')
GEMINI_KEYS2_PATH = os.getenv('GEMINI_KEYS2_PATH')
GEMINI_INVALID_KEYS_PATH = os.getenv('GEMINI_INVALID_KEYS_PATH')
GEMINI_URL = os.getenv('GEMINI_URL')
PROMPT3_PATH = os.getenv('PROMPT3_PATH')


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def load_gemini_keys():
    with open(GEMINI_KEYS2_PATH, 'r', encoding='utf-8') as f:
        return [line.strip() for line in f if line.strip()]


def load_gemini_invalid_keys():
    if os.path.exists(GEMINI_INVALID_KEYS_PATH):
        with open(GEMINI_INVALID_KEYS_PATH, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    return []


def save_gemini_invalid_key(key):
    with open(GEMINI_INVALID_KEYS_PATH, 'a') as f:
        f.write(f'{key}\n')


def get_response(model, model_key, prompt_content, base64_images, model_url=None):
    print('Model:', model)
    if model == GPT_4O_MODEL or model == GPT_O3_MODEL:
        client = OpenAI(api_key=model_key)
    else:
        client = OpenAI(base_url=model_url, api_key=model_key)
    content = [
        {
            'type': 'text',
            'text': prompt_content,
        }
    ]
    for base64_image in base64_images:
        content.append({
            'type': 'image_url',
            'image_url': {
                'url': f'data:image/jpeg;base64,{base64_image}'
            }
        })
    if model == GPT_O3_MODEL:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ],
            reasoning_effort='high'
        )
    elif model == GPT_4O_MODEL:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ],
            temperature=0
        )
    else:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    'role': 'user',
                    'content': content
                }
            ],
            reasoning_effort='high',
            temperature=0
        )
    return response.choices[0].message.content


def get_base64_images(image_path=None, frames=None):
    if image_path:
        with PILImage.open(image_path) as img:
            width, height = img.size
            if width < height:
                scale = min(768 / width, 2000 / height)
            else:
                scale = min(768 / height, 2000 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = img.resize((new_width, new_height), PILImage.Resampling.LANCZOS).convert('RGB')
            buffer = io.BytesIO()
            resized_img.save(buffer, format='JPEG')
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_str
    elif frames:
        base64_images = []
        for frame in frames:
            pil_img = PILImage.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            width, height = pil_img.size
            if width < height:
                scale = min(768 / width, 2000 / height)
            else:
                scale = min(768 / height, 2000 / width)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_img = pil_img.resize((new_width, new_height), PILImage.Resampling.LANCZOS).convert('RGB')
            buffer = io.BytesIO()
            resized_img.save(buffer, format='JPEG')
            base64_str = base64.b64encode(buffer.getvalue()).decode('utf-8')
            base64_images.append(base64_str)
        return base64_images


def show_frames(video_id, video_file, interval=0.5):
    frame_folder = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(frame_folder, exist_ok=True)
    pattern = os.path.join(frame_folder, f'frame_%06d.jpg')
    cmd = ['ffmpeg', '-i', video_file, '-vf', f'fps=1/{interval}', '-vsync', 'passthrough', pattern]
    try:
        print(f'{video_id} extract frames...')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'{video_id} extract frames error: {e}')
    frame_files = sorted(glob.glob(os.path.join(frame_folder, 'frame_*.jpg')))
    target_height = 200
    frames_per_row = 6
    resized_frames = []
    for frame_file in frame_files:
        img = cv2.imread(frame_file)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w, _ = img.shape
        new_w = int(target_height * (w / h))
        resized = cv2.resize(img, (new_w, target_height))
        resized_frames.append(resized)
    num_rows = (len(resized_frames) + frames_per_row - 1) // frames_per_row
    fig, axs = plt.subplots(num_rows, frames_per_row, figsize=(frames_per_row * 3, num_rows * 2))
    for i in range(num_rows * frames_per_row):
        row = i // frames_per_row
        col = i % frames_per_row
        ax = axs[row][col] if num_rows > 1 else axs[col]
        if i < len(resized_frames):
            ax.imshow(resized_frames[i])
            ax.set_title(f'Frame {i}')
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def extract_frames(video_id, video_file, interval=0.5):
    frame_folder = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(frame_folder, exist_ok=True)
    pattern = os.path.join(frame_folder, f'frame_%06d.jpg')
    cmd = ['ffmpeg', '-i', video_file, '-vf', f'fps=1/{interval}', '-vsync', 'passthrough', pattern]
    try:
        print(f'{video_id} extract frames...')
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f'{video_id} extract frames error: {e}')
    frame_files = sorted(glob.glob(os.path.join(frame_folder, 'frame_*.jpg')))
    frames = [cv2.imread(f) for f in frame_files]
    # for frame_file in frame_files:
    #     os.remove(frame_file)
    return frames


def extract_first_frame(video_id, video_file):
    frame_folder = os.path.join(OUTPUT_DIR, video_id)
    os.makedirs(frame_folder, exist_ok=True)
    frame_path = os.path.join(frame_folder, 'frame_000001.jpg')
    cmd = ['ffmpeg', '-i', video_file, '-frames:v', '1', frame_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        frame = cv2.imread(frame_path)
        os.remove(frame_path)
        return frame
    except subprocess.CalledProcessError as e:
        print(f'{video_id} extract frames error: {e}')
        return None


def run(video_id):
    full_response = ''
    video_path = os.path.join(ANIME_DIR, f'{video_id}.mp4')
    frames = extract_frames(video_id, video_path)
    # show_frames(video_id, video_path)
    if len(frames) == 0:
        frames = [extract_first_frame(video_id, video_path)]
    base64_images = get_base64_images(image_path=None, frames=frames)
    for i in range(0, len(base64_images), 10):
        group_base64_images = base64_images[i:i + 10]
        prompt_content = read_prompt(PROMPT3_PATH).format(ANIME, full_response)
        print(prompt_content)
        gemini_keys = load_gemini_keys()
        gemini_invalid_keys = load_gemini_invalid_keys()
        response = None
        is_all_invalid = True
        for key in gemini_keys:
            if key in gemini_invalid_keys:
                continue
            print('Gemini Key:', key)
            is_all_invalid = False
            is_error = False
            while True:
                try:
                    response = get_response(GEMINI_MODEL, key, prompt_content, group_base64_images, GEMINI_URL)
                    break
                except Exception as e:
                    error_message = str(e)
                    if 'Error code: 403' in error_message or 'Error code: 429' in error_message:
                        print('Error 403 / 429:', e)
                        save_gemini_invalid_key(key)
                        is_error = True
                        break
                    elif 'Error code: 503' in error_message:
                        print('Error 503:', e)
                        time.sleep(30)
                        continue
                    else:
                        print('Error:', e)
                        break
            if not is_error:
                break
        if is_all_invalid:
            raise ValueError('All gemini keys are invalid.')
        if response is None:
            response = get_response(GPT_O3_MODEL, GPT_KEY, prompt_content, group_base64_images)
        print('Response:', response)
        full_response = full_response + response + '\n'
        print('Full Response:', full_response)
        with open(os.path.join(OUTPUT_DIR, 'full.txt'), 'w', encoding='utf-8') as f:
            f.write(full_response)


if __name__ == '__main__':
    run(f'001')
