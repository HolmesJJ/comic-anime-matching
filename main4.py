import io
import os
import cv2
import time
import base64
import pandas as pd
import matplotlib.pyplot as plt

from openai import OpenAI
from dotenv import load_dotenv
from PIL import Image as PILImage
from openpyxl import Workbook
from openpyxl.styles import Alignment
from openpyxl.drawing.image import Image as XLImage


load_dotenv()

COMIC = os.getenv('COMIC')
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), COMIC, '1')
EXTENSION_DIR = os.path.join(os.getenv('EXTENSION_DIR'), COMIC, '1')
GPT_O3_MODEL = os.getenv('GPT_O3_MODEL')
GPT_4O_MODEL = os.getenv('GPT_4O_MODEL')
GPT_KEY = os.getenv('GPT_KEY')
GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# GEMINI_KEY = os.getenv('GEMINI_KEY')
GEMINI_KEYS_PATH = os.getenv('GEMINI_KEYS_PATH')
GEMINI_INVALID_KEYS_PATH = os.getenv('GEMINI_INVALID_KEYS_PATH')
GEMINI_URL = os.getenv('GEMINI_URL')
PROMPT1_PATH = os.getenv('PROMPT1_PATH')
PROMPT2_PATH = os.getenv('PROMPT2_PATH')


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def image_to_base64(image_path):
    with open(image_path, 'rb') as img_file:
        return base64.b64encode(img_file.read()).decode('utf-8')


def load_gemini_keys():
    with open(GEMINI_KEYS_PATH, 'r', encoding='utf-8') as f:
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
            pil_img = PILImage.fromarray(frame)
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


def show_video_frames(video_path, interval=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Invalid video: {video_path}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    start_frame = 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1
    cap.release()
    frames_per_row = 6
    target_height = 200
    resized_frames = []
    for frame in frames:
        h, w, _ = frame.shape
        aspect_ratio = w / h
        new_w = int(target_height * aspect_ratio)
        resized = cv2.resize(frame, (new_w, target_height))
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


def get_video_frames(video_path, interval=0.5):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f'Invalid video: {video_path}')
        return
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    start_frame = 10
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_count = start_frame
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_count % frame_interval == 0:
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)
        frame_count += 1
    cap.release()
    return frames


def run(anime):
    extension_path = os.path.join(EXTENSION_DIR, f'{anime}.pkl')
    if os.path.exists(extension_path):
        df = pd.read_pickle(extension_path)
        responses = df['response'].tolist()
    else:
        df = pd.DataFrame(columns=['comic_block_id', 'response'])
        responses = []
    for root, dirs, files in os.walk(os.path.join(OUTPUT_DIR, anime), topdown=True):
        for file in files:
            video_path = os.path.join(root, file)
            file_name = os.path.basename(video_path)
            comic_block_id = os.path.splitext(file_name)[0]
            parts = comic_block_id.split('_')
            comic_id = parts[0]
            page_folder = f'page_{parts[2]}'
            image_path = None
            image_path_jpg = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.jpg')
            image_path_png = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.png')
            if os.path.exists(image_path_jpg):
                image_path = image_path_jpg
            elif os.path.exists(image_path_png):
                image_path = image_path_png
            if not image_path:
                print(f'Image path is missing or invalid: {image_path_jpg, image_path_png}')
                continue
            # show_video_frames(video_path)
            print(video_path, image_path)
            base64_image = get_base64_images(image_path=image_path, frames=None)
            frames = get_video_frames(video_path)
            base64_images = get_base64_images(image_path=None, frames=frames)
            base64_images.insert(0, base64_image)
            prompt_content = read_prompt(PROMPT1_PATH).format(COMIC, 2000)
            print(prompt_content)
            gemini_keys = load_gemini_keys()
            gemini_invalid_keys = load_gemini_invalid_keys()
            response = None
            for key in gemini_keys:
                if key in gemini_invalid_keys:
                    continue
                print('Gemini Key:', key)
                is_error_429 = False
                while True:
                    try:
                        response = get_response(GEMINI_MODEL, key, prompt_content, base64_images, GEMINI_URL)
                        break
                    except Exception as e:
                        error_message = str(e)
                        if 'Error code: 429' in error_message:
                            print('Error 429:', e)
                            save_gemini_invalid_key(key)
                            is_error_429 = True
                            break
                        elif 'Error code: 503' in error_message:
                            print('Error 503:', e)
                            time.sleep(30)
                            continue
                        else:
                            print('Error:', e)
                            break
                if not is_error_429:
                    break
            if response is None:
                response = get_response(GPT_O3_MODEL, GPT_KEY, prompt_content, base64_images)
            print('Response:', response)
            df.loc[len(df)] = [comic_block_id, response]
            df.to_pickle(extension_path)
            responses.append(response)
            print(f'[Saved] {comic_block_id} -> pickle ({len(df)} total)')


def show_output(anime):
    extension_path = os.path.join(EXTENSION_DIR, f'{anime}.pkl')
    df = pd.read_pickle(extension_path)
    max_image_size = 256
    char_per_line = 80
    line_height = 18
    temp_images = []
    wb = Workbook()
    ws = wb.active
    ws.title = 'Comic Responses'
    ws.append(['comic_block_id', 'image', 'response', 'Anime'])
    ws.column_dimensions['A'].width = 20
    ws.column_dimensions['B'].width = max_image_size // 7
    ws.column_dimensions['C'].width = 80
    for idx, row in df.iterrows():
        comic_block_id = row['comic_block_id']
        response = row['response'] if row['response'] else ''
        parts = comic_block_id.split('_')
        comic_id = parts[0]
        page_folder = f'page_{parts[2]}'
        image_path = None
        image_path_jpg = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.jpg')
        image_path_png = os.path.join(COMIC_DIR, comic_id, page_folder, f'{parts[3]}.png')
        if os.path.exists(image_path_jpg):
            image_path = image_path_jpg
        elif os.path.exists(image_path_png):
            image_path = image_path_png
        if not image_path:
            print(image_path_jpg, image_path_png)
            raise ValueError('Image path is missing or invalid.')
        ws.append([comic_block_id, '', response])
        image_height = 0
        if image_path:
            pil_img = PILImage.open(image_path)
            width, height = pil_img.size
            scale = min(max_image_size / width, max_image_size / height, 1.0)
            new_width = int(width * scale)
            new_height = int(height * scale)
            resized_path = os.path.join(EXTENSION_DIR, f'tmp_resized_{idx}.png')
            pil_img.resize((new_width, new_height), PILImage.Resampling.LANCZOS).save(resized_path)
            image = XLImage(resized_path)
            temp_images.append(resized_path)
            image.anchor = f'B{idx + 2}'
            ws.add_image(image)
            image_height = new_height * 0.75
        for col_letter in ['C']:
            cell = ws[f'{col_letter}{idx + 2}']
            cell.alignment = Alignment(wrap_text=True, vertical='top')
        video_path = os.path.join(OUTPUT_DIR, anime,  f'{comic_id}_{page_folder}_{parts[3]}.mp4')
        frames = get_video_frames(video_path)
        base64_images = get_base64_images(image_path=None, frames=frames)
        pil_frames = []
        for base64_image in base64_images:
            image_data = base64.b64decode(base64_image)
            img = PILImage.open(io.BytesIO(image_data)).convert('RGB')
            pil_frames.append(img)
        target_height = 100
        resized_frames = []
        for img in pil_frames:
            w, h = img.size
            new_width = int(target_height * w / h)
            resized = img.resize((new_width, target_height), PILImage.Resampling.LANCZOS)
            resized_frames.append(resized)
        half = (len(resized_frames) + 1) // 2
        row1 = resized_frames[:half]
        row2 = resized_frames[half:]
        row1_width = sum(img.width for img in row1)
        row2_width = sum(img.width for img in row2)
        stitched_width = max(row1_width, row2_width)
        stitched_pil_img = PILImage.new('RGB', (stitched_width, max_image_size), color=(255, 255, 255))
        x_offset = 0
        for img in row1:
            stitched_pil_img.paste(img, (x_offset, 0))
            x_offset += img.width
        x_offset = 0
        for img in row2:
            stitched_pil_img.paste(img, (x_offset, target_height))
            x_offset += img.width
        stitched_path = os.path.join(EXTENSION_DIR, f'tmp_stitched_{idx}.png')
        stitched_pil_img.save(stitched_path)
        stitched_img = XLImage(stitched_path)
        temp_images.append(stitched_path)
        stitched_img.anchor = f'D{idx + 2}'
        ws.add_image(stitched_img)
        stitched_image_height = max_image_size * 0.75
        line_count = len(response) // char_per_line + 1
        text_height = line_count * line_height
        row_height = max(image_height, text_height, stitched_image_height)
        ws.row_dimensions[idx + 2].height = row_height
    extension_excel = os.path.splitext(extension_path)[0] + '.xlsx'
    wb.save(extension_excel)
    print(f'Saved Excel to: {extension_excel}')
    for path in temp_images:
        if os.path.exists(path):
            os.remove(path)


if __name__ == '__main__':
    run('145')
    show_output('145')
