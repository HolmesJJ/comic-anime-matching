import os
import time

from google import genai
from dotenv import load_dotenv


load_dotenv()

GEMINI_MODEL = os.getenv('GEMINI_MODEL')
# GEMINI_KEY = os.getenv('GEMINI_KEY')
GEMINI_KEYS_PATH = os.getenv('GEMINI_KEYS_PATH')
GEMINI_INVALID_KEYS_PATH = os.getenv('GEMINI_INVALID_KEYS_PATH')
COMIC = os.getenv('COMIC')
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), COMIC)


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


def get_response(model_key, video_path):
    client = genai.Client(api_key=model_key)
    video_file = client.files.upload(file=video_path)
    while video_file.state.name == 'PROCESSING':
        print('.', end='')
        time.sleep(10)
    print('Video State:', video_file.state.name)
    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=[
            video_file,
            'Summarize this video.'
        ]
    )
    return response.text


def run(anime):
    for root, dirs, files in os.walk(os.path.join(OUTPUT_DIR, anime), topdown=True):
        for file in files:
            video_path = os.path.join(root, file)
            print(video_path)
            gemini_keys = load_gemini_keys()
            gemini_invalid_keys = load_gemini_invalid_keys()
            response = None
            for key in gemini_keys:
                if key in gemini_invalid_keys:
                    continue
                print('Gemini Key:', key)
                try:
                    response = get_response(key, video_path)
                except Exception as e:
                    print(e)
                    error_message = str(e)
                    if 'Error code: 429' in error_message:
                        save_gemini_invalid_key(key)
                        continue
                    else:
                        break
                break
            print('Response:', response)


if __name__ == '__main__':
    run('001')
