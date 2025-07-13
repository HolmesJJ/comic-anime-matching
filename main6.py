import os

from openai import OpenAI
from dotenv import load_dotenv


load_dotenv()

ANIME = os.getenv('ANIME')
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), 'Tom and Jerry')
GPT_4O_AUDIO_MODEL = os.getenv('GPT_4O_AUDIO_MODEL')
GPT_KEY = os.getenv('GPT_KEY')
PROMPT5_PATH = os.getenv('PROMPT5_PATH')


def read_prompt(prompt_path):
    with open(prompt_path, 'r', encoding='utf-8') as file:
        content = file.read()
    return content


def get_response(prompt_content, input_content, output_path):
    client = OpenAI(api_key=GPT_KEY)
    response = client.audio.speech.create(
            model=GPT_4O_AUDIO_MODEL,
            voice='alloy',
            input=input_content,
            instructions=prompt_content
    )
    response.write_to_file(output_path)


def run(video_id):
    with open(os.path.join(OUTPUT_DIR, f'{video_id}_full.txt'), 'r', encoding='utf-8') as f:
        full_txt = f.read()
        lines = full_txt.splitlines()
    with open(os.path.join(OUTPUT_DIR, f'{video_id}_summary.txt'), 'r', encoding='utf-8') as f:
        summary_txt = f.read()
    prompt_content = read_prompt(PROMPT5_PATH).format(ANIME, full_txt)
    print(prompt_content)
    get_response(prompt_content, summary_txt, os.path.join(OUTPUT_DIR, f'{video_id}_summary.mp3'))
    for idx, line in enumerate(lines):
        audio_folder = os.path.join(OUTPUT_DIR, video_id)
        os.makedirs(audio_folder, exist_ok=True)
        get_response(prompt_content, line, os.path.join(audio_folder, f'{video_id}_{idx + 1}.mp3'))


if __name__ == '__main__':
    run(f'001')
