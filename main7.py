import os
import re
import subprocess

from dotenv import load_dotenv


load_dotenv()

ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), 'Tom and Jerry')
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), 'Tom and Jerry')


def extract_number(file_name):
    match = re.search(r'_(\d+)\.mp3$', file_name)
    return int(match.group(1)) if match else float('inf')


def run(video_id, start_time=0, interval=5, background_volume=0.3, voice_volume=1.3):
    input_cmds = ['-i', os.path.join(ANIME_DIR, f'{video_id}.mp4')]
    filter_cmds = [f'[0:a]volume={background_volume}[a0]']
    amix_inputs = '[a0]'
    audio_folder = os.path.join(OUTPUT_DIR, video_id)
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])
    sorted_files = sorted(audio_files, key=extract_number)
    for idx, file_name in enumerate(sorted_files):
        audio_path = os.path.join(audio_folder, file_name)
        delay = (start_time + idx * interval) * 1000
        input_cmds.extend(['-i', audio_path])
        label = f'a{idx+1}'
        filter_cmds.append(
            f'[{idx+1}:a]adelay={delay:.0f}|{delay:.0f},'
            f'volume={voice_volume}[{label}]'
        )
        amix_inputs += f'[{label}]'
    filter_complex = ';'.join(filter_cmds) + f';{amix_inputs}amix=inputs={len(sorted_files)+1}:normalize=0[aout]'
    cmd = (
        ['ffmpeg']
        + input_cmds
        + [
            '-filter_complex', filter_complex,
            '-map', '0:v',
            '-map', '[aout]',
            '-c:v', 'copy',
            '-c:a', 'aac',
            '-y',
            os.path.join(OUTPUT_DIR, f'{video_id}.mp4')
        ]
    )
    print(' '.join(cmd))
    subprocess.run(cmd, check=True)


def extract_audio(video_id, bit_rate='128k'):
    mp4_path = os.path.join(OUTPUT_DIR, f'{video_id}.mp4')
    mp3_path = os.path.join(OUTPUT_DIR, f'{video_id}.mp3')
    cmd = [
        'ffmpeg',
        '-i', mp4_path,
        '-vn',
        '-acodec', 'libmp3lame',
        '-ab', bit_rate,
        '-y',
        mp3_path
    ]
    print(cmd)
    subprocess.run(cmd, check=True)


if __name__ == '__main__':
    run(f'001')
    extract_audio(f'001')
