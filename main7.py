import os
import re
import shlex
import subprocess
import numpy as np

from dotenv import load_dotenv
from moviepy.editor import ColorClip
from moviepy.editor import AudioFileClip
from moviepy.editor import VideoFileClip
from moviepy.editor import concatenate_videoclips
from moviepy.audio.AudioClip import AudioArrayClip


load_dotenv()

ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), 'Tom and Jerry')
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), 'Tom and Jerry')


def extract_number(file_name):
    m = re.search(r'(\d+)(?=\.mp3$)', file_name, re.IGNORECASE)
    return int(m.group(1)) if m else float('inf')


def _ffprobe_dims_fps(mp4_path):
    cmd = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0',
        '-show_entries', 'stream=width,height,r_frame_rate',
        '-of', 'default=noprint_wrappers=1:nokey=1',
        mp4_path
    ]
    out = subprocess.check_output(cmd, text=True).strip().splitlines()
    width, height, fps = out
    return int(width), int(height), fps


def add_summary(video_id):
    summary_mp3 = os.path.join(OUTPUT_DIR, f'{video_id}_summary.mp3')
    core_mp4 = os.path.join(OUTPUT_DIR, f'{video_id}.mp4')
    final_mp4 = os.path.join(OUTPUT_DIR, f'{video_id}_with_summary.mp4')
    core_clip = VideoFileClip(core_mp4)
    width, height = core_clip.size
    vf_fps = core_clip.fps
    af_fps = core_clip.audio.fps
    n_channels = core_clip.audio.nchannels
    summary_audio = AudioFileClip(summary_mp3).set_fps(af_fps)
    summary_video = ColorClip(size=(width, height), color=(0, 0, 0), duration=summary_audio.duration)
    summary_video = summary_video.set_audio(summary_audio).set_fps(vf_fps).audio_fadeout(0.5)
    silence_audio = AudioArrayClip(np.zeros((af_fps, n_channels), dtype=np.float32), fps=af_fps)
    gap_video = ColorClip(size=(width, height), color=(0, 0, 0), duration=1)
    gap_video = gap_video.set_audio(silence_audio).set_fps(vf_fps)
    final_clip = concatenate_videoclips([summary_video, gap_video, core_clip])
    final_clip.write_videofile(final_mp4, codec='libx264', audio_codec='aac', preset='veryfast')
    core_clip.close()
    summary_audio.close()
    summary_video.close()
    silence_audio.close()
    gap_video.close()
    final_clip.close()


def run(video_id, start_time=0, interval=5, background_volume=0.3, voice_volume=1.3):
    input_cmds = ['-i', os.path.join(ANIME_DIR, f'{video_id}.mp4')]
    filter_cmds = [f'[0:a]volume={background_volume}[a0]']
    amix_inputs = '[a0]'
    audio_folder = os.path.join(OUTPUT_DIR, video_id)
    audio_files = sorted([f for f in os.listdir(audio_folder) if f.endswith('.mp3')])
    sorted_files = sorted(audio_files, key=extract_number)
    for idx, file_name in enumerate(sorted_files):
        n = extract_number(file_name)
        audio_path = os.path.join(audio_folder, file_name)
        delay = (start_time + (n - 1) * interval * 1000)
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
    print(' '.join(shlex.quote(str(x)) for x in cmd))
    subprocess.run(cmd, check=True)
    add_summary(video_id)


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
