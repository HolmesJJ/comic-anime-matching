# https://github.com/openai/CLIP

import os
import cv2
import torch

from PIL import Image
from clip import clip
from tqdm import tqdm


ANIME_DIR = 'anime'
COMIC_DIR = 'comic'


def find_best_matching_frame(template_path, frame_folder):
    # Load the CLIP model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model, preprocess = clip.load('ViT-B/32', device=device)
    print(model)
    # Load and preprocess the template image
    template_image = preprocess(Image.open(template_path)).unsqueeze(0).to(device)
    with torch.no_grad():
        template_features = model.encode_image(template_image).cpu()
    best_score = float('-inf')
    best_frame_path = None
    frame_files = os.listdir(frame_folder)
    for frame_name in tqdm(frame_files, desc='Processing frames'):
        frame_path = os.path.join(frame_folder, frame_name)
        try:
            frame_image = preprocess(Image.open(frame_path)).unsqueeze(0).to(device)
        except Exception as e:
            print(f'Warning: Could not load frame image from {frame_path}. Error: {e}')
            continue
        with torch.no_grad():
            frame_features = model.encode_image(frame_image).cpu()
        # Calculate similarity using cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            template_features.to(torch.float32),
            frame_features.to(torch.float32)
        ).item()
        if similarity > best_score:
            best_score = similarity
            best_frame_path = frame_path
    return best_score, best_frame_path


def display():
    template_path = os.path.join(COMIC_DIR, '01', 'page_13', '4.jpg')
    frame_folder = os.path.join(ANIME_DIR, '001')
    best_score, best_frame_path = find_best_matching_frame(template_path, frame_folder)
    print(best_score, best_frame_path)
    if best_frame_path is not None:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_resized = cv2.resize(template, (400, 400))
        best_frame = cv2.imread(best_frame_path, cv2.IMREAD_GRAYSCALE)
        best_frame_resized = cv2.resize(best_frame, (400, 400))
        combined = cv2.hconcat([template_resized, best_frame_resized])
        cv2.imshow('Template (Left) vs Best Match (Right)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    display()
