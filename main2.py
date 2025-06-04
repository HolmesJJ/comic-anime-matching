# https://github.com/facebookresearch/dino

import os
import cv2
import glob
import torch
import pickle

from PIL import Image
from tqdm import tqdm
from dotenv import load_dotenv
from torchvision import transforms


load_dotenv()

COMIC = os.getenv('COMIC')
ANIME_DIR = os.path.join(os.getenv('ANIME_DIR'), COMIC)
COMIC_DIR = os.path.join(os.getenv('COMIC_DIR'), COMIC)
OUTPUT_DIR = os.path.join(os.getenv('OUTPUT_DIR'), COMIC)
CHECKPOINTS_DIR = os.getenv('CHECKPOINTS_DIR')


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(model_name, model_path):
    model = torch.hub.load('facebookresearch/dino:main', model_name, pretrained=False)
    checkpoint = torch.load(model_path)
    model.load_state_dict(checkpoint, strict=True)
    model.eval()
    return model


def extract_features(model, image_tensor):
    with torch.no_grad():
        features = model.get_intermediate_layers(image_tensor, n=1)[0]
        features = features.mean(dim=1)
        return features.squeeze().cpu()


def find_matching_scores(template_path, frame_folder, model_name, model_path, step=None,
                         current_block=None, number_of_total_blocks=None, number_of_scanning_blocks=None):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name, model_path).to(device)
    template_image = transform(Image.open(template_path).convert('RGB')).unsqueeze(0).to(device)
    template_features = extract_features(model, template_image)
    frame_scores = {}
    frame_files = sorted(os.listdir(frame_folder))
    total_frames = len(frame_files)
    if current_block is not None and number_of_total_blocks is not None and number_of_scanning_blocks is not None:
        block_size = total_frames // number_of_total_blocks
        block_start = max(current_block - number_of_scanning_blocks // 2, 0)
        block_end = min(current_block + number_of_scanning_blocks // 2, number_of_total_blocks)
        print('current_block:', current_block, 'number_of_total_blocks:', number_of_total_blocks,
              'block_start:', block_start, 'block_end:', block_end)
        scan_start = max(block_start * block_size, 0)
        scan_end = min(block_end * block_size, total_frames)
        print('scan_start:', scan_start, 'scan_end:', scan_end)
        frame_files = frame_files[scan_start:scan_end]
    for idx, frame_name in enumerate(tqdm(frame_files, desc='Processing frames')):
        if step is not None and idx % step != 0:
            continue
        frame_path = os.path.join(frame_folder, frame_name)
        try:
            frame_image = transform(Image.open(frame_path).convert('RGB')).unsqueeze(0).to(device)
        except Exception as e:
            print(f'Warning: Could not load frame image from {frame_path}. Error: {e}')
            continue
        frame_features = extract_features(model, frame_image)
        # Calculate similarity using cosine similarity
        similarity = torch.nn.functional.cosine_similarity(
            template_features.unsqueeze(0).to(torch.float32),
            frame_features.unsqueeze(0).to(torch.float32)
        ).item()
        frame_scores[frame_path] = similarity
    return frame_scores


def display():
    template_path = os.path.join(COMIC_DIR, '01', 'page_13', '4.jpg')
    frame_folder = os.path.join(ANIME_DIR, '001')
    model_name = 'dino_vitb8'
    model_path = os.path.join(CHECKPOINTS_DIR, 'dino_vitbase8_pretrain.pth')
    frame_scores = find_matching_scores(template_path, frame_folder, model_name, model_path)
    sorted_scores = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
    best_frame_name, best_score = sorted_scores[0]
    print(best_frame_name, best_score)
    best_frame_path = os.path.join(frame_folder, best_frame_name)
    if best_frame_path is not None:
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_resized = cv2.resize(template, (400, 400))
        best_frame = cv2.imread(best_frame_path, cv2.IMREAD_GRAYSCALE)
        best_frame_resized = cv2.resize(best_frame, (400, 400))
        combined = cv2.hconcat([template_resized, best_frame_resized])
        cv2.imshow('Template (Left) vs Best Match (Right)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


def run():
    video = '001'
    start, end = 6, 35
    template_paths = []
    for i in range(start, end + 1):
        folder_path = os.path.join(COMIC_DIR, '01', f'page_{i}')
        template_paths.extend(glob.glob(os.path.join(folder_path, '*.jpg')))
        template_paths.extend(glob.glob(os.path.join(folder_path, '*.png')))
    frame_folder = os.path.join(ANIME_DIR, video)
    model_name = 'dino_vitb8'
    model_path = os.path.join(CHECKPOINTS_DIR, 'dino_vitbase8_pretrain.pth')
    scores = {}
    for template_idx, template_path in enumerate(template_paths):
        print('Processing template: ', template_path)
        frame_scores = find_matching_scores(template_path, frame_folder, model_name, model_path, step=None,
                                            current_block=template_idx, number_of_total_blocks=len(template_paths),
                                            number_of_scanning_blocks=50)
        sorted_scores = sorted(frame_scores.items(), key=lambda x: x[1], reverse=True)
        best_frame_name, best_score = sorted_scores[0]
        print(best_frame_name, best_score)
        scores[template_path] = frame_scores
        with open(os.path.join(OUTPUT_DIR, f'{video},pkl'), 'wb') as f:
            pickle.dump(scores, f)


def load():
    video = '001'
    with open(os.path.join(OUTPUT_DIR, f'{video},pkl'), 'rb') as file:
        scores = pickle.load(file)
    for template_path, frame_scores in scores.items():
        max_item = max(frame_scores.items(), key=lambda x: x[1])
        best_frame_path, best_score = max_item
        print(f'{template_path}: {best_frame_path} -> {best_score}')
        template = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        template_resized = cv2.resize(template, (400, 400))
        best_frame = cv2.imread(best_frame_path, cv2.IMREAD_GRAYSCALE)
        best_frame_resized = cv2.resize(best_frame, (400, 400))
        combined = cv2.hconcat([template_resized, best_frame_resized])
        cv2.imshow('Template (Left) vs Best Match (Right)', combined)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == '__main__':
    # display()
    run()
    # load()
