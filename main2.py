# https://github.com/facebookresearch/dino

import os
import cv2
import torch

from PIL import Image
from tqdm import tqdm
from torchvision import transforms


ANIME_DIR = 'anime'
COMIC_DIR = 'comic'
CHECKPOINTS_DIR = 'checkpoints'


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


def find_best_matching_frame(template_path, frame_folder, model_name, model_path):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = load_model(model_name, model_path).to(device)
    template_image = transform(Image.open(template_path).convert('RGB')).unsqueeze(0).to(device)
    template_features = extract_features(model, template_image)
    best_score = float('-inf')
    best_frame_path = None
    frame_files = os.listdir(frame_folder)
    for frame_name in tqdm(frame_files, desc='Processing frames'):
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
        if similarity > best_score:
            best_score = similarity
            best_frame_path = frame_path
    return best_score, best_frame_path


def display():
    template_path = os.path.join(COMIC_DIR, '01', '8', '7.jpg')
    frame_folder = os.path.join(ANIME_DIR, '001')
    model_name = 'dino_vitb8'
    model_path = os.path.join(CHECKPOINTS_DIR, 'dino_vitbase8_pretrain.pth')
    best_score, best_frame_path = find_best_matching_frame(template_path, frame_folder, model_name, model_path)
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
