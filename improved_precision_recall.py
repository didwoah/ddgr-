
import os

import torch
import torchvision.transforms as transforms
from torchvision.models import inception_v3

from sklearn.neighbors import NearestNeighbors

from PIL import Image

import numpy as np

# 이미지 로드 함수: 폴더 내의 모든 이미지 불러오기
def load_images_from_folder(folder, transform, max_images=None):
    images = []
    image_files = os.listdir(folder)
    if max_images is not None:
        image_files = image_files[:max_images]  # max_images만큼 이미지 제한

    for filename in image_files:
        img_path = os.path.join(folder, filename)
        try:
            img = Image.open(img_path).convert('RGB')  # 이미지를 RGB로 변환
            if transform:
                img = transform(img)
            images.append(img)
        except Exception as e:
            print(f"Error loading image {filename}: {e}")
    
    return torch.stack(images)

# 특징 벡터 추출 함수
def extract_features(images, model, batch_size=32):
    features_list = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = images[i:i+batch_size].cuda()  # GPU로 이동
            outputs = model(batch)  # 모델 출력
            features_list.append(outputs.cpu())  # CPU로 이동
    return torch.cat(features_list)

def feature_distance(feature_1, feature_b):
    return np.sum((feature_1-feature_b)**2)**0.5

def manifold_estimate(features_a, features_b, k=5):
    nn_a = NearestNeighbors(n_neighbors=k).fit(features_a)

    distance_a, _ = nn_a.kneighbors(features_a)
    radius_a = np.max(distance_a, axis=1)

    n = 0
    for pb in features_b:
        for i, pa in enumerate(features_a):
            if feature_distance(pa, pb) <= radius_a[i]:
                n += 1
                break
    
    return n/len(features_b)


# Precision-Recall 계산 함수
def compute_precision_recall(real_features, fake_features, k=5):
    precision = manifold_estimate(real_features, fake_features, k)
    recall = manifold_estimate(fake_features, real_features, k)

    return precision, recall

# 이미지 파일 폴더 경로를 받아 Precision-Recall 반환
def improved_precision_recall(real_images_folder, fake_images_folder, k=5):
    # 이미지 변환 (Inception 모델에 맞는 입력 크기와 정규화)
    transform = transforms.Compose([
        transforms.Resize((299, 299)),  # Inception 모델에 맞는 크기
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 실제 이미지와 생성된 이미지 로드
    real_images = load_images_from_folder(real_images_folder, transform)
    fake_images = load_images_from_folder(fake_images_folder, transform)

    # Inception V3 모델 (특징 벡터 추출용)
    inception_model = inception_v3(pretrained=True, transform_input=False)
    inception_model.eval()
    inception_model = inception_model.cuda()  # GPU로 이동 (가능한 경우)

    # 실제 데이터와 가짜 데이터의 특징 추출
    print(f"extracting features: {real_images_folder}")
    real_features = extract_features(real_images, inception_model).numpy()
    print(f"extracting features: {fake_images_folder}")
    fake_features = extract_features(fake_images, inception_model).numpy()

    return compute_precision_recall(real_features, fake_features, k)