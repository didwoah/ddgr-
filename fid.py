import os
from pytorch_fid import fid_score

from tqdm import tqdm

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import ToTensor, Resize
import torch
import numpy as np
from torchvision.models import inception_v3, Inception_V3_Weights
from scipy.linalg import sqrtm

from dataset_utils import RelabeledDataset

# *_data_path: 이미지들이 들어있는 폴더 경로
def __get_fid_value(real_data_path, fake_data_path, batch_size=50, dims=2048, device='cuda'):

    # FID 스코어 계산
    fid_value = fid_score.calculate_fid_given_paths(
        [real_data_path, fake_data_path], 
        batch_size=batch_size, 
        device=device, 
        dims=dims
        )

    return fid_value

def change_file_extensions(folder_path, old_extension, new_extension):
    # 폴더 내 모든 파일 검색
    for filename in os.listdir(folder_path):
        # 파일이 해당 확장자로 끝나는지 확인
        if filename.endswith(old_extension):
            # 파일의 전체 경로 생성
            old_file = os.path.join(folder_path, filename)
            # 새 파일 이름 생성 (확장자 변경)
            new_file = os.path.join(folder_path, filename.replace(old_extension, new_extension))
            # 파일 이름 변경
            os.rename(old_file, new_file)


# Inception 모델 준비
def get_inception_model():
    model = inception_v3(weights=Inception_V3_Weights.DEFAULT, transform_input=False).eval()
    model.fc = torch.nn.Identity()  # Feature vector만 추출
    return model


# 특징 추출 함수
def extract_feature(image, model, device="cuda"):
    image = Resize((299, 299))(image)  # Inception 모델 크기에 맞춤
    if not isinstance(image, torch.Tensor):
        image = ToTensor()(image)
    image = image.unsqueeze(0).to(device)  # 배치 추가
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

def extract_features(dataloader, model, device="cuda"):
    features = []
    model.eval()
    for images in dataloader:
        images = images.to(device)
        with torch.no_grad():
            feature = model(images)
        features.append(feature.cpu())
    return torch.cat(features, dim=0).numpy()


def calculate_fid(real_features, fake_features):
    mu_r, sigma_r = np.mean(real_features, axis=0), np.cov(real_features, rowvar=False)
    mu_f, sigma_f = np.mean(fake_features, axis=0), np.cov(fake_features, rowvar=False)
    
    diff = mu_r - mu_f
    covmean = sqrtm(sigma_r @ sigma_f)
    if np.iscomplexobj(covmean):
        covmean = covmean.real  # 허수 제거
    
    fid = diff.dot(diff) + np.trace(sigma_r + sigma_f - 2 * covmean)
    return fid

class InceptionNetDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

        self.transform = Resize((299, 299))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.transform(self.dataset[idx][0])


def get_fid_value(real_dataset: Dataset, fake_dataset: Dataset, device, batch_size=32):
    inception_model = get_inception_model().to(device)

    real_dataset = InceptionNetDataset(real_dataset)
    real_dataloader = DataLoader(real_dataset, batch_size=batch_size,shuffle=False)
    real_pbar = tqdm(real_dataloader, desc="extracting feature of real dataset", leave=False)
    real_features = extract_features(real_pbar, inception_model, device)

    fake_dataset = InceptionNetDataset(fake_dataset)
    fake_dataloader = DataLoader(fake_dataset, batch_size=batch_size,shuffle=False)
    fake_pbar = tqdm(fake_dataloader, desc="extracting feature of fake dataset", leave=False)
    fake_features = extract_features(fake_pbar, inception_model, device)

    return calculate_fid(real_features, fake_features)