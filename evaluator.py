import os
import torch
import numpy as np
from torchvision import models, transforms
from torchvision.datasets import ImageFolder
from torchvision.utils import save_image
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from tqdm import tqdm
from skimage.metrics import structural_similarity as compare_ssim
from sklearn.neighbors import NearestNeighbors
from scipy import linalg
from cleanfid import fid
import argparse

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==================== Dataset ====================
class ImageDataset(Dataset):
    def __init__(self, folder, transform):
        self.folder = folder
        self.files = sorted(os.listdir(folder))
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.folder, self.files[idx])
        image = Image.open(img_path).convert('RGB')
        image = self.transform(image)
        return image


# ==================== Feature Extractor ====================
class InceptionActivations(torch.nn.Module):
    def __init__(self):
        super().__init__()
        # Loading InceptionV3
        inception = models.inception_v3(pretrained=True, transform_input=False, aux_logits=True)
        inception.fc = torch.nn.Identity()
        self.features = inception
        self.features.eval()
        for p in self.features.parameters():
            p.requires_grad = False
        self.features.to(device)

    def forward(self, x):
        with torch.no_grad():
            x = self.features(x)
            return x.view(x.size(0), -1)


# ==================== SSIM ====================
def compute_ssim(dataset1, dataset2):
    ssim_list = []
    for i in range(len(dataset1)):
        img1 = np.array(Image.open(os.path.join(dataset1, sorted(os.listdir(dataset1))[i])).convert("L"))
        img2 = np.array(Image.open(os.path.join(dataset2, sorted(os.listdir(dataset2))[i])).convert("L"))
        ssim_score = compare_ssim(img1, img2)
        ssim_list.append(ssim_score)
    return np.mean(ssim_list)


# ==================== Precision & Recall ====================
def compute_pr(features_real, features_fake, k=3):

    real = features_real.cpu().numpy()
    fake = features_fake.cpu().numpy()

    nbr_real = NearestNeighbors(n_neighbors=k).fit(real)
    dist_real, _ = nbr_real.kneighbors(fake)
    precision = np.mean(dist_real[:, -1] < np.percentile(dist_real, 95))

    nbr_fake = NearestNeighbors(n_neighbors=k).fit(fake)
    dist_fake, _ = nbr_fake.kneighbors(real)
    recall = np.mean(dist_fake[:, -1] < np.percentile(dist_fake, 95))

    return precision, recall


# ==================== Main Evaluation Function ====================
def evaluate(real_dir, fake_dir, batch_size=32):
    # Transforms
    tf = transforms.Compose([
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
    ])

    real_dataset = ImageDataset(real_dir, tf)
    fake_dataset = ImageDataset(fake_dir, tf)

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False)
    fake_loader = DataLoader(fake_dataset, batch_size=batch_size, shuffle=False)

    # Extract features
    model = InceptionActivations()
    real_feats, fake_feats = [], []

    print("Extracting features...")
    for real_batch, fake_batch in zip(tqdm(real_loader), tqdm(fake_loader)):
        real_feats.append(model(real_batch.to(device)))
        fake_feats.append(model(fake_batch.to(device)))

    real_feats = torch.cat(real_feats, dim=0)
    fake_feats = torch.cat(fake_feats, dim=0)

    # Compute statistics
    mu1, sigma1 = real_feats.mean(dim=0), torch.cov(real_feats.T)
    mu2, sigma2 = fake_feats.mean(dim=0), torch.cov(fake_feats.T)

    mu1 = mu1.cpu().numpy()
    mu2 = mu2.cpu().numpy()
    sigma1 = sigma1.cpu().numpy()
    sigma2 = sigma2.cpu().numpy()

    # FID
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    fid_score = np.sum((mu1 - mu2) ** 2) + np.trace(sigma1 + sigma2 - 2 * covmean.real)

    # sFID using clean-fid
    # My clean-fid version is 0.1.35
    sfid = fid.compute_fid(real_dir, fake_dir, mode='clean')
    kid = fid.compute_kid(real_dir, fake_dir)

    # SSIM
    print("Computing SSIM...")
    ssim = compute_ssim(real_dir, fake_dir)

    # Precision and Recall
    print("Computing Precision and Recall...")
    precision, recall = compute_pr(real_feats, fake_feats)

    print(f"\n===== Evaluation Results =====")
    print(f"FID: {fid_score:.4f}")
    print(f"sFID: {sfid:.4f}")
    print(f"KID: {kid:.4f}")
    print(f"SSIM: {ssim:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    return {
        "FID": fid_score,
        "sFID": sfid,
        "KID": kid,
        "SSIM": ssim,
        "Precision": precision,
        "Recall": recall,
    }


def main(args):
    res = evaluate(args.reference_path, args.samples_path)
    print(res)


# ==================== Example Call ====================
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-path", type=str, required=True)
    parser.add_argument("--samples-path", type=str, required=True)
    args = parser.parse_args()
    main(args)
