import os
import argparse
import cv2
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

def circular_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    coords = np.argwhere(mask)
    
    if coords.size == 0:
        return image
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return image[y_min:y_max+1, x_min:x_max+1]

def get_transforms(is_train=True, normalize=True, to_tensor=True):
    base_transforms = [
        A.Resize(384, 384),
        #A.CLAHE(clip_limit=2.0, p=1.0), # p=1.0 lo rende deterministico per tutti
    ]
    
    transforms = base_transforms.copy()

    if is_train:
        # Passaggi solo per il Training (Augmentation)
        transforms.extend([
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.ShiftScaleRotate(
                shift_limit=0.05, 
                scale_limit=0.05, 
                rotate_limit=15, 
                p=0.5,
                # Explicitly fill empty space with black pixels
                border_mode=cv2.BORDER_CONSTANT, 
                value=(0, 0, 0)
            ),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
        ])

    # Normalizzazione finale (sempre presente)
    if normalize:
        transforms.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    if to_tensor:
        transforms.append(ToTensorV2())

    return A.Compose(transforms)
    
# Try with this but add before the transforms in the dataset class
def standardize_fundus_colors(image, sigmaX=10):
    """
    Applies Ben Graham's preprocessing to standardize lighting and color 
    across different fundus cameras.
    Input must be an RGB image.
    """
    # Apply Gaussian blur to estimate the background illumination
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX)
    
    # Subtract the background and add a neutral gray (128) 
    # Formula: alpha * image + beta * blurred + gamma
    standardized = cv2.addWeighted(image, 4, blurred, -4, 128)
    
    return standardized

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx, 0]) 
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'  # Default to .png if no extension
        img_path = os.path.join(self.img_dir, img_name)
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(
                f"Could not read image file: {img_path}. "
                "Check filename, extension, and dataset path."
            )
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = circular_crop(image)

        label = float(self.df.iloc[idx, 1])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.float32)


def _load_image_rgb(image_path):
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image file: {image_path}. Check filename and path."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def main():

    image = _load_image_rgb("C:\\Users\\frab0\\Downloads\\20051020_43808_0100_PP.png")
    image = circular_crop(image)

    transform = get_transforms(
        is_train=True,
        normalize=False,
        to_tensor=True,
    )
    augmented = transform(image=image)
    processed = augmented["image"]

    processed = processed.permute(1, 2, 0).cpu().numpy()

    plt.figure(figsize=(6, 6))
    plt.imshow(processed)
    plt.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()