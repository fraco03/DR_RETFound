import os
import argparse
import cv2
from PIL import Image, ImageFile
import contextlib
import io
import random
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, ConcatDataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


# Try with this but add before the transforms in the dataset class
def apply_ben_graham(image, sigma=10, **kwargs):
    # La logica originale di Ben Graham
    image_gaussian = cv2.GaussianBlur(image, (0, 0), sigma)
    # Calcolo: 4 * originale - 4 * gaussian + 128 (offset grigio)
    image = cv2.addWeighted(image, 4, image_gaussian, -4, 128)
    return image

def circular_crop(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    mask = gray > 7
    coords = np.argwhere(mask)
    
    if coords.size == 0:
        return image
        
    y_min, x_min = coords.min(axis=0)
    y_max, x_max = coords.max(axis=0)
    
    return image[y_min:y_max+1, x_min:x_max+1]

def get_transforms(is_train=True, normalize=True, to_tensor=True, clahe=False):
    ben_graham_transform = A.Lambda(name="BenGraham", image=apply_ben_graham, p=1.0)
    base_transforms = [
        A.Resize(224, 224),
        ben_graham_transform,
    ]
    
    if clahe:
        base_transforms.insert(1, A.CLAHE(clip_limit=2.0, p=1.0, tile_grid_size=(8, 8)))
   
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
                fill=(0, 0, 0)
            ),
        ])

    # Normalizzazione finale (sempre presente)
    if normalize:
        transforms.append(
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
        )
    if to_tensor:
        transforms.append(ToTensorV2())

    return A.Compose(transforms)
    

class RetinopathyDataset(Dataset):
    def __init__(self, df, img_dir, transform=None):
        self.df = df.reset_index(drop=True)
        if not isinstance(img_dir, (str, os.PathLike)):
            raise TypeError(
                f"img_dir must be a path-like string, got {type(img_dir).__name__}: {img_dir}"
            )
        self.img_dir = os.fspath(img_dir)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        img_name = str(self.df.iloc[idx, 0]) 
        if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
            img_name += '.png'  # Default to .png if no extension
        img_path = os.path.join(self.img_dir, img_name)
        image = safe_load_image_rgb(img_path)
        label_idx = idx

        # If image is corrupted (loader returned None), try other samples up to max_retries
        max_retries = 5
        retries = 0
        attempted = {idx}
        while image is None and retries < max_retries:
            retries += 1
            # sample another random index
            new_idx = random.randint(0, len(self.df) - 1)
            if new_idx in attempted:
                continue
            attempted.add(new_idx)
            img_name = str(self.df.iloc[new_idx, 0])
            if not img_name.lower().endswith(('.png', '.jpg', '.jpeg')):
                img_name += '.png'
            img_path = os.path.join(self.img_dir, img_name)
            image = safe_load_image_rgb(img_path)
            if image is not None:
                label_idx = new_idx

        if image is None:
            # last resort: return a black placeholder so training doesn't crash
            print(f"[dataset] Failed to load image after {max_retries} retries; returning placeholder for idx {idx}")
            image = np.zeros((512, 512, 3), dtype=np.uint8)

        image = circular_crop(image)

        label = float(self.df.iloc[label_idx, 1])

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented['image']

        return image, torch.tensor(label, dtype=torch.float32)


def _load_image_rgb(image_path):
    # Backwards-compatible loader (kept for external use)
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(
            f"Could not read image file: {image_path}. Check filename and path."
        )
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def safe_load_image_rgb(image_path):
    """Try to load an RGB image robustly.
    - If load succeeds returns an HxWx3 uint8 RGB numpy array.
    - If load fails, returns None so caller can skip.
    """
    ImageFile.LOAD_TRUNCATED_IMAGES = True

    try:
        with open(image_path, 'rb') as f:
            # suppress noisy libjpeg stderr messages during open
            with contextlib.redirect_stderr(io.StringIO()):
                img = Image.open(f)
                img = img.convert('RGB')
                arr = np.array(img)

        if arr.size == 0:
            raise ValueError('Empty image')

        return arr
    except Exception:
        print(f"[dataset] Corrupted image detected and skipped: {image_path}")
        return None


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