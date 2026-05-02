import pandas as pd
from sklearn.model_selection import train_test_split 
from torch.utils.data import ConcatDataset, DataLoader

from src.dataset import RetinopathyDataset, get_transforms
import PIL.Image as Image
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1. Load dataframes
    #df_aptos = pd.read_csv("data/raw/aptos_2019/train.csv")
    df_messidor = pd.read_csv("C:\\Users\\frab0\\OneDrive\\Desktop\\messidor-2\\messidor_data.csv")
    df_aptos = pd.read_csv("C:\\Users\\frab0\\OneDrive\\Desktop\\aptos2019-blindness-detection\\aptos\\train.csv")
    # 2. Stratified split for APTOS (80% train, 20% val)
    aptos_train, aptos_val = train_test_split(
       df_aptos, test_size=0.2, stratify=df_aptos['diagnosis'], random_state=42
    )

    # 3. Stratified split for Messidor
    messidor_train, messidor_val = train_test_split(
        df_messidor, test_size=0.2, stratify=df_messidor['diagnosis'], random_state=42
    )

    # 4. Create separate Dataset instances
    # Training datasets (with data augmentation)
    train_transform = get_transforms(is_train=True)
    ds_aptos_train = RetinopathyDataset(aptos_train, r"C:\Users\frab0\OneDrive\Desktop\aptos2019-blindness-detection\aptos\train_images", transform=train_transform)
    ds_messidor_train = RetinopathyDataset(messidor_train, r"C:\Users\frab0\OneDrive\Desktop\messidor-2\messidor-2\preprocess", transform=train_transform)

    # Validation datasets (NO data augmentation)
    val_transform = get_transforms(is_train=False)
    ds_aptos_val = RetinopathyDataset(aptos_val, r"C:\Users\frab0\OneDrive\Desktop\aptos2019-blindness-detection\aptos\train_images", transform=val_transform)
    ds_messidor_val = RetinopathyDataset(messidor_val, r"C:\Users\frab0\OneDrive\Desktop\messidor-2\messidor-2\preprocess", transform=val_transform)

    # 5. Concatenate them!
    train_dataset = ConcatDataset([ds_messidor_train, ds_aptos_train])
    val_dataset = ConcatDataset([ds_messidor_val, ds_aptos_val])

    # 6. Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, num_workers=4)

    # 7. Test the DataLoaders
    for images, labels in train_loader:
        print("Batch of images shape:", images.shape)
        print("Batch of labels shape:", labels.shape)
        
        # Move tensor to CPU and change shape from [C, H, W] to [H, W, C]
        img = images[0].permute(1, 2, 0).cpu().numpy()
        
        # 1. Denormalize using ImageNet stats
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img = std * img + mean
        
        # 2. Clip values between 0 and 1 to fix any floating point errors
        img = np.clip(img, 0, 1)
        
        # 3. Convert back to 0-255 scale
        img = (img * 255).astype(np.uint8)
        
        # Show the image using Matplotlib (better for Jupyter/Colab)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f"Label: {labels[0].item()}")
        plt.show()
        
        


if __name__ == "__main__":
    main()

