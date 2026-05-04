import os
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import ConcatDataset, DataLoader, WeightedRandomSampler

from src.dataset import RetinopathyDataset, get_transforms
import numpy as np
from src.loss import SmoothL1Loss
from src.model_setup import build_retfound_regression

def build_config():
    return {
        "aptos_csv": r"C:\Users\frab0\OneDrive\Desktop\aptos2019-blindness-detection\aptos\train.csv",
        "messidor_csv": r"C:\Users\frab0\OneDrive\Desktop\messidor-2\messidor_data.csv",
        "aptos_img_dir": r"C:\Users\frab0\OneDrive\Desktop\aptos2019-blindness-detection\aptos\train_images",
        "messidor_img_dir": r"C:\Users\frab0\OneDrive\Desktop\messidor-2\messidor-2\preprocess",
        "weights_path": r"C:\Users\frab0\OneDrive\Desktop\RETFound-MAE-Large.pth",
        "checkpoint_dir": "weights",
        "test_size": 0.2,
        "random_state": 42,
        "batch_size": 16,
        "num_workers": 4,
        "epochs": 10,
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "loss_beta": 1.0,
        "sampler": "weighted",
    }


def load_splits(cfg):
    df_messidor = pd.read_csv(cfg["messidor_csv"])
    df_aptos = pd.read_csv(cfg["aptos_csv"])

    aptos_train, aptos_val = train_test_split(
        df_aptos,
        test_size=cfg["test_size"],
        stratify=df_aptos["diagnosis"],
        random_state=cfg["random_state"],
    )

    messidor_train, messidor_val = train_test_split(
        df_messidor,
        test_size=cfg["test_size"],
        stratify=df_messidor["diagnosis"],
        random_state=cfg["random_state"],
    )

    return aptos_train, aptos_val, messidor_train, messidor_val


def build_datasets(cfg):
    aptos_train, aptos_val, messidor_train, messidor_val = load_splits(cfg)

    train_transform = get_transforms(is_train=True)
    val_transform = get_transforms(is_train=False)

    ds_aptos_train = RetinopathyDataset(
        aptos_train,
        cfg["aptos_img_dir"],
        transform=train_transform,
    )
    ds_messidor_train = RetinopathyDataset(
        messidor_train,
        cfg["messidor_img_dir"],
        transform=train_transform,
    )

    ds_aptos_val = RetinopathyDataset(
        aptos_val,
        cfg["aptos_img_dir"],
        transform=val_transform,
    )
    ds_messidor_val = RetinopathyDataset(
        messidor_val,
        cfg["messidor_img_dir"],
        transform=val_transform,
    )

    train_dataset = ConcatDataset([ds_messidor_train, ds_aptos_train])
    val_dataset = ConcatDataset([ds_messidor_val, ds_aptos_val])

    return train_dataset, val_dataset


def build_sampler(train_dataset):
    labels = []
    for _, label in train_dataset:
        labels.append(label.item())
    labels = np.array(labels)

    class_counts = np.bincount(np.round(labels).astype(int))
    print(f"Distribuzione immagini per classe: {class_counts}")

    class_weights = 1.0 / (class_counts + 1e-5)
    sample_weights = np.array([class_weights[int(round(l))] for l in labels])
    sample_weights = torch.from_numpy(sample_weights).float()

    return WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )


def build_loaders(cfg, train_dataset, val_dataset):
    train_sampler = None
    if cfg["sampler"] == "weighted":
        train_sampler = build_sampler(train_dataset)

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg["batch_size"],
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=cfg["num_workers"],
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=cfg["batch_size"],
        shuffle=False,
        num_workers=cfg["num_workers"],
    )

    return train_loader, val_loader


def build_checkpoint_name(cfg, epoch_idx, val_loss):
    parts = [
        f"bs{cfg['batch_size']}",
        f"lr{cfg['lr']}",
        f"wd{cfg['weight_decay']}",
        f"ep{cfg['epochs']}",
        f"beta{cfg['loss_beta']}",
        f"sampler{cfg['sampler']}",
        f"seed{cfg['random_state']}",
        f"epoch{epoch_idx}",
        f"valloss{val_loss:.4f}",
    ]
    return "retfound_" + "__".join(parts) + ".pt"


def save_checkpoint(cfg, model, optimizer, epoch_idx, val_loss):
    os.makedirs(cfg["checkpoint_dir"], exist_ok=True)
    ckpt_name = build_checkpoint_name(cfg, epoch_idx, val_loss)
    ckpt_path = os.path.join(cfg["checkpoint_dir"], ckpt_name)

    torch.save(
        {
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "epoch": epoch_idx,
            "val_loss": val_loss,
            "config": cfg,
        },
        ckpt_path,
    )
    return ckpt_path


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    total_loss = 0.0
    for images, labels in loader:
        images, labels = images.to(device), labels.float().to(device)
        optimizer.zero_grad(set_to_none=True)
        outputs = model(images).squeeze()
        batch_loss = loss_fn(outputs, labels)
        batch_loss.backward()
        optimizer.step()
        total_loss += batch_loss.item()
    return total_loss / max(1, len(loader))


def eval_one_epoch(model, loader, loss_fn, device):
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.float().to(device)
            outputs = model(images).squeeze()
            batch_loss = loss_fn(outputs, labels)
            total_loss += batch_loss.item()
    return total_loss / max(1, len(loader))


def main():
    cfg = build_config()
    train_dataset, val_dataset = build_datasets(cfg)
    train_loader, val_loader = build_loaders(cfg, train_dataset, val_dataset)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_retfound_regression(cfg["weights_path"], device=device)
    loss_fn = SmoothL1Loss(beta=cfg["loss_beta"])
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"],
    )

    for epoch_idx in range(1, cfg["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        val_loss = eval_one_epoch(model, val_loader, loss_fn, device)

        print(
            f"Epoch {epoch_idx}/{cfg['epochs']}, "
            f"Training Loss: {train_loss:.4f}, Validation Loss: {val_loss:.4f}"
        )

        ckpt_path = save_checkpoint(cfg, model, optimizer, epoch_idx, val_loss)
        print(f"Checkpoint salvato: {ckpt_path}")

if __name__ == "__main__":
    main()

