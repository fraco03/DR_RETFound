import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget

def visualize_attention_retfound(model, image_tensor, original_image, device):
    model.eval()
    
    # Estraiamo il modello dalla scatola DataParallel (se presente)
    base_model = model.module if hasattr(model, 'module') else model
    
    target_layers = [base_model.blocks[-1].norm1]
    
    def reshape_transform(tensor, height=14, width=14):
        # Il Class Token è a [:, 0, :], i patch iniziano da 1
        result = tensor[:, 1:, :].reshape(tensor.size(0), height, width, tensor.size(2))
        result = result.transpose(2, 3).transpose(1, 2)
        return result

    # --- LA CORREZIONE FINALE ---
    # Passiamo il reshape_transform DIRETTAMENTE all'interno del costruttore
    cam = GradCAM(
        model=model, 
        target_layers=target_layers,
        reshape_transform=reshape_transform # Eccolo qui!
    )
    
    targets = [RawScoresOutputTarget()]
    
    # Generiamo la heatmap
    grayscale_cam = cam(input_tensor=image_tensor.to(device), targets=targets)[0, :]
    
    cam_image = show_cam_on_image(original_image, grayscale_cam, use_rgb=True)
    
    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(original_image)
    ax[0].set_title("Immagine in Input (CLAHE)")
    ax[0].axis('off')
    
    ax[1].imshow(cam_image)
    ax[1].set_title("Heatmap di Attenzione RETFound")
    ax[1].axis('off')
    
    plt.show()