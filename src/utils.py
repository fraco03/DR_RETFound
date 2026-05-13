import cv2
import numpy as np
import torch
import matplotlib.pyplot as plt
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from pytorch_grad_cam.utils.model_targets import RawScoresOutputTarget
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.optimize import minimize
from sklearn.metrics import cohen_kappa_score, confusion_matrix
from scipy.optimize import minimize

def visualizza_attenzione_retfound(model, image_tensor, original_image, device):
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
    

def apply_thresholds(y_pred, thresholds):
    """Trasforma i valori continui in classi usando le soglie."""
    y_pred_cls = np.zeros_like(y_pred)
    y_pred_cls[y_pred < thresholds[0]] = 0
    y_pred_cls[(y_pred >= thresholds[0]) & (y_pred < thresholds[1])] = 1
    y_pred_cls[(y_pred >= thresholds[1]) & (y_pred < thresholds[2])] = 2
    y_pred_cls[(y_pred >= thresholds[2]) & (y_pred < thresholds[3])] = 3
    y_pred_cls[y_pred >= thresholds[3]] = 4
    return y_pred_cls

# STRATEGIA A: Ottimizzazione Kappa (Massima accuratezza)
def kappa_objective(thresholds, y_true, y_pred):
    # Forziamo le soglie a rimanere ordinate per evitare errori matematici
    if not np.all(np.diff(thresholds) > 0):
        return 1e6
    y_pred_cls = apply_thresholds(y_pred, thresholds)
    return -cohen_kappa_score(y_true, y_pred_cls, weights='quadratic')

# STRATEGIA B: Ottimizzazione Medica (Minimizzare Falsi Negativi)
def medical_objective(thresholds, y_true, y_pred, fp_weight=1.0, fn_weight=10.0):
    if not np.all(np.diff(thresholds) > 0):
        return 1e6
    
    y_pred_cls = apply_thresholds(y_pred, thresholds)
    cm = confusion_matrix(y_true, y_pred_cls, labels=[0, 1, 2, 3, 4])
    
    # Falsi Negativi Critici: Malati (1,2,3,4) scambiati per Sani (0)
    fn = cm[1:, 0].sum() 
    # Falsi Positivi: Sani (0) scambiati per Malati (1,2,3,4)
    fp = cm[0, 1:].sum()
    
    # La funzione obiettivo pesa molto di più il perdere un malato (FN)
    return (fn * fn_weight) + (fp * fp_weight)

# --- ESECUZIONE ---

# 1. Recupera i dati di validazione (esempio)
# y_true = all_val_labels
# y_pred = all_val_predictions 

initial_thresholds = [0.5, 1.5, 2.5, 3.5]

# Esegui l'ottimizzazione per la KAPPA
res_kappa = minimize(kappa_objective, initial_thresholds, args=(y_true, y_pred), method='Nelder-Mead')
t_kappa = res_kappa.x

# Esegui l'ottimizzazione MEDICA (Prudente)
res_med = minimize(medical_objective, initial_thresholds, args=(y_true, y_pred, 1.0, 15.0), method='Nelder-Mead')
t_med = res_med.x

print(f"Soglie Kappa: {t_kappa}")
print(f"Soglie Mediche (Prudenti): {t_med}")