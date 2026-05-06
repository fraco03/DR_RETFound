from functools import partial
import timm.models.vision_transformer
import torch
import torch.nn as nn

class VisionTransformer(timm.models.vision_transformer.VisionTransformer):
    """
    Vision Transformer wrapper for RETFound MAE.
    Uses timm's built-in pooling behavior.
    """
    def __init__(self, **kwargs):
        super(VisionTransformer, self).__init__(**kwargs)

def RETFound_mae(**kwargs):
    """
    Initializes the ViT-Large architecture used by RETFound.
    """
    model = VisionTransformer(
        patch_size=16, embed_dim=1024, depth=24, num_heads=16, mlp_ratio=4, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), **kwargs)
    return model

def build_retfound_regression(weights_path, device, *, trusted_checkpoint=True):
    """
    Builds the model, sets up the 1-neuron regression head, 
    and loads the pre-trained MAE weights safely.
    """
    # 1. Initialize for regression (1 continuous output)
    model = RETFound_mae(num_classes=1, global_pool="avg", img_size=384)
    
    # 2. Load the downloaded checkpoint
    # PyTorch 2.6 defaults to weights_only=True; disable it for trusted checkpoints.
    if trusted_checkpoint:
        checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    else:
        checkpoint = torch.load(weights_path, map_location='cpu')
    
    # Handle different checkpoint formats
    checkpoint_model = checkpoint['model'] if 'model' in checkpoint else checkpoint
    
    # 3. Filter out the head weights
    # The pre-trained head dimension will clash with our 1-neuron head, so we drop it.
    state_dict = model.state_dict()
    for key in ['head.weight', 'head.bias']:
        if key in checkpoint_model and checkpoint_model[key].shape != state_dict[key].shape:
            del checkpoint_model[key]
            
    # 4. Load remaining weights and move to device
    model.load_state_dict(checkpoint_model, strict=False)
    model.to(device)
    
    return model