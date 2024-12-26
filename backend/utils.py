import torch
import torch.nn as nn
from backend.models import Generator  # Import the Generator model

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def normalize_tensor(tensor):
    """
    Normalize a tensor to the range [0, 1].
    Args:
        tensor (torch.Tensor): Input tensor.
    Returns:
        torch.Tensor: Normalized tensor.
    """
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def load_model(path, device):
    model = Generator(ngpu=1)
    model.apply(weights_init)  # Initialize weights
    state_dict = torch.load(path, map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    return model



