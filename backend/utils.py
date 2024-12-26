import torch

def normalize_tensor(tensor):
    min_val = torch.min(tensor)
    max_val = torch.max(tensor)
    return (tensor - min_val) / (max_val - min_val)

def load_model(path, device):
    model = torch.load(path, map_location=device)
    return model