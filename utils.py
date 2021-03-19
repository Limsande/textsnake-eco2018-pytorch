import torch


def to_device(data, device):
    """Loads given data into the RAM of given device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_device() -> torch.device:
    """Return a torch.device of type 'cuda' if available, else of type 'cpu'"""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
