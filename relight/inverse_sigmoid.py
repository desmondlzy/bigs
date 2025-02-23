import torch

def inverse_sigmoid_safe(x, eps=1e-7):
    x_safe = torch.clamp(x, eps, 1 - eps)
    res = torch.log(x_safe / (1 - x_safe))

    return res
