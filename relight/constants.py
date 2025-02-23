import torch

C0 = 0.28209479177387814
flip_mitsuba = torch.tensor([-1.0, 1.0, -1.0], device="cuda")  # synthetic data generated using mitsuba
