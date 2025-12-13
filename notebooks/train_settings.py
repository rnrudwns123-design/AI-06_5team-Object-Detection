import os
import random
import numpy as np
import torch

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch. cuda. manual_seed_all(seed)
    # PyTorch 결정론 설정
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_aug():
    pass