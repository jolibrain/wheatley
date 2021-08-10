import sys

sys.path.append("..")

import torch

from models.mlp import MLP


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLP(3, 8, 16, 2, False, device)
    tensor = torch.Tensor(4, 8)
    print(tensor)
    print(mlp(tensor))
    mlp = MLP(3, 8, 16, 2, True, device)
    print(mlp(tensor))
