import torch

from models.mlp import MLP

from config import DEVICE


def test_forward():
    mlp = MLP(3, 8, 16, 2, False, DEVICE)
    tensor = torch.rand((4, 8), device=DEVICE)
    assert list(mlp(tensor).shape) == [4, 2]
    mlp = MLP(3, 8, 16, 2, True, DEVICE)
    assert list(mlp(tensor).shape) == [4, 2]
