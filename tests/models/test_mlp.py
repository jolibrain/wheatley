import torch

from models.mlp import MLP


def test_forward():
    mlp = MLP(3, 8, 16, 2, False, "relu", torch.device("cpu"))
    tensor = torch.rand((4, 8), device=torch.device("cpu"))
    assert list(mlp(tensor).shape) == [4, 2]
    mlp = MLP(3, 8, 16, 2, True, "relu", torch.device("cpu"))
    assert list(mlp(tensor).shape) == [4, 2]
