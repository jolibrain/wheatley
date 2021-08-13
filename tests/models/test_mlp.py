import torch

from models.mlp import MLP


def test_forward():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mlp = MLP(3, 8, 16, 2, False, device)
    tensor = torch.Tensor(4, 8, device=device)
    assert list(mlp(tensor).shape) == [4, 2]
    mlp = MLP(3, 8, 16, 2, True, device)
    assert list(mlp(tensor).shape) == [4, 2]
