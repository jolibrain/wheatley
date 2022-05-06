import torch


def test_forward(gym_observation, features_extractor):
    gym_observation["features"] = gym_observation["features"].to(torch.device("cpu")).float()
    gym_observation["edge_index"] = gym_observation["edge_index"].to(torch.device("cpu")).long()
    gym_observation["mask"] = gym_observation["mask"].to(torch.device("cpu")).float()
    features = features_extractor(gym_observation)
    assert list(features.shape) == [2, 9, (8 + 4 * 64) * 2]
