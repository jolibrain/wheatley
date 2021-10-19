import torch


def test_forward(gym_observation, features_extractor):
    gym_observation["features"] = gym_observation["features"].to(torch.device("cpu")).float()
    gym_observation["edge_index"] = gym_observation["edge_index"].to(torch.device("cpu")).long()
    gym_observation["mask"] = gym_observation["mask"].to(torch.device("cpu")).float()
    features = features_extractor(gym_observation)
    assert list(features.shape) == [2, 10, 73]
    mask = features[0, 1:10, 64:74]
    for i in range(9):
        for j in range(9):
            if i == j and i in [0, 3, 6]:
                assert mask[i, j].item() == 1
            else:
                assert mask[i, j].item() == 0
