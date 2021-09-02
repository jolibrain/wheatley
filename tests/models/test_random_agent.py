from models.random_agent import RandomAgent


def test_select_action(gym_observation):
    re = RandomAgent()
    for i in range(10):
        assert re.select_action(gym_observation) in [0, 1, 2]
