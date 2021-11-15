from models.random_agent import RandomAgent


def test_select_action(env_observation):
    re = RandomAgent(3, 3)
    for i in range(10):
        assert re.select_action(env_observation.to_gym_observation()) in [0, 3, 6]
