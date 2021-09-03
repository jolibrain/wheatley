from models.random_agent import RandomAgent


def test_select_action(env_observation):
    re = RandomAgent()
    for i in range(10):
        assert re.select_action(env_observation.to_gym_observation()) in [
            0,
            303,
            606,
        ]
