from models.random_agent import RandomAgent

from config import MAX_N_NODES


def test_select_action(env_observation):
    re = RandomAgent()
    for i in range(10):
        assert re.select_action(env_observation.to_gym_observation()) in [
            0,
            3 * MAX_N_NODES + 3,
            6 * MAX_N_NODES + 6,
        ]
