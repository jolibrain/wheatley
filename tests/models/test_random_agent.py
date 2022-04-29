from models.random_agent import RandomAgent


def test_select_action(env):
    re = RandomAgent(5, 5)
    for i in range(25):
        assert re.select_action(env) in [0, 5, 10, 15, 20]
