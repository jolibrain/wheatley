from copy import deepcopy


from env.sparse_reward_model import SparseRewardModel


def test_evaluate(env_observation):
    rm = SparseRewardModel()
    next_env_observation = deepcopy(env_observation)
    assert rm.evaluate(env_observation, None, next_env_observation) == 0
    next_env_observation.features[:, 0] = 0
    assert rm.evaluate(env_observation, None, next_env_observation) == -15
