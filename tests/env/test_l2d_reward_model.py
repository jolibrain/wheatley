from copy import deepcopy


from env.reward_models.l2d_reward_model import L2DRewardModel


def test_evaluate(env_observation):
    rm = L2DRewardModel()
    next_env_observation = deepcopy(env_observation)
    next_env_observation.features[0, 4:8] = 10
    next_env_observation.features[4, 4:8] = 14
    assert rm.evaluate(env_observation, None, next_env_observation) == 1
    next_env_observation.features[4, 4:8] = 10
    next_env_observation.features[5, 4:8] = 10
    assert rm.evaluate(env_observation, None, next_env_observation) == 5
