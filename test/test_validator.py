import os

import torch

from alg.ppo import PPO
from env.psp_env import PSPEnv
from models.agent_validator import AgentValidator


def test_validator_psp(
    problem_description_small,
    env_specification_small,
    training_specification,
    psp_agent,
    disable_visdom,
):
    validator = AgentValidator(
        problem_description_small,
        env_specification_small,
        torch.device("cpu"),
        training_specification,
        disable_visdom,
    )

    alg = PPO(training_specification, PSPEnv, validator)
    # normally initialzed in train
    alg.optimizer = alg.optimizer_class(psp_agent.parameters(), lr=0.1)
    alg.global_step = 0

    os.makedirs(validator.path, exist_ok=True)
    validator.validate(psp_agent, alg)
