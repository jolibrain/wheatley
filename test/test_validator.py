import torch
from models.agent_validator import AgentValidator
from env.psp_env import PSPEnv
from alg.ppo import PPO
import os


def test_validator_psp(
    problem_description_small,
    env_specification_small,
    training_specification,
    psp_agent,
    agent_specification,
):
    validator = AgentValidator(
        problem_description_small,
        env_specification_small,
        torch.device("cpu"),
        training_specification,
    )

    alg = PPO(agent_specification, PSPEnv, validator)
    # normally initialzed in train
    alg.optimizer = alg.optimizer_class(psp_agent.parameters(), lr=0.1)
    alg.global_step = 0

    os.makedirs(validator.path, exist_ok=True)
    validator.validate(psp_agent, alg)
