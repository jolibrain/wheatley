from utils.ortools import get_ortools_makespan
from models.agent_validator import AgentValidator


class PSPAgentValidator(AgentValidator):
    def __init__(
        self,
        problem_description,
        env_specification,
        env_cls,
        device,
        training_specification,
        verbose=2,
    ):
        super().__init__(
            problem_description,
            env_specification,
            env_cls,
            device,
            training_specification,
            verbose=2,
        )

    def _get_ortools_makespan(self, i):
        return get_ortools_makespan_psp(
            self.validation_envs[i],
            self.env_specification.n_features,
            self.max_time_ortools,
            self.scaling_constant_ortools,
            self.ortools_strategy,
        )
