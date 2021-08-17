from utils.utils import generate_problem


class ProblemDescription:
    def __init__(
        self,
        n_jobs,
        n_machines,
        max_duration,
        transition_model_config,
        reward_model_config,
        affectations=None,
        durations=None,
    ):
        if affectations is not None and durations is not None:
            self.affectations = affectations
            self.durations = durations
        elif affectations is not None and durations is None:
            raise Exception(
                "If you provide affectations, please provide durations"
            )
        elif affectations is None and durations is not None:
            raise Exception(
                "If you provide durations, please provide affectations"
            )
        else:
            self.affectations, self.durations = generate_problem(
                n_jobs, n_machines, max_duration
            )
        self.n_jobs = self.affectations.shape[0]
        self.n_machines = self.affectations.shape[1]
        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config
