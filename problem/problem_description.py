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
        self.affectations = affectations
        self.durations = durations
        self.max_duration = max_duration
        self.n_jobs = self.affectations.shape[0] if affectations is not None else n_jobs
        self.n_machines = self.affectations.shape[1] if affectations is not None else n_machines
        self.transition_model_config = transition_model_config
        self.reward_model_config = reward_model_config
