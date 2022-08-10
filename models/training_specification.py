class TrainingSpecification:
    def __init__(
        self,
        total_timesteps,
        n_validation_env,
        fixed_validation,
        validation_batch_size,
        validation_freq,
        display_env,
        path,
        custom_heuristic_name,
        ortools_strategy,
        max_time_ortools,
        scaling_constant_ortools,
        vecenv_type,
    ):
        self.total_timesteps = total_timesteps
        self.n_validation_env = n_validation_env
        self.fixed_validation = fixed_validation
        self.validation_batch_size = validation_batch_size
        self.validation_freq = validation_freq
        self.display_env = display_env
        self.path = path
        self.custom_heuristic_name = custom_heuristic_name
        self.ortools_strategy = ortools_strategy
        self.max_time_ortools = max_time_ortools
        self.scaling_constant_ortools = scaling_constant_ortools
        self.vecenv_type = vecenv_type

    def print_self(self):
        print(
            f"==========Training Description==========\n"
            f"Number of timesteps (total)       {self.total_timesteps}\n"
            f"Validation frequency:             {self.validation_freq}\n"
            f"Episodes per validation session:  {self.n_validation_env}\n"
        )
