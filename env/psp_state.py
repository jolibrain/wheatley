import numpy as np


class PSPState:
    # TODO
    def __init__(
        self,
        env_specification,
        problem_description,
        problem,
        deterministic=True,
        observe_conflicts_as_cliques=True,
    ):
        self.problem = problem
        self.problem_description = problem_description
        self.n_features = env_specification.n_features
        self.features = np.zeros((self.problem["n_modes"], self.n_features))

    def to_features_and_edge_index(self, normalize, input_list):
        return self.features, np.ones((2, 5))

    def get_features_wo_real_dur(self):
        return self.features

    def get_selectable(self):
        return self.features[:, 1]
