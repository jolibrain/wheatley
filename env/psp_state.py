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

    # features :
    # 0: is_affected
    # 1: is selectable
    # 2,3,4,5 : tct
    # 6.. 5+max_n_resources : level of resource i used by this mode
    # others: duration [optional] 6+max_n_resources : 6+max_n_resources +4

    def to_features_and_edge_index(self, normalize, input_list):
        return self.features, np.ones((2, 5))

    def get_features_wo_real_dur(self):
        return self.features

    def get_selectable(self):
        return self.features[:, 1]

    def done(self):
        return np.sum(self.features[:, 0]) == self.problem["n_jobs"]

    def tct():
        return self.features[:, 2:6]

    def render_solution(self, schedule, scaling):
        pass

    def get_solution(self):
        pass
