from copy import deepcopy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import visdom

from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_problem


def test_agent(agent, n_j, n_m, max_duration):
    affectations, durations = generate_problem(n_j, n_m, max_duration)
    problem_description = ProblemDescription(
        n_j,
        n_m,
        max_duration,
        "L2D",
        "L2D",
        affectations,
        durations,
    )
    solution = agent.predict(problem_description)
    makespan = np.max(solution + durations)
    return makespan


def get_ortools_makespan(n_j, n_m, max_duration):
    affectations, durations = generate_problem(n_j, n_m, max_duration)
    solution = solve_jssp(affectations, durations)
    makespan = np.max(solution.schedule + durations)
    return makespan


class TestCallback(BaseCallback):
    def __init__(self, verbose=2):
        super(TestCallback, self).__init__(verbose=verbose)
        self.vis = visdom.Visdom()
        self.makespans = []
        self.ortools_makespans = []

    def _init_callback(self):
        self.testing_env = deepcopy(self.training_env.envs[0])

    def _on_step(self):
        obs = self.testing_env.reset()
        done = False
        self.ortools_makespans.append(
            np.max(
                solve_jssp(
                    self.testing_env.affectations, self.testing_env.durations
                ).schedule
                + self.testing_env.durations
            )
        )
        while not done:
            action, _ = self.model.predict(obs, deterministic=False)
            obs, reward, done, info = self.testing_env.step(action)
        schedule = self.testing_env.get_solution().schedule
        durations = self.testing_env.durations
        makespan = np.max(schedule + durations)
        self.makespans.append(makespan)
        print(len(self.makespans))
        print(len(self.ortools_makespans))
        self.vis.line(
            Y=np.array([self.makespans, self.ortools_makespans]).T,
            X=np.arange(len(self.makespans)),
            win="test_makespan",
        )
        return True
