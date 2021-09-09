from copy import deepcopy
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback
import visdom

from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_problem

from config import MAX_DURATION


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
    def __init__(self, n_test_env, verbose=2):
        super(TestCallback, self).__init__(verbose=verbose)
        self.n_test_env = n_test_env
        self.vis = visdom.Visdom()
        self.makespans = []
        self.ortools_makespans = []
        self.random_makespans = []

    def _init_callback(self):
        self.testing_env = deepcopy(self.training_env.envs[0])

    def _on_step(self):
        random_agent = RandomAgent()
        mean_makespan = 0
        ortools_mean_makespan = 0
        random_mean_makespan = 0
        for _ in range(self.n_test_env):
            obs = self.testing_env.reset()
            done = False
            while not done:
                action, _ = self.model.predict(obs, deterministic=False)
                obs, reward, done, info = self.testing_env.step(action)
            schedule = self.testing_env.get_solution().schedule
            durations = self.testing_env.durations
            mean_makespan += np.max(schedule + durations) / self.n_test_env

            ortools_mean_makespan += (
                np.max(
                    solve_jssp(
                        self.testing_env.affectations,
                        self.testing_env.durations,
                    ).schedule
                    + self.testing_env.durations
                )
                / self.n_test_env
            )

            random_mean_makespan += (
                np.max(
                    random_agent.predict(
                        ProblemDescription(
                            self.testing_env.n_jobs,
                            self.testing_env.n_machines,
                            MAX_DURATION,
                            "L2D",
                            "L2D",
                            self.testing_env.affectations,
                            self.testing_env.durations,
                        )
                    ).schedule
                    + self.testing_env.durations
                )
                / self.n_test_env
            )

        self.makespans.append(mean_makespan)
        self.ortools_makespans.append(ortools_mean_makespan)
        self.random_makespans.append(random_mean_makespan)
        self.vis.line(
            Y=np.array(
                [self.makespans, self.ortools_makespans, self.random_makespans]
            ).T,
            win="test_makespan",
        )
        self.vis.line(
            Y=np.stack(
                [
                    np.array(self.makespans)
                    / np.array(self.ortools_makespans),
                    np.array(self.random_makespans)
                    / np.array(self.ortools_makespans),
                ]
            ).T,
            win="test_makespan_ratio",
        )
        return True
