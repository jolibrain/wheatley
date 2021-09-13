import gym
import numpy as np
from stable_baselines3 import PPO
import torch

from env.env import Env
from models.agent import Agent
from models.random_agent import RandomAgent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_problem
from utils.utils_testing import test_agent, get_ortools_makespan

from config import MAX_N_JOBS, MAX_N_MACHINES, MAX_DURATION, DEVICE
from args import args


def main():

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    print("Loading agent")
    agent = Agent.load(args.path)
    agent.model.verbose = 0

    random_agent = RandomAgent()
    print(
        "Launching inference.\n"
        f"Problem size : {args.n_j} jobs, {args.n_m} machines\n"
        f"Number of problem tested : {args.n_test_problems}"
    )

    diff_percentages = []
    rl_makespans = []
    or_tools_makespans = []
    random_makespans = []
    for i in range(args.n_test_problems):
        if (i + 1) % (args.n_test_problems // 10) == 0:
            print(f"{i+1}/{args.n_test_problems}")
        rl_makespan = test_agent(agent, args.n_j, args.n_m, MAX_DURATION)
        random_makespan = test_agent(
            random_agent, args.n_j, args.n_m, MAX_DURATION
        )
        or_tools_makespan = get_ortools_makespan(
            args.n_j, args.n_m, MAX_DURATION
        )
        diff_percentage = (
            100 * (rl_makespan - or_tools_makespan) / or_tools_makespan
        )
        rl_makespans.append(rl_makespan)
        or_tools_makespans.append(or_tools_makespan)
        random_makespans.append(random_makespan)
        diff_percentages.append(diff_percentage)

    print(
        f"Makespan for RL solution : {np.mean(rl_makespans):.0f}±{np.std(rl_makespans):.0f}"
    )
    print(
        f"Makespan for OR-tools solution : {np.mean(or_tools_makespans):.0f}±{np.std(or_tools_makespans):.0f}"
    )
    print(
        f"Makespan for random solution : {np.mean(random_makespans):.0f}±{np.std(random_makespans):.0f}"
    )
    print(
        f"Difference in percentage between OR-tools and RL : {np.mean(diff_percentages):.1f}±{np.std(diff_percentages):.1f}%"
    )


if __name__ == "__main__":
    main()
