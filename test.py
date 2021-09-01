import gym
import numpy as np
from stable_baselines3 import PPO
import torch

from env.env import Env
from models.agent import Agent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp
from utils.utils import generate_problem

from config import MAX_N_JOBS, MAX_N_MACHINES, DEVICE
from args import args


def main():
    testing_affectations, testing_durations = generate_problem(
        args.n_j_testing, args.n_m_testing, args.max_duration
    )
    problem_description = ProblemDescription(
        args.n_j_testing,
        args.n_m_testing,
        args.max_duration,
        "L2D",
        "L2D",
        testing_affectations,
        testing_durations,
    )
    testing_env = Env(problem_description)
    agent = Agent.load(args.path, testing_env)
    print(
        f"Launching inference. Problem size : {args.n_j_testing} jobs, {args.n_m_testing} machines"
    )

    solution = agent.predict(problem_description)
    solution_or_tools = solve_jssp(testing_affectations, testing_durations)

    print(testing_affectations)
    print(testing_durations)
    print("Solution found by the model")
    print(solution.schedule)
    print("Solution found by OR-tools solver : ")
    print(solution_or_tools.schedule)

    makespan = np.max(solution.schedule + testing_durations)
    makespan_or_tools = np.max(solution_or_tools.schedule + testing_durations)
    print(f"Makespan for found solution : {makespan}")
    print(f"Makespan for OR-tools solution : {makespan_or_tools}")
    print(
        f"Difference in percentage : {(makespan - makespan_or_tools) * 100 / makespan_or_tools:.1f}%"
    )


if __name__ == "__main__":
    main()
