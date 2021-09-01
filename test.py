import gym
from stable_baselines3 import PPO
import torch

from env.env import Env
from models.agent import Agent
from problem.problem_description import ProblemDescription
from utils.ortools_solver import solve_jssp_ortools
from utils.utils import generate_problem

from config import MAX_N_JOBS, MAX_N_MACHINES, DEVICE
from args import args


def main():
    agent = Agent.load(args.path)
    print(
        f"Launching inference. Problem size : {args.n_j_testing} jobs, {args.n_m_testing} machines"
    )

    testing_affectations, testing_durations = generate_problem(
        args.n_j_testing, args.n_m_testing, args.max_duration
    )
    solution = agent.predict(
        ProblemDescription(
            args.n_j_testing,
            args.n_m_testing,
            args.max_duration,
            "L2D",
            "L2D",
            testing_affectations,
            testing_durations,
        )
    )
    solution_or_tools = solve_jssp_ortools(
        testing_affectations, testing_durations
    )
    print(testing_affectations)
    print(testing_durations)
    print("Solution found by the model")
    print(solution.schedule)
    print("Solution found by OR-tools solver : ")
    print(solution_or_tools.schedule)


if __name__ == "__main__":
    main()
