import gym
from stable_baselines3 import PPO
import torch

from env.env import Env
from models.agent import Agent
from problem.problem_description import ProblemDescription
from utils.utils import generate_problem

from config import MAX_N_JOBS, MAX_N_MACHINES, DEVICE
from args import args


def main():

    if args.n_j > MAX_N_JOBS or args.n_m > MAX_N_MACHINES:
        raise Exception(
            "MAX_N_JOBS or MAX_N_MACHINES are too low for this setup"
        )
    
    print(f"Launching training. Problem size : {args.n_j} jobs, {args.n_m} machines")

    training_env = Env(
        ProblemDescription(args.n_j, args.n_m, args.max_duration, "L2D", "L2D")
    )

    problem_description = ProblemDescription(
        args.n_j, args.n_m, args.max_duration, "L2D", "L2D"
    )

    agent = Agent(training_env)
    agent.train(problem_description, total_timesteps=50)

    print(f"Launching inference. Problem size : {args.n_j_testing} jobs, {args.n_m_testing} machines")
    
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
    print(testing_affectations)
    print(testing_durations)
    print(solution.schedule)


if __name__ == "__main__":
    main()
