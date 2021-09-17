import sys

sys.path.append("../")

import argparse  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402


from models.agent import Agent  # noqa: E402
from problem.problem_description import ProblemDescription  # noqa: E402
from utils.ortools_solver import solve_jssp  # noqa: E402
from utils.utils import generate_problem  # noqa: E402

from config import MAX_DURATION  # noqa: E402


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    affectations, durations = generate_problem(args.n_j, args.n_m, MAX_DURATION)
    agent = Agent.load("../" + args.path)
    or_tools_schedule = solve_jssp(affectations, durations).schedule
    rl_schedule = agent.predict(
        ProblemDescription(
            args.n_j, args.n_m, MAX_DURATION, args.transition_model_config, args.reward_model_config, affectations, durations
        )
    ).schedule

    print(f"Affectations : \n{affectations}\nDurations: \n{durations}")
    print(f"OR-Tools schedule : \n{or_tools_schedule}")
    print(f"OR-Tools makespan : {np.max(or_tools_schedule + durations)}")
    print(f"RL schedule : \n{rl_schedule}")
    print(f"RL makespan : {np.max(rl_schedule + durations)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Launch an inference for or-tools and specified model, and print results")

    # Args
    parser.add_argument("--n_j", type=int, default=5, help="Number of jobs")
    parser.add_argument("--n_m", type=int, default=5, help="Number of machines")
    parser.add_argument("--transition_model_config", type=str, default="L2D", help="Which transition model to use")
    parser.add_argument("--reward_model_config", type=str, default="L2D", help="Which reward model to use")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--path", type=str, default="saved_networks/default_net", help="Path to saved model")

    main(parser.parse_args())
