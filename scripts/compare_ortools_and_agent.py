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


def main(cur_args):
    torch.manual_seed(cur_args.seed)
    np.random.seed(cur_args.seed)

    if cur_args.affectations and cur_args.durations:
        affectations = np.array(cur_args.affectations).reshape(cur_args.n_j, cur_args.n_m)
        durations = np.array(cur_args.durations).reshape(cur_args.n_j, cur_args.n_m)
    else:
        affectations, durations = generate_problem(cur_args.n_j, cur_args.n_m, MAX_DURATION)
    agent = Agent.load(
        "../" + cur_args.path, not cur_args.remove_machine_id, cur_args.one_hot_machine_id, cur_args.add_pdr_boolean
    )
    or_tools_schedule = solve_jssp(affectations, durations).schedule
    rl_schedule = agent.predict(
        ProblemDescription(
            cur_args.n_j,
            cur_args.n_m,
            MAX_DURATION,
            cur_args.transition_model_config,
            cur_args.reward_model_config,
            affectations,
            durations,
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
    parser.add_argument("--affectations", type=int, nargs="+", default=[], help="Problem affectations")
    parser.add_argument("--durations", type=int, nargs="+", default=[], help="Problem durations")
    parser.add_argument(
        "--remove_machine_id", default=False, action="store_true", help="Add the machine id in the node embedding"
    )
    parser.add_argument(
        "--one_hot_machine_id", default=False, action="store_true", help="Add machine id as one hot encoding"
    )
    parser.add_argument(
        "--add_pdr_boolean", default=False, action="store_true", help="Add a boolean in action space for PDR use"
    )

    main(parser.parse_args())
