"""Use a trained model to solve a given JSSP problem."""
from pathlib import Path

import numpy as np

from generic.utils import decode_mask
from jssp.description import Description
from jssp.env.env import Env
from jssp.models.agent import Agent
from jssp.solution import Solution


def solve_instance(
    agent: Agent,
    affectations: np.ndarray,
    durations: np.ndarray,
    deterministic: bool,
) -> Solution:
    problem_description = Description(
        transition_model_config="simple",
        reward_model_config="Sparse",
        deterministic=deterministic,
        fixed=True,
        seed=0,
        affectations=affectations,
        durations=durations,
    )
    env_specification = agent.env_specification
    env = Env(problem_description, env_specification)

    done = False
    obs, info = env.reset(soft=True)
    while not done:
        action_masks = info["mask"].reshape(1, -1)
        action_masks = decode_mask(action_masks)
        obs = agent.obs_as_tensor_add_batch_dim(obs)
        action = agent.predict(obs, deterministic=True, action_masks=action_masks)
        obs, reward, done, _, info = env.step(action.long().item())
        solution = env.get_solution()

    return solution


if __name__ == "__main__":
    from args import argument_parser, parse_args
    from jssp.utils.loaders import load_problem

    parser = argument_parser()
    args, _, _ = parse_args(parser)

    assert (
        args.load_problem is not None
    ), "You should provide a problem to solve (use --load_problem)."

    agent = Agent.load(args.path)
    n_j, n_m, affectations, durations = load_problem(
        args.load_problem,
        taillard_offset=args.first_machine_id_is_one,
        deterministic=args.duration_type == "deterministic",
    )

    agent.to(args.device)

    print(f"Solving a {n_j}x{n_m} JSSP instance.")

    assert (
        agent.env_specification.max_n_jobs >= n_j
    ), f"Too many jobs for the agent ({agent.env_specification.max_n_jobs} vs {n_j})."
    assert (
        agent.env_specification.max_n_machines >= n_m
    ), f"Too many jobs for the agent ({agent.env_specification.max_n_machines} vs {n_m})."

    solution = solve_instance(
        agent, affectations, durations, args.duration_type == "deterministic"
    )
    print(f"Makespan: {solution.get_makespan()}")

    problem_name = Path(args.load_problem).stem
    solution_filepath = Path("./") / f"{problem_name}_solution.txt"
    solution.save(solution_filepath)
    print(f"Schedule saved to {solution_filepath}")
