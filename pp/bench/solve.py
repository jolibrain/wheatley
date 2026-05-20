from pp.models.agent import Agent
from pp.domains.maze.env import Env
from generic.utils import decode_mask
from pp.description import Description
import os
import tqdm
import torch
import io
from pp.domains.maze.generators.maze_generator import generate_mazes
from maze_dataset.plotting import MazePlot, PathFormat
import matplotlib.pyplot as plt
import numpy as np
import cv2
import argparse
import pickle
from pp.domains.maze.problem import PathPlanningProblem
from pp.domains.maze.generators.maze_hard import preprocess_maze_hard
from maze_dataset import MazeDataset


def do_plot(venv, sols):
    m = venv.problem.solvedMaze
    mp = MazePlot(m)
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]
    for agent_id, sol in enumerate(sols):
        native = sol.get_native_sol()
        if native:
            color = colors[agent_id % len(colors)]
            path_fmt = PathFormat(
                label=f"agent {agent_id}", color=color, line_width=2.0
            )
            mp.add_predicted_path(native, path_fmt=path_fmt)
    if hasattr(venv, "_agent_start_coords"):
        for agent_id, start in enumerate(venv._agent_start_coords):
            color = colors[agent_id % len(colors)]
            mp.mark_coords(
                [start],
                marker="o",
                color=color,
                markeredgecolor="k",
                markersize=6,
                linestyle="None",
                label=f"start {agent_id}",
            )
    if hasattr(venv, "_goal_coords"):
        availability = getattr(venv, "goal_available", None)
        for goal_id, goal in enumerate(venv._goal_coords):
            available = True
            if availability is not None and len(availability) > goal_id:
                available = bool(availability[goal_id].item())
            color = "gold" if available else "gray"
            mp.mark_coords(
                [goal],
                marker="*",
                color=color,
                markeredgecolor="k",
                markersize=8,
                linestyle="None",
                label=f"goal {goal_id}",
            )
    danger_mask = getattr(venv, "_danger_zone_mask", None)
    danger_grid = None
    if danger_mask is not None and danger_mask.any():
        danger_grid = danger_mask.reshape(venv.problem.shape).cpu().numpy()
    mp.plot()
    if danger_grid is not None:
        extent = (
            0,
            venv.problem.shape[1] * mp.unit_length,
            venv.problem.shape[0] * mp.unit_length,
            0,
        )
        mp.ax.imshow(
            danger_grid,
            cmap="Reds",
            alpha=0.3,
            interpolation="nearest",
            extent=extent,
        )
    mp.fig.savefig("sol.png", format="png", dpi=150)


def solve_instance(
    problem_description,
    agent,
    pp_agent_types,
    nonchrono,
):
    # print("creating inference env")
    venv = Env(
        problem_description,
        agent.env_specification,
        [0],
        validate=True,
        walls=True,
        lappe=None,
        rwpe=None,
        pp_agent_types=pp_agent_types,
        nonchrono=nonchrono,
    )
    # print("reseting inference env")
    obs, info = venv.reset(soft=False)
    done = False
    action_num = 0
    while True:
        # print("ACTION ", action_num)
        action_num += 1
        action_masks = decode_mask([info["mask"]])
        obs = agent.obs_as_tensor_add_batch_dim(obs)
        action, _, _, _, _ = agent.get_action_and_value(
            agent.preprocess(obs),
            action_masks=action_masks,
            deterministic=True,
        )

        step_action = action[0].cpu().numpy()
        obs, reward, done, _, info = venv.step(step_action)
        if done:
            break
    sols = [venv.get_solution(agent_id) for agent_id in range(len(venv.states))]
    # do_plot(venv, sols)
    return sols


def load_dataset(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f)
        mds = MazeDataset.load(d)
    ds = [PathPlanningProblem(m) for m in mds]
    return ds


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument("--size", type=int, default=5, help="maze size")
    parser.add_argument("--n", type=int, default=100, help="number of mazes in dataset")
    parser.add_argument("--path", type=str, default="", help="agent path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--hard", default=False, action="store_true", help="use hard maze dataset"
    )

    args = parser.parse_args()

    maze_size = args.size if not args.hard else 30

    if args.hard:
        tpp = preprocess_maze_hard("test", aug=False)
    else:
        tpp = load_dataset(f"pp/bench/dataset_{maze_size}x{maze_size}_{args.n}.pkl")

    optimal_values = []
    agent_values = []
    n_optimal = 0
    n_failed = 0
    gap = 0
    total_optimal_lengths = 0
    for problem in tqdm.tqdm(tpp):
        pp = [problem]
        # pp = [tpp[10]]

        problem_description = Description(
            "simpleTransitionModel", "shortest_path_reward", [], pp, args.seed
        )
        # problem_description.print_self()

        agent = Agent.load(
            args.path + "/",
            max_n_modes=maze_size * maze_size,
        )

        agent.to(args.device)
        sol = solve_instance(
            problem_description,
            agent,
            pp_agent_types=agent.agent_specification.agent_types,
            nonchrono=agent.agent_specification.nonchrono,
        )
        optimal = int(sol[0].get_optimal_criterion() * args.size * args.size)
        total_optimal_lengths += optimal
        criterion = int(sol[0].get_criterion() * args.size * args.size)
        if criterion == maze_size * maze_size:
            n_failed += 1
        if optimal == criterion:
            n_optimal += 1
        gap += (criterion - optimal) / optimal
        optimal_values.append(optimal)
        agent_values.append(criterion)
        # print(f"agent path length {criterion}   /  optimal {optimal}")

    print(f"n optimal {n_optimal}")
    print(f"percent optimal values {100 * n_optimal / len(tpp)}")
    print(f"percent failed values {100 * n_failed / len(tpp)}")
    print(f"average gap (% length optimal path): {gap / len(tpp) * 100}")
    print(f"average optimal path length {total_optimal_lengths / len(tpp)}")
    print(
        f"average gap on non_optimal (% length optimal path) : {gap / (len(tpp) - n_optimal) * 100}"
    )
