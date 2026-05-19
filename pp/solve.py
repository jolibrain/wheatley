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
    walls,
    # pp_allow_stay,
    # pp_all_agents_must_finish,
    # pp_maze_gen,
    # pp_force_common_start,
    # pp_num_goals,
    # pp_danger_max_size,
    # pp_danger_max_num,
    # pp_danger_prob,
    # pp_danger_multiplier,
    # pp_goal_reward,
    # pp_k_lookahead,
    # pp_heading_motion,
    # pp_max_turn,
    # pp_protect_max_num,
    # pp_protect_max_radius,
    # pp_protect_kill_prob,
    nonchrono,
):
    print("creating inference env")
    venv = Env(
        problem_description,
        agent.env_specification,
        [0],
        validate=True,
        walls=walls,
        lappe=None,
        rwpe=None,
        pp_agent_types=pp_agent_types,
        # pp_allow_stay=pp_allow_stay,
        # pp_all_agents_must_finish=pp_all_agents_must_finish,
        # pp_maze_gen=pp_maze_gen,
        # pp_force_common_start=pp_force_common_start,
        # pp_num_goals=pp_num_goals,
        # pp_danger_max_size=pp_danger_max_size,
        # pp_danger_max_num=pp_danger_max_num,
        # pp_danger_prob=pp_danger_prob,
        # pp_danger_multiplier=pp_danger_multiplier,
        # pp_goal_reward=pp_goal_reward,
        # pp_k_lookahead=pp_k_lookahead,
        # pp_heading_motion=pp_heading_motion,
        # pp_max_turn=pp_max_turn,
        # pp_protect_max_num=pp_protect_max_num,
        # pp_protect_max_radius=pp_protect_max_radius,
        # pp_protect_kill_prob=pp_protect_kill_prob,
        nonchrono=nonchrono,
    )
    print("reseting inference env")
    obs, info = venv.reset(soft=False)
    done = False
    action_num = 0
    while True:
        print("ACTION ", action_num)
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
    do_plot(venv, sols)
    return sols


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    from pp.args_pp import argument_parser, parse_args

    parser = argument_parser()
    args = parse_args(parser)

    maze_size = 8 if args.pp_maze_gent != "maze_hard" else 30

    tpp, _ = generate_mazes(maze_size, 1, maze_gen=args.pp_maze_gen)

    pp = [tpp[0]]

    problem_description = Description(
        "simpleTransitionModel", "shortest_path_reward", [], pp, args.seed
    )
    problem_description.print_self()

    agent = Agent.load(
        args.path + "/",
        max_n_modes=maze_size * maze_size,
    )

    agent.to(args.device)
    sol = solve_instance(
        problem_description,
        agent,
        pp_agent_types=agent.agent_specification.agent_types,
        # args.pp_allow_stay,
        # args.pp_all_agents_must_finish,
        # args.pp_maze_gen,
        # args.pp_force_common_start,
        # args.pp_num_goals,
        # args.pp_danger_max_size,
        # args.pp_danger_max_num,
        # args.pp_danger_prob,
        # args.pp_danger_multiplier,
        # args.pp_goal_reward,
        # args.pp_k_lookahead,
        # args.pp_heading_motion,
        # args.pp_max_turn,
        # args.pp_protect_max_num,
        # args.pp_protect_max_radius,
        # args.pp_protect_kill_prob,
        nonchrono=agent.agent_specification.nonchrono,
        walls=not args.no_walls,
    )
