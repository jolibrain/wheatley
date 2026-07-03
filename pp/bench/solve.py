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
import csv
from pathlib import PurePath
import random


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


def random_instance(problem_description, agent, pp_agent_types, nonchrono, nsample):
    agent.env_specification.max_n_steps = -1  # cannot fail for too long traj
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

    optimal_steps = len(venv.optimal) - 1

    action_num = 0
    for i in tqdm.tqdm(range(nsample), leave=False, desc="instance"):
        obs, info = venv.reset(soft=False)
        done = False
        with torch.no_grad():
            while True:
                # print("ACTION ", action_num)
                action_num += 1
                action_masks = decode_mask([info["mask"]])
                step_action = random.choice(
                    (np.where(action_masks.squeeze(0)))[0].tolist()
                )
                step_action = np.array([step_action])
                obs, reward, done, _, info = venv.step(step_action)
                if done:
                    break
                # sols = [
                #     venv.get_solution(agent_id) for agent_id in range(len(venv.states))
                # ]
    return optimal_steps, action_num / nsample


def solve_instance(
    problem_description,
    agent,
    pp_agent_types,
    nonchrono,
    plot=False,
    nsample=0,
):
    # print("creating inference env")
    venv = Env(
        problem_description,
        agent.env_specification,
        [0],
        validate=True,
        walls=not agent.agent_specification.hierarchical,
        lappe=None,
        rwpe=None,
        pp_agent_types=pp_agent_types,
        nonchrono=nonchrono,
    )
    optimal_steps = len(venv.optimal) - 1
    action_num = 0
    nopt = 0
    ngap = 0
    gap_on_nonopt = 0
    gapstep = 0
    if nsample == 1:
        iterator = [0]
    else:
        iterator = tqdm.tqdm(range(nsample), leave=False, desc="instance")
    for i in iterator:
        obs, info = venv.reset(soft=False)
        done = False
        nac = 0
        with torch.no_grad():
            while True:
                # print("ACTION ", action_num)
                action_num += 1
                nac += 1
                action_masks = decode_mask([info["mask"]])
                obs = agent.obs_as_tensor_add_batch_dim(obs)
                action, _, _, _, _ = agent.get_action_and_value(
                    agent.preprocess(obs),
                    action_masks=action_masks,
                    deterministic=nsample == 1,
                    # temperature=0.1,
                )

                step_action = action[0].cpu().numpy()
                # print("action", step_action)
                obs, reward, done, _, info = venv.step(step_action)
                # print("partial sol ", venv.states[0].path)
                if done:
                    break
        sol = venv.get_solution(0)
        # print("sol ", sol.sol)
        # print("failed ", sol.failed)
        # print("nac ", nac)
        # print("optimal_steps ", optimal_steps)
        gapstep += (nac - optimal_steps) / optimal_steps

        if sol.get_criterion() == sol.optimal_criterion:
            # if nac == optimal_steps:
            nopt += 1
        else:
            ngap += 1
            # gap_on_nonopt += (nac - optimal_steps) / optimal_steps
            gap_on_nonopt += (
                sol.get_criterion() - sol.optimal_criterion
            ) / sol.optimal_criterion
    if nsample != 0:
        action_num /= nsample
        nopt /= nsample
        gapstep /= nsample
    sols = [venv.get_solution(agent_id) for agent_id in range(len(venv.states))]
    if plot:
        print("NUMSTEPS: ", action_num)
        do_plot(venv, sols)
    return (
        sols,
        action_num,
        nopt,
        (action_num - optimal_steps) / optimal_steps,
        gap_on_nonopt,
        ngap,
        gapstep,
    )


def load_dataset(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f)
        mds = MazeDataset.load(d)
    ds = [PathPlanningProblem(m) for m in mds]
    return ds


def find_bucket(buckets, val):
    for n, i in enumerate(buckets):
        if val < i:
            return n


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    parser = argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument("--size", type=int, default=5, help="maze size")
    parser.add_argument("--n", type=int, default=100, help="number of mazes in dataset")
    parser.add_argument("--path", type=str, default="", help="agent path")
    parser.add_argument("--seed", type=int, default=0, help="seed")
    parser.add_argument("--device", type=str, default="cuda:0", help="device")
    parser.add_argument(
        "--nlayers", type=int, default=None, help="number of layers for inference"
    )
    parser.add_argument(
        "--n_buckets", type=int, default=10, help="number of buckets for stats"
    )

    parser.add_argument(
        "--hard", default=False, action="store_true", help="use hard maze dataset"
    )
    parser.add_argument("--limit", default=None, type=int, help="max num steps")
    parser.add_argument("--random", type=int, default="1", help="random agent runs")

    args = parser.parse_args()

    maze_size = args.size if not args.hard else 30

    if args.hard:
        tpp = preprocess_maze_hard("test", aug=False)
    else:
        tpp = load_dataset(f"pp/bench/dataset_{maze_size}x{maze_size}_{args.n}.pkl")

    # if args.random != 0:
    #     agent = Agent.load(
    #         args.path + "/",
    #         max_n_modes=maze_size * maze_size,
    #     )
    #     gap = 0
    #     optimal = 0
    #     for problem in tqdm.tqdm(tpp, desc="problem"):
    #         pp = [problem]
    #         # pp = [tpp[10]]

    #         problem_description = Description(
    #             "simpleTransitionModel", "shortest_path_reward", [], pp, args.seed
    #         )
    #         # problem_description.print_self()
    #         optimal_steps, nsteps = random_instance(
    #             problem_description,
    #             agent,
    #             pp_agent_types=agent.agent_specification.agent_types,
    #             nonchrono=agent.agent_specification.nonchrono,
    #             nsample=args.random,
    #         )
    #         gap += (nsteps - optimal_steps) / optimal_steps
    #         if nsteps - optimal_steps == 0:
    #             optimal += 1
    #     print(f"average gap for random agent {gap / len(tpp) * 100}")
    #     print(f"number of optimal  {optimal}")
    #     exit(0)

    optimal_values = []
    agent_values = []
    step_values = []
    n_optimal = 0
    n_failed = 0
    n_gap = 0
    gap = 0
    total_optimal_lengths = 0
    max_path_length = 0
    min_path_length = args.size * args.size
    nsteps_optimal = 0
    nsteps_nonoptimal = 0
    nsteps_failed = 0
    buckets = []
    for i in range(1, args.n_buckets + 1):
        buckets.append(i * args.size * args.size / args.n_buckets)
    b_nelts = [0] * args.n_buckets
    b_failed = [0] * args.n_buckets
    b_optimal = [0] * args.n_buckets
    b_ngap = [0] * args.n_buckets
    b_gap = [0] * args.n_buckets
    b_nsteps_optimal = [0] * args.n_buckets
    b_nsteps_nonoptimal = [0] * args.n_buckets
    b_nsteps_failed = [0] * args.n_buckets
    agent = Agent.load(
        args.path + "/",
        max_n_modes=maze_size * maze_size,
    )
    if args.nlayers is not None:
        agent.gnn.gnn.n_layers = args.nlayers

    if args.limit is None:
        agent.env_specification.max_n_steps = maze_size * maze_size
    elif args.limit <= 0:
        agent.env_specification.max_n_steps = -1  # cannot fail for too long traj
    else:
        agent.env_specification.max_n_steps = (
            args.limit
        )  # cannot fail for too long traj

    if args.random != 1:
        agent.env_specification.max_n_steps = -1

    agent.to(args.device)

    print(f"NON CHRONO: {agent.agent_specification.nonchrono}")
    total_opt = 0
    total_gap = 0
    total_gap_on_nonopt = 0
    total_nonopt = 0
    total_gapstep = 0
    for problem in tqdm.tqdm(tpp):
        pp = [problem]
        # pp = [tpp[93]]

        problem_description = Description(
            "simpleTransitionModel", "shortest_path_reward", [], pp, args.seed
        )
        # problem_description.print_self()

        sol, nsteps, nopt, gapnondet, gapnondet_onnonopt, nonopt, gapstep = (
            solve_instance(
                problem_description,
                agent,
                pp_agent_types=agent.agent_specification.agent_types,
                nonchrono=agent.agent_specification.nonchrono,
                nsample=args.random,
            )
        )
        total_opt += nopt
        total_gap += gapnondet
        total_gap_on_nonopt += gapnondet_onnonopt
        total_nonopt += nonopt
        total_gapstep += gapstep
        # exit(0)
        optimal = round(sol[0].get_optimal_criterion() * args.size * args.size)
        if optimal > max_path_length:
            max_path_length = optimal
        if optimal < min_path_length:
            min_path_length = optimal

        total_optimal_lengths += optimal
        criterion = round(sol[0].get_criterion() * args.size * args.size)
        bi = find_bucket(buckets, optimal)
        b_nelts[bi] += 1
        if criterion >= 2.0 * maze_size * maze_size:
            n_failed += 1
            b_failed[bi] += 1
            nsteps_failed += nsteps / optimal
            b_nsteps_failed[bi] += nsteps / optimal
        elif criterion == optimal:
            n_optimal += 1
            b_optimal[bi] += 1
            nsteps_optimal += nsteps / optimal
            b_nsteps_optimal[bi] += nsteps / optimal
        else:
            gap += (criterion - optimal) / optimal
            n_gap += 1
            b_gap[bi] += (criterion - optimal) / optimal
            b_ngap[bi] += 1
            b_nsteps_nonoptimal[bi] += nsteps / optimal
            nsteps_nonoptimal += nsteps / optimal
        optimal_values.append(optimal)
        agent_values.append(criterion)
        step_values.append(nsteps)
        # print(f"agent path length {criterion}   /  optimal {optimal}")

    # print("per bucket stats:")
    # for i in range(len(buckets)):
    #     print(f"bucket {0 if i == 0 else buckets[i - 1]} -> {buckets[i]}")
    #     print(f"n_elts : {b_nelts[i]}")
    #     print(f"n_optimal : {b_optimal[i]}")
    #     print(f"n_failed : {b_failed[i]}")
    #     print(f"n_gap : {b_ngap[i]}")
    #     if b_ngap[i] != 0:
    #         print(f"gap average (%): {b_gap[i] / b_ngap[i] * 100}")
    #     else:
    #         print("gap average: 0.0")
    print(f"average optimal path length {total_optimal_lengths / len(tpp)}")
    print(f"min optimal path length {min_path_length}")
    print(f"max optimal path length {max_path_length}")
    print(f"percent optimal values {100 * n_optimal / len(tpp)}")
    if args.random != 1:
        print(f"NONDET: percent optimal {100 * total_opt / len(tpp)}")
        print(f"NONDET: percent GAP {100 * total_gap / len(tpp)}")
        if total_nonopt != 0:
            print(
                f"NONDET: percent GAP on non opt {100 * total_gap_on_nonopt / total_nonopt}"
            )
        else:
            print("NONDET: percent GAP on non opt 0.0")
    print(f"percent failed values {100 * n_failed / len(tpp)}")
    if n_gap != 0:
        print(f"n_steps on nonoptimal (%) {100 * nsteps_nonoptimal / n_gap}")
        print(
            f"average gap on non_optimal non_failed (% length optimal path) : {gap / n_gap * 100}"
        )
    if n_failed != 0:
        print(f"n_steps on failed (%) {100 * nsteps_failed / n_failed}")
    if n_optimal != 0:
        print(f"n_steps on optimal (%) {100 * nsteps_optimal / n_optimal}")

    print(f"GAPSTEP (%): {100 * total_gapstep / len(tpp)}")

    if args.hard:
        csvfile = open(f"results_hard_{PurePath(args.path).parts[-1]}.csv", "w")
    else:
        csvfile = open(f"results_{args.size}_{PurePath(args.path).parts[-1]}.csv", "w")
    reswriter = csv.writer(csvfile, delimiter=",")
    for ov, av, sv in zip(optimal_values, agent_values, step_values):
        optimal = "OPTIMAL" if ov == av else ""
        failed = "FAILED" if av >= args.size * args.size else ""
        reswriter.writerow([ov, av, sv, optimal, failed])
