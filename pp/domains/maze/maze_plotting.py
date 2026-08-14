import io
from typing import Iterable, Sequence

import cv2
import numpy as np
import torch
from maze_dataset.plotting import MazePlot, PathFormat
import matplotlib.pyplot as plt


def _is_empty_path(path) -> bool:
    """Return True if the provided path is empty."""
    if path is None:
        return True
    if isinstance(path, (list, tuple)) and len(path) == 0:
        return True
    try:
        if hasattr(path, "__len__") and len(path) == 0:
            return True
    except TypeError:
        return False
    return False


def plot_maze_solutions(
    validation_envs: Sequence,
    maze_solution: Sequence[Iterable],
    vis,
    base_win_name: str = "maze solution ",
) -> None:
    """
    Plot maze solutions for PP environments and push them to Visdom.

    Args:
        validation_envs: Environments matching the maze solutions (typically PPEnv instances).
        maze_solution: Iterable of per-environment agent paths.
        vis: Visdom client used to render images.
        base_win_name: Prefix for Visdom window names.
    """
    colors = [
        "tab:blue",
        "tab:orange",
        "tab:green",
        "tab:red",
        "tab:purple",
        "tab:brown",
    ]

    for i, agent_paths in enumerate(maze_solution):
        env = validation_envs[i]
        maze = env.problem.solvedMaze
        plot = MazePlot(maze)

        paths = agent_paths if isinstance(agent_paths, (list, tuple)) else [agent_paths]
        for agent_id, path in enumerate(paths):
            if _is_empty_path(path):
                continue
            fmt = PathFormat(
                label=f"agent {agent_id}",
                color=colors[agent_id % len(colors)],
            )
            plot.add_predicted_path(path, path_fmt=fmt)

        if hasattr(env, "_agent_start_coords"):
            for agent_id, start in enumerate(env._agent_start_coords):
                color = colors[agent_id % len(colors)]
                plot.mark_coords(
                    [start],
                    marker="o",
                    color=color,
                    markeredgecolor="k",
                    markersize=6,
                    linestyle="None",
                    label=f"start {agent_id}",
                )

        if hasattr(env, "_goal_coords"):
            availability = getattr(env, "goal_available", None)
            for goal_id, goal in enumerate(env._goal_coords):
                available = True
                if availability is not None and len(availability) > goal_id:
                    available = bool(availability[goal_id].item())
                marker = "*"
                color = "gold" if available else "gray"
                plot.mark_coords(
                    [goal],
                    marker=marker,
                    color=color,
                    markeredgecolor="k",
                    markersize=7,
                    linestyle="None",
                    label=f"goal {goal_id}",
                )

        danger_mask = getattr(env, "_danger_zone_mask", None)
        danger_grid = None
        if danger_mask is not None and danger_mask.any():
            danger_grid = danger_mask.reshape(env.problem.shape).cpu().numpy()

        plot.plot()
        if danger_grid is not None:
            extent = (
                0,
                env.problem.shape[1] * plot.unit_length,
                env.problem.shape[0] * plot.unit_length,
                0,
            )
            plot.ax.imshow(
                danger_grid,
                cmap="Reds",
                alpha=0.3,
                interpolation="nearest",
                extent=extent,
            )

        figimg = io.BytesIO()
        plot.fig.savefig(figimg, format="png", dpi=150)
        plt.clf()
        plt.close("all")
        figimg.seek(0)
        npimg = np.frombuffer(figimg.read(), dtype="uint8")
        cvimg = cv2.imdecode(npimg, cv2.IMREAD_UNCHANGED)
        npimg = np.transpose(cvimg, (2, 0, 1))
        torchimg = torch.from_numpy(npimg)

        win_name = f"{base_win_name}{i}"
        vis.image(torchimg, opts={"caption": win_name}, win=win_name)
