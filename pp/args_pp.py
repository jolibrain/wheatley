#
# Wheatley
# Copyright (c) 2024 Jolibrain
#
# This file is part of Wheatley.
#

from __future__ import annotations

import argparse
from typing import Iterable

from args import argument_parser as base_argument_parser
from args import parse_args as base_parse_args
from pp.domains.registry import DOMAIN_REGISTRY


def _extend_choices(action: argparse.Action, new_choices: Iterable[str]) -> None:
    """Add missing choices to an argparse action without duplicating entries."""
    if action is None:
        return

    existing = list(action.choices) if action.choices is not None else []
    for choice in new_choices:
        if choice not in existing:
            existing.append(choice)
    if action.choices is None or tuple(existing) != action.choices:
        action.choices = tuple(existing)


def argument_parser() -> argparse.ArgumentParser:
    """Return a parser extended with Path Planning specific options."""
    parser = base_argument_parser()

    transition_action = parser._option_string_actions.get("--transition_model_config")
    reward_action = parser._option_string_actions.get("--reward_model_config")

    _extend_choices(transition_action, ["simpleTransitionModel"])
    _extend_choices(reward_action, ["shortest_path_reward"])

    parser.set_defaults(
        transition_model_config="simpleTransitionModel",
        reward_model_config="shortest_path_reward",
        domain="maze",
    )

    parser.add_argument(
        "--domain",
        type=str,
        choices=tuple(DOMAIN_REGISTRY.keys()),
        default="maze",
        help="Domain to train on (currently only 'maze' is implemented).",
    )

    parser.add_argument(
        "--pp_agent_types",
        type=int,
        nargs="+",
        default=[0, 1],
        help="Types of agents to use. 0: no wall breaking, 1: can break 1 wall.",
    )
    # parser.add_argument(
    #     "--pp_allow_stay",
    #     action="store_true",
    #     help="Allow agents to stay on their current node (only effective in multi-agent settings).",
    # )
    # parser.add_argument(
    #     "--pp_all_agents_must_finish",
    #     action="store_true",
    #     help="End episodes only when every agent has reached its goal.",
    # )
    # parser.add_argument(
    #     "--pp_k_lookahead",
    #     dest="pp_k_lookahead",
    #     type=int,
    #     default=1,
    #     help="Max straight-line steps an agent can jump in navigation; treated as fixed speed when heading motion is enabled (k=1 disables lookahead).",
    # )
    # parser.add_argument(
    #     "--pp_heading_motion",
    #     action="store_true",
    #     help="Enable DIR8 heading-based motion with turning limits.",
    # )
    # parser.add_argument(
    #     "--pp_max_turn",
    #     type=int,
    #     default=4,
    #     help="Maximum heading delta (in DIR8 steps) allowed per tick when heading motion is enabled.",
    # )
    parser.add_argument(
        "--pp_maze_gen",
        choices=("dfs", "dfs_percolation", "maze_hard"),
        default="dfs_percolation",
        help="Maze generator to use when sampling training/test mazes.",
    )
    # parser.add_argument(
    #     "--pp_force_common_start",
    #     action="store_true",
    #     help="Force all agents to share the maze_dataset start instead of sampling per-agent starts.",
    # )
    # parser.add_argument(
    #     "--pp_num_goals",
    #     type=int,
    #     default=1,
    #     help="Maximum shared goals per maze/map (>=1); when pp_random_num_goals is true, sample in [1, pp_num_goals], else use the fixed value.",
    # )
    # parser.add_argument(
    #     "--pp_random_num_goals",
    #     dest="pp_random_num_goals",
    #     action="store_true",
    #     help="Sample the number of goals per problem in [1, pp_num_goals].",
    # )
    # parser.set_defaults(pp_random_num_goals=False)
    # parser.add_argument(
    #     "--pp_danger_max_size",
    #     type=int,
    #     default=0,
    #     help="Maximum side length of sampled danger zones (0 disables danger zones).",
    # )
    # parser.add_argument(
    #     "--pp_danger_max_num",
    #     type=int,
    #     default=0,
    #     help="Maximum number of danger zones sampled per maze.",
    # )
    # parser.add_argument(
    #     "--pp_danger_prob",
    #     type=float,
    #     default=1.0,
    #     help="Probability that a move from a danger cell scales its unit cost by pp_danger_multiplier.",
    # )
    # parser.add_argument(
    #     "--pp_danger_multiplier",
    #     type=float,
    #     default=2.0,
    #     help="Scaling factor applied to the unit cost when stepping from a danger cell.",
    # )
    parser.add_argument(
        "--pp_goal_reward",
        type=float,
        default=0.0,
        help="Reward given when a goal is reached.",
    )
    # parser.add_argument(
    #     "--pp_protect_max_num",
    #     type=int,
    #     default=0,
    #     help="Maximum number of protective zones to sample in navigation (0 disables).",
    # )
    # parser.add_argument(
    #     "--pp_protect_max_radius",
    #     type=int,
    #     default=1,
    #     help="Maximum radius (Chebyshev) of each protective square zone (>=1).",
    # )
    # parser.add_argument(
    #     "--pp_protect_kill_prob",
    #     type=float,
    #     default=0.0,
    #     help="Probability of agent death when entering a protective zone (0 disables).",
    # )

    return parser


def parse_args(parser: argparse.ArgumentParser):
    """Delegate parsing to the shared helper from args.py."""
    return base_parse_args(parser)
