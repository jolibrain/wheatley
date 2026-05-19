"""
Lightweight domain registry to keep environment plumbing centralized.
"""

from dataclasses import dataclass
from typing import Callable, Dict, Optional, Type

from pp.domains.maze import Env as MazeEnv
from pp.domains.maze import EnvSpecification as MazeEnvSpecification
from pp.domains.maze import PathPlanningProblem
from pp.domains.maze.generators import generate_mazes


@dataclass
class DomainDefinition:
    name: str
    env_cls: Type
    env_spec_cls: Type
    generator_fn: Callable
    problem_cls: Optional[Type]


DOMAIN_REGISTRY: Dict[str, DomainDefinition] = {
    "maze": DomainDefinition(
        name="maze",
        env_cls=MazeEnv,
        env_spec_cls=MazeEnvSpecification,
        generator_fn=generate_mazes,
        problem_cls=PathPlanningProblem,
    ),
}


def get_domain_definition(name: str) -> DomainDefinition:
    try:
        return DOMAIN_REGISTRY[name]
    except KeyError as exc:
        raise ValueError(f"Unknown domain '{name}'") from exc
