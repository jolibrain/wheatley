from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
from pp.domains.maze.problem import PathPlanningProblem
from pp.domains.maze.generators.maze_hard import preprocess_maze_hard
import warnings
import random

_MAZE_GENERATORS = {
    "dfs": LatticeMazeGenerators.gen_dfs,
    "dfs_percolation": LatticeMazeGenerators.gen_dfs_percolation,
}


def _resolve_maze_ctor(name):
    try:
        return _MAZE_GENERATORS[name]
    except KeyError as exc:
        raise ValueError(f"Unknown maze generator '{name}'") from exc


global_test_hashes = []


def generate_mazes(
    size_train,
    n_mazes_train,
    size_test=0,
    n_mazes_test=0,
    test_mazes=global_test_hashes,
    maze_gen="dfs",
    forbid_trivial=True,
):
    warnings.simplefilter("ignore")

    if maze_gen == "maze_hard":
        if not hasattr(generate_mazes, "hard_mazes"):
            generate_mazes.hard_mazes = {
                "train": preprocess_maze_hard("./data/maze_hard/train"),
                "test": preprocess_maze_hard("./data/maze_hard/test", aug=False),
            }
        train_pbs = []
        for i in range(n_mazes_train):
            train_pbs.append(
                generate_mazes.hard_mazes["train"][
                    random.randint(0, len(generate_mazes.hard_mazes["train"]))
                ]
            )
        test_pbs = []
        for i in range(n_mazes_test):
            test_pbs.append(
                generate_mazes.hard_mazes["test"][
                    random.randint(0, len(generate_mazes.hard_mazes["test"]))
                ]
            )

    else:
        maze_ctor = _resolve_maze_ctor(maze_gen)
        cfg_train = MazeDatasetConfig(
            name="train",
            grid_n=size_train,
            n_mazes=1 if forbid_trivial else n_mazes_train,
            maze_ctor=maze_ctor,
            seed=None,
        )
        if forbid_trivial:
            train_dataset = []
            while len(train_dataset) < n_mazes_train:
                d = MazeDataset.from_config(
                    cfg_train, load_local=False, save_local=False, do_download=False
                )
                if d[0].solution.shape[0] > 2:
                    train_dataset.append(d[0])
                else:
                    cfg_train = MazeDatasetConfig(
                        name="train",
                        grid_n=size_train,
                        n_mazes=1,
                        maze_ctor=maze_ctor,
                        seed=None,
                    )

        else:
            train_dataset = MazeDataset.from_config(
                cfg_train, load_local=False, save_local=False, do_download=False
            )

        if not test_mazes:
            train_pbs = [PathPlanningProblem(m) for m in train_dataset]
        else:
            while (
                True
            ):  # if all the mazes are in the test set, we generate a new set of mazes
                train_pbs = [
                    PathPlanningProblem(m)
                    for m in train_dataset
                    if PathPlanningProblem(m).data_hash() not in test_mazes
                ]
                if train_pbs != []:
                    break
                else:
                    warnings.warn(
                        "All the mazes are in the test set, generating a new set of mazes"
                    )
                    train_dataset = MazeDataset.from_config(
                        cfg_train, load_local=False, save_local=False, do_download=False
                    )

        if n_mazes_test != 0:
            cfg_test = MazeDatasetConfig(
                name="test",
                grid_n=size_test if size_test != 0 else size_train,
                n_mazes=n_mazes_test,
                maze_ctor=maze_ctor,
            )
            test_dataset = MazeDataset.from_config(cfg_test)
            test_pbs = [PathPlanningProblem(m) for m in test_dataset]

            for t in test_pbs:
                th = t.data_hash()
                global_test_hashes.append(th)
        else:
            test_pbs = []

    return train_pbs, test_pbs
