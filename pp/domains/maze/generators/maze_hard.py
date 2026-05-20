import csv
import numpy as np
import os
import json
from maze_dataset.maze import SolvedMaze
from maze_dataset.plotting import MazePlot
from pp.domains.maze.problem import PathPlanningProblem
import matplotlib

CHARSET = "# SGo"


def dihedral_transform(arr: np.ndarray, tid: int) -> np.ndarray:
    """8 dihedral symmetries by rotate, flip and mirror"""

    if tid == 0:
        return arr  # identity
    elif tid == 1:
        return np.rot90(arr, k=1)
    elif tid == 2:
        return np.rot90(arr, k=2)
    elif tid == 3:
        return np.rot90(arr, k=3)
    elif tid == 4:
        return np.fliplr(arr)  # horizontal flip
    elif tid == 5:
        return np.flipud(arr)  # vertical flip
    elif tid == 6:
        return arr.T  # transpose (reflection along main diagonal)
    elif tid == 7:
        return np.fliplr(np.rot90(arr, k=1))  # anti-diagonal reflection
    else:
        return arr


def c_from_xy(x, y):
    return np.array([x, y])


def conn_list_from_hard(maze):

    maze = maze.reshape(30, 30)
    grid_n = 30

    connection_list = np.zeros((2, grid_n, grid_n), dtype=np.bool_)

    for x in range(29):
        for y in range(29):
            if maze[x, y] != 1:
                if maze[x + 1, y] != 1:
                    connection_list[0, x, y] = True
                if maze[x, y + 1] != 1:
                    connection_list[1, x, y] = True
    for y in range(29):
        if maze[29, y] != 1 and maze[29, y + 1] != 1:
            connection_list[1, 29, y] = True
    for x in range(29):
        if maze[x, 29] != 1 and maze[x + 1, 29] != 1:
            connection_list[0, x, 29] = True
    return connection_list


def find_neigh(plist, pos):
    for p in plist:
        if abs(p[0] - pos[0]) + abs(p[1] - pos[1]) == 1:
            return p


def rebuild_path(start, end, path_list):
    clist = [(start[0].item(), start[1].item())]
    cur_pos = start
    while len(path_list) != 0:
        cur_pos = find_neigh(path_list, cur_pos)
        path_list.remove([cur_pos[0], cur_pos[1]])
        clist.append((cur_pos[0], cur_pos[1]))
    clist.append((end[0].item(), end[1].item()))
    return clist


def preprocess_maze_hard(set_name, aug=True, save_img=False):
    # Read CSV
    all_chars = set()
    grid_size = None
    inputs = []
    labels = []

    print(f"fname ./data/maze_hard/{set_name}.csv")
    with open(f"./data/maze_hard/{set_name}.csv", newline="") as csvfile:  # type: ignore
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for source, q, a, rating in reader:
            all_chars.update(q)
            all_chars.update(a)

            if grid_size is None:
                n = int(len(q) ** 0.5)
                grid_size = (n, n)

            inputs.append(np.frombuffer(q.encode(), dtype=np.uint8).reshape(grid_size))
            labels.append(np.frombuffer(a.encode(), dtype=np.uint8).reshape(grid_size))

    # If subsample_size is specified for the training set,
    # randomly sample the desired number of examples.
    # if set_name == "train" and config.subsample_size is not None:
    #     total_samples = len(inputs)
    #     if config.subsample_size < total_samples:
    #         indices = np.random.choice(
    #             total_samples, size=config.subsample_size, replace=False
    #         )
    #         inputs = [inputs[i] for i in indices]
    #         labels = [labels[i] for i in indices]

    # Generate dataset
    results = {
        k: []
        for k in [
            "inputs",
            "labels",
            "puzzle_identifiers",
            "puzzle_indices",
            "group_indices",
        ]
    }
    puzzle_id = 0
    example_id = 0

    results["puzzle_indices"].append(0)
    results["group_indices"].append(0)

    for inp, out in zip(inputs, labels):
        # Dihedral transformations for augmentation
        for aug_idx in range(8 if (set_name == "train" and aug) else 1):
            results["inputs"].append(dihedral_transform(inp, aug_idx))
            results["labels"].append(dihedral_transform(out, aug_idx))
            example_id += 1
            puzzle_id += 1

            results["puzzle_indices"].append(example_id)
            results["puzzle_identifiers"].append(0)

        # Push group
        results["group_indices"].append(puzzle_id)

    # Char mappings
    assert len(all_chars - set(CHARSET)) == 0

    char2id = np.zeros(256, np.uint8)
    char2id[np.array(list(map(ord, CHARSET)))] = np.arange(len(CHARSET)) + 1

    # To Numpy
    def _seq_to_numpy(seq):
        arr = np.vstack([char2id[s.reshape(-1)] for s in seq])

        return arr

    results = {
        "inputs": _seq_to_numpy(results["inputs"]),
        "labels": _seq_to_numpy(results["labels"]),
        "group_indices": np.array(results["group_indices"], dtype=np.int32),
        "puzzle_indices": np.array(results["puzzle_indices"], dtype=np.int32),
        "puzzle_identifiers": np.array(results["puzzle_identifiers"], dtype=np.int32),
    }

    pbs = []
    for i, s in enumerate(results["labels"]):
        conn_list = conn_list_from_hard(s)
        m = s.reshape(30, 30)
        sp = np.array(np.where(m == 3)).squeeze(-1)
        ep = np.array(np.where(m == 4)).squeeze(-1)
        in_path = np.array(np.where(m == 5)).transpose().tolist()
        sol = rebuild_path(sp, ep, in_path)

        sm = SolvedMaze(
            connection_list=conn_list, solution=sol, start_pos=sp, end_pos=ep
        )
        pbs.append(PathPlanningProblem(sm))

        if save_img:
            plot = MazePlot(sm)
            plot.plot()
            plot.fig.savefig(f"hard_test_{i}.png", format="png")
            matplotlib.pyplot.close()

    return pbs

    # Metadata
    # metadata = PuzzleDatasetMetadata(
    #     seq_len=int(math.prod(grid_size)),  # type: ignore
    #     vocab_size=len(CHARSET) + 1,  # PAD + Charset
    #     pad_id=0,
    #     ignore_label_id=0,
    #     blank_identifier_id=0,
    #     num_puzzle_identifiers=1,
    #     total_groups=len(results["group_indices"]) - 1,
    #     mean_puzzle_examples=1,
    #     sets=["all"],
    # )

    # Save metadata as JSON.
    # save_dir = os.path.join(output_dir, set_name)
    # os.makedirs(save_dir, exist_ok=True)

    # # with open(os.path.join(save_dir, "dataset.json"), "w") as f:
    # #     json.dump(metadata.model_dump(), f)

    # # Save data
    # for k, v in results.items():
    #     np.save(os.path.join(save_dir, f"all__{k}.npy"), v)

    # # Save IDs mapping (for visualization only)
    # with open(os.path.join(output_dir, "identifiers.json"), "w") as f:
    #     json.dump(["<blank>"], f)


if __name__ == "__main__":
    preprocess_maze_hard("test", aug=False, save_img=True)
