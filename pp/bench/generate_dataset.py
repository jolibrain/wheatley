from maze_dataset import MazeDataset, MazeDatasetConfig
from maze_dataset.generation import LatticeMazeGenerators
import pickle
import argparse


def generate_dataset(size, nmazes, seed):
    cfg_train = MazeDatasetConfig(
        name=f"dataset_{size}x{size}_{nmazes}",
        grid_n=size,
        n_mazes=nmazes,
        maze_ctor=LatticeMazeGenerators.gen_dfs_percolation,
        seed=seed,
    )
    d = MazeDataset.from_config(
        cfg_train, load_local=False, save_local=False, do_download=False
    )
    st = d.serialize()
    with open(f"dataset_{size}x{size}_{nmazes}.pkl", "wb") as f:
        pickle.dump(st, f)


def load_dataset(fname):
    with open(fname, "rb") as f:
        d = pickle.load(f)
    return d


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Dataset generator")
    parser.add_argument("--size", type=int, default=5, help="maze size")
    parser.add_argument("--seed", type=int, default=324, help="seed")
    parser.add_argument(
        "--n", type=int, default=100, help="number of mazes to generate"
    )
    args = parser.parse_args()
    generate_dataset(args.size, args.n, args.seed)
    # d = load_dataset("dataset_5x5_100.pkl")
