import sys

import numpy as np

from utils.utils import generate_data

from config import MAX_DURATION

sys.path.append("..")


def main():
    seed = 200
    for n_j, n_m in [(6, 6), (10, 10), (15, 15), (20, 20), (30, 20)]:
        data = generate_data(n_j, n_m, MAX_DURATION, seed, 100)
        np.save("generated_data{}_{}_seed{}.npy".format(n_j, n_m, seed), data)


if __name__ == "__main__":
    main()
