from __future__ import annotations

import argparse

import matplotlib.pyplot as plt
import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=50)
    args = parser.parse_args()

    arr = np.random.rand(args.n, args.n)
    plt.imshow(arr, cmap="viridis")
    plt.title("Placement Heatmap")
    plt.colorbar()
    plt.show()


if __name__ == "__main__":
    main()
