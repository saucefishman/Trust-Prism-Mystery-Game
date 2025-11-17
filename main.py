from metrics import sweep_hardness, export_metrics_json, export_metrics_csv
from puzzle import generate_puzzle, generate_puzzles_recursive, World, get_potential_worlds, TimeOrdering
import random


def main():
    # parameter grid
    alphas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]
    betas = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0]

    # generate metrics
    metrics = sweep_hardness(alphas, betas, max_clues=30, trials=10)

    # export to json/csv
    export_metrics_json(metrics)
    export_metrics_csv(metrics)

    print("Done.")


if __name__ == '__main__':
    main()