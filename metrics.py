import time
from puzzle import Puzzle, UnaryPositive, UnaryNegative, CrimePositive, CrimeNegative, TimeOrdering, IndirectPositive, \
    IndirectNegative, generate_puzzle, World, Clue


def timed(fn, *args, **kwargs):
    """Return (result, runtime_seconds)."""
    t0 = time.perf_counter()
    result = fn(*args, **kwargs)
    t1 = time.perf_counter()
    return result, (t1 - t0)


def measure_puzzle_hardness(puzzle: Puzzle):
    """
    Returns metrics describing the hardness of solving the puzzle.
    """
    # measure solver time
    pw, solve_time = timed(puzzle.get_potential_worlds)

    if pw is None:
        feasible = 0
        domain_size = 0
    else:
        feasible = pw.num_potential_murderers()
        domain_size = pw.get_total_domain_size()

    return {
        "solve_time": solve_time,
        "feasible_murderers": feasible,
        "domain_size": domain_size,
        "num_clues": len(puzzle.clues),
        "direct_clue_fraction": sum(
            isinstance(c, (UnaryPositive, UnaryNegative, CrimePositive, CrimeNegative, TimeOrdering))
            for c in puzzle.clues
        ) / len(puzzle.clues) if puzzle.clues else 0,
        "indirect_positive_fraction": sum(isinstance(c, IndirectPositive) for c in puzzle.clues) / len(
            puzzle.clues) if puzzle.clues else 0,
        "indirect_negative_fraction": sum(isinstance(c, IndirectNegative) for c in puzzle.clues) / len(
            puzzle.clues) if puzzle.clues else 0,
    }


def trace_info_gain(world: World, clues: list[Clue]):
    """
    Returns a list of dicts describing the hardness after each clue.
    Helps produce the plot:
        feasible murderers vs # clues
        domain size vs # clues
        runtime per step
    """
    acc = []
    for i in range(1, len(clues) + 1):
        partial = Puzzle(world, clues[:i])
        pw, runtime = timed(partial.get_potential_worlds)

        if pw is None:
            feasible = 0
            domain_size = 0
        else:
            feasible = pw.num_potential_murderers()
            domain_size = pw.get_total_domain_size()

        acc.append({
            "n_clues": i,
            "solve_time": runtime,
            "feasible_murderers": feasible,
            "domain_size": domain_size,
            "clue_type": clues[i - 1].__class__.__name__,
            "clue_repr": repr(clues[i - 1])
        })
    return acc


def sweep_hardness(alpha_values, beta_values, max_clues=10, num_suspects=5, trials=3):
    """
    Return matrix of hardness metrics over α × β.
    Each point aggregates multiple trials.
    """
    results = []

    for a in alpha_values:
        for b in beta_values:
            print('Testing alpha=', a, 'beta=', b)
            trial_metrics = []

            for _ in range(trials):
                puzzle, puzzle_gen_time = timed(generate_puzzle, alpha=a, beta=b, max_clues=max_clues, num_suspects=num_suspects)

                metrics = measure_puzzle_hardness(puzzle)
                metrics["puzzle_gen_time"] = puzzle_gen_time
                metrics["alpha"] = a
                metrics["beta"] = b

                # Add info-gain curve
                metrics["trace"] = trace_info_gain(puzzle.world, puzzle.clues)

                trial_metrics.append(metrics)

            # combine averages
            avg_puzzle_gen_time = sum(m["puzzle_gen_time"] for m in trial_metrics) / trials
            avg_solve_time = sum(m["solve_time"] for m in trial_metrics) / trials
            avg_feasible = sum(m["feasible_murderers"] for m in trial_metrics) / trials
            avg_domain = sum(m["domain_size"] for m in trial_metrics) / trials

            # fraction of clue types
            avg_direct = sum(m["direct_clue_fraction"] for m in trial_metrics) / trials
            avg_pos = sum(m["indirect_positive_fraction"] for m in trial_metrics) / trials
            avg_neg = sum(m["indirect_negative_fraction"] for m in trial_metrics) / trials

            results.append({
                "alpha": a,
                "beta": b,
                "avg_puzzle_gen_time": avg_puzzle_gen_time,
                "avg_solve_time": avg_solve_time,
                "avg_feasible_murderers": avg_feasible,
                "avg_domain_size": avg_domain,
                "avg_direct_clues": avg_direct,
                "avg_indirect_positive": avg_pos,
                "avg_indirect_negative": avg_neg,
                "trials": trial_metrics,
            })

    return results


import json


def export_metrics_json(metrics, filename="hardness_metrics.json"):
    with open(filename, "w") as f:
        json.dump(metrics, f, indent=4)


import csv


def export_metrics_csv(metrics, filename="hardness_metrics.csv"):
    keys = [
        "alpha", "beta",
        "avg_puzzle_gen_time",
        "avg_solve_time",
        "avg_feasible_murderers",
        "avg_domain_size",
        "avg_direct_clues",
        "avg_indirect_positive",
        "avg_indirect_negative",
    ]
    with open(filename, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in metrics:
            writer.writerow({k: row[k] for k in keys})

