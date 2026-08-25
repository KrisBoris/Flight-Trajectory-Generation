# benchmark.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from coordinates_grid.weights_grid import WeightsGrid
from coordinates_grid.test_data_generator import generate_random_terrain_coordinates, build_blocked_mask
from helpers.data_loader import load_mission_data, load_drone_params
from pathfinding_algorithms.trajectory_generator import TrajectoryGenerator, PATHFINDING_ALGORITHMS
from pathlib import Path
import csv
import numpy as np
import time


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SCENARIO_DIR = PROJECT_ROOT / "test_data"
DEFAULT_DRONE_CONFIG_PATH = PROJECT_ROOT / "drone_data" / "large_drone_config.json"
DEFAULT_RESULTS_PATH = Path(__file__).resolve().parent / "benchmark_results.csv"


def run_benchmark(
    scenario_dir: Path = DEFAULT_SCENARIO_DIR,
    drone_config_path: Path = DEFAULT_DRONE_CONFIG_PATH,
    algorithms: list = None,
    scenario_files: list = None,
    terrain_seed: int = 0,
    repetitions: int = 1,
    results_path: Path = DEFAULT_RESULTS_PATH,
) -> list:
    """
    Runs every pathfinding algorithm in PATHFINDING_ALGORITHMS (or just
    `algorithms`, if given) against every scenario*.json file found in
    scenario_dir (or just `scenario_files`, if given), `repetitions` times
    each, prints a results table to the console, and saves the same data as
    a CSV file at results_path. Returns the list of result dicts - the same
    rows written to the CSV.

    PERFORMANCE WARNING: this is a genuinely large sweep by default, and
    `repetitions` multiplies it further. Several algorithms in this project
    (ant_colony, tabu_search, variable_neighborhood_search, grasp,
    namoa_star) already take anywhere from a few seconds to over a minute
    PER SCENARIO on a 100-target scenario, and this project's own "_dense"
    scenario files have up to 510 targets - considerably more expensive
    again for every tournament-style algorithm. Running the full default
    set (every registered algorithm x every scenario file in test_data/)
    even once can easily take well over an hour. Pass a smaller
    `algorithms` and/or `scenario_files` list, and keep `repetitions` low,
    for a quick check instead of the full sweep.

    REPETITIONS: rerunning the exact same (scenario, algorithm) pair
    multiple times is only meaningful because most of the algorithms in
    this project are randomized (ant_colony, tabu_search, variable_
    neighborhood_search, grasp, simulated_annealing, genetic_algorithm, and
    the candidate ordering inside namoa_star) - each call already draws
    fresh randomness on its own (every algorithm defaults to seed=None), so
    simply calling run_benchmark's inner loop again naturally explores a
    different outcome each time, with no seed bookkeeping needed here. This
    parameter exists specifically so you don't have to wrap run_benchmark in
    your own outer loop (and re-merge several separate CSVs afterward) just
    to see how much an algorithm's result varies run to run - every
    repetition is tagged with a "repetition" column (1-indexed) in both the
    raw results and the CSV, and when repetitions > 1 an additional summary
    table (mean +/- standard deviation per scenario/algorithm) is printed
    after the raw one.

    The deterministic algorithms (greedy, direct_to_highest_value,
    value_cost_ratio, lowest_cost, and the three plain A* variants) will
    simply produce identical rows across every repetition, since nothing
    about them is randomized - that is expected, not a bug.

    Terrain is regenerated per scenario with a fixed terrain_seed (not one
    seed per algorithm or per repetition) so every algorithm on a given
    scenario - across every repetition - sees identical terrain/costs; only
    each algorithm's own internal randomness varies between repetitions, a
    fair, apples-to-apples comparison, the same convention used throughout
    this project's own algorithm comparisons.

    A (scenario, algorithm, repetition) combination that raises an
    exception is recorded with a non-empty "error" column instead of
    aborting the whole run, so one bad combination never costs every result
    gathered before it.
    """
    scenario_dir = Path(scenario_dir)
    if scenario_files is None:
        scenario_files = sorted(scenario_dir.glob("scenario*.json"))
    else:
        scenario_files = [Path(scenario_file) for scenario_file in scenario_files]

    if algorithms is None:
        algorithms = list(PATHFINDING_ALGORITHMS)

    drone_params = load_drone_params(Path(drone_config_path))

    results = []
    total_runs = len(scenario_files) * len(algorithms) * repetitions
    run_index = 0

    for scenario_file in scenario_files:
        mission_data = load_mission_data(scenario_file)
        grid, blocked_mask = _build_grid(mission_data, drone_params, terrain_seed)
        targets = [(int(area[0]), int(area[1])) for area in mission_data["search_areas"]]
        trajectory_generator = TrajectoryGenerator(grid=grid)

        for algorithm in algorithms:
            for repetition in range(1, repetitions + 1):
                run_index += 1
                repetition_suffix = f" (repetition {repetition}/{repetitions})" if repetitions > 1 else ""
                print(f"[{run_index}/{total_runs}] {scenario_file.name} / {algorithm}{repetition_suffix} ...", flush=True)
                start_time = time.time()

                try:
                    path, total_value, cost_used = trajectory_generator.find_best_path(
                        start_row=mission_data["start_row"],
                        start_col=mission_data["start_col"],
                        max_cost=drone_params["max_cost"],
                        require_return_to_base=drone_params["require_return_to_base"],
                        blocked_mask=blocked_mask,
                        algorithm=algorithm,
                    )
                    elapsed_seconds = time.time() - start_time
                    path_cells = set(path)
                    targets_hit = sum(1 for cell in targets if cell in path_cells)
                    budget_used_pct = 100 * cost_used / drone_params["max_cost"] if drone_params["max_cost"] > 0 else 0.0

                    results.append({
                        "scenario": scenario_file.name,
                        "algorithm": algorithm,
                        "repetition": repetition,
                        "steps": len(path),
                        "cost_used": round(cost_used, 2),
                        "budget_used_pct": round(budget_used_pct, 1),
                        "total_value": round(total_value, 2),
                        "targets_hit": targets_hit,
                        "targets_total": len(targets),
                        "runtime_seconds": round(elapsed_seconds, 2),
                        "error": "",
                    })
                except Exception as error:
                    elapsed_seconds = time.time() - start_time
                    results.append({
                        "scenario": scenario_file.name,
                        "algorithm": algorithm,
                        "repetition": repetition,
                        "steps": None,
                        "cost_used": None,
                        "budget_used_pct": None,
                        "total_value": None,
                        "targets_hit": None,
                        "targets_total": len(targets),
                        "runtime_seconds": round(elapsed_seconds, 2),
                        "error": str(error),
                    })

    _print_results_table(results)
    if repetitions > 1:
        _print_summary_table(results)
    _save_results_csv(results, Path(results_path))
    print(f"\nSaved {len(results)} results to {results_path}")

    return results


def _build_grid(mission_data: dict, drone_params: dict, terrain_seed: int):
    """
    Builds the same CoordinatesGrid + terrain + blocked_mask pipeline
    main.py uses, from an already-loaded mission_data/drone_params pair -
    factored out so run_benchmark can reuse it once per scenario file
    instead of duplicating main.py's setup inline for every algorithm run.

    Returns (grid, blocked_mask).
    """
    rows, cols = mission_data["rows"], mission_data["cols"]
    grid = CoordinatesGrid(
        coordinates_values=np.ones((rows, cols)),
        weights_grid=WeightsGrid(weights=np.ones((rows, cols, 8))),
    )
    grid.init_grids(rows, cols)
    grid.set_searched_areas(mission_data["search_areas"])

    np.random.seed(terrain_seed)
    terrain_coordinates = generate_random_terrain_coordinates(
        rows,
        cols,
        altitude_range=mission_data["altitude_range"],
        cell_size_meters=mission_data["cell_size_meters"],
        max_gradient=mission_data["max_gradient"],
    )
    grid.weights_grid.init_from_elevation(
        terrain_coordinates,
        climb_cost_per_meter=drone_params["climb_cost_per_meter"],
        descent_cost_per_meter=drone_params["descent_cost_per_meter"],
        base_cost=drone_params["base_cost"],
    )

    blocked_mask = build_blocked_mask(rows, cols, mission_data["blocked_cells"])

    return grid, blocked_mask


def _print_results_table(results: list) -> None:
    """
    Prints `results` (see run_benchmark) as a simple, aligned console table,
    with a blank line separating each scenario's block of algorithm rows.
    The "rep" column only appears when results actually contain more than
    one repetition, keeping a single-repetition run's output identical to
    before repetitions existed.
    """
    if not results:
        print("No results to display.")
        return

    show_repetition = max(row["repetition"] for row in results) > 1
    repetition_header = f"{'rep':>5}" if show_repetition else ""
    header = f"{'scenario':<28}{'algorithm':<30}{repetition_header}{'steps':>7}{'cost':>11}{'budget%':>9}{'value':>11}{'targets':>10}{'time(s)':>9}"
    print()
    print(header)
    print("-" * len(header))

    current_scenario = None
    for row in results:
        if row["scenario"] != current_scenario:
            if current_scenario is not None:
                print()
            current_scenario = row["scenario"]

        repetition_cell = f"{row['repetition']:>5}" if show_repetition else ""

        if row["error"]:
            print(f"{row['scenario']:<28}{row['algorithm']:<30}{repetition_cell}ERROR: {row['error'][:60]}")
            continue

        targets_str = f"{row['targets_hit']}/{row['targets_total']}"
        print(
            f"{row['scenario']:<28}{row['algorithm']:<30}{repetition_cell}{row['steps']:>7}{row['cost_used']:>11.1f}"
            f"{row['budget_used_pct']:>8.1f}%{row['total_value']:>11.2f}{targets_str:>10}{row['runtime_seconds']:>9.2f}"
        )


def _print_summary_table(results: list) -> None:
    """
    Prints one aggregated row per (scenario, algorithm) - mean +/- standard
    deviation of total_value, cost_used and runtime_seconds across every
    repetition, plus the best (max) total_value seen - so a multi-repetition
    run's variance is visible at a glance without opening the CSV. Rows
    whose "error" is non-empty are excluded from the statistics; a
    (scenario, algorithm) group where every repetition errored is skipped
    entirely rather than printed as an empty/NaN row.
    """
    groups = {}
    for row in results:
        if row["error"]:
            continue
        key = (row["scenario"], row["algorithm"])
        groups.setdefault(key, []).append(row)

    if not groups:
        return

    print()
    print("=== Summary across repetitions (mean +/- std, best) ===")
    header = f"{'scenario':<28}{'algorithm':<30}{'runs':>5}{'value (mean+/-std)':>26}{'best value':>12}{'cost (mean+/-std)':>26}{'time(s) (mean+/-std)':>24}"
    print()
    print(header)
    print("-" * len(header))

    current_scenario = None
    for (scenario, algorithm), rows in groups.items():
        if scenario != current_scenario:
            if current_scenario is not None:
                print()
            current_scenario = scenario

        values = [row["total_value"] for row in rows]
        costs = [row["cost_used"] for row in rows]
        runtimes = [row["runtime_seconds"] for row in rows]

        value_mean, value_std = float(np.mean(values)), float(np.std(values))
        cost_mean, cost_std = float(np.mean(costs)), float(np.std(costs))
        runtime_mean, runtime_std = float(np.mean(runtimes)), float(np.std(runtimes))

        print(
            f"{scenario:<28}{algorithm:<30}{len(rows):>5}"
            f"{value_mean:>14.2f} +/- {value_std:<7.2f}{max(values):>12.2f}"
            f"{cost_mean:>14.2f} +/- {cost_std:<7.2f}{runtime_mean:>13.2f} +/- {runtime_std:<7.2f}"
        )


def _save_results_csv(results: list, path: Path) -> None:
    """Saves `results` (see run_benchmark) as a CSV file, creating parent directories if needed."""
    if not results:
        return

    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(results[0].keys())

    with open(path, "w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(results)


if __name__ == "__main__":
    run_benchmark()
