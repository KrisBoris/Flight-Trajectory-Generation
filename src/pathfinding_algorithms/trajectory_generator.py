from coordinates_grid.coordinates_grid import CoordinatesGrid
from dataclasses import dataclass
from helpers.constants import Constants
from pathfinding_algorithms import greedy_pathfinding, metaheuristic_pathfinding, astar_pathfinding, exact_pathfinding
import numpy as np


# Maps an algorithm name to a "find the best path from one starting cell"
# function with signature (grid, start_row, start_col, max_cost,
# require_return_to_base, blocked_mask) -> (path, total_value, cost_used).
# TrajectoryGenerator.find_best_path tries every starting cell with whichever
# function is selected here and keeps the overall best result. Add a new
# algorithm by writing such a module next to greedy_pathfinding.py and
# registering its function here.
PATHFINDING_ALGORITHMS = {
    "greedy": greedy_pathfinding.find_path_to_highest_value_neighbor,
    "direct_to_highest_value": greedy_pathfinding.find_path_by_highest_value,
    "value_cost_ratio": greedy_pathfinding.find_path_by_value_cost_ratio,
    "lowest_cost": greedy_pathfinding.find_path_by_lowest_cost,
    "ant_colony": metaheuristic_pathfinding.find_path_by_ant_colony,
    "tabu_search": metaheuristic_pathfinding.find_path_by_tabu_search,
    "variable_neighborhood_search": metaheuristic_pathfinding.find_path_by_variable_neighborhood_search,
    "grasp": metaheuristic_pathfinding.find_path_by_grasp,
    "simulated_annealing": metaheuristic_pathfinding.find_path_by_simulated_annealing,
    "a_star_to_highest_value": astar_pathfinding.find_path_by_a_star_to_highest_value,
    "a_star_value_cost_ratio": astar_pathfinding.find_path_by_a_star_value_cost_ratio,
    "a_star_lowest_cost": astar_pathfinding.find_path_by_a_star_lowest_cost,
    "namoa_star": astar_pathfinding.find_path_by_namoa_star,
    "genetic_algorithm": metaheuristic_pathfinding.find_path_by_genetic_algorithm,
    "large_neighborhood_search": metaheuristic_pathfinding.find_path_by_large_neighborhood_search,
    "exact_solver": exact_pathfinding.find_path_by_exact_solver,
}


@dataclass
class TrajectoryGenerator():
    """
    Searches a CoordinatesGrid for the highest-value path from a fixed
    starting cell (e.g. the rescue team's base - see
    helpers.data_loader.load_mission_data's start_location), constrained by a total
    movement-cost budget taken from weights_grid. The search strategy is
    pluggable - see PATHFINDING_ALGORITHMS.
    """

    grid: CoordinatesGrid


    def find_best_path(
        self,
        start_row: int,
        start_col: int,
        max_cost: float,
        require_return_to_base: bool = True,
        blocked_mask: np.ndarray = None,
        algorithm: str = "greedy",
        actual_person_locations: list = None,
    ) -> tuple[list[tuple[int, int]], float, float, int]:
        """
        Runs the selected pathfinding algorithm (a key in
        PATHFINDING_ALGORITHMS) from (start_row, start_col) and returns its
        path, total collected value, cost used, and how many of
        actual_person_locations that path actually reached.

        actual_person_locations (optional) is the mission's ground-truth
        list of where the searched person(s) actually are - see
        helpers.data_loader.load_mission_data's "actual_person_locations" -
        as opposed to grid.coordinates_values, which only ever encodes a
        prior belief about where they might be. When given (and non-empty),
        see _stop_once_everyone_found for the additional, uniform stopping
        rule this adds on top of whichever algorithm was selected. Left as
        None (the default), every algorithm runs exactly as it did before
        this parameter existed, and the returned "people found" count is 0
        (there was nothing to look for).
        """
        if max_cost < 0:
            print(f"max_cost must be non-negative, not {max_cost}")
            return [], 0.0, 0.0, 0

        if not (0 <= start_row < self.grid.rows and 0 <= start_col < self.grid.cols):
            print(f"start_row/start_col ({start_row}, {start_col}) is outside the grid ({self.grid.rows}x{self.grid.cols})")
            return [], 0.0, 0.0, 0

        # The drone can't launch from a cell it isn't allowed to enter in
        # the first place.
        if blocked_mask is not None and blocked_mask[start_row, start_col]:
            print(f"start_row/start_col ({start_row}, {start_col}) is blocked")
            return [], 0.0, 0.0, 0

        find_path_from = PATHFINDING_ALGORITHMS.get(algorithm)
        if find_path_from is None:
            print(f"Unknown algorithm '{algorithm}', expected one of {list(PATHFINDING_ALGORITHMS)}")
            return [], 0.0, 0.0, 0

        path, total_value, cost_used = find_path_from(
            self.grid,
            start_row,
            start_col,
            max_cost,
            require_return_to_base=require_return_to_base,
            blocked_mask=blocked_mask,
        )

        if not actual_person_locations:
            return path, total_value, cost_used, 0

        return _stop_once_everyone_found(
            self.grid, path, total_value, cost_used,
            start_row, start_col, max_cost, require_return_to_base, blocked_mask,
            actual_person_locations,
        )


def _stop_once_everyone_found(
    grid: CoordinatesGrid,
    path: list,
    total_value: float,
    cost_used: float,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool,
    blocked_mask: np.ndarray,
    actual_person_locations: list,
) -> tuple[list[tuple[int, int]], float, float, int]:
    """
    Adds one additional, algorithm-independent stopping rule on top of
    whichever algorithm TrajectoryGenerator.find_best_path just ran: once
    the drone's path has actually passed through every cell in
    actual_person_locations - the mission's real, ground-truth targets, as
    opposed to grid.coordinates_values' prior probability belief - there is
    no more reason to keep flying toward whatever the algorithm's own
    objective was still chasing. If they are not all found by the time the
    algorithm's own path ends, nothing changes here at all - the original
    result, reached however that algorithm's own stopping condition
    naturally triggered, is returned untouched.

    WHY THIS WORKS FOR EVERY ALGORITHM UNCHANGED
    -----------------------------------------------
    Every algorithm in this project - from the simplest greedy walker to
    the metaheuristics' tour evaluation to the exact solver's stitched-
    together CP-SAT solution - already returns `path` as the FULL,
    cell-by-cell sequence the drone actually flies, not just a list of
    waypoints. That means the question "when, if ever, did the drone
    happen to pass by every real person" can be answered by scanning the
    ALREADY-COMPUTED path directly, without needing to know or modify
    anything about how any specific algorithm decided to build it. This is
    also strictly MORE precise than checking only at each algorithm's own
    internal decision points (e.g. only after finishing a whole leg to some
    other target) would be: it catches a person the exact instant their
    cell is entered, even if that happens partway through a longer walk
    toward a completely different destination.

    A ground-truth person whose cell coincides with a blocked cell can
    never actually be found (the drone can never legally enter it) - this
    is not treated as an error here (see helpers.data_loader.load_mission_
    data, which only prints a warning about it at load time): the scan
    below simply never finds all of them, so this function returns the
    original, untouched result, which is exactly the documented fallback
    behavior for "not everyone was found".

    WHAT "OPTIMAL" STILL MEANS FOR exact_solver
    -----------------------------------------------
    Truncating exact_pathfinding.find_path_by_exact_solver's output here is
    no different from truncating any other algorithm's - CP-SAT still
    proved its result optimal for its own actual objective (maximum
    grid.coordinates_values collected under budget), which has no idea
    actual_person_locations even exists. This function does not change
    that proof or make it stronger or weaker; it only decides, after the
    fact, that the mission stops being flown once it is realistically
    "done" - the same as for every other algorithm.

    Returns (path, total_value, cost_used, persons_found) - path truncated
    at the first point every real person was reached, plus a fresh walk
    home appended from there if require_return_to_base, with persons_found
    equal to len(actual_person_locations) in that case (everyone was
    found) - or the original, unmodified path/total_value/cost_used if
    that point never occurs before the path's own natural end, with
    persons_found counting however many (possibly not all) of
    actual_person_locations the path actually reached by then.
    """
    still_missing = set(actual_person_locations)

    found_at_index = None
    for index, cell in enumerate(path):
        still_missing.discard(cell)
        if not still_missing:
            found_at_index = index
            break

    persons_found = len(actual_person_locations) - len(still_missing)

    if found_at_index is None or found_at_index >= len(path) - 1:
        return path, total_value, cost_used, persons_found  # never all found, or only right at the path's own natural end - nothing to trim

    truncated_path = path[:found_at_index + 1]
    truncated_cost = _replay_path_cost(grid, truncated_path)

    visited = set()
    truncated_value = 0.0
    for cell in truncated_path:
        if cell not in visited:
            visited.add(cell)
            truncated_value += float(grid.coordinates_values[cell])

    if not require_return_to_base:
        return truncated_path, truncated_value, truncated_cost, persons_found

    row, col = truncated_path[-1]
    rows, cols = grid.rows, grid.cols
    max_steps = 8 * (rows + cols)
    visited_mask = np.zeros((rows, cols), dtype=bool)
    for cell in visited:
        visited_mask[cell] = True

    # Every algorithm's own step-acceptance logic already guarantees
    # "the shortest route home from wherever I am now still fits in
    # max_cost" at every cell it actually visited - truncated_path[-1] is
    # one such cell, so this walk home is guaranteed to fit within
    # max_cost - truncated_cost; `reached` is only checked defensively.
    reached, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, max_cost - truncated_cost, None, blocked_mask, visited_mask, max_steps,
    )
    if not reached:
        return truncated_path, truncated_value, truncated_cost, persons_found

    return truncated_path + return_path_cells, truncated_value + return_value_gained, truncated_cost + return_cost, persons_found


def _replay_path_cost(grid: CoordinatesGrid, path: list) -> float:
    """
    Recomputes the total movement cost of an already-computed path by
    replaying it one step at a time through grid.weights_grid.weights.
    Needed because every algorithm only ever returns one aggregate cost for
    its WHOLE path, never a per-step breakdown - so this is the only way to
    get the cost of just a PREFIX of a path some algorithm already produced
    (see _stop_once_everyone_found). Assumes - true of every path any
    algorithm in this project produces - that each consecutive pair of
    cells is a single legal compass step apart.
    """
    weights = grid.weights_grid.weights
    cost = 0.0

    for (row, col), (next_row, next_col) in zip(path, path[1:]):
        direction = Constants.DIRECTIONS.index((next_row - row, next_col - col))
        cost += weights[row, col, direction]

    return cost


if __name__ == "__main__":
    pass
