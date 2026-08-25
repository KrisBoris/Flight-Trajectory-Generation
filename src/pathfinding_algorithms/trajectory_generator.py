# trajectory_generator.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from dataclasses import dataclass
from pathfinding_algorithms import greedy_pathfinding, metaheuristic_pathfinding, astar_pathfinding
import numpy as np


# Maps an algorithm name to a "find the best path from one starting cell"
# function with signature (grid, start_row, start_col, max_cost,
# require_return_to_base, blocked_mask) -> (path, total_value, cost_used).
# TrajectoryGenerator.find_best_path tries every starting cell with whichever
# function is selected here and keeps the overall best result. Add a new
# algorithm by writing such a module next to greedy_pathfinding.py and
# registering its function here.
PATHFINDING_ALGORITHMS = {
    "greedy": greedy_pathfinding.find_path_for_highest_neighbor_value,
    "direct_to_highest_value": greedy_pathfinding.find_path_to_highest_value,
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
    ) -> tuple[list[tuple[int, int]], float, float]:
        """
        Runs the selected pathfinding algorithm (a key in
        PATHFINDING_ALGORITHMS) from (start_row, start_col) and returns its
        path, total collected value, and cost used.
        """
        if max_cost < 0:
            print(f"max_cost must be non-negative, not {max_cost}")
            return [], 0.0, 0.0

        if not (0 <= start_row < self.grid.rows and 0 <= start_col < self.grid.cols):
            print(f"start_row/start_col ({start_row}, {start_col}) is outside the grid ({self.grid.rows}x{self.grid.cols})")
            return [], 0.0, 0.0

        # The drone can't launch from a cell it isn't allowed to enter in
        # the first place.
        if blocked_mask is not None and blocked_mask[start_row, start_col]:
            print(f"start_row/start_col ({start_row}, {start_col}) is blocked")
            return [], 0.0, 0.0

        find_path_from = PATHFINDING_ALGORITHMS.get(algorithm)
        if find_path_from is None:
            print(f"Unknown algorithm '{algorithm}', expected one of {list(PATHFINDING_ALGORITHMS)}")
            return [], 0.0, 0.0

        return find_path_from(
            self.grid,
            start_row,
            start_col,
            max_cost,
            require_return_to_base=require_return_to_base,
            blocked_mask=blocked_mask,
        )


if __name__ == "__main__":
    pass
