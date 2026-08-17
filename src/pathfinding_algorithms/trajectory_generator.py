# trajectory_generator.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from dataclasses import dataclass
from pathfinding_algorithms import greedy_pathfinding
import numpy as np


# Maps an algorithm name to a "find the best path from one starting cell"
# function with signature (grid, start_row, start_col, max_cost,
# require_return_to_base, blocked_mask) -> (path, total_value, cost_used).
# TrajectoryGenerator.find_best_path tries every starting cell with whichever
# function is selected here and keeps the overall best result. Add a new
# algorithm by writing such a module next to greedy_pathfinding.py and
# registering its function here.
PATHFINDING_ALGORITHMS = {
    "greedy": greedy_pathfinding.find_path_from,
}


@dataclass
class TrajectoryGenerator():
    """
    Searches a CoordinatesGrid for the highest-value path, constrained by a
    total movement-cost budget taken from weights_grid. The single-start
    search strategy is pluggable - see PATHFINDING_ALGORITHMS.
    """

    grid: CoordinatesGrid


    def find_best_path(
        self,
        max_cost: float,
        require_return_to_base: bool = True,
        blocked_mask: np.ndarray = None,
        algorithm: str = "greedy",
    ) -> tuple[list[tuple[int, int]], float, float]:
        """
        Runs the selected pathfinding algorithm (a key in
        PATHFINDING_ALGORITHMS) from every cell in the grid and returns the
        path with the highest total collected value.
        """
        if max_cost < 0:
            print(f"max_cost must be non-negative, not {max_cost}")
            return [], 0.0, 0.0

        find_path_from = PATHFINDING_ALGORITHMS.get(algorithm)
        if find_path_from is None:
            print(f"Unknown algorithm '{algorithm}', expected one of {list(PATHFINDING_ALGORITHMS)}")
            return [], 0.0, 0.0

        best_path: list[tuple[int, int]] = []
        best_total_value = -np.inf
        best_cost_used = 0.0

        for row in range(self.grid.rows):
            for col in range(self.grid.cols):
                # Can't launch the search from a cell the drone isn't allowed
                # to enter in the first place.
                if blocked_mask is not None and blocked_mask[row, col]:
                    continue

                path, total_value, cost_used = find_path_from(
                    self.grid,
                    row,
                    col,
                    max_cost,
                    require_return_to_base=require_return_to_base,
                    blocked_mask=blocked_mask,
                )

                if total_value > best_total_value:
                    best_path = path
                    best_total_value = total_value
                    best_cost_used = cost_used

        return best_path, best_total_value, best_cost_used


if __name__ == "__main__":
    pass
