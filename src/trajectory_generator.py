# trajectory_generator.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from constants import Constants
from dataclasses import dataclass
import numpy as np


@dataclass
class TrajectoryGenerator():
    """
    Greedy search for the highest-value path through a CoordinatesGrid,
    constrained by a total movement-cost budget taken from weights_grid.
    """

    grid: CoordinatesGrid


    def _greedy_path_from(
        self,
        start_row: int,
        start_col: int,
        max_cost: float,
        require_return_to_base: bool = True,
        blocked_mask: np.ndarray = None,
    ) -> tuple[list[tuple[int, int]], float, float]:

        rows, cols = self.grid.rows, self.grid.cols
        weights = self.grid.weights_grid.weights
        values = self.grid.coordinates_values

        visited = np.zeros((rows, cols), dtype=bool)
        visited[start_row, start_col] = True

        path = [(start_row, start_col)]
        total_value = values[start_row, start_col]
        remaining_budget = max_cost
        row, col = start_row, start_col

        # Running cost of retracing the outbound path flown so far, edge by
        # edge, back to base. Updated incrementally (one edge per accepted
        # move) instead of being recomputed from scratch every step.
        return_trip_cost = 0.0

        while True:
            best_value = None
            best_move = None  # (next_row, next_col, cost, reverse_cost)

            for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
                next_row = row + delta_row
                next_col = col + delta_col

                if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                    continue
                if visited[next_row, next_col]:
                    continue
                # A no-fly cell (storm cell, restricted airspace, terrain the
                # drone can't overfly) is removed from consideration outright,
                # regardless of how cheap it would otherwise be to reach.
                if blocked_mask is not None and blocked_mask[next_row, next_col]:
                    continue

                cost = weights[row, col, direction] * Constants.DIRECTION_COST_MULTIPLIERS[direction]

                reverse_cost = 0.0
                if require_return_to_base:
                    # In an 8-direction compass the opposite of `direction` is
                    # always 4 slots away (up <-> down, up-right <-> down-left,
                    # ...). reverse_cost is what flying straight back along
                    # this same edge - from the candidate cell to the one
                    # we're standing on now - would cost.
                    reverse_direction = (direction + 4) % 8
                    reverse_cost = weights[next_row, next_col, reverse_direction] * Constants.DIRECTION_COST_MULTIPLIERS[reverse_direction]

                # The budget must cover this move AND retracing every edge
                # flown so far - including this new one - back to base.
                # Otherwise the drone could strand itself past the point of
                # no return.
                if cost + return_trip_cost + reverse_cost > remaining_budget:
                    continue

                value = values[next_row, next_col]
                if best_value is None or value > best_value:
                    best_value = value
                    best_move = (next_row, next_col, cost, reverse_cost)

            if best_move is None:
                break

            next_row, next_col, cost, reverse_cost = best_move
            visited[next_row, next_col] = True
            path.append((next_row, next_col))
            total_value += best_value
            remaining_budget -= cost
            return_trip_cost += reverse_cost
            row, col = next_row, next_col

        if not require_return_to_base:
            return path, total_value, max_cost - remaining_budget

        # Append the actual return leg: retrace the outbound path in reverse,
        # from the current cell back to base. path[-2::-1] is every cell
        # except the last one, reversed - e.g. [start, a, b] -> [a, start].
        path_with_return = path + path[-2::-1]
        total_cost_used = (max_cost - remaining_budget) + return_trip_cost

        return path_with_return, total_value, total_cost_used


    def find_best_path(
        self,
        max_cost: float,
        require_return_to_base: bool = True,
        blocked_mask: np.ndarray = None,
    ) -> tuple[list[tuple[int, int]], float, float]:
        """
        Runs the greedy search from every cell in the grid and returns the
        path with the highest total collected value.
        """
        if max_cost < 0:
            print(f"max_cost must be non-negative, not {max_cost}")
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

                path, total_value, cost_used = self._greedy_path_from(
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
