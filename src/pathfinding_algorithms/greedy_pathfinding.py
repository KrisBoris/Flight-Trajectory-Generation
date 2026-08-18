# greedy_pathfinding.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from constants import Constants
import numpy as np


def find_path_for_highest_neighbor_value(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Greedy search: starting at (start_row, start_col), repeatedly steps to
    the highest-value unvisited, reachable neighbor until no move is left
    that fits the remaining budget. See
    TrajectoryGenerator.find_best_path, which calls this once per candidate
    starting cell and keeps the best result overall.
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

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

            # weights already encodes real per-direction cost (e.g. climb vs
            # descent, from WeightsGrid.init_from_elevation) - no flat,
            # compass-direction-only multiplier on top, since grid-compass
            # direction ("up" = north on the grid) has no fixed relationship
            # to real elevation change, and stacking one on would fight the
            # real terrain-based cost instead of reflecting it.
            cost = weights[row, col, direction]

            reverse_cost = 0.0
            if require_return_to_base:
                # In an 8-direction compass the opposite of `direction` is
                # always 4 slots away (up <-> down, up-right <-> down-left,
                # ...). reverse_cost is what flying straight back along
                # this same edge - from the candidate cell to the one
                # we're standing on now - would cost.
                reverse_direction = (direction + 4) % 8
                reverse_cost = weights[next_row, next_col, reverse_direction]

            # The budget must cover this move AND retracing every edge
            # flown so far - including this new one - back to base.
            # Otherwise the drone could strand itself past the point of
            # no return.
            if cost + return_trip_cost + reverse_cost > remaining_budget:
                continue

            value = values[next_row, next_col]
            # Prefer strictly higher probability; among equal probabilities,
            # prefer the cheaper move.
            if best_value is None or value > best_value or (value == best_value and cost < best_move[2]):
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


def find_path_to_highest_value(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    1. Find the highest-value unvisited, unblocked cell in
       grid.coordinates_values.
    2. Head toward it one step at a time. At each step, the "ideal"
       direction is whichever closes both the row and column distance at
       once (diagonal) until one axis is aligned, then whichever closes the
       remaining axis (straight) - the same shortest-path logic as before.
       If that direction's cell is blocked, rotate to the next direction
       (direction + 1, wrapping around the 8-direction compass) and check
       again, up to all 8 directions, until an in-bounds, unblocked cell is
       found.
    3. Take that step - cost-checked the same way as
       find_path_for_highest_neighbor_value, including the
       require_return_to_base reserve - and go back to step 2 (the ideal
       direction is recalculated from the new position; a detour doesn't
       change the target).
    4. Once the target is reached, repeat from step 1. The algorithm stops
       the moment a step can't be taken at all - every direction is
       blocked/out of bounds, or even the cheapest available direction would
       exceed the remaining budget.

    Rotating around one obstacle cell at a time is a simple heuristic, not a
    full pathfinding search - it can fail to find a way around a large or
    maze-like blocked_mask even when one exists, unlike
    find_path_for_highest_neighbor_value. A generous step cap per target
    (see max_steps_per_target below) guards against looping indefinitely if
    that happens.
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    # Running cost of retracing every edge committed so far, back to base.
    return_trip_cost = 0.0

    max_steps_per_target = 8 * (rows + cols)

    while True:
        # 1. Find the highest-value unvisited, unblocked cell.
        unavailable = visited if blocked_mask is None else visited | blocked_mask
        candidate_values = np.where(unavailable, -np.inf, values)
        if not np.isfinite(candidate_values).any():
            break

        target_row, target_col = (int(index) for index in np.unravel_index(np.argmax(candidate_values), candidate_values.shape))

        stuck = False
        steps_taken = 0

        # 2-3. Head toward (target_row, target_col) one step at a time,
        # rotating around obstacles as they're encountered.
        while row != target_row or col != target_col:
            if steps_taken >= max_steps_per_target:
                stuck = True
                break
            steps_taken += 1

            delta_row = 0 if row == target_row else (1 if target_row > row else -1)
            delta_col = 0 if col == target_col else (1 if target_col > col else -1)
            ideal_direction = Constants.DIRECTIONS.index((delta_row, delta_col))

            direction = None
            for attempt in range(8):
                candidate_direction = (ideal_direction + attempt) % 8
                candidate_delta_row, candidate_delta_col = Constants.DIRECTIONS[candidate_direction]
                candidate_row, candidate_col = row + candidate_delta_row, col + candidate_delta_col

                if candidate_row < 0 or candidate_row >= rows or candidate_col < 0 or candidate_col >= cols:
                    continue
                if blocked_mask is not None and blocked_mask[candidate_row, candidate_col]:
                    continue

                direction = candidate_direction
                next_row, next_col = candidate_row, candidate_col
                break

            if direction is None:
                # Boxed in - every direction is blocked or off-grid.
                stuck = True
                break

            cost = weights[row, col, direction]

            reverse_cost = 0.0
            if require_return_to_base:
                # In an 8-direction compass the opposite of `direction` is
                # always 4 slots away. This is what flying straight back
                # along this same edge would cost.
                reverse_direction = (direction + 4) % 8
                reverse_cost = weights[next_row, next_col, reverse_direction]

            # The budget must cover this step AND retracing every edge
            # flown so far - including this new one - back to base.
            if cost + return_trip_cost + reverse_cost > remaining_budget:
                stuck = True
                break

            path.append((next_row, next_col))
            if not visited[next_row, next_col]:
                visited[next_row, next_col] = True
                total_value += values[next_row, next_col]

            remaining_budget -= cost
            return_trip_cost += reverse_cost
            row, col = next_row, next_col

        # 4. Couldn't finish reaching this target - stop entirely, no
        # fallback to a lesser one.
        if stuck:
            break

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    # Append the actual return leg: retrace the outbound path in reverse,
    # from the current cell back to base. path[-2::-1] is every cell
    # except the last one, reversed - e.g. [start, a, b] -> [a, start].
    path_with_return = path + path[-2::-1]
    total_cost_used = (max_cost - remaining_budget) + return_trip_cost

    return path_with_return, total_value, total_cost_used
