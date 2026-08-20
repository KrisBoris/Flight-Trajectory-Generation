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
    2. Walk toward it (see _walk_toward_target: shortest path, rotating
       around obstacles one cell at a time), cost-checked the same way as
       find_path_for_highest_neighbor_value, including the
       require_return_to_base reserve.
    3. If reached, commit the walk (every cell passed through is added to
       the path). Repeat from step 1.
    4. The algorithm stops the moment a target can't be fully reached -
       boxed in, or unaffordable - there is no fallback to a lesser target.

    Rotating around one obstacle cell at a time is a simple heuristic, not a
    full pathfinding search - it can fail to find a way around a large or
    maze-like blocked_mask even when one exists, unlike
    find_path_for_highest_neighbor_value.
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col
    return_trip_cost = 0.0

    max_steps = 8 * (rows + cols)

    while True:
        # 1. Find the highest-value unvisited, unblocked cell.
        unavailable = visited if blocked_mask is None else visited | blocked_mask
        candidate_values = np.where(unavailable, -np.inf, values)
        if not np.isfinite(candidate_values).any():
            break

        target_row, target_col = (int(index) for index in np.unravel_index(np.argmax(candidate_values), candidate_values.shape))

        # 2-4. Walk toward it; stop entirely if it can't be reached.
        reached, path_cells, cost, return_cost, value_gained = _walk_toward_target(
            grid, row, col, target_row, target_col, remaining_budget, return_trip_cost,
            require_return_to_base, blocked_mask, visited, max_steps,
        )

        if not reached:
            break

        # 3. Commit the walk.
        for next_row, next_col in path_cells:
            path.append((next_row, next_col))
            visited[next_row, next_col] = True
        total_value += value_gained
        remaining_budget -= cost
        return_trip_cost += return_cost
        row, col = target_row, target_col

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    # Append the actual return leg: retrace the outbound path in reverse,
    # from the current cell back to base. path[-2::-1] is every cell
    # except the last one, reversed - e.g. [start, a, b] -> [a, start].
    path_with_return = path + path[-2::-1]
    total_cost_used = (max_cost - remaining_budget) + return_trip_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_value_cost_ratio(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    A tournament between candidate targets, picking whichever gives the
    better value-collected/cost ratio rather than always the raw highest
    value:

    1. Sort every cell in grid.coordinates_values from highest to lowest
       value (once, up front) - this is the same candidate ordering
       find_path_to_highest_value uses one at a time. Cells at or below
       Constants.DEFAULT_PROBABILITY (CoordinatesGrid.init_empty_grid's
       "no information yet" fill value) are never candidates - see
       _next_fresh_candidate - since this function is specifically about
       ranking targets by value, and an untouched background cell isn't a
       real one.
    2. Take the first CANDIDATES_PER_ROUND not-yet-visited, unblocked cells
       from that sorted order. For each, walk to it (see
       _walk_toward_target: shortest path, rotating around obstacles - the
       same approach find_path_to_highest_value uses) and compute
       value_collected / cost for that walk. A candidate that can't be
       reached at all (boxed in or unaffordable) is dropped outright rather
       than kept for a rematch, since positions only move forward and
       budget only shrinks.
    3. Commit the walk with the best ratio among the reachable candidates.
       The rest of the reachable-but-not-chosen candidates are kept and
       re-challenged next round, topped back up to CANDIDATES_PER_ROUND with
       fresh candidates from the sorted order - e.g. with
       CANDIDATES_PER_ROUND = 2, if the 2nd-highest value won round 1, round
       2 compares the 1st-highest (the round-1 loser) against the 3rd-
       highest.
    4. Repeat from step 2 until none of the round's candidates (held-over or
       fresh) can be reached - the algorithm stops there, with no fallback
       to a lesser target.
    """
    # How many candidates are compared each round. 2 reproduces the simplest
    # "compare this one against the next one" tournament; a higher number
    # widens each round's search at the cost of more speculative walks per
    # round (see _evaluate_candidate).
    CANDIDATES_PER_ROUND = 2

    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col
    return_trip_cost = 0.0

    max_steps = 8 * (rows + cols)

    sorted_target_indices = np.argsort(-values, axis=None)
    candidate_pointer = 0
    pending = []  # candidates held over from the previous round

    while True:
        # A candidate held over from last round may have been swallowed by
        # the winning path that was just committed (visited as a
        # pass-through cell) - drop it rather than re-challenge with it.
        pending = [candidate for candidate in pending if not visited[candidate[0], candidate[1]]]

        candidates = list(pending)
        while len(candidates) < CANDIDATES_PER_ROUND:
            fresh_candidate, candidate_pointer = _next_fresh_candidate(sorted_target_indices, candidate_pointer, values, visited, blocked_mask)
            if fresh_candidate is None:
                break
            candidates.append(fresh_candidate)

        if not candidates:
            break  # every cell has been visited or attempted

        evaluated = [
            (candidate, _evaluate_candidate(grid, candidate, row, col, remaining_budget, return_trip_cost, require_return_to_base, blocked_mask, visited, max_steps))
            for candidate in candidates
        ]
        reachable = [(candidate, result) for candidate, result in evaluated if result is not None]

        if not reachable:
            break  # none of this round's candidates are reachable within the budget - stop

        winning_candidate, winner = max(reachable, key=lambda entry: entry[1]["ratio"])
        pending = [candidate for candidate, _ in reachable if candidate != winning_candidate]

        for next_row, next_col in winner["path_cells"]:
            path.append((next_row, next_col))
            visited[next_row, next_col] = True
        total_value += winner["value_gained"]
        remaining_budget -= winner["cost"]
        return_trip_cost += winner["return_cost"]
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    path_with_return = path + path[-2::-1]
    total_cost_used = (max_cost - remaining_budget) + return_trip_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_lowest_cost(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    The same tournament structure as find_path_by_value_cost_ratio, but
    picking whichever candidate is cheapest to reach rather than whichever
    has the best value/cost ratio:

    1. Sort every cell in grid.coordinates_values from highest to lowest
       value (once, up front) - purely to pick a consistent, deterministic
       set of candidates each round; value itself doesn't otherwise factor
       into the choice here. Cells at or below Constants.DEFAULT_PROBABILITY
       (CoordinatesGrid.init_empty_grid's "no information yet" fill value)
       are never candidates - see _next_fresh_candidate. Without this, a
       grid mostly filled with tied background cells would have this
       function spend nearly every round comparing meaningless candidates,
       and since a straight-line hop is measurably cheaper than a diagonal
       one of the same length (see WeightsGrid.init_from_elevation), it
       would systematically prefer whichever tied background cell happened
       to be reachable in a straight line - producing a path that hugs rows
       and columns instead of heading toward real targets.
    2. Take the first CANDIDATES_PER_ROUND not-yet-visited, unblocked cells
       from that sorted order. For each, walk to it (see
       _walk_toward_target: shortest path, rotating around obstacles) and
       note its cost. A candidate that can't be reached at all (boxed in or
       too expensive for the remaining budget) is dropped outright rather
       than kept for a rematch, since positions only move forward and
       budget only shrinks.
    3. Commit the walk with the lowest cost among the reachable candidates.
       The rest of the reachable-but-not-chosen candidates are kept and
       re-challenged next round, topped back up to CANDIDATES_PER_ROUND with
       fresh candidates from the sorted order.
    4. Repeat from step 2 until none of the round's candidates (held-over or
       fresh) can be reached within the remaining budget - the algorithm
       stops there, with no fallback to a lesser target.
    """
    # How many candidates are compared each round - see
    # find_path_by_value_cost_ratio's CANDIDATES_PER_ROUND for the tradeoff.
    CANDIDATES_PER_ROUND = 10

    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col
    return_trip_cost = 0.0

    max_steps = 8 * (rows + cols)

    sorted_target_indices = np.argsort(-values, axis=None)
    candidate_pointer = 0
    pending = []  # candidates held over from the previous round

    while True:
        # A candidate held over from last round may have been swallowed by
        # the winning path that was just committed (visited as a
        # pass-through cell) - drop it rather than re-challenge with it.
        pending = [candidate for candidate in pending if not visited[candidate[0], candidate[1]]]

        candidates = list(pending)
        while len(candidates) < CANDIDATES_PER_ROUND:
            fresh_candidate, candidate_pointer = _next_fresh_candidate(sorted_target_indices, candidate_pointer, values, visited, blocked_mask)
            if fresh_candidate is None:
                break
            candidates.append(fresh_candidate)

        if not candidates:
            break  # every cell has been visited or attempted

        evaluated = [
            (candidate, _evaluate_candidate(grid, candidate, row, col, remaining_budget, return_trip_cost, require_return_to_base, blocked_mask, visited, max_steps))
            for candidate in candidates
        ]
        reachable = [(candidate, result) for candidate, result in evaluated if result is not None]

        if not reachable:
            break  # every candidate this round costs more than the remaining budget (or is unreachable) - stop

        winning_candidate, winner = min(reachable, key=lambda entry: entry[1]["cost"])
        pending = [candidate for candidate, _ in reachable if candidate != winning_candidate]

        for next_row, next_col in winner["path_cells"]:
            path.append((next_row, next_col))
            visited[next_row, next_col] = True
        total_value += winner["value_gained"]
        remaining_budget -= winner["cost"]
        return_trip_cost += winner["return_cost"]
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    path_with_return = path + path[-2::-1]
    total_cost_used = (max_cost - remaining_budget) + return_trip_cost

    return path_with_return, total_value, total_cost_used


# Rotation offsets tried, in order, when the ideal direction is blocked -
# ordered by absolute angular distance from that ideal direction (itself,
# then its immediate neighbor on each side, working outward to the
# opposite direction) rather than always spinning the same way. Without
# this, a walker deflected off its ideal diagonal by an obstacle would keep
# rotating in a single direction (see _walk_toward_target), which can make
# it hug a wall running parallel to an edge for much longer than necessary
# instead of cutting back toward the target as soon as an opening appears.
_ROTATION_OFFSETS = (0, 1, -1, 2, -2, 3, -3, 4)


def _walk_toward_target(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
    remaining_budget: float,
    return_trip_cost: float,
    require_return_to_base: bool,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_steps: int,
):
    """
    Speculatively walks from (start_row, start_col) toward (target_row,
    target_col) one step at a time: the "ideal" direction closes both the
    row and column distance at once (diagonal) until one axis is aligned,
    then closes the remaining axis (straight) - the shortest path on this
    8-connected grid. If that direction's cell is blocked or off-grid, the
    next-closest direction to it is tried instead - see _ROTATION_OFFSETS -
    until an open one is found, so a detour always bends back toward the
    target as gently as possible rather than spinning off in a fixed
    direction.

    This does NOT mutate `visited`, `remaining_budget` or `return_trip_cost`
    - it only reads them, so a caller can evaluate several candidate targets
    from the same position before deciding which one (if any) to actually
    commit. `visited` is used only to avoid double-counting a cell's value
    if the walk passes through somewhere already collected on an earlier,
    already-committed leg.

    Returns (reached, path_cells, cost, return_cost, value_gained):
      - reached: False if the walk got stuck (boxed in on all 8 sides, hit
        max_steps, or the next step would exceed remaining_budget) before
        arriving at the target - the other values are meaningless if so.
      - path_cells: the (row, col) cells stepped onto, in order, NOT
        including the starting cell.
      - cost: total forward cost of the walk.
      - return_cost: total cost of retracing this walk's own edges back
        (0.0 if require_return_to_base is False) - added to, not replacing,
        return_trip_cost from any previously committed legs.
      - value_gained: sum of grid.coordinates_values over path_cells, minus
        any cells already in `visited` (or repeated within this same walk).
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

    path_cells = []
    cost = 0.0
    return_cost = 0.0
    value_gained = 0.0
    seen_this_walk = set()

    row, col = start_row, start_col
    steps = 0

    while row != target_row or col != target_col:
        if steps >= max_steps:
            return False, [], 0.0, 0.0, 0.0
        steps += 1

        delta_row = 0 if row == target_row else (1 if target_row > row else -1)
        delta_col = 0 if col == target_col else (1 if target_col > col else -1)
        ideal_direction = Constants.DIRECTIONS.index((delta_row, delta_col))

        direction = None
        next_row = next_col = None
        for offset in _ROTATION_OFFSETS:
            candidate_direction = (ideal_direction + offset) % 8
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
            return False, [], 0.0, 0.0, 0.0

        step_cost = weights[row, col, direction]

        step_reverse_cost = 0.0
        if require_return_to_base:
            reverse_direction = (direction + 4) % 8
            step_reverse_cost = weights[next_row, next_col, reverse_direction]

        # The budget must cover every edge walked so far this leg AND the
        # committed return-trip reserve AND retracing this new edge too.
        if cost + step_cost + return_trip_cost + return_cost + step_reverse_cost > remaining_budget:
            return False, [], 0.0, 0.0, 0.0

        cost += step_cost
        return_cost += step_reverse_cost
        path_cells.append((next_row, next_col))

        if not visited[next_row, next_col] and (next_row, next_col) not in seen_this_walk:
            value_gained += values[next_row, next_col]
            seen_this_walk.add((next_row, next_col))

        row, col = next_row, next_col

    return True, path_cells, cost, return_cost, value_gained


def _next_fresh_candidate(
    sorted_target_indices: np.ndarray,
    candidate_pointer: int,
    values: np.ndarray,
    visited: np.ndarray,
    blocked_mask: np.ndarray,
):
    """
    Scans sorted_target_indices (flat indices into `values`, highest value
    first) starting at candidate_pointer for the next cell that's neither
    already visited nor blocked - and whose value is above
    Constants.DEFAULT_PROBABILITY, the "no information yet" fill value
    CoordinatesGrid.init_empty_grid gives every cell. Since these two
    functions specifically reason about value (a ratio to cost, or picking
    the cheapest among value-ranked candidates), an untouched background
    cell isn't a meaningful target - including it just means comparing
    against noise, which is what was producing the row/column-hugging
    pattern seen when tied background cells vastly outnumbered real search
    areas. As a plain function rather than a closure, it can't remember
    candidate_pointer itself between calls - the caller must keep track of
    the returned pointer and pass it back in next time.

    Returns ((row, col), next_pointer), or (None, next_pointer) if no
    candidate remains.
    """
    while candidate_pointer < len(sorted_target_indices):
        flat_index = sorted_target_indices[candidate_pointer]
        candidate_pointer += 1
        candidate_row, candidate_col = (int(index) for index in np.unravel_index(flat_index, values.shape))
        if values[candidate_row, candidate_col] <= Constants.DEFAULT_PROBABILITY:
            # Every cell after this one in the sorted order is <= this one's
            # value too, so none of the rest can be real targets either.
            break
        if visited[candidate_row, candidate_col]:
            continue
        if blocked_mask is not None and blocked_mask[candidate_row, candidate_col]:
            continue
        return (candidate_row, candidate_col), candidate_pointer
    return None, candidate_pointer


def _evaluate_candidate(
    grid: CoordinatesGrid,
    candidate: tuple[int, int],
    row: int,
    col: int,
    remaining_budget: float,
    return_trip_cost: float,
    require_return_to_base: bool,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_steps: int,
):
    """
    Speculatively walks from (row, col) toward candidate (see
    _walk_toward_target) and summarizes the outcome as a dict, or returns
    None if the target can't be reached within remaining_budget.
    """
    target_row, target_col = candidate
    reached, path_cells, cost, return_cost, value_gained = _walk_toward_target(
        grid, row, col, target_row, target_col, remaining_budget, return_trip_cost,
        require_return_to_base, blocked_mask, visited, max_steps,
    )
    if not reached or cost <= 0:
        return None
    return {
        "target": candidate,
        "path_cells": path_cells,
        "cost": cost,
        "return_cost": return_cost,
        "value_gained": value_gained,
        "ratio": value_gained / cost,
    }
