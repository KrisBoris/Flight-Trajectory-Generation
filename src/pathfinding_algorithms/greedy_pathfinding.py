from coordinates_grid.coordinates_grid import CoordinatesGrid
from helpers.constants import Constants
import numpy as np


def find_path_to_highest_value_neighbor(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_terrain: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Greedy search (one-step-lookahead greedy heuristic): starting at 
    (start_row, start_col), repeatedly steps to the highest-value 
    reachable neighbor until no move is left that fits the remaining budget. 
    An already-visited neighbor is never preferred over an unvisited one - 
    revisiting adds no new value, so its value is only counted once, the 
    first time - but it's allowed as a fallback move when every unvisited 
    neighbor is off-grid, blocked or unaffordable.    

    Returns:
        Found path build of sequence of visited cells (x, y): list[tuple[int, int]]
        Total value collected from visited cells: float
        Total cost of traveling through found path: float        
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values
    max_steps = 8 * (rows + cols)

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    # return_cost_grid[r, c] is the cost of the shortest route from (r, c) back to base -
    # calculated once for each node
    return_cost_grid = _build_return_cost_grid(grid, start_row, start_col, blocked_terrain, max_steps) if require_return_to_base else None

    while True:
        best_value = None
        best_move = None  # (next_row, next_col, cost, already_visited)

        for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
            next_row = row + delta_row
            next_col = col + delta_col

            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue

            if blocked_terrain is not None and blocked_terrain[next_row, next_col]:
                continue

            cost = weights[row, col, direction]

            # The budget must cover this move + the shortest route home
            return_reserve = return_cost_grid[next_row, next_col] if return_cost_grid is not None else 0.0
            if cost + return_reserve > remaining_budget:
                continue

            already_visited = visited[next_row, next_col]

            # Revisiting adds no new value - it's only ever 
            # chosen when nothing unvisited is reachable at all.
            new_value = -np.inf if already_visited else values[next_row, next_col]

            # Prefers strictly higher value; among equal values (including
            # ties between revisits), prefers the cheaper move.
            if best_value is None or new_value > best_value or (new_value == best_value and cost < best_move[2]):
                best_value = new_value
                best_move = (next_row, next_col, cost, already_visited)

        if best_move is None:
            break

        next_row, next_col, cost, already_visited = best_move
        if not already_visited:
            visited[next_row, next_col] = True
            total_value += values[next_row, next_col]
        path.append((next_row, next_col))
        remaining_budget -= cost
        row, col = next_row, next_col

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    # Fly the shortest route home
    _, return_path_cells, return_cost, return_value_gained = _walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_terrain, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_highest_value(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_terrain: np.ndarray = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    1. Find a node with the highest-value, unvisited, unblocked, with 
       value above Constants.DEFAULT_PROBABILITY.
    2. Calculate the cost of traveling to it using the shortest path
       possible - straightforward with collision avoidance.
    3. If reachedable, travel to it (every cell passed through is added to
       the path). Repeat from step 1.
    4. The algorithm stops the moment a target can't be fully reached -
       boxed in, or unaffordable - there is no fallback to a lesser target.    
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col
    max_steps = 8 * (rows + cols)

    # return_cost_grid[r, c] is the cost of the shortest route from (r, c) back to base -
    # calculated once for each node
    return_cost_grid = _build_return_cost_grid(grid, start_row, start_col, blocked_terrain, max_steps) if require_return_to_base else None

    while True:
        # 1. Find the highest-value unvisited, unblocked, real cell - see
        # STOPPING ON BACKGROUND CELLS above. Without the DEFAULT_PROBABILITY
        # floor, once every real target is gone this would fall back to
        # np.argmax's tie-break (the first cell in row-major order) among
        # every remaining background cell, all tied at the same value - an
        # arbitrary, cost-blind detour rather than stopping.
        unavailable = visited if blocked_terrain is None else visited | blocked_terrain
        candidate_values = np.where(unavailable | (values <= Constants.DEFAULT_PROBABILITY), -np.inf, values)
        if not np.isfinite(candidate_values).any():
            break

        target_row, target_col = (int(index) for index in np.unravel_index(np.argmax(candidate_values), candidate_values.shape))

        # 2-4. Walk toward it; stop entirely if it can't be reached.
        reached, path_cells, cost, value_gained = _walk_toward_target(
            grid, row, col, target_row, target_col, remaining_budget, return_cost_grid, blocked_terrain, visited, max_steps,
        )

        if not reached:
            break

        # 3. Commit the walk.
        for next_row, next_col in path_cells:
            path.append((next_row, next_col))
            visited[next_row, next_col] = True
        total_value += value_gained
        remaining_budget -= cost
        row, col = target_row, target_col

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    # Fly the actual shortest route home rather than retracing the outbound
    # path - see find_path_for_highest_neighbor_value for why this is
    # guaranteed to fit the remaining budget.
    _, return_path_cells, return_cost, return_value_gained = _walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_terrain, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

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
    better value-collected/cost ratio.

    1. Sort every cell in grid.coordinates_values from highest to lowest value. 
       Cells at or below Constants.DEFAULT_PROBABILITY are never candidates.
    2. Take the first CANDIDATES_PER_ROUND not-yet-visited, unblocked cells
       from that sorted order. For each, calculate the cost of traveling to it
       using the shortest path possible - straightforward with collision avoidance.
       Then compute value_collected / cost for that travel. A candidate that can't be
       reached at all (boxed in or unaffordable) is dropped outright rather
       than kept for a rematch.
    3. Travel to it candidate with the best ratio among the reachable candidates
       (every cell passed through is added to the path). The rest of the reachable-but-not-chosen
       candidates are kept and re-challenged next round.
    4. Repeat from step 2 until none of the round's candidates (held-over or
       fresh) can be reached - the algorithm stops there, with no fallback
       to a lesser target.
    """
    # How many candidates are compared each round
    CANDIDATES_PER_ROUND = 2

    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    max_steps = 8 * (rows + cols)
    return_cost_grid = _build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None

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
            (candidate, _evaluate_candidate(grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps))
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
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = _walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

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
    A tournament between candidate targets, picking whichever gives the lowest cost.

    1. Sort every cell in grid.coordinates_values from highest to lowest
       value (once, up front) - purely to pick a consistent, deterministic
       set of candidates each round; value itself doesn't otherwise factor
       into the choice here. Cells at or below Constants.DEFAULT_PROBABILITY
       (CoordinatesGrid.set_searched_area_values's "no information yet" fill value)
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
    # How many candidates are compared each round
    CANDIDATES_PER_ROUND = 50

    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    max_steps = 8 * (rows + cols)
    return_cost_grid = _build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None

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
            (candidate, _evaluate_candidate(grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps))
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
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = _walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


# Rotation offsets tried, in order, when the ideal direction is blocked -
# ordered by absolute angular distance from that ideal direction (itself,
# then its immediate neighbor on each side, working outward to the
# opposite direction) rather than always spinning the same way. Without
# this, a walker deflected off its ideal diagonal by an obstacle would keep
# rotating in a single direction (see _pick_next_step), which can make it
# hug a wall running parallel to an edge for much longer than necessary
# instead of cutting back toward the target as soon as an opening appears.
_ROTATION_OFFSETS = (0, 1, -1, 2, -2, 3, -3, 4)


def _pick_next_step(row: int, col: int, target_row: int, target_col: int, blocked_mask: np.ndarray, rows: int, cols: int):
    """
    Single-step routing decision shared by _walk_toward_target and
    _build_return_cost_grid: from (row, col), the "ideal" direction closes
    both the row and column distance toward (target_row, target_col) at
    once (diagonal) until one axis is aligned, then closes the remaining
    axis (straight) - the shortest path on this 8-connected grid, ignoring
    obstacles. If that direction's cell is blocked or off-grid, the
    next-closest direction to it is tried instead - see _ROTATION_OFFSETS -
    until an open one is found, so a detour always bends back toward the
    target as gently as possible rather than spinning off in a fixed
    direction.

    Returns (direction, next_row, next_col), or None if every direction out
    of (row, col) is off-grid or blocked.
    """
    delta_row = 0 if row == target_row else (1 if target_row > row else -1)
    delta_col = 0 if col == target_col else (1 if target_col > col else -1)
    ideal_direction = Constants.DIRECTIONS.index((delta_row, delta_col))

    for offset in _ROTATION_OFFSETS:
        candidate_direction = (ideal_direction + offset) % 8
        candidate_delta_row, candidate_delta_col = Constants.DIRECTIONS[candidate_direction]
        candidate_row, candidate_col = row + candidate_delta_row, col + candidate_delta_col

        if candidate_row < 0 or candidate_row >= rows or candidate_col < 0 or candidate_col >= cols:
            continue
        if blocked_mask is not None and blocked_mask[candidate_row, candidate_col]:
            continue

        return candidate_direction, candidate_row, candidate_col

    return None


def _build_return_cost_grid(grid: CoordinatesGrid, start_row: int, start_col: int, blocked_mask: np.ndarray, max_steps: int) -> np.ndarray:
    """
    For every cell, the cost of the shortest route back to (start_row,
    start_col) - the same diagonal-then-straight, rotate-around-obstacles
    routing _walk_toward_target uses (via _pick_next_step) - rather than
    retracing whatever specific route the drone actually took to get there.
    A cell that can't reach the start within max_steps (boxed in by
    blocked_mask) gets np.inf.

    This depends only on grid geometry, weights and blocked_mask - never on
    `visited`, since _pick_next_step's routing decision doesn't look at it
    either - so it's computed once per pathfinding call (see each
    find_path_* function) and reused via O(1) lookups for every candidate
    move afterward, rather than being re-walked from scratch on each one -
    which, multiplied across every step of every candidate of a long path
    (and, for find_path_by_ant_colony, every ant and every iteration), would
    be far too slow to redo live.
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights

    return_cost = np.full((rows, cols), np.inf, dtype=np.float64)
    return_cost[start_row, start_col] = 0.0

    for row in range(rows):
        for col in range(cols):
            if np.isfinite(return_cost[row, col]):
                continue

            current_row, current_col = row, col
            total = 0.0
            steps = 0

            while (current_row, current_col) != (start_row, start_col):
                if steps >= max_steps:
                    total = np.inf
                    break
                steps += 1

                step = _pick_next_step(current_row, current_col, start_row, start_col, blocked_mask, rows, cols)
                if step is None:
                    total = np.inf
                    break

                direction, next_row, next_col = step
                total += weights[current_row, current_col, direction]
                current_row, current_col = next_row, next_col

            return_cost[row, col] = total

    return return_cost


def _walk_toward_target(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
    remaining_budget: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_steps: int,
):
    """
    Speculatively walks from (start_row, start_col) toward (target_row,
    target_col) one step at a time - see _pick_next_step for the routing
    rule. At every step, the budget must cover the walk's own cost so far
    AND - if return_cost_grid is given - the cost of the shortest route home
    from wherever that step would land (see _build_return_cost_grid), so a
    long speculative walk toward a target can't itself strand the drone.
    Pass return_cost_grid=None when no further reserve is needed - e.g. when
    this walk IS the trip home, or require_return_to_base is False.

    This does NOT mutate `visited` or `remaining_budget` - it only reads
    them, so a caller can evaluate several candidate targets from the same
    position before deciding which one (if any) to actually commit.
    `visited` is used only to avoid double-counting a cell's value if the
    walk passes through somewhere already collected on an earlier,
    already-committed leg.

    Returns (reached, path_cells, cost, value_gained):
      - reached: False if the walk got stuck (boxed in on all 8 sides, hit
        max_steps, or the next step would exceed remaining_budget) before
        arriving at the target - the other values are meaningless if so.
      - path_cells: the (row, col) cells stepped onto, in order, NOT
        including the starting cell.
      - cost: total cost of the walk.
      - value_gained: sum of grid.coordinates_values over path_cells, minus
        any cells already in `visited` (or repeated within this same walk).
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

    path_cells = []
    cost = 0.0
    value_gained = 0.0
    seen_this_walk = set()

    row, col = start_row, start_col
    steps = 0

    while row != target_row or col != target_col:
        if steps >= max_steps:
            return False, [], 0.0, 0.0
        steps += 1

        step = _pick_next_step(row, col, target_row, target_col, blocked_mask, rows, cols)
        if step is None:
            return False, [], 0.0, 0.0
        direction, next_row, next_col = step

        step_cost = weights[row, col, direction]
        return_reserve = return_cost_grid[next_row, next_col] if return_cost_grid is not None else 0.0

        if cost + step_cost + return_reserve > remaining_budget:
            return False, [], 0.0, 0.0

        cost += step_cost
        path_cells.append((next_row, next_col))

        if not visited[next_row, next_col] and (next_row, next_col) not in seen_this_walk:
            value_gained += values[next_row, next_col]
            seen_this_walk.add((next_row, next_col))

        row, col = next_row, next_col

    return True, path_cells, cost, value_gained


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
    CoordinatesGrid.set_searched_area_values gives every cell. Since these two
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
    return_cost_grid: np.ndarray,
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
    reached, path_cells, cost, value_gained = _walk_toward_target(
        grid, row, col, target_row, target_col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps,
    )
    if not reached or cost <= 0:
        return None
    return {
        "target": candidate,
        "path_cells": path_cells,
        "cost": cost,
        "value_gained": value_gained,
        "ratio": value_gained / cost,
    }
