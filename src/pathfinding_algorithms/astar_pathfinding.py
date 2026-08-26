# astar_pathfinding.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from helpers.constants import Constants
from pathfinding_algorithms import greedy_pathfinding
import heapq
import numpy as np


def _min_real_edge_cost(weights: np.ndarray) -> float:
    """
    The cheapest cost of any edge that could actually be taken - used as the
    per-step lower bound in _a_star_path's and _namoa_star_pareto_frontier's
    admissible heuristics.

    Deliberately NOT weights.min(): WeightsGrid.init_from_elevation leaves
    every border cell's off-grid-pointing directions at their unused
    default fill value (see that function's docstring) - those are never
    real moves (there is no cell there to step onto), but a plain .min()
    would still pick one up if it happened to be the smallest number in the
    array. Using that phantom value as the heuristic's per-step cost would
    make it needlessly - and harmfully - pessimistic: an admissible
    heuristic only needs to never OVERestimate, but the weaker (lower) it
    is, the less it actually guides the search, and the gap between a
    genuine minimum and an unused fill value can easily be several times
    over on real terrain, turning a fast, focused search into a slow,
    near-undirected one.
    """
    rows, cols, _ = weights.shape
    min_cost = np.inf

    for direction_index, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
        row_start, row_stop = max(0, -delta_row), rows - max(0, delta_row)
        col_start, col_stop = max(0, -delta_col), cols - max(0, delta_col)
        if row_start >= row_stop or col_start >= col_stop:
            continue

        direction_min = weights[row_start:row_stop, col_start:col_stop, direction_index].min()
        min_cost = min(min_cost, float(direction_min))

    return min_cost


def find_path_by_a_star_to_highest_value(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    max_expanded_cells: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    A* to the highest-value cell, repeated - this otherwise exactly mirrors
    greedy_pathfinding.find_path_to_highest_value:

    1. Find the highest-value unvisited, unblocked, real cell (value above
       Constants.DEFAULT_PROBABILITY - see greedy_pathfinding.find_path_to_
       highest_value's STOPPING ON BACKGROUND CELLS, which applies here
       identically) in grid.coordinates_values.
    2. Search for the CHEAPEST possible route to it using true A* search
       (see _a_star_path) - rather than find_path_to_highest_value's
       cheaper, but not necessarily optimal, "diagonal, then straight,
       rotate around one obstacle at a time" heuristic
       (greedy_pathfinding._walk_toward_target).
    3. If that cheapest route - plus getting home afterward - still fits the
       budget, commit it (every cell passed through is added to the path).
       Repeat from step 1.
    4. The algorithm stops the moment a target can't be fully reached -
       sealed off, or unaffordable even by the cheapest possible route -
       there is no fallback to a lesser target.

    Because A* is a full search rather than a fixed rule, it can navigate a
    large or maze-like blocked_mask that find_path_to_highest_value's
    simpler heuristic can fail to route around, and it is guaranteed to find
    the actual cheapest path whenever more than one exists - see
    _a_star_path for what that correctness guarantee costs in return.
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
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None
    expand_cap = max_expanded_cells if max_expanded_cells is not None else rows * cols

    while True:
        # 1. Find the highest-value unvisited, unblocked, real cell - see
        # greedy_pathfinding.find_path_to_highest_value's STOPPING ON
        # BACKGROUND CELLS. Without the DEFAULT_PROBABILITY floor, once
        # every real target is gone this would fall back to np.argmax's
        # tie-break (the first cell in row-major order) among every
        # remaining background cell, all tied at the same value - an
        # arbitrary, cost-blind detour rather than stopping.
        unavailable = visited if blocked_mask is None else visited | blocked_mask
        candidate_values = np.where(unavailable | (values <= Constants.DEFAULT_PROBABILITY), -np.inf, values)
        if not np.isfinite(candidate_values).any():
            break

        target_row, target_col = (int(index) for index in np.unravel_index(np.argmax(candidate_values), candidate_values.shape))

        # 2. Search for the cheapest possible route.
        reached, path_cells, cost = _a_star_path(grid, row, col, target_row, target_col, blocked_mask, expand_cap)
        if not reached:
            break

        # 3. Only commit if it (plus the trip home) still fits the budget.
        return_reserve = return_cost_grid[target_row, target_col] if return_cost_grid is not None else 0.0
        if cost + return_reserve > remaining_budget:
            break

        value_gained = _value_gained_along_path(grid, path_cells, visited)
        for cell in path_cells:
            path.append(cell)
            visited[cell] = True
        total_value += value_gained
        remaining_budget -= cost
        row, col = target_row, target_col

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_a_star_value_cost_ratio(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    candidates_per_round: int = 2,
    max_expanded_cells: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    The same tournament greedy_pathfinding.find_path_by_value_cost_ratio
    runs - repeatedly comparing candidates_per_round candidates by
    value-collected/cost and committing whichever wins, holding reachable
    losers over for a rematch next round - but every candidate's route (and
    its cost) is found via true A* search (_a_star_path) instead of the
    cheaper "diagonal, then straight, rotate around one obstacle" heuristic
    (greedy_pathfinding._walk_toward_target) find_path_by_value_cost_ratio
    uses. See find_path_by_a_star_to_highest_value and _a_star_path for what
    that trades off, and find_path_by_value_cost_ratio's own docstring for
    the full tournament mechanics (candidate sourcing, held-over candidates,
    stopping condition) - identical here except for the walking method.

    candidates_per_round is "only a given amount of cells" considered each
    round - the same knob as find_path_by_value_cost_ratio's
    CANDIDATES_PER_ROUND, exposed here as a parameter instead of a fixed
    module constant.
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
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None
    expand_cap = max_expanded_cells if max_expanded_cells is not None else rows * cols

    sorted_target_indices = np.argsort(-values, axis=None)
    candidate_pointer = 0
    pending = []  # candidates held over from the previous round

    while True:
        pending = [candidate for candidate in pending if not visited[candidate[0], candidate[1]]]

        candidates = list(pending)
        while len(candidates) < candidates_per_round:
            fresh_candidate, candidate_pointer = greedy_pathfinding._next_fresh_candidate(sorted_target_indices, candidate_pointer, values, visited, blocked_mask)
            if fresh_candidate is None:
                break
            candidates.append(fresh_candidate)

        if not candidates:
            break  # every cell has been visited or attempted

        evaluated = [
            (candidate, _evaluate_candidate_a_star(grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, expand_cap))
            for candidate in candidates
        ]
        reachable = [(candidate, result) for candidate, result in evaluated if result is not None]

        if not reachable:
            break  # none of this round's candidates are reachable within the budget - stop

        winning_candidate, winner = max(reachable, key=lambda entry: entry[1]["ratio"])
        pending = [candidate for candidate, _ in reachable if candidate != winning_candidate]

        for cell in winner["path_cells"]:
            path.append(cell)
            visited[cell] = True
        total_value += winner["value_gained"]
        remaining_budget -= winner["cost"]
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_a_star_lowest_cost(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    candidates_per_round: int = 50,
    max_expanded_cells: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    The same tournament greedy_pathfinding.find_path_by_lowest_cost runs -
    repeatedly comparing candidates_per_round candidates by raw cost and
    committing whichever is cheapest, holding reachable losers over for a
    rematch next round - but every candidate's route (and its cost) is
    found via true A* search (_a_star_path) instead of the cheaper
    "diagonal, then straight, rotate around one obstacle" heuristic
    (greedy_pathfinding._walk_toward_target) find_path_by_lowest_cost uses.
    See find_path_by_a_star_to_highest_value and _a_star_path for what that
    trades off, and find_path_by_lowest_cost's own docstring for the full
    tournament mechanics - identical here except for the walking method.

    candidates_per_round is "only a given amount of cells" considered each
    round - the same knob as find_path_by_lowest_cost's
    CANDIDATES_PER_ROUND, exposed here as a parameter instead of a fixed
    module constant, and defaulted higher (50) than the ratio version's (2)
    for the same reason find_path_by_lowest_cost's own CANDIDATES_PER_ROUND
    is higher - see that function's docstring (tied or near-tied background
    cells need a wider comparison to avoid a row/column-hugging pattern).
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
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None
    expand_cap = max_expanded_cells if max_expanded_cells is not None else rows * cols

    sorted_target_indices = np.argsort(-values, axis=None)
    candidate_pointer = 0
    pending = []  # candidates held over from the previous round

    while True:
        pending = [candidate for candidate in pending if not visited[candidate[0], candidate[1]]]

        candidates = list(pending)
        while len(candidates) < candidates_per_round:
            fresh_candidate, candidate_pointer = greedy_pathfinding._next_fresh_candidate(sorted_target_indices, candidate_pointer, values, visited, blocked_mask)
            if fresh_candidate is None:
                break
            candidates.append(fresh_candidate)

        if not candidates:
            break  # every cell has been visited or attempted

        evaluated = [
            (candidate, _evaluate_candidate_a_star(grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, expand_cap))
            for candidate in candidates
        ]
        reachable = [(candidate, result) for candidate, result in evaluated if result is not None]

        if not reachable:
            break  # every candidate this round costs more than the remaining budget (or is unreachable) - stop

        winning_candidate, winner = min(reachable, key=lambda entry: entry[1]["cost"])
        pending = [candidate for candidate, _ in reachable if candidate != winning_candidate]

        for cell in winner["path_cells"]:
            path.append(cell)
            visited[cell] = True
        total_value += winner["value_gained"]
        remaining_budget -= winner["cost"]
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


def find_path_by_namoa_star(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    candidates_per_round: int = 2,
    max_expanded_labels: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    NAMOA* (New Approach to Multi-Objective A*).

    GENERAL IDEA
    ------------
    Imagine walking to school, caring about TWO things at once: how tired
    you'll be (distance/effort) and how many stickers you'll collect from
    machines scattered around town along the way. A normal shortest-path
    finder only thinks about tiredness - it will always take the least
    tiring route, even if a slightly-more-tiring route would let you scoop
    up five stickers for almost no extra effort, because it never considers
    stickers a factor at all.

    NAMOA* is smarter about this: instead of finding just ONE route, it
    finds a whole MENU of genuinely good routes - for every worthwhile
    "how many stickers" target, the LEAST tiring route that gets you that
    many. The menu might read: "0 stickers costs 10 energy", "3 stickers
    costs 12 energy", "7 stickers costs 20 energy". No single entry is
    objectively "the best" - that depends on how much you personally value
    stickers versus energy - but the menu is guaranteed complete and
    honest: an option only appears on it if no OTHER route beats it in both
    energy AND stickers at once (that comparison is called "domination" -
    route A dominates route B if A costs no more AND collects no fewer
    stickers, and strictly wins on at least one of those). Anything
    dominated gets thrown off the menu, since it would never be a sensible
    choice - something else is always at least as good, in every way.

    Once the menu exists, whoever is choosing (here, this function itself,
    scoring by stickers-per-energy) picks whichever entry actually suits
    the goal - instead of being stuck with whatever a single-minded
    "least tiring" or "most stickers" search would have handed over on its
    own.

    Mechanically, NAMOA* is A* (see astar_pathfinding._a_star_path) but
    keeping track of TWO running scores per step instead of one - cost so
    far, AND value collected so far - and instead of keeping only the single
    best-known way to reach a cell (as plain A*/Dijkstra do), it keeps every
    non-dominated (cost, value) combination reached there, since a costlier
    route that picked up more value along the way is a genuine trade-off,
    not simply worse. It keeps exploring, always working on whichever
    partial route looks most promising by a cost estimate (see
    _namoa_star_pareto_frontier), and does not stop the moment it first
    reaches the destination - it keeps going until every non-dominated way
    of getting there has been found, then hands back the whole menu (the
    "Pareto frontier").

    IN THIS CASE
    ------------
    Same tournament structure as astar_pathfinding.find_path_by_a_star_
    value_cost_ratio: repeatedly compare candidates_per_round real targets
    and commit whichever wins, holding reachable losers over for a rematch
    (see greedy_pathfinding.find_path_by_value_cost_ratio for the full
    tournament mechanics - identical here). The difference is what "wins" a
    candidate comparison: astar_pathfinding._evaluate_candidate_a_star finds
    only the single CHEAPEST route to a candidate; _evaluate_candidate_
    namoa_star (see below) instead asks _namoa_star_pareto_frontier for
    EVERY non-dominated cost/value trade-off reaching that candidate, then
    picks whichever point on that menu has the best value/cost ratio. A
    plain A* search can never even consider taking a costlier detour that
    happens to sweep up extra incidental value along the way - it only ever
    optimizes cost - so NAMOA* can find a genuinely better answer for this
    project's actual goal (collecting value under a budget) than pure
    cheapest-route A* can, at the cost of a considerably more expensive
    search (see _namoa_star_pareto_frontier's PERFORMANCE NOTE).

    candidates_per_round is "only a given amount of cells" considered each
    round, the same knob find_path_by_a_star_value_cost_ratio exposes.

    A CANDIDATE THAT FAILS IS NEVER RETRIED: like every tournament in this
    project, a candidate that comes back unreachable this round is dropped
    for good rather than held over - correct reasoning for the plain
    heuristic walk, where "unreachable" is a deterministic fact about
    distance, budget and obstacles. For NAMOA*, though, an unreachable
    verdict can also mean _namoa_star_pareto_frontier gave up because it hit
    max_expanded_labels, not because no affordable route genuinely exists -
    a real trade-off, not a bug: a target far enough away, or beyond
    whatever label budget is set, can be dropped from consideration for the
    rest of this run even though a larger max_expanded_labels might have
    found it. Raising max_expanded_labels (or candidates_per_round, to
    compare fewer distant long-shots per round) trades runtime for fewer
    of these false negatives.
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
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None
    # Multi-label search needs more headroom than plain A*'s
    # max_expanded_cells (see _namoa_star_pareto_frontier's PERFORMANCE
    # NOTE) - rows * cols alone is not reliably enough to avoid the search
    # falsely giving up on a genuinely reachable, distant candidate. This is
    # a middle ground, not a value that eliminates false negatives on a
    # large grid (see A CANDIDATE THAT FAILS IS NEVER RETRIED above) -
    # raise it for more thoroughness at the cost of runtime.
    label_cap = max_expanded_labels if max_expanded_labels is not None else 2 * rows * cols

    sorted_target_indices = np.argsort(-values, axis=None)
    candidate_pointer = 0
    pending = []  # candidates held over from the previous round

    while True:
        pending = [candidate for candidate in pending if not visited[candidate[0], candidate[1]]]

        candidates = list(pending)
        while len(candidates) < candidates_per_round:
            fresh_candidate, candidate_pointer = greedy_pathfinding._next_fresh_candidate(sorted_target_indices, candidate_pointer, values, visited, blocked_mask)
            if fresh_candidate is None:
                break
            candidates.append(fresh_candidate)

        if not candidates:
            break  # every cell has been visited or attempted

        evaluated = [
            (candidate, _evaluate_candidate_namoa_star(grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, label_cap))
            for candidate in candidates
        ]
        reachable = [(candidate, result) for candidate, result in evaluated if result is not None]

        if not reachable:
            break  # none of this round's candidates have an affordable, reachable trade-off - stop

        winning_candidate, winner = max(reachable, key=lambda entry: entry[1]["ratio"])
        pending = [candidate for candidate, _ in reachable if candidate != winning_candidate]

        for cell in winner["path_cells"]:
            path.append(cell)
            visited[cell] = True
        total_value += winner["value_gained"]
        remaining_budget -= winner["cost"]
        row, col = winner["target"]

    if not require_return_to_base:
        return path, total_value, max_cost - remaining_budget

    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used


def _a_star_path(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
    blocked_mask: np.ndarray,
    max_expanded_cells: int,
):
    """
    True A* search from (start_row, start_col) to (target_row, target_col)
    on the 8-connected grid, honoring the real per-direction terrain costs
    in grid.weights_grid.weights - unlike greedy_pathfinding._walk_toward_
    target's cheap "diagonal, then straight, rotate around one obstacle at a
    time" heuristic, this explores however many alternative routes are
    needed and is GUARANTEED to return the cheapest possible path (subject
    to max_expanded_cells, a safety cap - see below), including around
    obstacle layouts the simpler heuristic can get stuck on.

    Tracks g(cell) - the cheapest cost found so far from the start to cell -
    and explores cells in order of f(cell) = g(cell) + h(cell), where h is
    a heuristic ESTIMATE of the remaining cost to the target:
    h(cell) = chebyshev_distance(cell, target) * (the cheapest cost of any
    real, in-bounds move anywhere on the grid - see _min_real_edge_cost).
    This is a guaranteed lower bound on the true remaining cost - no real
    route can need fewer than chebyshev_distance edges (the fewest
    8-directional steps possible, ignoring obstacles), and no real edge
    anywhere can cost less than the grid's own cheapest real edge - which
    is exactly what makes the result provably
    optimal (an "admissible" heuristic) rather than merely a good guess.

    Returns (reached, path_cells, cost): path_cells is the (row, col) cells
    stepped onto in order, NOT including the start cell (matching
    _walk_toward_target's convention) - reached is False (other values
    meaningless) if the target is unreachable at all (sealed off by
    blocked_mask) or the search exhausts max_expanded_cells first, which can
    only ever under-report reachability, never claim a cheaper route exists
    than really does.

    Unlike _walk_toward_target, this does not consider a travel budget at
    all - it always finds the cheapest possible route regardless of cost,
    leaving budget/return-trip feasibility to the caller (see
    _evaluate_candidate_a_star): since A* already finds the true minimum
    cost, no amount of budget-aware pruning mid-search could find anything
    cheaper - if the minimum doesn't fit, nothing would have.
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights

    if (start_row, start_col) == (target_row, target_col):
        return True, [], 0.0

    min_edge_cost = _min_real_edge_cost(weights)

    def heuristic(row, col):
        """Admissible lower-bound estimate of the remaining cost from (row, col) to the target - see the docstring above."""
        return max(abs(row - target_row), abs(col - target_col)) * min_edge_cost

    g_score = {(start_row, start_col): 0.0}
    came_from = {}
    open_heap = [(heuristic(start_row, start_col), 0, start_row, start_col, 0.0)]
    push_count = 0
    expanded = 0

    while open_heap:
        _, _, row, col, g = heapq.heappop(open_heap)

        if g > g_score.get((row, col), np.inf):
            continue  # a cheaper route to this cell was already found - stale entry

        if (row, col) == (target_row, target_col):
            path_cells = []
            cell = (row, col)
            while cell != (start_row, start_col):
                path_cells.append(cell)
                cell = came_from[cell]
            path_cells.reverse()
            return True, path_cells, g

        expanded += 1
        if expanded > max_expanded_cells:
            return False, [], 0.0

        for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
            next_row, next_col = row + delta_row, col + delta_col

            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue
            if blocked_mask is not None and blocked_mask[next_row, next_col]:
                continue

            tentative_g = g + weights[row, col, direction]
            if tentative_g < g_score.get((next_row, next_col), np.inf):
                g_score[(next_row, next_col)] = tentative_g
                came_from[(next_row, next_col)] = (row, col)
                push_count += 1
                f_score = tentative_g + heuristic(next_row, next_col)
                heapq.heappush(open_heap, (f_score, push_count, next_row, next_col, tentative_g))

    return False, [], 0.0


def _namoa_star_pareto_frontier(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    target_row: int,
    target_col: int,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_expanded_labels: int,
):
    """
    Multi-objective A* (NAMOA*) search for the Pareto-optimal frontier of
    (cost, value_gained) trade-offs when traveling from (start_row,
    start_col) to (target_row, target_col) - see find_path_by_namoa_star's
    docstring for the general idea.

    Unlike _a_star_path, which tracks a single best cost per cell, this
    tracks a whole SET of non-dominated (cost, value) "labels" per cell,
    since a costlier route that picks up more incidental value along the
    way is a genuine trade-off, not simply worse. Label A is kept over label
    B at the same cell only if A costs less OR has collected more value than
    B - a new label that merely TIES an existing one in both respects is
    also discarded, not just one that is strictly worse, since keeping two
    identically-scoring routes on the menu adds nothing a chooser could ever
    use. This tie-collapsing matters a lot in practice: on flat or
    low-relief terrain, many geometrically different routes cost exactly
    the same, and without collapsing those ties the label set at nearby
    cells multiplies uncontrollably (thousands of distinct-but-identically-
    scored routes to the same cell) long before the search gets anywhere
    near the target.

    The priority for which label to expand next is its estimated total cost
    g + h (the same admissible Chebyshev-distance-based cost heuristic
    _a_star_path uses) - lower estimated cost first. This does not affect
    which labels ultimately survive the dominance checks, only the order
    they are found in.

    GOAL PRUNING: without any further help, this would still explore an
    enormous number of labels on a large grid, since a label is only ever
    discarded once something at least as good at the SAME cell turns up -
    nothing stops the search from fanning out toward distant, ultimately
    unhelpful cells in the meantime. Once at least one label has reached
    the target, every OTHER open label is checked against an optimistic
    best case for it: (its cost so far, its value so far + an upper bound
    on how much more value it could possibly still collect - the same
    "every remaining step lands on the single most valuable cell in the
    grid" bound value_heuristic uses below). If even that best case is
    already beaten by a solution found at the goal, the label can never
    contribute anything new to the frontier and is dropped without being
    expanded further. This is the single most important optimization here -
    without it, even a modest 100x100 grid can take tens of seconds (or
    exhaust max_expanded_labels outright) to reach a target 80-or-so cells
    away, since undirected multi-label search fans out far more aggressively
    than the single label _a_star_path tracks per cell.

    Each label carries its own path_cells and a running value_gained that
    only credits a cell's value the first time this specific label's path
    (or the `visited` cells from earlier, already-committed legs) touches
    it, so a label that loops back on itself never double-counts - it would
    cost more for no additional value and so always ends up dominated by
    the non-looping alternative by the time either reaches the target.

    Returns a list of (cost, value_gained, path_cells) triples, one per
    non-dominated way of reaching the target - path_cells excludes the
    start cell, the same convention as _a_star_path - or an empty list if
    the target is unreachable, or max_expanded_labels is exhausted first
    (which can only under-report options, never invent a trade-off that
    does not really exist).

    PERFORMANCE NOTE: tracking a whole non-dominated set per cell, rather
    than a single best value, means this explores substantially more state
    than _a_star_path for the same trip - this is, in practice, by far the
    most expensive algorithm in this project. On a small grid, or one where
    real targets sit close together, it runs quickly (comparable to the
    other A* variants); on a large, spread-out scenario (this project's own
    100x100 test scenarios included) a single candidate evaluation can take
    seconds even with goal pruning, and find_path_by_namoa_star's own
    tournament calls this once per candidate, every round.

    max_expanded_labels bounds the worst case (the same role _a_star_path's
    max_expanded_cells plays for plain A*), but there is no reliable "safe"
    value to raise it to for a large scenario: because find_path_by_namoa_
    star's tournament is a greedy, round-by-round process (see A CANDIDATE
    THAT FAILS IS NEVER RETRIED there), a bigger label budget does not
    monotonically improve the final answer - it can change which Pareto
    point (or even which candidate) wins an early round, cascading into a
    genuinely different, not necessarily better, sequence of later choices.
    For a large grid with many spread-out targets, prefer
    find_path_by_a_star_value_cost_ratio or greedy_pathfinding.find_path_
    by_value_cost_ratio for reliably fast, complete coverage; reach for
    NAMOA* where the value/cost trade-off itself is the point, or the
    search space is small enough to explore thoroughly.
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

    if (start_row, start_col) == (target_row, target_col):
        return [(0.0, 0.0, ())]

    min_edge_cost = _min_real_edge_cost(weights)
    max_cell_value = float(values.max())

    def cost_heuristic(row, col):
        """Admissible lower-bound estimate of the remaining cost from (row, col) to the target."""
        return max(abs(row - target_row), abs(col - target_col)) * min_edge_cost

    def value_heuristic(row, col):
        """
        An optimistic (never-underestimating) upper bound on remaining
        value: the best possible case is every remaining step landing on
        the single most valuable cell anywhere on the grid.
        """
        return max(abs(row - target_row), abs(col - target_col)) * max_cell_value

    def at_least_as_good(a, b):
        """
        True if `a` costs no more AND has collected no less value than
        `b` - used both to reject a new label that a's existing presence
        makes redundant (including an exact tie), and to prune old labels
        a strictly new one supersedes.
        """
        return a[0] <= b[0] and a[1] >= b[1]

    # A label is (cost, value, row, col, path_cells, label_id) - label_id is
    # a unique tag used to recognize (via lazy deletion) whether a label
    # popped off the heap is still considered non-dominated, since a label
    # already in the heap can be superseded after it was pushed but before
    # it is popped.
    label_id = 0
    start_label = (0.0, 0.0, start_row, start_col, (), label_id)
    frontier_by_cell = {(start_row, start_col): [start_label]}
    open_heap = [(cost_heuristic(start_row, start_col), 0, start_label)]
    push_count = 0
    expanded = 0
    goal_labels = []  # (cost, value, path_cells) triples found so far - see GOAL PRUNING

    while open_heap:
        _, _, label = heapq.heappop(open_heap)
        cost, value, row, col, path_cells, this_label_id = label

        if not any(existing[5] == this_label_id for existing in frontier_by_cell.get((row, col), [])):
            continue  # this label has since been dominated away - stale entry

        if (row, col) == (target_row, target_col):
            goal_labels.append((cost, value, path_cells))
            continue  # a label at the goal has nowhere further to usefully go

        if goal_labels:
            optimistic = (cost, value + value_heuristic(row, col))
            if any(at_least_as_good(goal, optimistic) for goal in goal_labels):
                continue  # even this label's best possible future is already beaten - see GOAL PRUNING

        expanded += 1
        if expanded > max_expanded_labels:
            break

        for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
            next_row, next_col = row + delta_row, col + delta_col

            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue
            if blocked_mask is not None and blocked_mask[next_row, next_col]:
                continue

            next_cost = cost + weights[row, col, direction]
            already_counted = visited[next_row, next_col] or (next_row, next_col) in path_cells
            next_value = value if already_counted else value + values[next_row, next_col]
            next_path_cells = path_cells + ((next_row, next_col),)

            existing_labels = frontier_by_cell.setdefault((next_row, next_col), [])
            candidate_label = (next_cost, next_value, next_row, next_col, next_path_cells, None)
            if any(at_least_as_good(existing, candidate_label) for existing in existing_labels):
                continue  # an existing label is at least as good in both respects (or ties) - redundant

            existing_labels[:] = [existing for existing in existing_labels if not at_least_as_good(candidate_label, existing)]

            label_id += 1
            next_label = (next_cost, next_value, next_row, next_col, next_path_cells, label_id)
            existing_labels.append(next_label)

            push_count += 1
            priority = next_cost + cost_heuristic(next_row, next_col)
            heapq.heappush(open_heap, (priority, push_count, next_label))

    return goal_labels


def _value_gained_along_path(grid: CoordinatesGrid, path_cells: list, visited: np.ndarray) -> float:
    """
    Sum of grid.coordinates_values over path_cells, excluding any cell
    already in `visited` - the same "don't double-count value already
    collected on an earlier leg" rule greedy_pathfinding._walk_toward_target
    applies. Unlike that function, no within-this-walk duplicate tracking is
    needed: an A* shortest path never revisits a cell (doing so could only
    add cost, never reduce it), so path_cells is always a simple path.
    """
    values = grid.coordinates_values
    return sum(values[cell] for cell in path_cells if not visited[cell])


def _evaluate_candidate_a_star(
    grid: CoordinatesGrid,
    candidate: tuple[int, int],
    row: int,
    col: int,
    remaining_budget: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_expanded_cells: int,
):
    """
    Like greedy_pathfinding._evaluate_candidate, but the walk to `candidate`
    is computed with true A* search (_a_star_path) instead of
    _walk_toward_target's rotate-around-obstacles heuristic - see
    find_path_by_a_star_value_cost_ratio's docstring for why that matters.
    Returns the same shaped dict, or None if candidate can't be reached at
    all, or its cheapest possible route still doesn't fit remaining_budget
    once the trip home from there is reserved for.
    """
    target_row, target_col = candidate
    reached, path_cells, cost = _a_star_path(grid, row, col, target_row, target_col, blocked_mask, max_expanded_cells)
    if not reached or cost <= 0:
        return None

    return_reserve = return_cost_grid[target_row, target_col] if return_cost_grid is not None else 0.0
    if cost + return_reserve > remaining_budget:
        return None

    value_gained = _value_gained_along_path(grid, path_cells, visited)
    return {
        "target": candidate,
        "path_cells": path_cells,
        "cost": cost,
        "value_gained": value_gained,
        "ratio": value_gained / cost,
    }


def _evaluate_candidate_namoa_star(
    grid: CoordinatesGrid,
    candidate: tuple[int, int],
    row: int,
    col: int,
    remaining_budget: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    max_expanded_labels: int,
):
    """
    Runs NAMOA* (_namoa_star_pareto_frontier) from (row, col) to `candidate`
    and picks whichever non-dominated (cost, value) trade-off gives the best
    value/cost ratio among the ones that still fit remaining_budget (after
    reserving the trip home) - see find_path_by_namoa_star's docstring for
    why scanning the whole frontier, rather than taking the single cheapest
    route the way _evaluate_candidate_a_star does, can find a better answer.
    Returns the same shaped dict as _evaluate_candidate_a_star, or None if
    no point on the frontier is both reachable and affordable.
    """
    target_row, target_col = candidate
    frontier = _namoa_star_pareto_frontier(grid, row, col, target_row, target_col, blocked_mask, visited, max_expanded_labels)
    if not frontier:
        return None

    return_reserve = return_cost_grid[target_row, target_col] if return_cost_grid is not None else 0.0

    best = None
    for cost, value_gained, path_cells in frontier:
        if cost <= 0 or cost + return_reserve > remaining_budget:
            continue
        ratio = value_gained / cost
        if best is None or ratio > best["ratio"]:
            best = {
                "target": candidate,
                "path_cells": list(path_cells),
                "cost": cost,
                "value_gained": value_gained,
                "ratio": ratio,
            }

    return best


if __name__ == "__main__":
    pass
