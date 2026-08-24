# metaheuristic_pathfinding.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from constants import Constants
from pathfinding_algorithms import greedy_pathfinding
import numpy as np


# When an ant's immediate neighbors are all used up, _attempt_jump looks for
# a still-reachable frontier cell (unvisited, but touching already-visited
# ground) to resume from instead of ending the path there - see
# SELF-TRAPPING REPAIR in find_path_by_ant_colony's docstring. This caps how
# many of the highest-attraction frontier cells are actually speculatively
# walked to and compared - each such walk costs real computation via
# greedy_pathfinding._walk_toward_target.
_JUMP_CANDIDATES_TO_TRY = 5


def find_path_by_ant_colony(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    ant_count: int = 15,
    iterations: int = 30,
    alpha: float = 1.0,
    beta: float = 3.0,
    evaporation_rate: float = 0.3,
    pheromone_deposit: float = 1.0,
    initial_pheromone: float = 1.0,
    attraction_range: float = 20.0,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Ant Colony Optimization (ACO).

    GENERAL IDEA
    ------------
    ACO is a constructive metaheuristic modeled on how real ant colonies find
    short paths between their nest and food: each ant wanders semi-randomly,
    laying down pheromone as it goes; pheromone evaporates over time; because
    a shorter/better path lets more ants complete a round trip per unit time,
    it accumulates pheromone faster than a longer one, which makes it more
    attractive to the next ants, which reinforces it further. No single ant
    ever "sees" the whole colony's knowledge - the pheromone trail is a
    shared, indirect memory (stigmergy) that lets the colony converge on a
    good path collectively.

    The algorithmic version repeats, for a fixed number of iterations:
      1. Every ant independently builds one complete candidate solution by
         moving step by step. At each step, standing at node i, it picks the
         next node j from the feasible options at random, but weighted by
         (pheromone_ij ^ alpha) * (heuristic_ij ^ beta) - so a step is more
         likely to be taken if the edge either has a strong pheromone trail
         (past ants found it useful) or looks good by itself (the heuristic,
         e.g. "short" or "cheap"). alpha/beta tune how much each of those two
         signals matters relative to the other.
      2. Once every ant has finished, each edge's pheromone evaporates by a
         factor (1 - evaporation_rate), so trails nobody keeps reusing fade
         out - this is what stops the colony from freezing onto the first
         decent solution it finds.
      3. Every ant then deposits fresh pheromone on the edges of the solution
         it built, proportional to how good that solution was. Edges shared
         by many good solutions end up reinforced from multiple directions
         each iteration; edges only used by poor solutions decay away.
      4. The best solution seen across all ants and all iterations is kept
         and returned once the iteration budget runs out.

    IN THIS CASE
    ------------
    The "nodes" are grid cells and the "edges" are the 8 compass moves out of
    each cell, exactly as in greedy_pathfinding.py. Each ant plays the same
    role as TrajectoryGenerator's single-start search: starting at
    (start_row, start_col), it grows a path one cell at a time, subject to
    the same budget rule as find_path_for_highest_neighbor_value - a move is
    only taken if paying for it AND (when require_return_to_base is set) the
    shortest route home from wherever it lands still fits inside max_cost
    (see RETURN-TRIP COST below). An ant with no feasible move left simply
    stops there, and (if require_return_to_base) flies that shortest route
    home as its final leg, the same as every function in
    greedy_pathfinding.py.

    Where an ant differs from the greedy searches is the one choice it makes
    at each step: instead of always taking the highest-value or best-ratio
    neighbor, it randomly picks among every feasible neighbor, weighted by:
      - pheromone[row, col, direction] ^ alpha - how much past ants favored
        this exact edge, out of this exact cell, in this exact direction
        (pheromone is asymmetric and per-direction, mirroring
        WeightsGrid.weights - climbing north out of a cell and descending
        south into it are tracked as separate trails);
      - (attraction_of_destination / cost_of_edge) ^ beta - see ATTRACTION
        FIELD below. This plays the same role as the plain value/cost ratio
        find_path_by_value_cost_ratio uses, so even an ant walking on a
        still-uninformative, near-uniform pheromone field is nudged toward
        cheap, promising cells rather than wandering blind.

    ATTRACTION FIELD
    ----------------
    A step's heuristic only ever compares the handful of immediately
    adjacent cells - on its own that gives an ant no way to sense a valuable
    cell a few steps away but not directly next to it, since that cell's
    high value doesn't show up in any neighbor's score until the ant happens
    to be standing right beside it. With a sparse, spread-out set of search
    targets and a limited number of ants/iterations, an ant is unlikely to
    stumble that close by chance, and pheromone can't fill the gap either -
    it only accumulates on edges some ant has already walked, so a target
    off the path nobody has taken yet stays permanently invisible to the
    colony.

    To give ants a longer-range sense of "something good is over there",
    every cell's desirability is drawn from a precomputed attraction field
    rather than its raw value: for every notable target cell (value above
    Constants.DEFAULT_PROBABILITY - the same "is this a real signal or just
    unexplored background" test greedy_pathfinding._next_fresh_candidate
    uses), its value radiates outward with exponential distance decay,
    controlled by attraction_range (the distance, in cells, at which a
    target's pull has fallen to ~37% = 1/e of its raw value). Cells near
    several targets, or one very high-value target, end up with a locally
    elevated attraction even far from any of them, which steers ants toward
    that neighborhood well before any target cell is actually adjacent -
    at which point the field's contribution from that target is exactly its
    real value, so the step-by-step choice still ultimately favors the
    genuine target cell over its surroundings. A larger attraction_range
    reaches further but blurs together targets that are actually far apart;
    a smaller one stays closer to plain greedy value/cost behavior. This
    field only steers ant choices - the value actually collected into
    total_value is always each cell's real value, never the attraction
    field.

    RETURN-TRIP COST
    -----------------
    Every candidate move is checked against
    greedy_pathfinding._build_return_cost_grid: the cost of the shortest
    route (diagonal until aligned, then straight, rotating around obstacles
    - see greedy_pathfinding._pick_next_step) from every cell back to
    (start_row, start_col), computed once up front and reused as an O(1)
    lookup for the rest of this call - rather than assuming the drone must
    retrace whatever specific, possibly meandering route it actually took.
    The same table is reused for every ant and every iteration, since it
    depends only on grid geometry, weights and blocked_mask, never on which
    cells happen to be visited.

    SELF-TRAPPING REPAIR
    ---------------------
    An ant only ever steps onto a touching, not-yet-visited cell, so it can
    box itself in with its own trail on an open, mostly-uniform grid well
    before its travel budget runs out - the exact same weakness
    find_path_for_highest_neighbor_value has. This is a side effect of
    modeling ants as an 8-connected grid walk (needed so per-direction
    pheromone means anything) rather than the classical ACO-for-Orienteering
    -Problem setup, where every unvisited point of interest is always a
    legal next move regardless of adjacency.

    Rather than switch the whole construction over to that model (and lose
    the per-direction pheromone trail), an ant that finds every neighbor
    blocked, visited, off-grid or unaffordable does not stop there. Instead
    (see _attempt_jump) it looks at the frontier - every unvisited,
    unblocked cell that touches at least one cell it has already visited,
    computed by _frontier_mask - and picks a handful of the
    highest-attraction frontier cells to walk toward, using the same
    rotate-around-obstacles routing greedy_pathfinding._walk_toward_target
    already uses for the direct-target algorithms. Among whichever of those
    turn out reachable within the remaining budget, it picks one at random
    weighted by (value_gained/cost)^beta, the same style of weighted choice
    ordinary steps use. Only if none of the tried candidates are reachable
    does the ant actually stop.

    Restricting jump targets to the frontier - rather than any unvisited
    cell anywhere on the grid, ranked purely by attraction - keeps a jump
    behaving like resuming a search sweep from the edge of ground already
    covered, instead of teleporting to a disconnected hotspot elsewhere on
    the map (which is what the direct-target algorithms already do, and
    would make this repair redundant with them rather than a genuine fix for
    the grid-walk's own weakness). It's also never a wasted search: because
    the ant only reaches this branch when every immediate neighbor has
    already been ruled out, at least one earlier cell along its own trail
    generally still borders unvisited ground, so a frontier cell is normally
    there to be found - the only way _attempt_jump comes up empty is if
    every bit of frontier within reach costs more than the remaining budget.

    After all ant_count ants for this iteration have finished:
      - every pheromone entry decays by (1 - evaporation_rate);
      - every ant deposits pheromone_deposit * (total_value / cost_used) onto
        each edge it actually used - a path that gathered a lot of value for
        little cost reinforces its edges more than one that gathered little
        for a lot, regardless of how long either path happened to be.

    Across iterations, edges that repeatedly show up in high value/cost
    paths accumulate pheromone and get chosen more often, so later ants are
    increasingly steered toward the routes earlier ants discovered were
    worthwhile - while evaporation keeps a one-off lucky detour from
    permanently biasing the search. The single best (path, total_value,
    cost_used) seen from any ant in any iteration is returned, matching the
    return shape of every function in greedy_pathfinding.py so this can be
    registered in trajectory_generator.PATHFINDING_ALGORITHMS like any other
    algorithm.

    PERFORMANCE NOTE: this function runs ant_count * iterations path
    constructions per call. TrajectoryGenerator.find_best_path calls it once,
    from the mission's fixed starting cell, so cost scales with ant_count and
    iterations alone - not with grid size - but a bigger grid still means
    each ant's walk (and thus each iteration) takes longer, so ant_count/
    iterations may need lowering for very large grids to keep runtime
    reasonable.
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values
    rng = np.random.default_rng(seed)

    attraction_field = _build_attraction_field(values, attraction_range)
    max_steps = 8 * (rows + cols)
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None

    # pheromone[row, col, direction] mirrors the shape of
    # grid.weights_grid.weights - one trail strength per directed edge.
    pheromone = np.full((rows, cols, 8), initial_pheromone, dtype=np.float64)

    best_path = [(start_row, start_col)]
    best_total_value = values[start_row, start_col]
    best_cost_used = 0.0

    for _ in range(iterations):
        ant_results = []

        for _ in range(ant_count):
            path, total_value, cost_used, edges_used = _construct_ant_path(
                grid, start_row, start_col, max_cost, return_cost_grid,
                blocked_mask, pheromone, attraction_field, alpha, beta, rng, max_steps,
            )
            ant_results.append((total_value, cost_used, edges_used))

            if total_value > best_total_value:
                best_path, best_total_value, best_cost_used = path, total_value, cost_used

        pheromone *= (1.0 - evaporation_rate)

        for total_value, cost_used, edges_used in ant_results:
            if cost_used <= 0:
                continue
            deposit = pheromone_deposit * total_value / cost_used
            for row, col, direction in edges_used:
                pheromone[row, col, direction] += deposit

    return best_path, best_total_value, best_cost_used


def find_path_by_tabu_search(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    iterations: int = 150,
    tabu_tenure: int = 10,
    swap_samples_per_round: int = 20,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Tabu Search.

    GENERAL IDEA
    ------------
    Unlike Ant Colony Optimization, which juggles a whole population of ants,
    Tabu Search follows a SINGLE evolving solution and repeatedly asks "what
    is the best nearby tweak I can make to what I already have?" Its
    defining trick is a short-term memory (the "tabu list") of moves it just
    made, which it refuses to immediately undo - this stops it endlessly
    flip-flopping between the same two solutions, and is what lets it accept
    a WORSE move on purpose sometimes, to climb out of a dead end (a "local
    optimum") that a purely greedy hill-climber would get stuck in forever.

    Each iteration:
      1. From the current solution, generate a set of "neighbor" solutions -
         ones reachable by one small, well-defined kind of change ("move").
      2. Throw out any neighbor that isn't actually feasible.
      3. Throw out any neighbor reached by a move that's currently "tabu"
         (its reverse was made recently) - UNLESS that neighbor would be a
         new best solution ever seen (the "aspiration criterion": a good
         enough result is allowed to break the rule).
      4. Move to whichever surviving neighbor scores best - even if that is
         worse than the current solution. Record the move's reverse as tabu
         for a fixed number of rounds (tabu_tenure).
      5. If this is the best solution seen so far across the whole run - not
         just this step - remember it.
    Repeat for `iterations` rounds, then return the best solution ever seen,
    not wherever the search happens to be standing at the end - it may have
    wandered downhill again since then.

    IN THIS CASE
    ------------
    The "solution" is a plan: an ORDERED LIST of which real search targets
    (cells with value above Constants.DEFAULT_PROBABILITY - the same test
    greedy_pathfinding._next_fresh_candidate uses) to fly to, and in what
    order - not a list of every grid cell. Given that list, the actual grid
    path is worked out mechanically (see _evaluate_tour) by walking start ->
    target_1 -> target_2 -> ... -> target_k -> back to start, using the same
    shortest-path routing (greedy_pathfinding._walk_toward_target) every
    other algorithm in this project uses to travel between two points, with
    the same return-trip accounting find_path_by_ant_colony uses (see
    greedy_pathfinding._build_return_cost_grid). This turns "find a good
    grid path" into "find a good short list of stops, in a good order" - a
    far smaller, more tractable search space than reasoning about individual
    grid cells directly.

    Three kinds of moves make up a solution's neighborhood every round:
      - ADD a target that isn't in the plan yet, tacked onto the end.
      - DROP a target that's currently in the plan.
      - SWAP the order of two targets already in the plan (a randomly
        sampled subset of all possible pairs, capped at
        swap_samples_per_round, once the plan gets long enough that trying
        every pair would be expensive).
    A move is only ever considered if the resulting plan still fits
    max_cost - infeasible neighbors are dropped outright rather than
    penalized.

    The tabu rule: after ADDing target T, DROPping T is forbidden for
    tabu_tenure rounds (so the search cannot immediately regret and undo
    it); after DROPping T, ADDing T is forbidden the same way; after
    SWAPping two targets, swapping them straight back is forbidden. This is
    what lets the search accept a round that makes total_value temporarily
    worse - it cannot just retreat next round, so it is pushed to find a
    genuinely different way forward instead, and that occasional acceptance
    of a worse plan is exactly what lets it climb out of a plan that looks
    locally unbeatable but is not actually the best one on the map.

    PERFORMANCE NOTE: evaluating a single neighbor means re-walking the
    ENTIRE plan from scratch (see _evaluate_tour) - there is no incremental
    shortcut for "what if I changed just this one stop." With P real targets
    and a plan of length k, one iteration costs roughly O((P + k) * average
    leg length), for up to `iterations` rounds - fine for the tens-to-low-
    hundreds of targets a search-and-rescue scenario typically has, but this
    will not scale gracefully to a grid with thousands of individually
    valuable cells.
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values
    rng = np.random.default_rng(seed)
    max_steps = 8 * (rows + cols)
    return_cost_grid = greedy_pathfinding._build_return_cost_grid(grid, start_row, start_col, blocked_mask, max_steps) if require_return_to_base else None

    candidate_rows, candidate_cols = np.nonzero(values > Constants.DEFAULT_PROBABILITY)
    candidate_pool = [
        (int(candidate_row), int(candidate_col))
        for candidate_row, candidate_col in zip(candidate_rows, candidate_cols)
        if (int(candidate_row), int(candidate_col)) != (start_row, start_col)
        and (blocked_mask is None or not blocked_mask[candidate_row, candidate_col])
    ]

    def evaluate(tour):
        return _evaluate_tour(grid, start_row, start_col, tour, max_cost, return_cost_grid, blocked_mask, max_steps)

    current_tour = []
    _, best_path, best_total_value, best_cost_used = evaluate(current_tour)

    tabu_list = {}  # move -> rounds remaining before it's allowed again

    for _ in range(iterations):
        neighbors = []  # (move, candidate_tour)

        in_tour = set(current_tour)
        for target in candidate_pool:
            if target not in in_tour:
                neighbors.append((("add", target), current_tour + [target]))

        for target in current_tour:
            neighbors.append((("drop", target), [cell for cell in current_tour if cell != target]))

        if len(current_tour) >= 2:
            sample_count = min(swap_samples_per_round, len(current_tour) * (len(current_tour) - 1) // 2)
            for _ in range(sample_count):
                first_index, second_index = rng.choice(len(current_tour), size=2, replace=False)
                swapped_tour = list(current_tour)
                swapped_tour[first_index], swapped_tour[second_index] = swapped_tour[second_index], swapped_tour[first_index]
                neighbors.append((("swap", current_tour[first_index], current_tour[second_index]), swapped_tour))

        best_neighbor_move = None
        best_neighbor_value = None
        best_neighbor_result = None

        for move, candidate_tour in neighbors:
            feasible, path, total_value, cost_used = evaluate(candidate_tour)
            if not feasible:
                continue

            # A move is tabu if it would undo a recent one - unless it's
            # good enough to beat the best solution ever found (aspiration).
            if tabu_list.get(_reverse_move(move), 0) > 0 and total_value <= best_total_value:
                continue

            if best_neighbor_value is None or total_value > best_neighbor_value:
                best_neighbor_move = move
                best_neighbor_value = total_value
                best_neighbor_result = (candidate_tour, path, total_value, cost_used)

        if best_neighbor_move is None:
            break  # every neighbor was infeasible or blocked by the tabu list

        current_tour, current_path, current_total_value, current_cost_used = best_neighbor_result

        for move in list(tabu_list):
            tabu_list[move] -= 1
            if tabu_list[move] <= 0:
                del tabu_list[move]
        tabu_list[_reverse_move(best_neighbor_move)] = tabu_tenure

        if current_total_value > best_total_value:
            best_path, best_total_value, best_cost_used = current_path, current_total_value, current_cost_used

    return best_path, best_total_value, best_cost_used


def _reverse_move(move: tuple) -> tuple:
    """
    The move that would undo `move` - what find_path_by_tabu_search marks
    tabu after taking a step, so the search can't immediately regret and
    reverse its own last decision. Adding a target is undone by dropping it
    (and vice versa); swapping two targets' order is undone by swapping them
    right back.
    """
    kind = move[0]
    if kind == "add":
        return ("drop", move[1])
    if kind == "drop":
        return ("add", move[1])
    return move  # ("swap", a, b) undoes itself


def _evaluate_tour(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    tour: list,
    max_cost: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    max_steps: int,
):
    """
    Walks start -> tour[0] -> tour[1] -> ... -> tour[-1] -> (back to start,
    if return_cost_grid is not None) using greedy_pathfinding._walk_toward_
    target for each leg, the same shortest-path routing every other
    algorithm in this project uses. Returns (feasible, path, total_value,
    cost_used); feasible is False (other values meaningless) if any leg -
    including the final trip home - can't be completed within max_cost.
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True
    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    for target_row, target_col in tour:
        reached, path_cells, cost, value_gained = greedy_pathfinding._walk_toward_target(
            grid, row, col, target_row, target_col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps,
        )
        if not reached:
            return False, None, 0.0, 0.0

        for cell in path_cells:
            visited[cell] = True
            path.append(cell)
        total_value += value_gained
        remaining_budget -= cost
        row, col = target_row, target_col

    if return_cost_grid is None:
        return True, path, total_value, max_cost - remaining_budget

    reached, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    if not reached:
        return False, None, 0.0, 0.0

    path = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return True, path, total_value, total_cost_used


def _build_attraction_field(values: np.ndarray, attraction_range: float) -> np.ndarray:
    """
    See ATTRACTION FIELD in find_path_by_ant_colony's docstring: every cell
    with a value above Constants.DEFAULT_PROBABILITY radiates that value
    outward with exponential distance decay (rate 1/attraction_range), and
    every grid cell's field strength is the sum of those contributions from
    every such target. A target's own cell always gets at least its raw
    value back (distance 0 -> decay factor 1), so it never looks less
    attractive than its surroundings.

    Only a handful of cells are ever real targets (see
    greedy_pathfinding._next_fresh_candidate for the same test), so this
    loops over targets - typically far fewer than the rows * cols cells in
    the grid - rather than over every cell pair.
    """
    rows, cols = values.shape
    target_rows, target_cols = np.nonzero(values > Constants.DEFAULT_PROBABILITY)

    if target_rows.size == 0:
        return values.copy()

    row_indices, col_indices = np.indices((rows, cols))
    attraction_field = np.zeros((rows, cols), dtype=np.float64)

    for target_row, target_col in zip(target_rows, target_cols):
        distance = np.hypot(row_indices - target_row, col_indices - target_col)
        attraction_field += values[target_row, target_col] * np.exp(-distance / attraction_range)

    return attraction_field


def _frontier_mask(visited: np.ndarray, blocked_mask: np.ndarray) -> np.ndarray:
    """
    Boolean mask of every unvisited, unblocked cell that touches (in any of
    the 8 directions) at least one already-visited cell - the boundary of
    the region an ant has explored so far. Used by _attempt_jump to keep a
    repair jump anchored to ground the ant has actually been near, rather
    than letting it land anywhere on the grid - see SELF-TRAPPING REPAIR in
    find_path_by_ant_colony's docstring.

    Computed fresh each time (visited grows every step, so it can't be
    precomputed once like the attraction field) via 8 shifted copies of
    `visited` - one per compass direction - rather than a per-cell Python
    loop, since this needs to run once per stuck ant, potentially many times
    over a long path.
    """
    rows, cols = visited.shape
    frontier = np.zeros((rows, cols), dtype=bool)

    for delta_row, delta_col in Constants.DIRECTIONS:
        # shifted[r, c] ends up True exactly where visited[r + delta_row, c
        # + delta_col] is True and in-bounds - i.e. "this cell has an
        # already-visited neighbor in this direction".
        shifted = np.zeros((rows, cols), dtype=bool)
        row_start, row_stop = max(0, -delta_row), min(rows, rows - delta_row)
        col_start, col_stop = max(0, -delta_col), min(cols, cols - delta_col)
        shifted[row_start:row_stop, col_start:col_stop] = visited[
            row_start + delta_row:row_stop + delta_row, col_start + delta_col:col_stop + delta_col,
        ]
        frontier |= shifted

    frontier &= ~visited
    if blocked_mask is not None:
        frontier &= ~blocked_mask

    return frontier


def _attempt_jump(
    grid: CoordinatesGrid,
    row: int,
    col: int,
    remaining_budget: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    visited: np.ndarray,
    attraction_field: np.ndarray,
    rng: np.random.Generator,
    beta: float,
    max_steps: int,
):
    """
    See SELF-TRAPPING REPAIR in find_path_by_ant_colony's docstring. Called
    when an ant standing at (row, col) has no adjacent feasible move left.
    Takes the up to _JUMP_CANDIDATES_TO_TRY highest-attraction cells in
    _frontier_mask (unvisited cells bordering ground already visited) and
    speculatively walks toward each with greedy_pathfinding._walk_toward_
    target (the same rotate-around-obstacles routing the direct-target
    algorithms use - it may well cross other already-visited cells along the
    way, which is fine and realistic: reaching a fresh patch of frontier can
    genuinely require flying back over ground already swept). This does not
    mutate `visited`, so several candidates can be tried from the same state
    before one is committed - mirroring greedy_pathfinding._evaluate_candidate.

    Among the candidates that turn out reachable within remaining_budget,
    picks one at random weighted by (value_gained/cost)^beta - the same
    style of weighted choice ordinary steps use - rather than always taking
    the single best one, so a jump still leans toward better options
    without behaving like a rigid greedy rule.

    Returns (path_cells, cost, value_gained) for the chosen jump - the same
    shape _walk_toward_target returns, minus the `reached` flag since only
    reachable candidates make it this far - or None if the frontier is
    empty or none of the tried candidates are reachable.
    """
    frontier_rows, frontier_cols = np.nonzero(_frontier_mask(visited, blocked_mask))
    if frontier_rows.size == 0:
        return None

    frontier_attraction = attraction_field[frontier_rows, frontier_cols]
    candidate_count = min(_JUMP_CANDIDATES_TO_TRY, frontier_rows.size)
    # argpartition only needs to separate the top candidate_count entries
    # from the rest - O(frontier size) - rather than fully sorting the
    # frontier, which can be a sizeable fraction of the grid once an ant has
    # covered a lot of ground.
    top_indices = np.argpartition(-frontier_attraction, candidate_count - 1)[:candidate_count]
    candidate_targets = [(int(frontier_rows[i]), int(frontier_cols[i])) for i in top_indices]

    reachable = []
    for target_row, target_col in candidate_targets:
        reached, path_cells, cost, value_gained = greedy_pathfinding._walk_toward_target(
            grid, row, col, target_row, target_col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps,
        )
        if reached and cost > 0:
            reachable.append((path_cells, cost, value_gained))

    if not reachable:
        return None

    # Every reachable candidate gets at least a tiny share of the odds, even
    # one with value_gained == 0 (e.g. it only crosses cells another
    # already-committed leg touched first) - it's still a legitimate way to
    # escape the trap and keep spending the budget.
    jump_weights = np.array([max(value_gained / cost, 1e-9) ** beta for _, cost, value_gained in reachable], dtype=np.float64)
    chosen_index = rng.choice(len(reachable), p=jump_weights / jump_weights.sum())

    return reachable[chosen_index]


def _edges_from_path(start_row: int, start_col: int, path_cells: list):
    """
    Reconstructs the (row, col, direction) edge for each step of path_cells
    (a sequence of one-grid-cell-at-a-time moves, as _walk_toward_target
    produces), starting from (start_row, start_col) - so a multi-step jump
    can have pheromone deposited on every real edge it crosses, the same as
    an ordinary step, rather than as one lumped-together jump.
    """
    row, col = start_row, start_col
    for next_row, next_col in path_cells:
        direction = Constants.DIRECTIONS.index((next_row - row, next_col - col))
        yield row, col, direction
        row, col = next_row, next_col


def _construct_ant_path(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    pheromone: np.ndarray,
    attraction_field: np.ndarray,
    alpha: float,
    beta: float,
    rng: np.random.Generator,
    max_steps: int,
):
    """
    One ant's walk: same step-by-step budget accounting as
    find_path_for_highest_neighbor_value in greedy_pathfinding.py - see
    RETURN-TRIP COST in find_path_by_ant_colony's docstring for how
    return_cost_grid (None if require_return_to_base was False) is used -
    but the next cell is sampled at random among feasible neighbors -
    weighted by pheromone^alpha * (attraction_field/cost)^beta - instead of
    always taking the single best one. If no adjacent move is feasible,
    tries _attempt_jump before giving up - see SELF-TRAPPING REPAIR.
    Returns (path, total_value, cost_used, edges_used), where edges_used is
    the list of (row, col, direction) edges the ant actually walked (jumps
    included, one entry per real grid edge crossed), for the caller to
    deposit pheromone onto, and total_value is accumulated from the grid's
    real values (not the attraction field, which only steers the random
    choice).
    """
    rows, cols = grid.rows, grid.cols
    weights = grid.weights_grid.weights
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True

    path = [(start_row, start_col)]
    edges_used = []
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col

    while True:
        feasible_moves = []  # (direction, next_row, next_col, cost)

        for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
            next_row = row + delta_row
            next_col = col + delta_col

            if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                continue
            if visited[next_row, next_col]:
                continue
            if blocked_mask is not None and blocked_mask[next_row, next_col]:
                continue

            cost = weights[row, col, direction]

            return_reserve = return_cost_grid[next_row, next_col] if return_cost_grid is not None else 0.0
            if cost + return_reserve > remaining_budget:
                continue

            feasible_moves.append((direction, next_row, next_col, cost))

        if not feasible_moves:
            jump = _attempt_jump(
                grid, row, col, remaining_budget, return_cost_grid,
                blocked_mask, visited, attraction_field, rng, beta, max_steps,
            )
            if jump is None:
                break

            jump_path_cells, jump_cost, jump_value_gained = jump
            for edge_row, edge_col, edge_direction in _edges_from_path(row, col, jump_path_cells):
                edges_used.append((edge_row, edge_col, edge_direction))
            for cell_row, cell_col in jump_path_cells:
                visited[cell_row, cell_col] = True
                path.append((cell_row, cell_col))
            total_value += jump_value_gained
            remaining_budget -= jump_cost
            row, col = jump_path_cells[-1]
            continue

        move_weights = np.empty(len(feasible_moves), dtype=np.float64)
        for index, (direction, next_row, next_col, cost) in enumerate(feasible_moves):
            attraction = attraction_field[next_row, next_col]
            desirability = attraction / cost if cost > 0 else attraction
            move_weights[index] = (pheromone[row, col, direction] ** alpha) * (desirability ** beta)

        total_weight = move_weights.sum()
        if total_weight <= 0 or not np.isfinite(total_weight):
            # Every candidate looks equally (un)promising to both pheromone
            # and heuristic - fall back to a uniform pick rather than
            # dividing by zero.
            chosen_index = rng.integers(len(feasible_moves))
        else:
            chosen_index = rng.choice(len(feasible_moves), p=move_weights / total_weight)

        direction, next_row, next_col, cost = feasible_moves[chosen_index]

        visited[next_row, next_col] = True
        path.append((next_row, next_col))
        edges_used.append((row, col, direction))
        total_value += values[next_row, next_col]
        remaining_budget -= cost
        row, col = next_row, next_col

    if return_cost_grid is None:
        return path, total_value, max_cost - remaining_budget, edges_used

    # Fly the actual shortest route home (see greedy_pathfinding._walk_
    # toward_target) rather than retracing the outbound path - the
    # invariant enforced above (return_cost_grid[cell] <= remaining_budget
    # after every accepted move or jump) guarantees this fits.
    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used, edges_used


if __name__ == "__main__":
    pass
