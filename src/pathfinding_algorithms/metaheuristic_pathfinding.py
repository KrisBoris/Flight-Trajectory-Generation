# metaheuristic_pathfinding.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from helpers.constants import Constants
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
        """Shorthand for _evaluate_tour with this call's fixed grid/start/budget/blocked_mask arguments bound in."""
        return _evaluate_tour(grid, start_row, start_col, tour, max_cost, return_cost_grid, blocked_mask, max_steps)

    current_tour = []
    _, best_path, best_total_value, best_cost_used = evaluate(current_tour)

    tabu_list = {}  # move -> rounds remaining before it's allowed again

    for _ in range(iterations):
        neighbors = _generate_tour_neighbors(current_tour, candidate_pool, rng, swap_samples_per_round)

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


def _generate_tour_neighbors(current_tour: list, candidate_pool: list, rng: np.random.Generator, swap_samples_per_round: int) -> list:
    """
    The neighborhood of a tour (an ordered list of target cells - see
    find_path_by_tabu_search and find_path_by_variable_neighborhood_search)
    under three move types: ADD an unvisited target to the end, DROP a
    target already in the tour, or SWAP the order of two targets already in
    it. SWAP moves are randomly sampled (capped at swap_samples_per_round)
    rather than exhaustively enumerated, since the number of possible pairs
    grows quadratically with tour length.

    Returns a list of (move, candidate_tour) pairs, where move is ("add",
    target), ("drop", target) or ("swap", target_a, target_b).
    """
    neighbors = []

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

    return neighbors


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


def find_path_by_variable_neighborhood_search(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    iterations: int = 40,
    max_neighborhood: int = 4,
    local_search_rounds: int = 15,
    swap_samples_per_round: int = 20,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Variable Neighborhood Search (VNS).

    GENERAL IDEA
    ------------
    Picture standing on a hilly map made of "how good is this plan", trying
    to find the highest hill. A plain hill-climber takes one tiny step at a
    time toward whatever looks better right now - which works fine until it
    is standing on top of A hill that is not the TALLEST hill around, with
    no single small step that looks better. It is stuck there for good.

    VNS's trick: when small steps stop helping, take a bigger, RANDOM jump
    instead - far enough that it might land somewhere completely different,
    maybe at the foot of a taller hill - then go back to small careful steps
    from wherever it lands, to climb whatever is nearby. If that jump didn't
    lead anywhere better after climbing, take an even BIGGER random jump
    next time. The moment a jump does lead somewhere better, go back to
    small jumps again, since it is probably close to a good hilltop now.

    Each round:
      1. SHAKE: from the current solution, make a random jump of size k (a
         gentle jump if k is small, a disruptive one if k is large) -
         landing on some solution that was not carefully chosen, just picked
         at random from "k steps away".
      2. LOCAL SEARCH: from that randomly-jumped-to solution, repeatedly take
         the single best small tidying step available until none helps
         anymore (a local hilltop).
      3. COMPARE: if this hilltop beats the solution the round started from,
         move there for good and reset k back to its smallest value - the
         next round only takes small, careful jumps again. If it does not
         beat the starting solution, stay put, but grow k - the next round's
         jump will be bigger, since a small jump clearly was not enough to
         shake things up.
    Repeat for `iterations` rounds, then return the best solution seen at
    any point - not necessarily wherever the search ends up standing.

    The key difference from find_path_by_tabu_search: Tabu Search stays
    disciplined, taking one careful step at a time and using a "no backsies"
    memory to force itself past a dead end. VNS instead leans on
    RANDOMNESS - a sometimes-small, sometimes-large random jump - to
    physically relocate out of a dead end, then tidies up locally from
    wherever it happens to land.

    IN THIS CASE
    ------------
    The solution, and how it is evaluated, is exactly
    find_path_by_tabu_search's: an ordered list of real search targets (see
    _evaluate_tour), with the grid-level flight path worked out mechanically
    between them via greedy_pathfinding._walk_toward_target, subject to the
    same return-trip accounting (greedy_pathfinding._build_return_cost_grid).

    NEIGHBORHOODS
    -------------
    _shake(tour, k) is the "jump of size k": drop k random targets from the
    tour and add k random not-yet-included ones - the same ADD/DROP moves
    find_path_by_tabu_search takes one at a time, just k of them applied at
    once, and picked randomly rather than for their score. k = 1 is a small,
    gentle reshuffle; k = max_neighborhood swaps out that many targets in
    one go. A shake can land on a temporarily infeasible tour (too
    expensive) - that is fine, since the local search step right after it
    will usually repair it with a DROP move, the same way it would clean up
    any other unproductive addition.

    _local_search is a plain greedy hill-climb, not a miniature Tabu Search:
    from the shaken tour it repeatedly takes whichever single ADD/DROP/SWAP
    move (find_path_by_tabu_search's move set again - see
    _generate_tour_neighbors, shared by both algorithms) most improves
    total_value, stopping the moment no move helps (local_search_rounds caps
    how many such steps it is allowed to take, as a safety net against a
    pathologically long climb).

    PERFORMANCE NOTE: like find_path_by_tabu_search, every neighbor
    evaluation re-walks a candidate tour from scratch (see _evaluate_tour),
    so cost scales with the number of real targets on the map. A single
    shake-then-local-search round here does strictly more work than one
    find_path_by_tabu_search round (the shake, plus a whole burst of hill-
    climbing afterward), so `iterations` defaults lower to keep total
    runtime in the same ballpark.
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

    current_tour = []
    _, best_path, best_total_value, best_cost_used = _evaluate_tour(
        grid, start_row, start_col, current_tour, max_cost, return_cost_grid, blocked_mask, max_steps,
    )
    current_total_value = best_total_value

    neighborhood_size = 1

    for _ in range(iterations):
        shaken_tour = _shake(current_tour, candidate_pool, neighborhood_size, rng)

        local_tour, local_feasible, local_path, local_total_value, local_cost_used = _local_search(
            grid, start_row, start_col, shaken_tour, candidate_pool, max_cost, return_cost_grid,
            blocked_mask, max_steps, swap_samples_per_round, rng, local_search_rounds,
        )

        if local_feasible and local_total_value > current_total_value:
            current_tour, current_total_value = local_tour, local_total_value
            neighborhood_size = 1  # this shake paid off - go back to gentle ones

            if local_total_value > best_total_value:
                best_path, best_total_value, best_cost_used = local_path, local_total_value, local_cost_used
        else:
            neighborhood_size += 1  # that shake didn't help - try a bigger one next time
            if neighborhood_size > max_neighborhood:
                neighborhood_size = 1

    return best_path, best_total_value, best_cost_used


def find_path_by_grasp(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    iterations: int = 25,
    rcl_size: int = 5,
    local_search_rounds: int = 15,
    swap_samples_per_round: int = 20,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Greedy Randomized Adaptive Search Procedure (GRASP).

    GENERAL IDEA
    ------------
    Imagine building a LEGO tower as tall and sturdy as possible, one piece
    at a time. A purely greedy builder always grabs the single best piece
    available at every step - but that locks in the exact same choices every
    time, and an early "best-looking" piece does not always lead to the best
    finished tower. GRASP's fix: at each step, look at a handful of the best
    pieces available - not just the single best one - and pick ONE OF THOSE
    AT RANDOM. This still builds a genuinely good tower (every piece
    considered was a strong one), but a different tower nearly every time,
    since the random pick varies. Once a full tower is built this way, spend
    a bit of effort fine-tuning it - swap a piece, add one, remove one,
    whatever helps - until nothing helps anymore. THEN throw the plan away
    and build an entirely new tower from scratch the same randomized-greedy
    way, fine-tune that one too, and so on. After many such attempts, keep
    whichever finished tower turned out best overall.

    Each iteration is two phases:
      1. CONSTRUCTION (greedy + randomized + adaptive): build a solution one
         piece at a time. At every step, score every still-possible next
         piece by how good it looks RIGHT NOW given what has been built so
         far (the "adaptive" part - scores are recalculated fresh every
         step, never fixed in advance), gather the best few into a short
         "Restricted Candidate List" (RCL), and pick one of them at random -
         not always the single best. Repeat until nothing more can be added.
      2. LOCAL SEARCH: from that constructed solution, hill-climb (see
         _local_search) to a local optimum - the same tidying-up step
         find_path_by_variable_neighborhood_search uses right after a shake.
      3. Keep this iteration's result if it is the best seen across every
         iteration so far.
    Repeat for `iterations` independent restarts, then return the best
    solution found across all of them.

    Where this differs from every other metaheuristic in this file: Ant
    Colony Optimization and Tabu Search/VNS each maintain and evolve ONE
    solution (or, for ACO, a shared pheromone-guided population) over time,
    learning from its own history as they go. GRASP instead restarts from
    scratch every iteration - there is no memory carried between
    iterations - and relies purely on "many independent, randomized-but-
    still-smart attempts, keep the best" to find a good answer. That makes
    it embarrassingly parallel in principle (every iteration is fully
    independent of every other) and immune to ever being trapped by a bad
    early decision inherited from a previous iteration, at the cost of not
    being able to build on partial progress the way the other algorithms do.

    IN THIS CASE
    ------------
    Same solution representation as find_path_by_tabu_search and
    find_path_by_variable_neighborhood_search: an ordered list of real
    search targets (see _evaluate_tour), with the grid-level flight path
    worked out via greedy_pathfinding._walk_toward_target and the same
    return-trip accounting (greedy_pathfinding._build_return_cost_grid).

    CONSTRUCTION
    ------------
    _construct_greedy_randomized_tour builds the tour by repeatedly walking
    from wherever it currently ends toward one more target - exactly what
    find_path_by_value_cost_ratio's tournament does (see
    greedy_pathfinding._evaluate_candidate for the walk-and-score-by-
    value/cost-ratio logic) - but instead of always taking the single
    best-ratio candidate, it ranks every reachable remaining target by that
    ratio, keeps the top rcl_size of them (the Restricted Candidate List),
    and picks one uniformly at random. Since which targets are still
    reachable, and what their ratios are, depends on where the tour
    currently stands - not fixed at the start - re-evaluating this at every
    step is the "adaptive" half of GRASP. Construction stops the moment no
    remaining target is reachable within budget.

    LOCAL SEARCH
    ------------
    Reuses find_path_by_variable_neighborhood_search's _local_search
    unchanged: the same ADD/DROP/SWAP hill-climb (see
    _generate_tour_neighbors), run until no move improves the constructed
    tour or local_search_rounds is reached.

    PERFORMANCE NOTE: construction alone costs roughly O(P^2 * average leg
    length) in the worst case, since each of up to P rounds re-scores every
    still-unused candidate (P = size of the real-target candidate pool) -
    and local search adds its own per-iteration cost on top of that, on the
    same order as find_path_by_tabu_search's. With `iterations` independent
    restarts, this is the most compute-hungry algorithm in this file for a
    large candidate pool; lower `iterations` or rcl_size for a big scenario
    if runtime matters more than the last bit of quality.
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

    best_path = [(start_row, start_col)]
    best_total_value = values[start_row, start_col]
    best_cost_used = 0.0

    for _ in range(iterations):
        constructed_tour = _construct_greedy_randomized_tour(
            grid, start_row, start_col, max_cost, return_cost_grid, blocked_mask, max_steps, candidate_pool, rcl_size, rng,
        )

        _, feasible, local_path, local_total_value, local_cost_used = _local_search(
            grid, start_row, start_col, constructed_tour, candidate_pool, max_cost, return_cost_grid,
            blocked_mask, max_steps, swap_samples_per_round, rng, local_search_rounds,
        )

        if feasible and local_total_value > best_total_value:
            best_path, best_total_value, best_cost_used = local_path, local_total_value, local_cost_used

    return best_path, best_total_value, best_cost_used


def _construct_greedy_randomized_tour(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    max_steps: int,
    candidate_pool: list,
    rcl_size: int,
    rng: np.random.Generator,
) -> list:
    """
    See CONSTRUCTION in find_path_by_grasp's docstring. Starting at
    (start_row, start_col), repeatedly walks toward one more target using
    greedy_pathfinding._evaluate_candidate (the same walk-and-score-by-
    value/cost-ratio helper find_path_by_value_cost_ratio's tournament
    uses), but each round narrows every still-reachable remaining candidate
    down to the top rcl_size by ratio (the Restricted Candidate List) and
    picks one of them uniformly at random, rather than always taking the
    single best. Stops the moment no remaining candidate is reachable within
    budget.

    Returns just the tour (the ordered list of chosen target cells) - not a
    full path/value/cost - so it can be fed into _local_search the same way
    _shake's output is.
    """
    rows, cols = grid.rows, grid.cols

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True
    remaining_budget = max_cost
    row, col = start_row, start_col
    remaining_candidates = set(candidate_pool)
    tour = []

    while remaining_candidates:
        evaluated = []
        for candidate in remaining_candidates:
            result = greedy_pathfinding._evaluate_candidate(
                grid, candidate, row, col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps,
            )
            if result is not None:
                evaluated.append((candidate, result))

        if not evaluated:
            break  # nothing left is reachable within the remaining budget

        evaluated.sort(key=lambda entry: entry[1]["ratio"], reverse=True)
        restricted_candidate_list = evaluated[:rcl_size]
        chosen_candidate, chosen_result = restricted_candidate_list[rng.integers(len(restricted_candidate_list))]

        for cell in chosen_result["path_cells"]:
            visited[cell] = True
        remaining_budget -= chosen_result["cost"]
        row, col = chosen_candidate
        tour.append(chosen_candidate)
        remaining_candidates.discard(chosen_candidate)

    return tour


def find_path_by_simulated_annealing(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    iterations: int = 3000,
    initial_temperature: float = 20.0,
    cooling_rate: float = 0.995,
    min_temperature: float = 1e-3,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Simulated Annealing (SA).

    GENERAL IDEA
    ------------
    Picture a bouncy ball dropped onto a bumpy landscape of hills and
    valleys, trying to settle on the tallest hill (higher is better here).
    A ball that only ever moves to something better gets stuck on the very
    first small hill it climbs - even if a far taller hill is sitting just
    past a dip in between, since reaching it would first require going the
    "wrong" way, downhill, and a strictly-better-only mover never allows
    that.

    Simulated Annealing fixes this by making the ball extra bouncy at the
    start - as if the whole landscape is being shaken hard - so it can hop
    over small hills and out of shallow dips, wandering fairly freely (even
    sometimes ending up somewhere clearly worse than where it started).
    Then, very gradually, the shaking is turned down ("cooling"), so the
    ball's bounces get smaller and smaller, until near the end it barely
    bounces at all and just settles wherever it happens to be - hopefully a
    tall hill it stumbled onto while it still had plenty of bounce left to
    explore broadly.

    The name comes from annealing metal: heat it up (its atoms jiggle
    around energetically, shaking loose from a poor, defect-ridden
    arrangement), then cool it down SLOWLY so the atoms have time to settle
    into a low-energy, orderly crystal structure. Cool it too fast (quench
    it) and the atoms freeze into a poor structure - the metal equivalent of
    getting permanently stuck on a mediocre hill.

    Each iteration:
      1. Propose ONE random small change to the current solution (a
         "neighbor").
      2. If it is better, always accept it.
      3. If it is worse, accept it ANYWAY with a probability that depends on
         both how much worse it is and the current "temperature" - a small
         worsening is often still accepted, a big one rarely is, and either
         is far more likely to be accepted early (high temperature) than
         late (low temperature). This is the Metropolis criterion:
         probability = exp(how much worse / temperature).
      4. Remember this as the best solution ever seen if it beats everything
         found so far - the current solution can wander to something worse
         at any moment, so what actually gets returned at the end is
         tracked separately, exactly like every other algorithm in this
         file.
      5. Cool down a little: temperature = temperature * cooling_rate
         (never letting it drop below min_temperature).
    Repeat for `iterations` rounds - typically far more than find_path_by_
    tabu_search or find_path_by_variable_neighborhood_search use, since each
    round here is much cheaper, see IN THIS CASE - then return the best
    solution ever seen.

    IN THIS CASE
    ------------
    Same solution representation as every other search-based metaheuristic
    in this file: an ordered list of real search targets (see
    _evaluate_tour), with the grid-level flight path worked out via
    greedy_pathfinding._walk_toward_target and the same return-trip
    accounting (greedy_pathfinding._build_return_cost_grid).

    Where this differs from find_path_by_tabu_search, find_path_by_
    variable_neighborhood_search and find_path_by_grasp: those all generate
    a whole batch of candidate neighbors every round and deliberately pick
    among them (the best one, or the best after a random shake). Simulated
    Annealing instead proposes exactly ONE random neighbor per round (see
    _random_neighbor - a single random ADD, DROP or SWAP, the same three
    move types used everywhere else in this file) and simply decides
    accept-or-reject on that one proposal. That is a much cheaper single
    round, which is exactly why it is normally run for many more rounds than
    the others.

    TEMPERATURE SCHEDULE
    ---------------------
    initial_temperature sets how freely the search wanders early on;
    cooling_rate (multiplied into the temperature every round) controls how
    quickly that freedom fades - a cooling_rate close to 1 cools slowly
    (more exploration, needs more iterations to fully cool), a smaller one
    cools fast (less exploration, converges sooner but risks settling
    early). Temperature never drops below min_temperature, both to avoid
    dividing by zero and because a temperature of exactly zero would make
    the search purely greedy for every remaining iteration - fine in
    principle, but a small floor keeps a sliver of randomness alive
    throughout.

    PERFORMANCE NOTE: every iteration still calls _evaluate_tour, which
    re-walks the whole tour from scratch (see find_path_by_tabu_search's
    PERFORMANCE NOTE) - so this is not free per iteration, just far cheaper
    per iteration than generating and scoring a whole neighborhood.
    `iterations` defaults much higher than the other tour-based
    metaheuristics in this file to compensate.
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

    current_tour = []
    _, best_path, best_total_value, best_cost_used = _evaluate_tour(
        grid, start_row, start_col, current_tour, max_cost, return_cost_grid, blocked_mask, max_steps,
    )
    current_total_value = best_total_value

    temperature = initial_temperature

    for _ in range(iterations):
        candidate_tour = _random_neighbor(current_tour, candidate_pool, rng)
        feasible, path, total_value, cost_used = _evaluate_tour(
            grid, start_row, start_col, candidate_tour, max_cost, return_cost_grid, blocked_mask, max_steps,
        )

        if feasible:
            delta = total_value - current_total_value
            # A strictly better proposal is always taken; a worse one is
            # still taken with probability exp(delta / temperature) - delta
            # is <= 0 here, so this is in (0, 1], shrinking as temperature
            # cools or the proposal gets worse.
            if delta > 0 or rng.random() < np.exp(delta / temperature):
                current_tour, current_total_value = candidate_tour, total_value

                if total_value > best_total_value:
                    best_path, best_total_value, best_cost_used = path, total_value, cost_used

        temperature = max(temperature * cooling_rate, min_temperature)

    return best_path, best_total_value, best_cost_used


def _random_neighbor(current_tour: list, candidate_pool: list, rng: np.random.Generator) -> list:
    """
    Picks ONE random move - ADD an unvisited target, DROP a target already
    in the tour, or SWAP two targets' order (the same three move types
    _generate_tour_neighbors builds a whole batch of) - and returns the
    resulting tour. find_path_by_simulated_annealing only ever needs a
    single random proposal per iteration, so this picks one move directly
    rather than generating and discarding every other possible neighbor.
    """
    in_tour = set(current_tour)
    not_in_tour = [cell for cell in candidate_pool if cell not in in_tour]

    available_move_kinds = []
    if not_in_tour:
        available_move_kinds.append("add")
    if current_tour:
        available_move_kinds.append("drop")
    if len(current_tour) >= 2:
        available_move_kinds.append("swap")

    if not available_move_kinds:
        return list(current_tour)  # nothing possible to change

    move_kind = available_move_kinds[rng.integers(len(available_move_kinds))]

    if move_kind == "add":
        target = not_in_tour[rng.integers(len(not_in_tour))]
        return current_tour + [target]

    if move_kind == "drop":
        drop_index = rng.integers(len(current_tour))
        return [cell for index, cell in enumerate(current_tour) if index != drop_index]

    # "swap"
    first_index, second_index = rng.choice(len(current_tour), size=2, replace=False)
    swapped_tour = list(current_tour)
    swapped_tour[first_index], swapped_tour[second_index] = swapped_tour[second_index], swapped_tour[first_index]
    return swapped_tour


def _shake(tour: list, candidate_pool: list, neighborhood_size: int, rng: np.random.Generator) -> list:
    """
    See NEIGHBORHOODS in find_path_by_variable_neighborhood_search's
    docstring: randomly perturbs `tour` by dropping up to neighborhood_size
    of its targets and adding the same number of random not-yet-included
    targets from candidate_pool. Larger neighborhood_size means a bigger,
    more disruptive random jump.
    """
    tour = list(tour)

    remove_count = min(neighborhood_size, len(tour))
    if remove_count > 0:
        remove_indices = set(rng.choice(len(tour), size=remove_count, replace=False).tolist())
        tour = [cell for index, cell in enumerate(tour) if index not in remove_indices]

    in_tour = set(tour)
    not_in_tour = [cell for cell in candidate_pool if cell not in in_tour]
    add_count = min(neighborhood_size, len(not_in_tour))
    if add_count > 0:
        add_indices = rng.choice(len(not_in_tour), size=add_count, replace=False)
        tour = tour + [not_in_tour[index] for index in add_indices]

    return tour


def _local_search(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    tour: list,
    candidate_pool: list,
    max_cost: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    max_steps: int,
    swap_samples_per_round: int,
    rng: np.random.Generator,
    max_rounds: int,
):
    """
    Greedy hill-climb from `tour`: repeatedly applies whichever single
    ADD/DROP/SWAP move (see _generate_tour_neighbors) most improves
    total_value, stopping as soon as no move improves it (a local optimum)
    or after max_rounds. Used by find_path_by_variable_neighborhood_search
    to tidy up a solution right after _shake has randomly perturbed it.

    A tour that starts infeasible (_shake can produce one, e.g. by adding an
    expensive target) is treated as having value -inf, so any feasible
    neighbor - most commonly a DROP - counts as an improvement; if no
    feasible neighbor is ever found, this returns feasible=False.

    Returns (tour, feasible, path, total_value, cost_used).
    """
    current_tour = tour
    feasible, path, total_value, cost_used = _evaluate_tour(
        grid, start_row, start_col, current_tour, max_cost, return_cost_grid, blocked_mask, max_steps,
    )
    current_value = total_value if feasible else -np.inf

    for _ in range(max_rounds):
        neighbors = _generate_tour_neighbors(current_tour, candidate_pool, rng, swap_samples_per_round)

        best_result = None
        best_value = current_value

        for _, candidate_tour in neighbors:
            candidate_feasible, candidate_path, candidate_value, candidate_cost = _evaluate_tour(
                grid, start_row, start_col, candidate_tour, max_cost, return_cost_grid, blocked_mask, max_steps,
            )
            if candidate_feasible and candidate_value > best_value:
                best_value = candidate_value
                best_result = (candidate_tour, candidate_path, candidate_value, candidate_cost)

        if best_result is None:
            break  # local optimum - no move improves on the current tour

        current_tour, path, total_value, cost_used = best_result
        current_value = total_value
        feasible = True

    return current_tour, feasible, path, total_value, cost_used


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


def find_path_by_genetic_algorithm(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    population_size: int = 30,
    generations: int = 60,
    tournament_size: int = 3,
    crossover_rate: float = 0.8,
    mutation_rate: float = 0.3,
    elite_count: int = 2,
    seed_fraction: float = 0.2,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Genetic Algorithm (GA) - introduced by John Holland ("Adaptation in
    Natural and Artificial Systems", 1975) and popularized by David
    Goldberg's 1989 book "Genetic Algorithms in Search, Optimization, and
    Machine Learning".

    GENERAL IDEA
    ------------
    Imagine breeding pea plants for the tallest, most productive garden. You
    would not just pick the single best plant you have and stop - you would
    let your best plants cross-pollinate with each other, combining their
    traits, let a few random mutations happen naturally, and grow a whole
    new generation from that. Repeat for many generations, and the garden as
    a whole tends to keep improving, because good traits from two DIFFERENT
    parents can combine into an offspring better than either parent alone -
    something no single plant, however good, could ever do by itself.

    A Genetic Algorithm does exactly this with candidate solutions instead
    of plants:
      1. Start with a whole population of candidate solutions - mostly
         randomly built, some given a head start from a quick greedy guess.
      2. Score every one of them ("how good is this?") - its "fitness".
      3. The better-scoring ones are more likely to become "parents" for
         the next generation - here via tournament selection: grab a few
         candidates at random, let the fittest of that small group win a
         chance to be a parent (the same style of "compare a handful, take
         the best" tournament greedy_pathfinding.find_path_by_value_cost_
         ratio already uses, just applied to picking parents instead of
         picking targets).
      4. Two parents "have a child" (crossover): the child borrows part of
         one parent's plan and fills in the rest from the other's,
         hopefully combining the best of both.
      5. Every so often, a child gets a small random mutation - a tweak
         that has nothing to do with either parent - keeping some fresh
         randomness alive so the population does not converge into a sea
         of near-identical copies.
      6. The single best solution ever found is always kept safe
         ("elitism"), so a bad generation can never accidentally lose it.
      7. Repeat for `generations` rounds, then return the best solution
         ever seen - the same "track the best across the whole run"
         convention every algorithm in this file follows.

    IN THIS CASE
    ------------
    A "chromosome" is exactly the same solution representation find_path_
    by_tabu_search, find_path_by_variable_neighborhood_search, find_path_
    by_grasp and find_path_by_simulated_annealing already use: an ordered
    list of real search targets (cells with value above Constants.
    DEFAULT_PROBABILITY). The grid-level flight path is worked out
    mechanically from that list via greedy_pathfinding._walk_toward_target,
    with the same return-trip accounting (greedy_pathfinding._build_
    return_cost_grid) - see REPAIR RATHER THAN REJECT for the one place
    this differs from those other algorithms' _evaluate_tour.

    - Initial population: seed_fraction of it is built with _construct_
      greedy_randomized_tour (the same randomized-greedy construction
      find_path_by_grasp uses); the rest is uniformly random subsets of the
      candidate pool in random order (see _random_chromosome) - a mix of
      "informed" and "blind" starting material, rather than only one or
      the other.
    - Fitness: total_value from _evaluate_chromosome_with_repair.
    - Selection: tournament selection (_tournament_select), tournament_size
      candidates compared per pick.
    - Crossover: _crossover - order-preserving, adapted for variable-length
      subsets rather than full permutations (see its own docstring).
    - Mutation: reuses _random_neighbor UNCHANGED - the exact same single
      random ADD/DROP/SWAP move find_path_by_simulated_annealing proposes
      every iteration. Every metaheuristic in this file that operates on
      this tour representation - Tabu Search, VNS, GRASP, Simulated
      Annealing, and now this - ultimately explores the same three moves;
      what differs between them is only the STRATEGY for choosing among
      those moves (memory-guided local search, shake-then-climb,
      randomized-greedy-then-climb, single-proposal random walk, and here,
      population + selection + crossover).
    - Elitism: the elite_count best chromosomes (already fitness-sorted)
      survive into the next generation unchanged, guaranteeing the
      population's best individual never gets worse from one generation to
      the next - though the single best-EVER solution is still tracked
      separately and returned, exactly like every other algorithm here,
      since even an elite generation is regenerated with fresh crossover/
      mutation for everyone else.

    REPAIR RATHER THAN REJECT
    ---------------------------
    Every other tour-based algorithm in this file uses _evaluate_tour,
    which fails a tour OUTRIGHT (feasible=False) the moment any single leg
    does not fit the budget. That is a fine rule when there is only one
    candidate tour to judge at a time (as in a single local-search move),
    but a GA's crossover and mutation routinely produce chromosomes with
    more targets than the budget could ever afford, and simply discarding
    every such individual would waste most of the population on every
    single generation. _evaluate_chromosome_with_repair instead walks a
    chromosome one target at a time and just STOPS at the first target that
    would not fit, keeping everything up to that point - "repairing" an
    over-ambitious chromosome into the best feasible prefix of itself,
    rather than throwing the whole thing away for being too greedy. This
    repaired, shorter chromosome (not the original) is what is used for
    crossover and mutation henceforth once a generation is scored, so the
    population's genetic material stays realistic.

    PERFORMANCE NOTE: like find_path_by_grasp, every individual's fitness
    evaluation walks its whole chromosome from scratch, and this runs that
    evaluation for every individual, every generation - population_size *
    generations evaluations in total, the same order of cost as GRASP's
    iterations * (candidate pool size) construction work. Expect a runtime
    in the same ballpark as find_path_by_grasp or find_path_by_variable_
    neighborhood_search for a similarly sized candidate pool.
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

    best_path = [(start_row, start_col)]
    best_total_value = values[start_row, start_col]
    best_cost_used = 0.0

    if not candidate_pool:
        return best_path, best_total_value, best_cost_used

    seeded_count = max(1, round(population_size * seed_fraction))
    population = [
        _construct_greedy_randomized_tour(grid, start_row, start_col, max_cost, return_cost_grid, blocked_mask, max_steps, candidate_pool, rcl_size=5, rng=rng)
        for _ in range(min(seeded_count, population_size))
    ]
    while len(population) < population_size:
        population.append(_random_chromosome(candidate_pool, rng))

    for _ in range(generations):
        evaluated = []
        for chromosome in population:
            path, total_value, cost_used, repaired_chromosome = _evaluate_chromosome_with_repair(
                grid, start_row, start_col, chromosome, max_cost, return_cost_grid, blocked_mask, max_steps,
            )
            evaluated.append((repaired_chromosome, total_value, path, cost_used))

            if total_value > best_total_value:
                best_path, best_total_value, best_cost_used = path, total_value, cost_used

        evaluated.sort(key=lambda entry: entry[1], reverse=True)
        repaired_population = [entry[0] for entry in evaluated]
        fitnesses = [entry[1] for entry in evaluated]

        next_population = list(repaired_population[:elite_count])

        while len(next_population) < population_size:
            parent_a = _tournament_select(repaired_population, fitnesses, tournament_size, rng)
            parent_b = _tournament_select(repaired_population, fitnesses, tournament_size, rng)

            child = _crossover(parent_a, parent_b, rng) if rng.random() < crossover_rate else list(parent_a)
            if rng.random() < mutation_rate:
                child = _random_neighbor(child, candidate_pool, rng)

            next_population.append(child)

        population = next_population

    return best_path, best_total_value, best_cost_used


def _random_chromosome(candidate_pool: list, rng: np.random.Generator) -> list:
    """
    A uniformly random chromosome for find_path_by_genetic_algorithm's
    initial population: a random-size subset of candidate_pool (from 1
    target up to every target), in random order. This is the "blind"
    counterpart to _construct_greedy_randomized_tour's informed seeding -
    together they give the starting population a mix of head-started and
    genuinely unbiased material.
    """
    subset_size = rng.integers(1, len(candidate_pool) + 1)
    chosen_indices = rng.choice(len(candidate_pool), size=subset_size, replace=False)
    chromosome = [candidate_pool[index] for index in chosen_indices]
    rng.shuffle(chromosome)
    return chromosome


def _tournament_select(population: list, fitnesses: list, tournament_size: int, rng: np.random.Generator) -> list:
    """
    Tournament selection: samples tournament_size individuals at random
    (with replacement) from `population` and returns whichever of them has
    the highest fitness - the same "compare a handful, take the best"
    pattern used throughout this project's tournament-style algorithms,
    applied here to picking a parent rather than picking a target.
    """
    candidate_indices = rng.integers(0, len(population), size=tournament_size)
    best_index = max(candidate_indices, key=lambda index: fitnesses[index])
    return population[best_index]


def _crossover(parent_a: list, parent_b: list, rng: np.random.Generator) -> list:
    """
    Order-preserving crossover, adapted for find_path_by_genetic_
    algorithm's variable-length subset chromosomes rather than the
    fixed-length full permutations classic TSP-style crossover assumes:
    keeps a random-length prefix of parent_a as-is, then appends whichever
    of parent_b's targets are not already in that prefix, in parent_b's own
    order. The child is always duplicate-free and never longer than
    len(parent_a) + len(parent_b).
    """
    if not parent_a:
        return list(parent_b)
    if not parent_b:
        return list(parent_a)

    prefix_length = rng.integers(1, len(parent_a) + 1)
    prefix = list(parent_a[:prefix_length])
    prefix_set = set(prefix)

    return prefix + [target for target in parent_b if target not in prefix_set]


def _evaluate_chromosome_with_repair(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    chromosome: list,
    max_cost: float,
    return_cost_grid: np.ndarray,
    blocked_mask: np.ndarray,
    max_steps: int,
):
    """
    See REPAIR RATHER THAN REJECT in find_path_by_genetic_algorithm's
    docstring. Walks chromosome (an ordered list of target cells) one
    target at a time via greedy_pathfinding._walk_toward_target, exactly
    like _evaluate_tour, but instead of failing the whole chromosome the
    moment one leg does not fit the budget, simply stops there and keeps
    whatever prefix of targets was actually reachable.

    Returns (path, total_value, cost_used, repaired_chromosome) - always
    feasible (cost_used <= max_cost) - where repaired_chromosome is the
    prefix of chromosome that was actually kept, for the caller to use as
    this individual's real genetic material going forward.
    """
    rows, cols = grid.rows, grid.cols
    values = grid.coordinates_values

    visited = np.zeros((rows, cols), dtype=bool)
    visited[start_row, start_col] = True
    path = [(start_row, start_col)]
    total_value = values[start_row, start_col]
    remaining_budget = max_cost
    row, col = start_row, start_col
    repaired_chromosome = []

    for target_row, target_col in chromosome:
        reached, path_cells, cost, value_gained = greedy_pathfinding._walk_toward_target(
            grid, row, col, target_row, target_col, remaining_budget, return_cost_grid, blocked_mask, visited, max_steps,
        )
        if not reached:
            break  # this target - and, since positions only move forward, everything after it - doesn't fit

        for cell in path_cells:
            visited[cell] = True
            path.append(cell)
        total_value += value_gained
        remaining_budget -= cost
        row, col = target_row, target_col
        repaired_chromosome.append((target_row, target_col))

    if return_cost_grid is None:
        return path, total_value, max_cost - remaining_budget, repaired_chromosome

    _, return_path_cells, return_cost, return_value_gained = greedy_pathfinding._walk_toward_target(
        grid, row, col, start_row, start_col, remaining_budget, None, blocked_mask, visited, max_steps,
    )
    path_with_return = path + return_path_cells
    total_value += return_value_gained
    total_cost_used = (max_cost - remaining_budget) + return_cost

    return path_with_return, total_value, total_cost_used, repaired_chromosome


_DESTROY_OPERATORS = ("random", "worst", "related")
_REPAIR_OPERATORS = ("greedy", "regret")

# Scores handed to whichever (destroy, repair) pair produced a round's
# result - see ADAPTIVE WEIGHTS in find_path_by_large_neighborhood_search's
# docstring. A new best-ever solution is worth the most, being accepted
# despite being worse is worth the least (but still more than an outright
# rejection), matching the classic Ropke & Pisinger ALNS scoring scheme.
_SCORE_NEW_BEST = 3.0
_SCORE_IMPROVED_CURRENT = 2.0
_SCORE_ACCEPTED_WORSE = 1.0
_SCORE_REJECTED = 0.0


def find_path_by_large_neighborhood_search(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    iterations: int = 400,
    min_removal_fraction: float = 0.1,
    max_removal_fraction: float = 0.3,
    repair_candidate_sample_size: int = 15,
    construction_rcl_size: int = 3,
    initial_temperature: float = 15.0,
    cooling_rate: float = 0.997,
    min_temperature: float = 1e-3,
    reaction_factor: float = 0.3,
    segment_length: int = 20,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Adaptive Large Neighborhood Search (ALNS) - introduced by Stefan Ropke
    and David Pisinger ("An Adaptive Large Neighborhood Search Heuristic for
    the Pickup and Delivery Problem with Time Windows", 2006), building on
    the plain Large Neighborhood Search idea from Paul Shaw (1998).

    GENERAL IDEA
    ------------
    Imagine you already built a decent LEGO castle (a full flight plan),
    but you suspect a few rooms could be rearranged to fit a couple more
    towers in. Instead of tearing the whole castle down and starting over
    (that's what find_path_by_grasp does every iteration), ALNS knocks out
    just a HANDFUL of rooms at a time - leaving most of the good castle
    standing - and then rebuilds only the missing part, as cleverly as it
    can. If the rebuilt castle is better, it becomes the new starting
    castle for the next round of knock-out-and-rebuild; if not, it is
    sometimes kept anyway (see ACCEPTANCE below) so the search doesn't get
    stuck forever polishing the exact same castle.

    Every round:
      1. DESTROY: pick a "destroy operator" and remove a chunk of stops
         from the current plan.
      2. REPAIR: pick a "repair operator" and plug the resulting hole back
         up as well as possible.
      3. DECIDE: keep the rebuilt plan, or throw it away (see ACCEPTANCE).
      4. LEARN: operators that tend to produce good rebuilds get picked
         more often afterward; ones that don't get picked less (see
         ADAPTIVE WEIGHTS) - this is the "Adaptive" in ALNS, and what
         separates it from plain, fixed-recipe LNS.

    Why remove a whole chunk instead of the single ADD/DROP/SWAP moves
    find_path_by_tabu_search, find_path_by_variable_neighborhood_search and
    find_path_by_grasp already use in this project? A one-move-at-a-time
    search can only ever discover an improvement reachable by changing ONE
    stop - if the best rearrangement genuinely needs three stops swapped
    out together (three stops that only pay off as a matched set), such a
    search may never stumble onto it, because every in-between state
    (changing just 1 of the 3) looks worse and gets rejected before the
    3rd change ever gets a chance to pay off. Ripping out several stops at
    once and rebuilding them together lets the search "see" that whole
    combination directly - precisely the gap ALNS is designed to fill, and
    why it's a natural next addition alongside this project's existing
    one-move-at-a-time and restart-from-scratch metaheuristics.

    IN THIS CASE
    ------------
    Same solution representation as find_path_by_tabu_search, find_path_by_
    variable_neighborhood_search, find_path_by_grasp and find_path_by_
    simulated_annealing: an ordered list of real search targets (a "tour"),
    evaluated by walking start -> target_1 -> ... -> target_k -> (back to
    start) via greedy_pathfinding._walk_toward_target (_evaluate_tour),
    subject to the same return-trip accounting
    (greedy_pathfinding._build_return_cost_grid). Unlike those four, the
    STARTING tour is not empty - it's built once up front by GRASP's own
    construction routine (_construct_greedy_randomized_tour, see
    find_path_by_grasp's CONSTRUCTION), so ALNS spends its whole iteration
    budget improving an already-reasonable plan rather than also having to
    discover one from nothing.

    DESTROY OPERATORS
    ------------------
    Each removes `removal_count` stops (a random fraction of the current
    tour, redrawn every round between min_removal_fraction and
    max_removal_fraction) from the tour:
      - "random" (_random_removal): removes a uniformly random subset - the
        simplest possible chunk, a baseline against the two smarter ones.
      - "worst" (_worst_removal): removes whichever stops are contributing
        the LEAST value to the plan right now - for each stop, that's "how
        much total_value would I lose if this one stop, and only this one,
        were removed", found by literally re-evaluating the tour without
        it. This targets exactly the stops most likely to be a poor use of
        the budget.
      - "related" (_related_removal, a Shaw removal): picks one random
        stop, then removes whichever OTHER stops sit geographically
        closest to it. Nearby stops tend to be interchangeable (there are
        usually several other real targets in the same neighborhood
        competing for the same "swing by here" decision), so ripping out a
        cluster at once gives the repair step real freedom to pick a
        genuinely different combination from that area - removing
        scattered, unrelated stops would just get plugged back with the
        same stops, since nothing else was ever competing for their slot.

    REPAIR OPERATORS
    ------------------
    Each adds back up to `removal_count` stops - drawn not just from the
    ones just removed, but from every real target not currently in the
    tour (the budget freed up by a destroy round might now afford a
    completely different, better target instead of whichever one used to
    sit there):
      - "greedy" (_greedy_repair): repeatedly inserts whichever (candidate,
        position) pairing gives the single best resulting total_value, one
        stop at a time, stopping early the moment nothing more fits.
      - "regret" (_regret_repair, "regret-2" insertion): for every
        candidate, compares its BEST possible insertion against its
        SECOND-best. A candidate whose best spot only barely beats its
        second-best can safely wait a round, but one with only ONE decent
        spot (or a huge gap to its next-best) needs to be grabbed now, or
        that opportunity may be gone once other insertions have used up
        the budget. Regret insertion reacts to exactly this "act now or
        regret it" signal - something plain greedy insertion, which only
        ever looks at each candidate's single best option, cannot see.

    Both operators only consider inserting a candidate at TWO positions -
    right at the end of the tour, and right after whichever existing stop
    sits geographically closest to it (_candidate_positions) - rather than
    every possible position, and only look at a random sample of
    repair_candidate_sample_size not-yet-included targets per insertion
    rather than literally every one (_sample_candidates) - see PERFORMANCE
    NOTE for why.

    ACCEPTANCE
    -----------
    Like find_path_by_simulated_annealing (see that function's docstring
    for the full mechanics), a repaired tour that's better than the
    current one is always accepted; a worse one is still sometimes
    accepted, with a probability that shrinks as a "temperature" cools
    over the run (initial_temperature, cooling_rate, min_temperature play
    identical roles to their find_path_by_simulated_annealing
    counterparts). This is what stops ALNS from being a pure hill-climber
    that locks onto the first local optimum its destroy/repair pairs
    happen to find. As with every algorithm in this project, the single
    best (path, total_value, cost_used) ever seen - not wherever the
    search happens to end up - is what gets returned.

    ADAPTIVE WEIGHTS
    ------------------
    Each destroy operator and each repair operator starts with an equal
    chance of being picked (roulette-wheel selection weighted by these
    scores - see _roulette_pick). Every round scores the (destroy, repair)
    pairing that produced it using the _SCORE_* constants above: finding a
    new best-ever solution scores highest, merely improving the current
    solution scores less, being accepted anyway despite being worse scores
    less still, and outright rejection scores nothing. Every
    segment_length rounds (_update_operator_weights), each operator's
    weight is nudged toward its average recent score - blended in by
    reaction_factor (closer to 1 forgets older performance faster, closer
    to 0 barely reacts at all) - and its scoreboard resets for the next
    segment. Over a whole run this lets ALNS gradually lean on whichever
    destroy/repair combination is actually paying off for THIS specific
    scenario, rather than committing to one fixed strategy for the whole
    run the way this project's other metaheuristics do.

    PERFORMANCE NOTE: every insertion attempt re-walks the whole tour from
    scratch (see find_path_by_tabu_search's PERFORMANCE NOTE) - restricting
    each repair round to repair_candidate_sample_size sampled candidates
    and 2 insertion positions apiece (rather than every remaining candidate
    at every possible position) is what keeps a single destroy+repair
    round's cost roughly comparable to one find_path_by_grasp construction
    step, instead of scaling with the full candidate pool squared - this
    project's own "_dense" scenario files have up to 510 targets, where an
    unrestricted search would be far too slow to run `iterations` times.
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

    def evaluate_tour(tour):
        """Shorthand for _evaluate_tour with this call's fixed grid/start/budget/blocked_mask arguments bound in."""
        return _evaluate_tour(
            grid, start_row, start_col, tour, max_cost, return_cost_grid, blocked_mask, max_steps,
        )

    if not candidate_pool:
        _, path, total_value, cost_used = evaluate_tour([])
        return path, total_value, cost_used

    current_tour = _construct_greedy_randomized_tour(
        grid, start_row, start_col, max_cost, return_cost_grid, blocked_mask, max_steps,
        candidate_pool, construction_rcl_size, rng,
    )
    feasible, current_path, current_total_value, current_cost_used = evaluate_tour(current_tour)
    if not feasible:
        current_tour = []  # construction itself should never fail, but fall back safely if it somehow does
        _, current_path, current_total_value, current_cost_used = evaluate_tour(current_tour)

    best_tour = list(current_tour)
    best_path, best_total_value, best_cost_used = current_path, current_total_value, current_cost_used

    destroy_weights = {name: 1.0 for name in _DESTROY_OPERATORS}
    repair_weights = {name: 1.0 for name in _REPAIR_OPERATORS}
    destroy_scores = {name: 0.0 for name in _DESTROY_OPERATORS}
    repair_scores = {name: 0.0 for name in _REPAIR_OPERATORS}
    destroy_uses = {name: 0 for name in _DESTROY_OPERATORS}
    repair_uses = {name: 0 for name in _REPAIR_OPERATORS}

    temperature = initial_temperature

    for iteration_index in range(1, iterations + 1):
        destroy_name = _roulette_pick(destroy_weights, rng)
        repair_name = _roulette_pick(repair_weights, rng)

        tour_length = len(current_tour)
        if tour_length > 0:
            min_removal = max(1, round(min_removal_fraction * tour_length))
            max_removal = max(min_removal, round(max_removal_fraction * tour_length))
            removal_count = int(rng.integers(min_removal, max_removal + 1))
        else:
            removal_count = 0

        if destroy_name == "random":
            partial_tour = _random_removal(current_tour, removal_count, rng)
        elif destroy_name == "worst":
            partial_tour = _worst_removal(current_tour, removal_count, evaluate_tour, current_total_value)
        else:  # "related"
            partial_tour = _related_removal(current_tour, removal_count, rng)

        # Restore roughly as many stops as were removed - if the tour was
        # empty (e.g. construction found nothing affordable), still make a
        # modest attempt to seed it from scratch instead of stalling forever.
        insertion_budget = (tour_length - len(partial_tour)) if tour_length > 0 else min(5, len(candidate_pool))

        if repair_name == "greedy":
            repaired_tour = _greedy_repair(partial_tour, insertion_budget, candidate_pool, repair_candidate_sample_size, evaluate_tour, rng)
        else:  # "regret"
            repaired_tour = _regret_repair(partial_tour, insertion_budget, candidate_pool, repair_candidate_sample_size, evaluate_tour, rng)

        feasible, path, total_value, cost_used = evaluate_tour(repaired_tour)

        if not feasible:
            score = _SCORE_REJECTED
        else:
            delta = total_value - current_total_value
            # A strictly better repair is always kept; a worse one is still
            # sometimes kept, with probability exp(delta / temperature) -
            # the same Metropolis criterion find_path_by_simulated_annealing
            # uses, so ALNS can wander past a round that didn't pay off
            # instead of freezing onto the first local optimum it finds.
            accept = delta > 0 or rng.random() < np.exp(delta / temperature)

            if accept:
                current_tour, current_path, current_total_value, current_cost_used = repaired_tour, path, total_value, cost_used

                if total_value > best_total_value:
                    best_tour = list(repaired_tour)
                    best_path, best_total_value, best_cost_used = path, total_value, cost_used
                    score = _SCORE_NEW_BEST
                elif delta > 0:
                    score = _SCORE_IMPROVED_CURRENT
                else:
                    score = _SCORE_ACCEPTED_WORSE
            else:
                score = _SCORE_REJECTED

        destroy_scores[destroy_name] += score
        repair_scores[repair_name] += score
        destroy_uses[destroy_name] += 1
        repair_uses[repair_name] += 1

        if iteration_index % segment_length == 0:
            _update_operator_weights(destroy_weights, destroy_scores, destroy_uses, reaction_factor)
            _update_operator_weights(repair_weights, repair_scores, repair_uses, reaction_factor)

        temperature = max(temperature * cooling_rate, min_temperature)

    return best_path, best_total_value, best_cost_used


def _random_removal(tour: list, count: int, rng: np.random.Generator) -> list:
    """See DESTROY OPERATORS ("random") in find_path_by_large_neighborhood_search's docstring."""
    if count <= 0 or not tour:
        return list(tour)

    count = min(count, len(tour))
    remove_indices = set(rng.choice(len(tour), size=count, replace=False).tolist())
    return [cell for index, cell in enumerate(tour) if index not in remove_indices]


def _worst_removal(tour: list, count: int, evaluate_tour, current_total_value: float) -> list:
    """
    See DESTROY OPERATORS ("worst") in find_path_by_large_neighborhood_
    search's docstring. For each stop, evaluates the tour with just that
    one stop taken out and compares total_value against the full tour's -
    the drop is that stop's "contribution". Removes the `count` stops with
    the smallest contribution (least valuable to keep).
    """
    if count <= 0 or not tour:
        return list(tour)

    count = min(count, len(tour))
    contributions = []
    for index in range(len(tour)):
        reduced_tour = tour[:index] + tour[index + 1:]
        _, _, total_value_without, _ = evaluate_tour(reduced_tour)
        contributions.append((current_total_value - total_value_without, index))

    contributions.sort(key=lambda entry: entry[0])  # smallest contribution (least valuable stop) first
    remove_indices = {index for _, index in contributions[:count]}
    return [cell for index, cell in enumerate(tour) if index not in remove_indices]


def _related_removal(tour: list, count: int, rng: np.random.Generator) -> list:
    """
    See DESTROY OPERATORS ("related") in find_path_by_large_neighborhood_
    search's docstring - a Shaw removal. Picks one random seed stop, then
    removes the `count` - 1 other stops geographically closest to it (the
    seed itself, at distance 0, is always among them).
    """
    if count <= 0 or not tour:
        return list(tour)

    count = min(count, len(tour))
    seed_row, seed_col = tour[int(rng.integers(len(tour)))]
    distances = [(np.hypot(row - seed_row, col - seed_col), index) for index, (row, col) in enumerate(tour)]
    distances.sort(key=lambda entry: entry[0])  # nearest (most related) to the seed first

    remove_indices = {index for _, index in distances[:count]}
    return [cell for index, cell in enumerate(tour) if index not in remove_indices]


def _sample_candidates(candidates: list, sample_size: int, rng: np.random.Generator) -> list:
    """Uniformly samples up to sample_size candidates without replacement - all of them if there are fewer."""
    if len(candidates) <= sample_size:
        return list(candidates)

    indices = rng.choice(len(candidates), size=sample_size, replace=False)
    return [candidates[index] for index in indices]


def _candidate_positions(tour: list, candidate: tuple) -> list:
    """
    The (small, fixed) set of tour positions _greedy_repair and
    _regret_repair actually try for inserting `candidate` - see
    PERFORMANCE NOTE in find_path_by_large_neighborhood_search's docstring
    for why this isn't every position. Always includes the end of the tour
    (a plain append, like every ADD move elsewhere in this project); if the
    tour isn't empty, also includes the slot right after whichever existing
    stop is geographically closest to `candidate` - inserting a new stop
    next to its nearest neighbor is usually cheap, since the two are
    already close together.
    """
    if not tour:
        return [0]

    candidate_row, candidate_col = candidate
    distances = [np.hypot(candidate_row - row, candidate_col - col) for row, col in tour]
    nearest_index = int(np.argmin(distances))

    return sorted({len(tour), nearest_index + 1})


def _greedy_repair(partial_tour: list, insertion_budget: int, candidate_pool: list, sample_size: int, evaluate_tour, rng: np.random.Generator) -> list:
    """
    See REPAIR OPERATORS ("greedy") in find_path_by_large_neighborhood_
    search's docstring. Runs up to insertion_budget rounds; each round
    samples sample_size not-yet-included candidates (_sample_candidates),
    tries each at its _candidate_positions, and commits whichever single
    (candidate, position) pairing gives the best resulting total_value.
    Stops early the moment a round finds nothing feasible to insert.
    """
    tour = list(partial_tour)

    for _ in range(insertion_budget):
        in_tour = set(tour)
        not_in_tour = [candidate for candidate in candidate_pool if candidate not in in_tour]
        if not not_in_tour:
            break

        sample = _sample_candidates(not_in_tour, sample_size, rng)

        best_value = None
        best_tour = None
        for candidate in sample:
            for position in _candidate_positions(tour, candidate):
                candidate_tour = tour[:position] + [candidate] + tour[position:]
                feasible, _, total_value, _ = evaluate_tour(candidate_tour)
                if feasible and (best_value is None or total_value > best_value):
                    best_value = total_value
                    best_tour = candidate_tour

        if best_tour is None:
            break  # nothing sampled this round was both feasible and reachable
        tour = best_tour

    return tour


def _regret_repair(partial_tour: list, insertion_budget: int, candidate_pool: list, sample_size: int, evaluate_tour, rng: np.random.Generator) -> list:
    """
    See REPAIR OPERATORS ("regret") in find_path_by_large_neighborhood_
    search's docstring. Like _greedy_repair, but each round picks the
    sampled candidate with the largest "regret" - the gap between its best
    and second-best feasible insertion - rather than the candidate with the
    single best insertion outright. A candidate with only one feasible
    position gets infinite regret (there is no second-best to fall back on
    - it must be taken now or the chance is gone), matching the classic
    regret-2 insertion heuristic.
    """
    tour = list(partial_tour)

    for _ in range(insertion_budget):
        in_tour = set(tour)
        not_in_tour = [candidate for candidate in candidate_pool if candidate not in in_tour]
        if not not_in_tour:
            break

        sample = _sample_candidates(not_in_tour, sample_size, rng)

        best_regret = None
        best_candidate_tour = None
        for candidate in sample:
            scored_positions = []
            for position in _candidate_positions(tour, candidate):
                candidate_tour = tour[:position] + [candidate] + tour[position:]
                feasible, _, total_value, _ = evaluate_tour(candidate_tour)
                if feasible:
                    scored_positions.append((total_value, candidate_tour))

            if not scored_positions:
                continue  # no feasible spot at all for this candidate this round

            scored_positions.sort(key=lambda entry: entry[0], reverse=True)
            best_value, best_position_tour = scored_positions[0]
            second_best_value = scored_positions[1][0] if len(scored_positions) > 1 else -np.inf
            regret = best_value - second_best_value

            if best_regret is None or regret > best_regret:
                best_regret = regret
                best_candidate_tour = best_position_tour

        if best_candidate_tour is None:
            break  # nothing sampled this round had even one feasible spot
        tour = best_candidate_tour

    return tour


def _roulette_pick(weights: dict, rng: np.random.Generator) -> str:
    """
    Picks one operator name from `weights`, at random, with probability
    proportional to its current weight - see ADAPTIVE WEIGHTS in
    find_path_by_large_neighborhood_search's docstring. Falls back to a
    plain uniform pick if every weight has somehow collapsed to zero.
    """
    names = list(weights.keys())
    raw_weights = np.array([weights[name] for name in names], dtype=np.float64)
    total_weight = raw_weights.sum()

    if total_weight <= 0:
        return names[int(rng.integers(len(names)))]

    probabilities = raw_weights / total_weight
    return names[rng.choice(len(names), p=probabilities)]


def _update_operator_weights(weights: dict, scores: dict, uses: dict, reaction_factor: float) -> None:
    """
    See ADAPTIVE WEIGHTS in find_path_by_large_neighborhood_search's
    docstring. Called once per segment_length rounds for the destroy
    weights, and separately for the repair weights. For every operator used
    at least once this segment, blends its weight toward its average score
    this segment by reaction_factor; an operator never picked this segment
    keeps its existing weight unchanged. Mutates `weights` in place and
    resets `scores`/`uses` to zero for the next segment.
    """
    for name in weights:
        if uses[name] > 0:
            average_score = scores[name] / uses[name]
            weights[name] = max(weights[name] * (1.0 - reaction_factor) + reaction_factor * average_score, 1e-6)
        scores[name] = 0.0
        uses[name] = 0


if __name__ == "__main__":
    pass
