# exact_pathfinding.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from helpers.constants import Constants
from pathfinding_algorithms import astar_pathfinding
import numpy as np

try:
    from ortools.sat.python import cp_model
except ImportError as import_error:
    cp_model = None
    _IMPORT_ERROR = import_error
else:
    _IMPORT_ERROR = None


# CP-SAT works over integer-coefficient linear expressions, not floats - see
# INTEGER SCALING in find_path_by_exact_solver's docstring. Costs and values
# in this project are floats (terrain-derived costs, 0-10 probabilities), so
# every cost/value fed into the model is multiplied by this factor and
# rounded to the nearest integer first.
_SCALE = 1000


def find_path_by_exact_solver(
    grid: CoordinatesGrid,
    start_row: int,
    start_col: int,
    max_cost: float,
    require_return_to_base: bool = True,
    blocked_mask: np.ndarray = None,
    time_limit_seconds: float = 60.0,
    max_targets: int = 60,
    seed: int = None,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    Exact solver, via Google OR-Tools' CP-SAT constraint solver - not a
    heuristic like every other function in this project, but a genuine
    optimality-seeking search: branch-and-bound with cutting planes, the
    same family of technique real MIP solvers (CBC, SCIP, Gurobi, CPLEX)
    use. Every other algorithm here can only ever tell you "this is the
    best answer I happened to find"; this one can tell you "this is
    PROVABLY the best answer that exists" - or, if it runs out of time
    first, "the best I found is at most this many points below whatever
    the true best is" (see SOLVE STATUS AND OPTIMALITY GAP).

    WHY THIS EXISTS
    -----------------
    None of this project's own algorithms (greedy, A*/NAMOA*, ACO, Tabu
    Search, VNS, GRASP, Simulated Annealing, the Genetic Algorithm, ALNS)
    can answer "how close to optimal is this?" - benchmark.py can only ever
    compare them against EACH OTHER. This function exists purely to answer
    that question on small/medium scenarios: run it alongside the others in
    benchmark.py and every heuristic's score can finally be read as "X% of
    the true optimum" instead of just "better or worse than last time."

    NOT REIMPLEMENTED FROM SCRATCH, DELIBERATELY
    ------------------------------------------------
    The actual search - branch-and-bound, LP relaxations, cutting planes,
    presolve - is entirely handled inside OR-Tools' compiled CP-SAT engine,
    not written here. This is deliberate: the whole point of an "exact"
    baseline is that its answer can be trusted as ground truth, and a
    hand-rolled branch-and-bound implementation would be slower, harder to
    get right, and - critically - could silently report a wrong "optimum"
    if it had a subtle bug, which would be worse than having no baseline at
    all. What IS written here is the modeling/glue layer: turning this
    project's grid-and-budget problem into the variables and constraints
    CP-SAT understands, and turning its solution back into a grid path.

    PROBLEM REDUCTION
    -------------------
    Solving directly on the raw grid (up to rows*cols nodes, 8 directed
    edges each) is hopeless for a general-purpose solver. Instead this
    reuses the exact same reduction every tour-based algorithm in this
    project already makes (see metaheuristic_pathfinding._evaluate_tour):
    collapse the problem down to just the start cell plus the real search
    targets (value > Constants.DEFAULT_PROBABILITY - the same candidate_
    pool construction find_path_by_tabu_search, find_path_by_grasp, etc.
    all use), and precompute the true cheapest travel cost between every
    pair of them with astar_pathfinding._a_star_path (_build_pairwise_
    costs) - already provably optimal for a single point-to-point trip, so
    nothing is lost by reducing to this smaller "complete graph" of K+1
    nodes (K real targets + the start). What CP-SAT then decides is which
    subset of targets to include and in what order - a well-studied problem
    in the operations-research literature (this is the classical
    Orienteering Problem; see Vansteenwegen et al., "The Orienteering
    Problem: A Survey", 2011).

    THE CP-SAT MODEL
    -------------------
    Every candidate target gets a boolean "self-loop" arc (node i -> node
    i), which CP-SAT's AddCircuit constraint treats as "this node opts out
    of the tour" - this is what lets the model choose a SUBSET of targets,
    not just an order for all of them. Every other pair of nodes gets a
    boolean "travel" arc, weighted by its true A* cost. AddCircuit then
    enforces, in one constraint, everything find_path_by_tabu_search's
    ADD/DROP/SWAP moves and _evaluate_tour's walking logic have to
    painstakingly maintain by hand: whichever nodes don't self-loop must
    form exactly one connected route (no disconnected mini-loops of
    targets that never link back to the start).

      - CLOSED TOUR (require_return_to_base=True): nodes are [start] +
        targets. The circuit naturally starts and ends at the start node,
        exactly like this project's other algorithms' return-trip
        accounting. Only real targets get a self-loop (opt-out) option -
        the start node is deliberately never allowed to self-loop, see
        SAFETY NET for why.

      - OPEN PATH (require_return_to_base=False): AddCircuit always
        produces a closed loop, so an "I don't have to come home" trip
        needs a small trick - a second, purely notional "end" node at the
        same coordinates as start is added. Every real node gets a free
        (cost 0) arc into "end", and "end" gets one free arc back to
        "start" to close the circuit for CP-SAT's bookkeeping. None of
        those free arcs represent real flight, so path reconstruction
        below stops the moment "end" is reached rather than walking them.
        A direct, always-available start->end arc (cost 0) guarantees the
        model is solvable even if literally nothing is reachable.

    Budget (max_cost) is one linear constraint: the sum of every CHOSEN
    real arc's cost (self-loops and the free end-node arcs contribute 0)
    must not exceed max_cost. The objective maximizes the sum of every
    non-self-looped target's value - this, and only this, is what CP-SAT
    is proving optimal (see WHAT "OPTIMAL" ACTUALLY MEANS HERE).

    SAFETY NET
    -----------
    Every other algorithm in this project has a trivial, always-feasible
    fallback: a single-cell path that never leaves the start (see e.g.
    find_path_by_tabu_search's `current_tour = []` starting point). The
    start node here is deliberately NEVER given a self-loop option, in
    either variant - AddCircuit only requires that whichever nodes do NOT
    self-loop form one connected circuit AMONG THEMSELVES, so letting start
    opt out would let the solver pick a circuit made entirely of profitable
    targets that never actually connects back to the depot: a valid-looking
    objective value with no physically sensible flight path behind it (an
    earlier version of this function had exactly this bug). Making start
    (and, in the open case, the "end" node) strictly mandatory is what
    forces every valid circuit to actually include the depot.

    For the OPEN case, a direct, always-available free start->end arc still
    guarantees "visit nothing" is a valid fallback solution. For the CLOSED
    case, a scenario where every real target is genuinely unreachable
    within budget (a tiny max_cost, or a blocked_mask that seals the start
    in) leaves start with no arc it can legally take at all - CP-SAT
    reports this as INFEASIBLE rather than finding a trivial empty circuit,
    and this function's own status check (see SOLVE STATUS AND OPTIMALITY
    GAP) catches exactly that and falls back to the same single-cell path
    every other algorithm in this project uses, printing a message instead
    of raising.

    WHAT "OPTIMAL" ACTUALLY MEANS HERE
    -------------------------------------
    The objective only credits a target's value if it is explicitly chosen
    as one of the circuit's real stops. It does NOT award incidental value
    for cells a chosen leg's shortest path merely happens to pass near or
    through on the way somewhere else - unlike this project's other
    tour-based algorithms, whose walks (greedy_pathfinding._walk_toward_
    target) DO opportunistically pick up any newly-touched cell's value
    along the way (see _evaluate_tour). This function's final total_value
    (see PATH RECONSTRUCTION) is computed the exact same opportunistic way
    as every other algorithm, for a fair, apples-to-apples benchmark.py
    comparison - so it can, occasionally, come out a little HIGHER than
    the model's own internal objective value, never lower. What CP-SAT is
    provably optimal about, precisely, is: "no other choice of which
    targets to schedule, and in what order, could have collected more
    target value under this budget, given the true cheapest travel cost
    between any two chosen stops." That is a meaningful, honest baseline
    even though it is not a proof about incidental pickups too - chasing
    that stronger (and messier) guarantee was judged not worth it for what
    is meant to be a comparison baseline, not a ninth production algorithm.

    PATH RECONSTRUCTION
    ----------------------
    CP-SAT hands back which arcs were chosen, not a grid path. _build_
    pairwise_costs already cached each pair's actual A*-found path_cells
    alongside its cost, so reconstruction just follows the chosen arcs from
    the start node to build the sequence of real stops, then stitches
    together each leg's cached path_cells (and, for a closed tour, the
    final real leg back to the start) exactly the way astar_pathfinding.
    _value_gained_along_path/metaheuristic_pathfinding._evaluate_tour
    already do for every other algorithm - a cell's value is only ever
    credited the first time any leg actually reaches it.

    SOLVE STATUS AND OPTIMALITY GAP
    -----------------------------------
    Every call prints CP-SAT's solve status (OPTIMAL vs FEASIBLE-but-time-
    limited), its objective value, its best remaining bound, and the gap
    between them - this project's return-value contract is a plain (path,
    total_value, cost_used) tuple, matching every other PATHFINDING_
    ALGORITHMS entry so TrajectoryGenerator/benchmark.py can call this
    identically, so this diagnostic information is printed rather than
    returned. OPTIMAL means the result is provably the best possible;
    FEASIBLE means time_limit_seconds ran out first, and the printed gap
    is how far from optimal the returned answer could still be (this is
    still far more informative than any other algorithm in this project
    can offer, which report no such bound at all).

    SCALING - WHERE THIS STOPS BEING PRACTICAL
    -----------------------------------------------
    Two costs stack: the pairwise cost precompute is O(K^2) A* searches on
    the real grid, and the Orienteering Problem CP-SAT is solving underneath
    is itself NP-hard. For this project's normal scenarios (~15 targets)
    both are fast. For its "_dense" scenario files (up to 510 targets), the
    precompute ALONE would mean on the order of a quarter-million A*
    searches before the solver even starts - impractical. max_targets
    (default 60, chosen to keep the precompute in the tens-of-thousands-of-
    A*-calls range) guards against silently attempting that: if
    len(candidate_pool) exceeds it, this prints a message and returns the
    same trivial single-cell fallback rather than hanging - pass a larger
    max_targets explicitly to attempt a bigger scenario anyway, at your own
    patience's risk.

    INTEGER SCALING
    -----------------
    CP-SAT's linear constraints/objective require integer coefficients, but
    this project's costs and values are floats. Every cost and value is
    multiplied by _SCALE and rounded to the nearest integer before being
    handed to the model (and divided back by _SCALE when printing solve
    diagnostics) - a standard, minor precision trade-off for using an
    integer solver on a naturally continuous-valued problem.
    """
    if cp_model is None:
        print(f"find_path_by_exact_solver requires the 'ortools' package, which isn't installed ({_IMPORT_ERROR}) - run `pip install ortools`.")
        return [(start_row, start_col)], float(grid.coordinates_values[start_row, start_col]), 0.0

    values = grid.coordinates_values

    candidate_rows, candidate_cols = np.nonzero(values > Constants.DEFAULT_PROBABILITY)
    candidate_pool = [
        (int(candidate_row), int(candidate_col))
        for candidate_row, candidate_col in zip(candidate_rows, candidate_cols)
        if (int(candidate_row), int(candidate_col)) != (start_row, start_col)
        and (blocked_mask is None or not blocked_mask[candidate_row, candidate_col])
    ]

    if not candidate_pool:
        return [(start_row, start_col)], float(values[start_row, start_col]), 0.0

    if len(candidate_pool) > max_targets:
        print(
            f"find_path_by_exact_solver: {len(candidate_pool)} real targets exceeds max_targets={max_targets} "
            "- exact solving would likely take far too long, skipping. Pass a larger max_targets explicitly "
            "to attempt it anyway."
        )
        return [(start_row, start_col)], float(values[start_row, start_col]), 0.0

    # index 0 = start; 1..len(candidate_pool) = real targets - see PROBLEM REDUCTION.
    nodes = [(start_row, start_col)] + candidate_pool
    node_count = len(nodes)
    target_indices = range(1, node_count)

    cost_matrix, path_matrix = _build_pairwise_costs(grid, nodes, blocked_mask)

    model = cp_model.CpModel()
    circuit_arcs = []
    arc_cost_terms = []

    def add_arc(tail, head, cost):
        """
        Creates one CP-SAT boolean "is this arc used" literal, registers it
        for AddCircuit and (if cost is nonzero) adds its scaled cost to the
        running budget sum - see INTEGER SCALING above for why costs are
        scaled/rounded to integers first.
        """
        literal = model.NewBoolVar(f"arc_{tail}_{head}")
        circuit_arcs.append((tail, head, literal))
        if cost:
            arc_cost_terms.append(int(round(cost * _SCALE)) * literal)
        return literal

    self_loop_literals = {}

    if require_return_to_base:
        end_index = None
        for i in target_indices:  # only real targets may opt out - start is mandatory, see SAFETY NET
            self_loop_literals[i] = add_arc(i, i, 0.0)
    else:
        end_index = node_count
        for i in target_indices:  # only real targets may opt out - start/end are mandatory
            self_loop_literals[i] = add_arc(i, i, 0.0)

    for i in range(node_count):
        for j in range(node_count):
            if i == j:
                continue
            cost = cost_matrix[i][j]
            if cost is None:
                continue  # j is unreachable from i at all - see _build_pairwise_costs
            add_arc(i, j, cost)

    if end_index is not None:
        add_arc(0, end_index, 0.0)  # always-available empty-tour arc - see SAFETY NET
        for i in target_indices:
            add_arc(i, end_index, 0.0)
        add_arc(end_index, 0, 0.0)  # structural wraparound, never walked in PATH RECONSTRUCTION

    model.AddCircuit(circuit_arcs)
    model.Add(sum(arc_cost_terms) <= int(round(max_cost * _SCALE)))

    visited_literals = {}
    objective_terms = []
    for i in target_indices:
        visited = model.NewBoolVar(f"visited_{i}")
        model.Add(visited + self_loop_literals[i] == 1)
        visited_literals[i] = visited
        target_row, target_col = nodes[i]
        objective_terms.append(int(round(float(values[target_row, target_col]) * _SCALE)) * visited)

    model.Maximize(sum(objective_terms))

    solver = cp_model.CpSolver()
    solver.parameters.max_time_in_seconds = time_limit_seconds
    if seed is not None:
        solver.parameters.random_seed = seed
    status = solver.Solve(model)

    if status not in (cp_model.OPTIMAL, cp_model.FEASIBLE):
        print(f"find_path_by_exact_solver: solve status {solver.StatusName(status)} - falling back to a no-op path.")
        return [(start_row, start_col)], float(values[start_row, start_col]), 0.0

    best_bound = solver.BestObjectiveBound() / _SCALE
    objective_value = solver.ObjectiveValue() / _SCALE
    gap_pct = 0.0 if best_bound <= 0 else 100.0 * (best_bound - objective_value) / best_bound
    print(
        f"find_path_by_exact_solver: status={solver.StatusName(status)}, "
        f"objective={objective_value:.2f}, best_bound={best_bound:.2f}, "
        f"gap={gap_pct:.2f}%, wall_time={solver.WallTime():.2f}s"
    )

    return _reconstruct_path(grid, nodes, solver, circuit_arcs, cost_matrix, path_matrix, require_return_to_base, end_index)


def _build_pairwise_costs(grid: CoordinatesGrid, nodes: list, blocked_mask: np.ndarray):
    """
    The true cheapest travel cost (and the grid cells actually walked)
    between every ordered pair of `nodes`, via astar_pathfinding._a_star_
    path - see PROBLEM REDUCTION in find_path_by_exact_solver's docstring.
    Costs are NOT assumed symmetric (WeightsGrid tracks a separate cost per
    direction, so i->j and j->i are computed independently, exactly like
    every other algorithm in this project treats terrain cost).

    Returns (cost_matrix, path_matrix): node_count x node_count nested
    lists (diagonal and any genuinely unreachable pair left as None in
    both).
    """
    rows, cols = grid.rows, grid.cols
    expand_cap = rows * cols
    node_count = len(nodes)

    cost_matrix = [[None] * node_count for _ in range(node_count)]
    path_matrix = [[None] * node_count for _ in range(node_count)]

    for i in range(node_count):
        start_row_i, start_col_i = nodes[i]
        for j in range(node_count):
            if i == j:
                continue
            target_row_j, target_col_j = nodes[j]
            reached, path_cells, cost = astar_pathfinding._a_star_path(
                grid, start_row_i, start_col_i, target_row_j, target_col_j, blocked_mask, expand_cap,
            )
            if reached:
                cost_matrix[i][j] = cost
                path_matrix[i][j] = path_cells

    return cost_matrix, path_matrix


def _reconstruct_path(
    grid: CoordinatesGrid,
    nodes: list,
    solver,
    circuit_arcs: list,
    cost_matrix: list,
    path_matrix: list,
    require_return_to_base: bool,
    end_index,
) -> tuple[list[tuple[int, int]], float, float]:
    """
    See PATH RECONSTRUCTION in find_path_by_exact_solver's docstring. Reads
    which arcs CP-SAT actually chose (solver.Value(literal) == 1), follows
    them from the start node (0) to recover the ordered sequence of real
    stops, then stitches each leg's cached path_cells together - crediting
    a cell's value only the first time any leg reaches it, matching
    astar_pathfinding._value_gained_along_path's convention.
    """
    values = grid.coordinates_values
    node_count = len(nodes)

    chosen_next = {}
    for tail, head, literal in circuit_arcs:
        if tail != head and solver.Value(literal):
            chosen_next[tail] = head

    stop_sequence = [0]
    current = 0
    for _ in range(node_count + 1):  # a valid circuit can never need more hops than this - safety cap
        next_node = chosen_next.get(current)
        if next_node is None or next_node == 0 or next_node == end_index:
            break
        stop_sequence.append(next_node)
        current = next_node

    start_row, start_col = nodes[0]
    path_cells = [(start_row, start_col)]
    total_value = float(values[start_row, start_col])
    cost_used = 0.0
    visited_mask = np.zeros_like(values, dtype=bool)
    visited_mask[start_row, start_col] = True

    def walk_leg(tail_index, head_index):
        """
        Appends the cached leg from tail_index to head_index (see
        _build_pairwise_costs) onto path_cells, crediting a cell's value
        only the first time any leg reaches it (visited_mask), and adds
        the leg's real cost to cost_used.
        """
        nonlocal total_value, cost_used
        for cell in path_matrix[tail_index][head_index]:
            if not visited_mask[cell]:
                total_value += float(values[cell])
                visited_mask[cell] = True
            path_cells.append(cell)
        cost_used += cost_matrix[tail_index][head_index]

    for tail_index, head_index in zip(stop_sequence, stop_sequence[1:]):
        walk_leg(tail_index, head_index)

    if require_return_to_base and len(stop_sequence) > 1:
        walk_leg(stop_sequence[-1], 0)

    return path_cells, total_value, cost_used


if __name__ == "__main__":
    pass
