# Flight Trajectory Generation — Project Guide

*A Polish translation of this document is available at [`guide.pl.md`](guide.pl.md).*

## 1. What this project does

This project plans a single drone's search flight for a search-and-rescue (SAR) mission. The area to search is modeled as a grid of cells. Each cell carries a **value**: the probability that the missing person is located there. Moving from one cell to a neighboring one costs a certain amount of energy/time (a **cost**), derived from the real terrain the drone would fly over — climbing costs more than descending, and a longer hop costs more than a short one. The drone has a fixed **budget** (a maximum total cost it can spend) and, in most missions, must also make it back to its launch point before running out of budget.

The task the project solves is therefore: **starting from a fixed base cell, find a route through the grid that collects as much probability value as possible without spending more than the available budget** (and, if required, while still affording the trip home). This is a classic problem in operations research called the **[Orienteering Problem](https://en.wikipedia.org/wiki/Orienteering_problem)** — picking a profitable *subset* of prizes to collect under a budget, rather than a Travelling-Salesman-style requirement to visit everything.

There is no single "correct" way to solve this problem well and fast at the same time — it is NP-hard, so an algorithm that is guaranteed to find the true best answer can be slow on a large map, while a fast algorithm can only ever promise a *good* answer, not necessarily the *best* one. This project implements **16 different algorithms** that make that trade-off differently, from the simplest possible greedy rules up to a genuine mathematically-optimal solver, so they can be run and compared against one another on the same scenario.

## 2. Core concepts

Understanding a few shared building blocks makes every algorithm description below much easier to follow.

- **The grid.** A `rows × cols` array of cells. Every cell is connected to up to 8 neighbors (the compass directions: up, up-right, right, down-right, down, down-left, left, up-left).
- **Value (`coordinates_values`).** Every cell's probability of hiding the searched person, between `0.0` and `Constants.MAX_PROBABILITY` (`10.0`). A cell nobody has set an explicit probability for defaults to `Constants.DEFAULT_PROBABILITY` (`0.0`) — "no information", not a real signal. Almost every algorithm treats a cell as a genuine **target** only if its value is *above* this threshold; untouched background cells are never deliberately sought out.
- **Cost / weights (`WeightsGrid`).** A `rows × cols × 8` array giving the cost of moving from a cell in each of the 8 directions. It is built from real local terrain coordinates (`x, y, z` in meters): a move costs a flat rate per meter of horizontal distance, plus a grade penalty proportional to `(elevation change)² / horizontal distance` — climbing adds to that cost, descending discounts it (down to a floor, so a long descent is never free or "profitable"). This is why a diagonal move, or a climb, is never assumed to cost the same as a short flat step — the cost model reflects genuine terrain.
- **`blocked_mask`.** A boolean grid of no-fly cells (storm cells, restricted airspace) that no algorithm is ever allowed to step onto.
- **Budget (`max_cost`) and `require_return_to_base`.** The total cost ceiling for the whole mission. When return is required, every algorithm reserves enough remaining budget for the cheapest possible route home before committing to any further move — so the drone can never strand itself past the point of no return.
- **The start cell.** Fixed by the mission (e.g. the rescue team's base). Every algorithm begins here; the project also validates at scenario-load time that the start cell isn't itself listed as blocked.
- **Targets / candidate pool.** Cells whose value is above `Constants.DEFAULT_PROBABILITY`. Most algorithms in this project (everything except the simplest greedy walker) restrict their target-picking specifically to this pool, rather than considering literally every grid cell — this is what keeps them from wasting search effort comparing meaningless, information-free background cells against one another.
- **Ground truth vs. prior belief.** `searched_person_locations` (with their `probability`) is only ever a *belief* about where the person might be — it is what every algorithm's target-seeking logic actually sees and searches over. `actual_person_locations` is a separate, optional list of where the person(s) *really* are. It plays no part in any algorithm's own decisions; it only adds one extra, algorithm-independent stopping rule on top of whatever algorithm ran — see **Stopping once everyone is found**, right before section 4.1.

## 3. How to use the project

### 3.1 Project layout

```
src/
  coordinates_grid/        the grid data structures + terrain/mask generation
  pathfinding_algorithms/  every algorithm described in section 4
  helpers/                 constants, JSON loading, benchmarking
  gui/                     the matplotlib/PyQt5 visualizer
  main.py                  a single end-to-end example run
test_data/                 example mission scenarios (JSON)
drone_data/                example drone configurations (JSON)
requirements.txt           numpy, ortools, matplotlib, PyQt5
```

### 3.2 Describing a mission (scenario JSON, `test_data/*.json`)

```json
{
  "terrain": {
    "rows": 100, "cols": 100,
    "altitude_range": [0.0, 20.0],
    "cell_size_meters": 5.0,
    "max_gradient": 0.001,
    "seed": 1
  },
  "start_location": {"row": 0, "col": 0},
  "searched_person_locations": [
    {"row": 3, "col": 5, "probability": 9.5}
  ],
  "blocked_cells": [
    {"row": 10, "col": 10}
  ],
  "actual_person_locations": [
    {"row": 3, "col": 5}
  ]
}
```

`terrain` describes the grid size and how its random elevation is generated (`max_gradient` caps how steep two neighboring cells can be — a slope ratio, not a percentage; `seed` makes that random elevation reproducible — the same scenario file always regenerates the exact same map). `start_location` is the fixed launch/return point. `searched_person_locations` are the real targets (a prior belief, with probabilities). `blocked_cells` is optional and lists permanent no-fly cells — **the start cell may never be one of them**; the loader raises an error immediately if it is. `actual_person_locations` is optional and lists where the person(s) actually are (ground truth, no probability needed, zero or more entries) — see **Stopping once everyone is found** before section 4.1.

### 3.3 Describing a drone (`drone_data/*.json`)

```json
{
  "climb_cost_per_meter": 0.5,
  "descent_cost_per_meter": 0.3,
  "base_cost": 1.0,
  "max_cost": 5000.0,
  "require_return_to_base": true
}
```

`base_cost` is the flat travel-cost-per-meter rate; `climb_cost_per_meter`/`descent_cost_per_meter` scale the grade penalty; `max_cost` is the mission budget.

### 3.4 Running a single mission

`src/main.py` loads a scenario and a drone config, builds the grid, runs one algorithm (`TrajectoryGenerator.find_best_path(..., algorithm="grasp")`), prints a one-line summary, and opens a GUI window with a 2D top-down view (and a 3D terrain view, if terrain coordinates were generated) showing the search-probability heat map, the chosen path, its direction of travel, and any blocked cells.

```bash
cd src
python main.py
```

Any of the 16 keys listed in section 4 can be passed as `algorithm=` — see `pathfinding_algorithms/trajectory_generator.py`'s `PATHFINDING_ALGORITHMS` dictionary for the authoritative list.

### 3.5 Comparing algorithms (`helpers/benchmark.py`)

`run_benchmark(...)` runs a chosen set of algorithms (default: all of them) against a chosen set of scenario files (default: everything in `test_data/`), any number of repetitions each, and prints/saves a results table (steps, cost used, % of budget used, total value collected, targets hit, runtime). Since several algorithms are randomized, running more than one repetition also prints a mean ± standard deviation summary, so you can see how much a given algorithm's result actually varies from run to run.

```python
from helpers.benchmark import run_benchmark
run_benchmark(algorithms=["grasp", "tabu_search", "exact_solver"], scenario_files=["../test_data/scenario1.json"])
```

## 4. How every pathfinding algorithm works

All 16 algorithms share the exact same call signature and return shape — `(grid, start_row, start_col, max_cost, require_return_to_base=True, blocked_mask=None, ...) → (path, total_value, cost_used)` — so any of them can be dropped into `TrajectoryGenerator` or `run_benchmark` interchangeably. What differs is entirely *how* each one decides where to go next.

They fall into four families: simple **greedy heuristics**, **A\* search** variants that route optimally between chosen targets, **metaheuristics** that iteratively improve a whole tour, and one **exact solver** that proves optimality outright.

**Stopping once everyone is found.** `TrajectoryGenerator.find_best_path` accepts an optional `actual_person_locations` argument — the mission's ground truth (see **Ground truth vs. prior belief** in section 2) — and adds exactly one extra rule on top of whichever algorithm just ran, uniformly, without modifying any algorithm's own logic at all: once the returned path has actually passed through every one of those cells, everything after that point is trimmed off and replaced with a fresh, short walk straight home (if `require_return_to_base`). This works for all 16 algorithms unchanged because every one of them already returns the *full*, cell-by-cell route the drone actually flies, not just a list of waypoints — so the check is just a scan over an already-computed path. It also means a person is recognized as found the instant their cell is entered, even mid-flight toward a completely different target, not only when an algorithm finishes whatever leg it happened to be on. If `actual_person_locations` is empty or omitted, nothing here changes anything — every algorithm behaves exactly as documented below. Left unresolved-by-the-time-the-path-ends locations (including ones that coincide with a blocked cell, and so can never actually be entered) simply mean this rule never triggers, and the algorithm's own original result is returned untouched.

One nuance worth being precise about: for `exact_solver`, this does not change what it proved optimal. Its optimization has no notion of `actual_person_locations` at all — it still maximizes probability-weighted value collected under budget, and that proof stands regardless. Trimming its returned path once it happens to pass every real person is exactly the same cosmetic, after-the-fact step applied to every other algorithm's output, not a change to the underlying search or its guarantee.

---

### 4.1 Greedy heuristics (`pathfinding_algorithms/greedy_pathfinding.py`)

These four algorithms never look more than one decision ahead, and never reconsider a choice once made. They are the fastest and simplest algorithms in the project, and serve as an easy-to-understand baseline for everything else.

#### 4.1.1 `greedy` — highest-value neighbor

**Simple explanation:** at every step, look at the (up to) 8 cells right next to the drone, and step onto whichever one is worth the most — like a hiker who, at every fork, always takes the path that visibly looks best from where they're standing.

**In detail:** from the current cell, every one of its 8 neighbors is scored by its raw value (an already-visited neighbor is scored as "worthless", so revisiting is only ever a last resort). Whichever affordable, unblocked neighbor scores highest is taken. If two or more neighbors tie on value — extremely common, since most of the grid is untouched background — the **cheaper** of the tied moves is preferred.

**How it picks targets:** it does not pick a "target" in the way the other algorithms do at all — there is no destination in mind, only a one-step-ahead comparison of the immediate neighborhood, repeated over and over.

**When it finishes:** the moment no affordable, unblocked move is left (accounting for the reserved cost of the trip home, if required) — including moves that only revisit already-covered ground. Unlike every other algorithm below, it never stops early just because nothing *fresh* is reachable: as long as *some* affordable move exists, even a pure re-tread of old ground, it keeps going until the budget is genuinely exhausted. This is a deliberate design choice — a real sweep of unmapped territory can still be worth flying, even where no positive-probability signal has been set yet.

**Other notes:** because it never looks beyond its immediate neighbors, it has no way to plan around being boxed in by its own visited trail, and can end up wandering through already-covered ground for a while before finding fresh territory (or running out of budget first).

**Reference:** the general "always take the locally best option" greedy paradigm is a standard building block of algorithm design — see e.g. T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, ch. "Greedy Algorithms"; and for its application specifically to orienteering-style problems, B. L. Golden, L. Levy, R. Vohra, "The orienteering problem", *Naval Research Logistics*, 1987.

#### 4.1.2 `direct_to_highest_value` — walk straight to the best cell, repeat

**Simple explanation:** find the single most valuable unvisited cell on the whole map, walk straight there, then repeat — like planning a shopping trip stop by stop, always heading next to the best remaining item on the list.

**In detail:** the entire grid's cells are scanned for the current highest-value one that's still unvisited, unblocked, and *above the "real target" threshold* (background cells are never chased once every real target is gone — see section 4.6 for why this specific guard matters). The drone walks toward it using a simple, fast routing rule: move diagonally until aligned on one axis, then move straight, rotating step-by-step around any obstacle in the way. If that walk (plus the reserved trip home) still fits the budget, it's committed and every cell passed through — not just the destination — has its value credited. Then the search repeats for the next-highest remaining target.

**How it picks targets:** always the single highest-value real target left on the entire grid, regardless of how far away it is.

**When it finishes:** the first time the current highest-value target can't be fully reached — whether because it's boxed in by obstacles the routing rule can't work around, or because the walk plus the trip home wouldn't fit the remaining budget. There is no fallback to a lesser target; it simply stops.

**Other notes:** the routing rule (rotate around one obstacle cell at a time) is a cheap heuristic, not a full search — it can fail to find a way around a large or maze-like obstacle even when a route does exist.

**Reference:** T. Tsiligirides, "Heuristic Methods Applied to Orienteering", *Journal of the Operational Research Society*, 35(9), 1984 — one of the original "always go to the best remaining prize" orienteering heuristics.

#### 4.1.3 `value_cost_ratio` — best value for the cost, tournament-style

**Simple explanation:** instead of always chasing the single richest target regardless of how far it is, this algorithm asks "which nearby target gives the best payoff *per unit of effort*?" — like choosing your next errand by how much it accomplishes relative to how far out of the way it is, not just by which errand is biggest.

**In detail:** the highest-value cells are pre-sorted once. Each round, a small number of not-yet-visited candidates (2 by default) are drawn from that sorted order, each is speculatively walked to, and its **value collected ÷ cost of the walk** ratio is computed. Whichever candidate has the best ratio this round is committed; the round's other candidate(s), if still reachable, are kept and re-challenged next round against a freshly drawn one — so a candidate isn't eliminated just for losing once.

**How it picks targets:** whichever reachable candidate — among a small, rotating tournament pool — currently offers the best value-collected-per-cost-spent, not simply the highest raw value or the lowest cost.

**When it finishes:** once every candidate in a round is unreachable within the remaining budget, with no fallback.

**Other notes:** a candidate that fails to be reached in a given round is dropped permanently rather than retried, since the drone's position only moves forward and the remaining budget only shrinks — so nothing about its reachability could improve later.

**Reference:** B. L. Golden, L. Levy, R. Vohra, "The orienteering problem", *Naval Research Logistics*, 34(3), 1987 — value/cost-ratio-based greedy construction is a standard orienteering-problem heuristic from this line of work.

#### 4.1.4 `lowest_cost` — cheapest reachable target first

**Simple explanation:** always fly to whichever real target is currently *cheapest* to reach, saving the expensive ones for later (or never).

**In detail:** the same tournament structure as `value_cost_ratio` above, but candidates are compared on raw walking **cost** alone rather than a value/cost ratio — whichever reachable candidate this round is cheapest to reach wins. A wider tournament pool (50 candidates by default, vs. 2 for the ratio version) is used specifically because plain cost comparisons among many similarly-cheap background-adjacent cells would otherwise make the walker systematically prefer whichever tied cell happened to be reachable in a perfectly straight line, producing an unnatural path that hugs grid rows/columns instead of heading toward genuinely valuable areas.

**How it picks targets:** the reachable candidate with the lowest walking cost, full stop — value plays no role in the choice itself (only in which cells count as candidates at all).

**When it finishes:** once every candidate in a round is unreachable (too expensive, or physically unreachable) within the remaining budget.

**Other notes:** this tends to front-load a mission with easy, nearby wins and can leave a lot of budget on the table if the remaining unvisited targets are all comparatively expensive — it never weighs "is this worth its cost", only "is this cheap".

**Reference:** same orienteering-problem heuristic family as 4.1.2/4.1.3 above (Golden, Levy & Vohra, 1987); this variant corresponds to a pure nearest-neighbor-style construction rule.

---

### 4.2 A* search variants (`pathfinding_algorithms/astar_pathfinding.py`)

The four greedy algorithms above route between cells with a cheap, approximate "rotate around one obstacle at a time" heuristic — fast, but not guaranteed to find the truly cheapest route, and prone to getting stuck on complex obstacle layouts. The four algorithms in this family swap that heuristic out for a genuine **[A\* search](https://en.wikipedia.org/wiki/A*_search_algorithm)**: a real graph search that is *guaranteed* to find the cheapest possible route between any two cells (never overestimating the true remaining cost thanks to an *admissible heuristic* — the straight-line distance times the cheapest possible per-step cost anywhere on the map), at the price of exploring more of the grid to find it.

Three of the four (`a_star_to_highest_value`, `a_star_value_cost_ratio`, `a_star_lowest_cost`) are otherwise identical in their target-picking logic to their greedy counterparts in section 4.1 — only the routing between chosen points changes.

#### 4.2.1 `a_star_to_highest_value`

Same target-picking logic as `direct_to_highest_value` (4.1.2): repeatedly seek the single highest-value real target left on the grid. The only difference is *how* it gets there — via true A* search instead of the "rotate around an obstacle" heuristic — so it can navigate a large or maze-like `blocked_mask` the simpler heuristic might fail to route around, and is guaranteed to find the actual cheapest path whenever more than one route exists. It finishes under the same condition as 4.1.2: the moment the current best target can't be fully, affordably reached, with no fallback to a lesser one.

#### 4.2.2 `a_star_value_cost_ratio`

Same tournament logic as `value_cost_ratio` (4.1.3) — small rotating pool of candidates, best value-collected-per-cost wins each round — but every candidate's cost is the true A*-optimal cost rather than the cheaper heuristic's estimate. Finishes the same way: once no candidate in a round is reachable within budget.

#### 4.2.3 `a_star_lowest_cost`

Same tournament logic as `lowest_cost` (4.1.4) — wide rotating pool (50 by default), cheapest reachable candidate wins each round — with true A*-optimal costs. Finishes the same way.

#### 4.2.4 `namoa_star` — multi-objective A*

**Simple explanation:** imagine planning a walk where you care about *two* things at once — how tired you'll get, and how many stickers you'll collect along the way from machines scattered around town. A normal shortest-path planner only ever minimizes tiredness, and would never consider a slightly longer route that scoops up five stickers for almost no extra effort — it doesn't even think about stickers. **[NAMOA\*](https://en.wikipedia.org/wiki/Multi-objective_optimization)** ("New Approach to Multi-Objective A\*") is smarter: instead of finding *one* route, it finds a whole menu of genuinely good trade-offs — for every worthwhile "how many stickers" amount, the least-tiring route that achieves it — and then picks whichever entry on that menu suits the goal best.

**In detail:** this uses the same tournament structure as `a_star_value_cost_ratio` (4.2.2), but instead of finding only the single *cheapest* route to a candidate (as plain A* does), it runs a full multi-objective search that tracks every **non-dominated** (cost, value-collected) trade-off reaching that candidate — a route is kept only if no other route to the same point is both no more expensive *and* no less valuable. Among that whole menu of trade-offs for a candidate, whichever point has the best value/cost ratio is used to score it for the round. Because a plain shortest-path A* search can never even consider taking a costlier detour that happens to sweep up extra incidental value along the way, NAMOA* can occasionally find a genuinely better answer for this project's actual goal (collect value under budget) than pure cheapest-route A* can.

**How it picks targets:** the same small rotating tournament as 4.2.2, scored by the best available value/cost point on each candidate's full Pareto-optimal trade-off menu rather than a single cheapest-route number.

**When it finishes:** once no candidate in a round has an affordable, reachable trade-off point left — the same stopping rule as the other tournament-style algorithms.

**Other notes:** tracking a whole set of trade-offs per cell, rather than one best number, makes this by far the most computationally expensive algorithm in the project — on a large, spread-out scenario a single candidate evaluation can take seconds even with aggressive pruning of clearly-inferior partial routes. It's the right tool when the value/cost trade-off itself is genuinely the point, or the search space is small enough to explore thoroughly; for large scenarios with many spread-out targets, `a_star_value_cost_ratio` or plain `value_cost_ratio` give reliably faster, still-strong results.

**Reference:** L. Mandow, J. L. Pérez-de-la-Cruz, "A New Approach to Multiobjective A\* Search", *Proceedings of IJCAI 2005*; building on the original A* paper: P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 1968.

---

### 4.3 Metaheuristics (`pathfinding_algorithms/metaheuristic_pathfinding.py`)

Every algorithm so far commits to a decision and never revisits it. The seven algorithms in this family instead work with a **whole candidate tour at once** — an ordered list of real targets to visit — and repeatedly try to *improve* it, sometimes even accepting a temporarily worse tour on purpose, specifically so they can escape a decision that looked locally unbeatable but wasn't actually the best available. Six of the seven (all but Ant Colony Optimization) share the exact same solution representation and the exact same three basic moves — **ADD** an unvisited target, **DROP** a visited one, **SWAP** the order of two — and the grid-level flight path between chosen targets is always worked out the same way (the same "rotate around an obstacle" routing the greedy algorithms use). What differs between them is purely the *strategy* for exploring that space of possible tours.

Because most of these involve randomness, running them more than once on the same scenario (e.g. via `run_benchmark`'s `repetitions`) will generally produce slightly different — though usually similarly good — results each time.

#### 4.3.1 `ant_colony` — Ant Colony Optimization (ACO)

**Simple explanation:** modeled on how real ant colonies find short paths between their nest and food. Each ant wanders somewhat randomly, laying down a scent trail (pheromone) as it goes. The scent fades over time, but a shorter or better path lets more ants complete a round trip per unit of time, so it accumulates scent faster than a worse one — which makes it more attractive to the next ants, which reinforces it further. No single ant ever sees the whole colony's knowledge; the shared, fading trail is what lets the colony converge on a good path collectively.

**In detail:** for a fixed number of iterations, every one of several ants independently builds one complete candidate path step by step, starting from base. At each step, standing at a cell, the next cell is chosen **randomly among every feasible neighbor**, but weighted by `(pheromone on that edge)^alpha × (attraction of the destination ÷ cost of the move)^beta` — so a step is more likely if the edge has either a strong existing pheromone trail, or looks good in its own right. Because a step-by-step walk can only ever "see" its immediate 8 neighbors, an **attraction field** is precomputed once up front: every real target's value radiates outward with exponential distance decay, so a cell several steps away from a valuable target still looks locally attractive well before the ant is actually standing next to it — without this, a sparse map of targets would be nearly invisible to a purely local, one-step-at-a-time decision rule. Once every ant in a round has finished, all pheromone evaporates by a fixed fraction, then every ant deposits fresh pheromone on the edges it actually used, proportional to how good (value collected per unit cost) its finished path was.

**How it picks targets:** it doesn't explicitly "pick targets" the way the tour-based algorithms below do — each ant just walks step by step, guided by the combination of pheromone and the attraction field, naturally gravitating toward valuable areas of the map without ever committing to a specific destination in advance.

**When it finishes:** each ant's walk stops when it has no feasible neighbor left (see *other notes* for the fallback when this happens prematurely); the overall search finishes after a fixed number of iterations, and returns the single best path any ant found across the whole run.

**Other notes:** because an ant only ever steps onto an adjacent, not-yet-visited cell, it can in principle trap itself inside its own already-covered trail well before its budget runs out. A repair mechanism kicks in when this happens: the ant looks at the "frontier" (unvisited cells that border ground it's already covered), tries walking toward a handful of the most attractive frontier cells, and resumes from whichever turns out reachable — only giving up entirely if none of those attempts succeed either.

**Reference:** M. Dorigo, *Optimization, Learning and Natural Algorithms*, PhD thesis, Politecnico di Milano, 1992; M. Dorigo, L. M. Gambardella, "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem", *IEEE Transactions on Evolutionary Computation*, 1(1), 1997.

#### 4.3.2 `tabu_search` — Tabu Search

**Simple explanation:** follows a **single** evolving candidate tour and repeatedly asks "what's the best nearby tweak I can make to what I already have?" Its defining trick is a short-term memory of moves it just made, which it refuses to immediately undo — this is what lets it accept a *worse* move on purpose sometimes, to climb out of a dead end a purely uphill-only search would get stuck in forever.

**In detail:** each round, every ADD/DROP/SWAP move reachable from the current tour is generated, infeasible ones (over budget) are discarded, and whichever *feasible, non-tabu* move scores best is taken — even if that's a step backward in value. The move just taken has its reverse marked "tabu" (forbidden) for a fixed number of rounds, specifically so the search can't immediately regret and undo its own last decision — unless that reversal would actually beat the best tour ever seen in the whole run, in which case it's allowed anyway (the "aspiration criterion": a good enough result is allowed to break the rule).

**How it picks targets:** among every reachable ADD/DROP/SWAP neighbor of the current tour, whichever scores best this round, subject to the tabu list.

**When it finishes:** after a fixed number of rounds, or immediately if every candidate move in a round is either infeasible or blocked by the tabu list — whichever comes first. The single best tour seen at *any* point during the run is what's returned, not necessarily wherever the search happens to be standing at the end (it may have wandered to a worse spot since).

**Other notes:** evaluating every candidate move means re-walking the entire tour from scratch each time — fine for the tens-to-low-hundreds of real targets a typical scenario has, but this doesn't scale gracefully to a grid with thousands of individually valuable cells.

**Reference:** F. Glover, "Future Paths for Integer Programming and Links to Artificial Intelligence", *Computers & Operations Research*, 13(5), 1986; F. Glover, M. Laguna, *Tabu Search*, Kluwer Academic Publishers, 1997.

#### 4.3.3 `variable_neighborhood_search` — Variable Neighborhood Search (VNS)

**Simple explanation:** picture standing on a hilly landscape of "how good is this tour", trying to find the tallest hill. A plain hill-climber takes one tiny step at a time toward whatever looks better right now — which works fine until it's standing on top of *a* hill that isn't the *tallest* one around, with no single small step that looks better. VNS's fix: when small steps stop helping, take a bigger, **random** jump instead — far enough that it might land somewhere completely different — then go back to small, careful steps from wherever it lands. If that didn't lead anywhere better, take an even bigger jump next time; the moment a jump *does* pay off, go back to small jumps again.

**In detail:** each round has two phases. **Shake:** randomly drop `k` targets from the current tour and add `k` random not-yet-included ones — a disruptive random jump whose size, `k`, starts small. **Local search:** from that shaken tour, repeatedly take whichever single ADD/DROP/SWAP move most improves it, until nothing helps anymore (a local hilltop). If the resulting tour beats where the round started, it's kept and `k` resets to its smallest value; if not, the tour is left unchanged but `k` grows, so the next round's jump is bigger.

**How it picks targets:** the shake phase picks randomly (not by any score); the local-search phase that follows every shake picks the single best-scoring ADD/DROP/SWAP move available, exactly like one round of Tabu Search's move selection (minus the tabu memory).

**When it finishes:** after a fixed number of rounds; the best tour seen at any point is returned.

**Other notes:** the key difference from Tabu Search is *how* each escapes a dead end — Tabu Search stays disciplined with one careful step and a "no backsies" memory; VNS instead physically relocates via a sometimes-large random jump, then tidies up locally from wherever it lands. A shaken tour can start out over budget; the local-search phase right after usually repairs this on its own (typically via a DROP move).

**Reference:** N. Mladenović, P. Hansen, "Variable Neighborhood Search", *Computers & Operations Research*, 24(11), 1997.

#### 4.3.4 `grasp` — Greedy Randomized Adaptive Search Procedure (GRASP)

**Simple explanation:** imagine building a tower one piece at a time. A purely greedy builder always grabs the single best piece available — but that locks in the exact same choices every time, and an early "best-looking" piece doesn't always lead to the best finished tower. GRASP's fix: at each step, look at a *handful* of the best pieces available, not just the single best one, and pick one of those at random. This still builds a genuinely good tower, but a different one nearly every time. Once built, fine-tune it, then throw the whole plan away and build an entirely new tower from scratch the same way — repeatedly — keeping whichever finished attempt turned out best overall.

**In detail:** each of a fixed number of independent iterations has two phases. **Construction:** starting from the current tour end, every reachable remaining target is scored by its value-collected/cost ratio, the top few (a "Restricted Candidate List") are kept, and one is picked *uniformly at random* from that shortlist — repeated, re-scoring fresh every step (the "adaptive" part), until nothing more is reachable. **Local search:** the exact same hill-climb VNS uses after a shake, applied to the freshly constructed tour. Whichever iteration's result is best overall is kept.

**How it picks targets:** during construction, one of the top-`rcl_size` reachable targets by value/cost ratio, chosen at random rather than always the single best; during local search, whichever single ADD/DROP/SWAP move scores best (same as Tabu Search/VNS).

**When it finishes:** after a fixed number of independent construction+local-search restarts; the best result across all of them is returned.

**Other notes:** unlike Tabu Search, VNS or Simulated Annealing, GRASP carries **no memory** between iterations — every restart is fully independent, which makes it naturally parallelizable and immune to ever being trapped by a bad early decision inherited from a previous attempt, at the cost of never building on partial progress the way the others do.

**Reference:** T. A. Feo, M. G. C. Resende, "Greedy Randomized Adaptive Search Procedures", *Journal of Global Optimization*, 6(2), 1995.

#### 4.3.5 `simulated_annealing` — Simulated Annealing (SA)

**Simple explanation:** picture a bouncy ball dropped onto that same hilly "how good is this tour" landscape. A ball that only ever moves to something better gets stuck on the very first small hill it climbs. Simulated Annealing makes the ball extra bouncy at the start — so it can hop over small hills and shallow dips, wandering fairly freely, even sometimes to somewhere clearly worse. The bounciness is then very gradually turned down ("cooling"), so it settles, hopefully, on a tall hill it stumbled onto while it still had plenty of bounce left. The name comes from annealing metal: heat it, then cool it *slowly* so its atoms have time to settle into a good structure; cool it too fast and it freezes into a poor one.

**In detail:** each iteration proposes exactly **one** random ADD/DROP/SWAP move. If it's better, it's always accepted. If it's worse, it's still accepted with a probability that shrinks both as the move gets worse and as a "temperature" value cools over the course of the run (`probability = exp(how much worse ÷ temperature)`) — the *Metropolis criterion*. Temperature starts high (freely exploring) and is multiplied down by a fixed cooling rate every round, never below a small floor.

**How it picks targets:** a single random ADD, DROP or SWAP move is proposed each round — not scored against alternatives the way Tabu Search/VNS/GRASP compare a whole batch, just accepted or rejected on its own merits (and the temperature).

**When it finishes:** after a fixed number of iterations — typically far more than Tabu Search or VNS use, since each individual round here is much cheaper (one proposal, not a whole neighborhood). The best tour seen at any point is returned.

**Other notes:** temperature never drops all the way to zero, both to avoid a division by zero and because a temperature of exactly zero would make the search purely greedy for the rest of the run — a small floor keeps a sliver of randomness alive throughout.

**Reference:** S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by Simulated Annealing", *Science*, 220(4598), 1983.

#### 4.3.6 `genetic_algorithm` — Genetic Algorithm (GA)

**Simple explanation:** imagine breeding plants for the tallest, most productive garden. You wouldn't just pick the single best plant and stop — you'd let your best plants cross-pollinate, combining their traits, let a few random mutations happen, and grow a new generation from that. Repeat for many generations, and the garden as a whole tends to keep improving, because good traits from two *different* parents can combine into an offspring better than either parent alone.

**In detail:** a whole population of candidate tours ("chromosomes") is maintained at once — some seeded from a quick randomized-greedy construction (the same one GRASP uses), the rest fully random subsets. Each generation: every chromosome is scored by its collected value (a chromosome too ambitious for the budget is "repaired" — walked one target at a time and simply cut off at the first target that doesn't fit, rather than discarded outright, so an over-eager crossover result isn't wasted). Parents are chosen via tournament selection (grab a few chromosomes at random, the fittest wins a chance to breed). Two parents produce a child by crossover — the child keeps a random-length prefix of one parent's plan and fills in the rest with whichever of the other parent's targets aren't already included. A child is occasionally mutated by the exact same single random ADD/DROP/SWAP move Simulated Annealing uses. The very best chromosomes ("elites") always survive unchanged into the next generation, so the population's best individual can never accidentally get worse from one generation to the next.

**How it picks targets:** indirectly, through the whole population + selection + crossover + mutation process, rather than any single explicit "pick the best candidate" step — good combinations of targets that happen to appear in fit parents get propagated forward and recombined.

**When it finishes:** after a fixed number of generations; the single best chromosome ever seen (not necessarily in the final generation) is returned.

**Other notes:** every metaheuristic in this project that shares the "ordered list of targets" tour representation — Tabu Search, VNS, GRASP, Simulated Annealing, and this one — ultimately explores the same three underlying moves (ADD/DROP/SWAP); what differs between all of them is purely the *strategy* for choosing among those moves.

**Reference:** J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975; D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989.

#### 4.3.7 `large_neighborhood_search` — Adaptive Large Neighborhood Search (ALNS)

**Simple explanation:** imagine a decent LEGO castle already built, but a few rooms could probably be rearranged to fit more towers in. Instead of tearing the whole thing down and starting over (GRASP's approach), or changing a single room at a time (Tabu Search/VNS's approach), ALNS knocks out a *handful* of rooms at once — leaving most of the good castle standing — and rebuilds only the missing part as cleverly as it can. Multiple different ways of "knocking out" and "rebuilding" a chunk are tried over the run, and the ones that keep paying off get used more often as the search learns what's actually working for this specific map.

**In detail:** starts from a real, already-decent tour (built the same randomized-greedy way GRASP constructs one). Each round: a **destroy operator** removes a chunk of the current tour (a random subset; the least-valuable-contributing stops; or a cluster of *geographically related* stops, since nearby targets tend to compete for the same "swing by here" decision, so removing them together as a group gives real freedom to reshuffle). A **repair operator** then adds stops back — drawn from *every* real target not currently in the tour, not just the ones just removed, since the freed-up budget might now afford something better — either by always inserting whichever candidate/position pairing scores best, or by prioritizing candidates that have only one good remaining insertion spot (and would lose that chance if not grabbed now). The repaired tour is accepted if it's better, and sometimes accepted even if it's worse (the same Metropolis criterion Simulated Annealing uses, with its own cooling temperature). Each destroy/repair operator's recent track record is scored, and periodically the odds of picking each operator again are nudged toward whichever ones have actually been paying off.

**How it picks targets:** destroy operators remove a chunk of the *current* tour's targets (randomly, by lowest contribution, or by geographic clustering); repair operators add back targets from the *entire* pool of real targets not currently included, prioritizing by best resulting value or by "act now or lose the chance" urgency.

**When it finishes:** after a fixed number of rounds; the best tour seen at any point is returned.

**Other notes:** unlike Tabu Search/VNS's single ADD/DROP/SWAP moves, this can genuinely discover an improvement that requires changing *several* targets together, since the destroy step removes them as a group before the repair step ever has to choose a replacement combination — a rearrangement that would look worse at every single intermediate one-move-at-a-time step, and so would never be reached by an algorithm that only ever changes one thing at a time, can be found directly here.

**Reference:** P. Shaw, "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems", *Proceedings of CP 1998*; S. Ropke, D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows", *Transportation Science*, 40(4), 2006.

---

### 4.4 Exact solver (`pathfinding_algorithms/exact_pathfinding.py`)

#### 4.4.1 `exact_solver` — provably optimal solution via CP-SAT

**Simple explanation:** every algorithm above searches for a *good* answer without ever being able to prove it's the *best possible* one. This algorithm is different in kind: it hands the problem to a genuine mathematical optimization engine — [Google OR-Tools' CP-SAT solver](https://developers.google.com/optimization) — which explores the space of possible answers in a structured way that lets it *prove* when it has found the true optimum, or, if it runs out of time first, report exactly how far its best answer could still be from optimal.

**In detail:** the grid is first reduced to just the start cell plus every real target (the same target set every metaheuristic in section 4.3 uses), and the true cheapest travel cost between every pair of them is precomputed with real A* search (section 4.2). This turns "find a good grid path" into the much smaller, classical **[Orienteering Problem](https://en.wikipedia.org/wiki/Orienteering_problem)**: choose a subset of these targets and an order to visit them. That problem is then modeled as a **[branch-and-bound](https://en.wikipedia.org/wiki/Branch_and_bound)** search: every target gets a "skip this one" option, every pair of targets gets a "travel directly between these" option, and a single constraint (CP-SAT's `AddCircuit`) enforces that whichever targets aren't skipped form one connected route, starting and ending at the depot. The solver repeatedly relaxes the problem to get a fast upper bound, and only ever explores a branch of possibilities further if that bound says it could still beat the best answer found so far anywhere else in the search — everything else is mathematically ruled out without needing to be checked. This is exactly what lets it eventually *prove* optimality, rather than merely reporting "this is the best I happened to find".

**How it picks targets:** it doesn't pick incrementally at all — the solver considers every possible subset and ordering of the real target pool at once (via the constraint structure, not by brute force) and returns the genuinely best one under the budget constraint.

**When it finishes:** either the moment it has proven no better answer can exist ("OPTIMAL"), or when a configurable time limit runs out first, in which case it still returns its best answer found so far, along with a printed optimality gap — a guaranteed bound on how far from the true optimum that answer could still be. This project's return contract is the same plain `(path, total_value, cost_used)` tuple every other algorithm uses, so the extra diagnostic information (solve status, objective value, best remaining bound, gap) is printed to the console rather than returned.

**Other notes:** this is deliberately **not** a hand-written search algorithm — the actual branch-and-bound engine is Google's production-grade CP-SAT solver, not code written for this project; only the translation from "grid and budget" into CP-SAT's variables and constraints is project-specific. Two costs make this impractical on very large scenarios: precomputing every pairwise travel cost is quadratic in the number of real targets, and the underlying Orienteering Problem is itself NP-hard — so a safety guard (`max_targets`, 60 by default) refuses to even attempt a scenario with more real targets than that unless explicitly told to. What "optimal" specifically means here is also worth being precise about: the objective only credits a target's value if it's explicitly chosen as a stop — not any *incidental* value a chosen route happens to sweep past on the way somewhere else, the way the tour-based metaheuristics' cheaper walking heuristic sometimes opportunistically picks up. It is provably the best possible choice of *which targets to visit, and in what order*, under the true cheapest travel costs between them — a meaningful, honest guarantee, even though it stops short of also being a proof about incidental pickups.

**Reference:** A. H. Land, A. G. Doig, "An Automatic Method of Solving Discrete Programming Problems", *Econometrica*, 28(3), 1960 (the original branch-and-bound paper); P. Vansteenwegen, W. Souffriau, D. Van Oudheusden, "The Orienteering Problem: A Survey", *European Journal of Operational Research*, 209(1), 2011; [Google OR-Tools documentation](https://developers.google.com/optimization) for the CP-SAT solver itself.

## 5. Quick-reference comparison

| Key | Family | Randomized? | Picks by | Stops when |
|---|---|---|---|---|
| `greedy` | Greedy | No | Highest-value immediate neighbor (cost tie-break) | No affordable move left at all |
| `direct_to_highest_value` | Greedy | No | Single highest-value target on the grid | Current best target unreachable |
| `value_cost_ratio` | Greedy | No | Best value/cost among a small rotating pool | No candidate reachable this round |
| `lowest_cost` | Greedy | No | Cheapest among a wide rotating pool | No candidate reachable this round |
| `a_star_to_highest_value` | A* | No | Same as `direct_to_highest_value`, true-optimal routing | Same as `direct_to_highest_value` |
| `a_star_value_cost_ratio` | A* | No | Same as `value_cost_ratio`, true-optimal routing | Same as `value_cost_ratio` |
| `a_star_lowest_cost` | A* | No | Same as `lowest_cost`, true-optimal routing | Same as `lowest_cost` |
| `namoa_star` | A* | No | Best point on each candidate's full cost/value trade-off menu | No candidate has an affordable trade-off left |
| `ant_colony` | Metaheuristic | Yes | Pheromone + distance-decayed attraction, per step | Fixed iteration budget |
| `tabu_search` | Metaheuristic | Partial | Best non-tabu ADD/DROP/SWAP move | Fixed rounds, or no valid move left |
| `variable_neighborhood_search` | Metaheuristic | Yes | Random shake, then best local move | Fixed rounds |
| `grasp` | Metaheuristic | Yes | Random pick among top-ratio candidates, then best local move | Fixed independent restarts |
| `simulated_annealing` | Metaheuristic | Yes | One random move, accepted/rejected by cooling temperature | Fixed iterations |
| `genetic_algorithm` | Metaheuristic | Yes | Population + tournament selection + crossover + mutation | Fixed generations |
| `large_neighborhood_search` | Metaheuristic | Yes | Adaptive destroy + repair operator pair, cooling-temperature acceptance | Fixed rounds |
| `exact_solver` | Exact | No | Every subset/order at once, via CP-SAT branch-and-bound | Proven optimal, or a time limit |

## 6. Full bibliography

- T. H. Cormen, C. E. Leiserson, R. L. Rivest, C. Stein, *Introduction to Algorithms*, MIT Press.
- T. Tsiligirides, "Heuristic Methods Applied to Orienteering", *Journal of the Operational Research Society*, 35(9), 1984.
- B. L. Golden, L. Levy, R. Vohra, "The Orienteering Problem", *Naval Research Logistics*, 34(3), 1987.
- P. E. Hart, N. J. Nilsson, B. Raphael, "A Formal Basis for the Heuristic Determination of Minimum Cost Paths", *IEEE Transactions on Systems Science and Cybernetics*, 4(2), 1968.
- L. Mandow, J. L. Pérez-de-la-Cruz, "A New Approach to Multiobjective A* Search", *Proceedings of IJCAI 2005*.
- M. Dorigo, *Optimization, Learning and Natural Algorithms*, PhD thesis, Politecnico di Milano, 1992.
- M. Dorigo, L. M. Gambardella, "Ant Colony System: A Cooperative Learning Approach to the Traveling Salesman Problem", *IEEE Transactions on Evolutionary Computation*, 1(1), 1997.
- F. Glover, "Future Paths for Integer Programming and Links to Artificial Intelligence", *Computers & Operations Research*, 13(5), 1986.
- F. Glover, M. Laguna, *Tabu Search*, Kluwer Academic Publishers, 1997.
- N. Mladenović, P. Hansen, "Variable Neighborhood Search", *Computers & Operations Research*, 24(11), 1997.
- T. A. Feo, M. G. C. Resende, "Greedy Randomized Adaptive Search Procedures", *Journal of Global Optimization*, 6(2), 1995.
- S. Kirkpatrick, C. D. Gelatt, M. P. Vecchi, "Optimization by Simulated Annealing", *Science*, 220(4598), 1983.
- J. H. Holland, *Adaptation in Natural and Artificial Systems*, University of Michigan Press, 1975.
- D. E. Goldberg, *Genetic Algorithms in Search, Optimization, and Machine Learning*, Addison-Wesley, 1989.
- P. Shaw, "Using Constraint Programming and Local Search Methods to Solve Vehicle Routing Problems", *Proceedings of CP 1998*.
- S. Ropke, D. Pisinger, "An Adaptive Large Neighborhood Search Heuristic for the Pickup and Delivery Problem with Time Windows", *Transportation Science*, 40(4), 2006.
- A. H. Land, A. G. Doig, "An Automatic Method of Solving Discrete Programming Problems", *Econometrica*, 28(3), 1960.
- P. Vansteenwegen, W. Souffriau, D. Van Oudheusden, "The Orienteering Problem: A Survey", *European Journal of Operational Research*, 209(1), 2011.
- [Google OR-Tools documentation](https://developers.google.com/optimization) (CP-SAT solver).
