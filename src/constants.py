# constants.py

from dataclasses import dataclass


@dataclass(frozen=True)
class Constants():
    """
    Central place for constant values shared across the project. Not meant
    to be instantiated - access fields directly on the class, e.g.
    Constants.MAX_PROBABILITY.
    """

    MAX_PROBABILITY: float = 10.0

    # The probability CoordinatesGrid.init_empty_grid fills every cell with
    # before any real search area is set - i.e. "no information yet", not a
    # genuine signal. Pathfinding algorithms that reason about value (see
    # greedy_pathfinding._next_fresh_candidate) use this to skip untouched
    # background cells rather than treating them as real targets.
    DEFAULT_PROBABILITY: float = 1.0

    # Offsets (delta_row, delta_col) for the 8 directions stored in the last
    # dimension of WeightsGrid.weights: index 0 is "up", the rest follow
    # clockwise.
    DIRECTIONS: tuple = (
        (-1, 0),   # 0: up
        (-1, 1),   # 1: up-right
        (0, 1),    # 2: right
        (1, 1),    # 3: down-right
        (1, 0),    # 4: down
        (1, -1),   # 5: down-left
        (0, -1),   # 6: left
        (-1, -1),  # 7: up-left
    )
