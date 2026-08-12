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

    # Climbing (up / up-right / up-left) costs the drone more battery than
    # the weights_grid base cost; descending (down / down-right / down-left)
    # costs less. This is a flat, direction-only approximation - if
    # weights_grid was populated via WeightsGrid.init_from_elevation, the
    # climb/descend asymmetry is already baked into the base cost from real
    # terrain, and stacking this multiplier on top would double-penalize it.
    DIRECTION_COST_MULTIPLIERS: tuple = (1.1, 1.1, 1.0, 0.9, 0.9, 0.9, 1.0, 1.1)
