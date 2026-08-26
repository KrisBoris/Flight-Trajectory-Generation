# coordinates_grid.py

from coordinates_grid.weights_grid import WeightsGrid
from helpers.constants import Constants
from dataclasses import dataclass, field
import numpy as np


@dataclass
class CoordinatesGrid():
    """
    Data class holding the grid of cell coordinates that can be visited,
    together with each cell's value - the probability of finding the
    searched person there - and the WeightsGrid describing the movement
    cost between neighboring cells.
    """

    coordinates_values: np.ndarray = field()
    weights_grid: WeightsGrid = field()


    def __post_init__(self):
        """
        Runs right after the dataclass's generated __init__: coerces
        coordinates_values to a 2D float64 numpy array if it wasn't
        already one, raising ValueError if it isn't 2D to begin with (a
        shape no pathfinding algorithm in this project could make sense
        of).
        """
        if not isinstance(self.coordinates_values, np.ndarray):
            self.coordinates_values = np.array(self.coordinates_values, dtype=np.float64)

        if self.coordinates_values.ndim != 2:
            raise ValueError(f"Coordinates matrix must be two-dimensional, not {self.coordinates_values.ndim}")

        if not np.issubdtype(self.coordinates_values.dtype, np.floating):
            self.coordinates_values = self.coordinates_values.astype(np.float64)


    @property
    def rows(self):
        """Number of rows in the grid, read from coordinates_values' own shape."""
        return self.coordinates_values.shape[0]


    @property
    def cols(self):
        """Number of columns in the grid, read from coordinates_values' own shape."""
        return self.coordinates_values.shape[1]


    def init_empty_grid(self, x: int, y: int) -> bool:
        """
        (Re)initializes coordinates_values as an x by y grid, every cell
        filled with Constants.DEFAULT_PROBABILITY - i.e. "no information
        yet", not a genuine signal (see set_searched_areas for setting
        real values). Returns False (and prints a message, leaving the
        grid untouched) if x or y isn't positive.
        """
        if x <= 0 or y <= 0:
            print(f"Grid size must be greater than zero, not {x}x{y}")
            return False

        self.coordinates_values = np.full([x, y], Constants.DEFAULT_PROBABILITY, dtype=np.float64)
        return True


    def init_grids(self, x: int, y: int) -> bool:
        """ Initializes coordinates_values and the nested weights_grid together """
        
        if not self.init_empty_grid(x, y):
            return False

        if not self.weights_grid.init_empty_grid(x, y):
            return False

        return True


    def set_searched_areas(self, areas_coords: list) -> bool:
        """
        Writes each (row, col, probability) entry in areas_coords into
        coordinates_values, clamping probability to at most
        Constants.MAX_PROBABILITY. An entry with the wrong shape/dtype
        aborts the whole call (returns False without applying anything);
        an individual entry whose row/col falls outside the grid, or whose
        probability is negative, is silently skipped instead - the same
        tolerance build_blocked_mask gives an out-of-range blocked cell -
        so a scenario file authored for one grid size doesn't hard-crash
        if reused with a smaller one.
        """
        for coords in areas_coords:

            if not isinstance(coords, np.ndarray) or coords.ndim != 1 or coords.shape[0] != 3:
                print(f"Each area must be an ndarray of shape (3,) - (row, col, probability), not {coords}")
                return False

            if not np.issubdtype(coords.dtype, np.floating):
                print(f"Coordinates values must be of float type, not {coords.dtype}")
                return False

            # coords[0] -> row
            # coords[1] -> col
            # coords[2] -> probability

            if (coords[0] < 0 or coords[0] > self.coordinates_values.shape[0] - 1
                or coords[1] < 0 or coords[1] > self.coordinates_values.shape[1] - 1
                or coords[2] < 0.0):
                continue

            probability = min(coords[2], Constants.MAX_PROBABILITY)
            row = int(coords[0])
            col = int(coords[1])
            self.coordinates_values[row, col] = probability

        return True