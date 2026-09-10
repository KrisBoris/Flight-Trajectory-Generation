from coordinates_grid.weights_grid import WeightsGrid
from helpers.constants import Constants
from dataclasses import dataclass
import numpy as np


@dataclass
class CoordinatesGrid():
    """
    Data class holding the grid of cell coordinates that can be visited,
    together with each cell's value - the probability of finding the
    searched person there - and the WeightsGrid describing the movement
    cost between neighboring cells.
    """

    # Grid holding the costs of traveling between each connected node
    weights_grid: WeightsGrid = None

    # Value (probability of finding searched person) of each node
    coordinates_values: np.ndarray = None


    def __post_init__(self):
        """
        Runs right after the dataclass's generated __init__. If
        coordinates_values was given directly, converts it to a 2D float64
        numpy array (raising ValueError if it isn't 2D) - otherwise leaves
        it as None.
        """
        if self.coordinates_values is None:
            return

        if not isinstance(self.coordinates_values, np.ndarray):
            self.coordinates_values = np.array(self.coordinates_values, dtype=np.float64)

        if self.coordinates_values.ndim != 2:
            raise ValueError(f"Coordinates matrix must be two-dimensional, not {self.coordinates_values.ndim}")

        if not np.issubdtype(self.coordinates_values.dtype, np.floating):
            self.coordinates_values = self.coordinates_values.astype(np.float64)


    @property
    def rows(self):
        """Number of rows in the grid, read from coordinates_values' own shape"""
        return self.coordinates_values.shape[0]


    @property
    def cols(self):
        """Number of columns in the grid, read from coordinates_values' own shape"""
        return self.coordinates_values.shape[1]    


    def set_searched_area_values(self, x: int, y: int, areas_coords: list) -> bool:
        """
        Initializes coordinates_values as an x by y grid, every cell
        starting at Constants.DEFAULT_PROBABILITY, then writes each (row, col,
        probability) entry in areas_coords on top of it
        """
        if x <= 0 or y <= 0:
            print(f"Grid size must be greater than zero, not {x}x{y}")
            return False

        self.coordinates_values = np.full([x, y], Constants.DEFAULT_PROBABILITY, dtype=np.float64)

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


    def set_searched_area_cost(
        self,
        coordinates: np.ndarray,
        climb_cost_per_meter: float,
        descent_cost_per_meter: float,
        base_cost: float = 1.0,
    ) -> bool:
        """
        Calls WeightsGrid method that calculates the
        costs of traveling between each connected node
        based on terrain elevation
        """
        if self.weights_grid is None:
            self.weights_grid = WeightsGrid()
        
        return self.weights_grid.init_from_elevation(
            coordinates,
            climb_cost_per_meter,
            descent_cost_per_meter,
            base_cost
        )
