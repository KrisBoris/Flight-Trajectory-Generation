# weights_grid.py

from dataclasses import dataclass, field
import numpy as np


# Offsets (delta_row, delta_col) for the 8 directions stored in the last
# dimension of `weights`: index 0 is "up", the rest follow clockwise.
DIRECTIONS = (
    (-1, 0),   # 0: up
    (-1, 1),   # 1: up-right
    (0, 1),    # 2: right
    (1, 1),    # 3: down-right
    (1, 0),    # 4: down
    (1, -1),   # 5: down-left
    (0, -1),   # 6: left
    (-1, -1),  # 7: up-left
)


@dataclass
class WeightsGrid():
    """
    Data class przechowująca wagi krawędzi łączących wierzchołki grafu ze współrzędnymi 
    """

    weights: np.ndarray = field()


    def __post_init__(self):
        
        if not isinstance(self.weights, np.ndarray):
            self.weights = np.array(self.weights, dtype=np.float64)

        if self.weights.ndim != 3:
            raise ValueError(f"Weights matrix must be three-dimensional, not {self.coordinates.ndim}")
        
        if not np.issubdtype(self.weights.dtype, np.floating):
            self.weights = self.weights.astype(np.float64)


    def init_empty_grid(self, x: int, y: int) -> bool:
        if x <= 0 or y <= 0:
            print(f"Grid size must be greater than zero, not {x}x{y}")
            return False
                
        self.weights = np.ones([x, y, 8], dtype=np.float64)
        return True


    def init_from_elevation(
        self,
        elevation: np.ndarray,
        climb_cost_per_meter: float,
        descent_cost_per_meter: float,
        base_cost: float = 1.0,
    ) -> bool:
        """
        Populates weights from a real (or synthetic, see
        test_data_generator.generate_random_elevation_grid) elevation grid
        instead of a flat guess.

        For every cell and direction, cost is base_cost adjusted by the actual
        altitude change to that neighbor: climbing adds climb_cost_per_meter
        per meter gained, descending subtracts descent_cost_per_meter per
        meter lost (floored at 0.9 * base_cost so a steep descent is
        discounted but never absurdly cheap). This is a more physically
        grounded stand-in for a flat direction bias (e.g. "north always costs
        more") since it reacts to the terrain that is actually there, not
        just which compass direction the drone flies.
        """
        if elevation.ndim != 2:
            print(f"Elevation matrix must be two-dimensional, not {elevation.ndim}")
            return False

        rows, cols = elevation.shape
        self.weights = np.full((rows, cols, 8), base_cost, dtype=np.float64)

        for row in range(rows):
            for col in range(cols):
                for direction, (delta_row, delta_col) in enumerate(DIRECTIONS):
                    next_row = row + delta_row
                    next_col = col + delta_col

                    if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                        continue  # off-grid; left at base_cost since a path search will never take this edge

                    elevation_change = elevation[next_row, next_col] - elevation[row, col]

                    if elevation_change > 0:
                        self.weights[row, col, direction] = base_cost + elevation_change * climb_cost_per_meter
                    else:
                        self.weights[row, col, direction] = max(base_cost * 0.9, base_cost + elevation_change * descent_cost_per_meter)

        return True
