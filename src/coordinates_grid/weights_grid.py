# weights_grid.py

from constants import Constants
from dataclasses import dataclass, field
import numpy as np


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
        coordinates: np.ndarray,
        climb_cost_per_meter: float,
        descent_cost_per_meter: float,
        base_cost: float = 1.0,
    ) -> bool:
        """
        Populates weights from each cell's real-world local coordinates
        (a rows x cols x 3 array of (x, y, z) in meters - see
        test_data_generator.generate_random_terrain_coordinates - where the
        grid's bottom-left cell sits at x = y = 0 and z is altitude above sea
        level) instead of a flat direction guess.

        For every cell and direction, cost has two parts:
          - a travel cost of base_cost per meter of actual horizontal
            distance to that neighbor, so a diagonal move - which covers more
            real ground than a cardinal one - costs proportionally more;
          - a grade penalty of elevation_change^2 / horizontal_distance
            (i.e. slope^2 * horizontal_distance, slope = elevation_change /
            horizontal_distance). This grows quickly for a steep, short hop
            and shrinks for the same altitude change spread over a longer,
            gentler run, rewarding gradual climbs/descents over abrupt ones.
        Climbing adds climb_cost_per_meter * grade_penalty; descending
        subtracts descent_cost_per_meter * grade_penalty (floored at
        0.1 * travel_cost so a long, gentle descent is discounted but never
        made absurdly - or negatively - cheap).
        """
        if coordinates.ndim != 3 or coordinates.shape[2] != 3:
            print(f"Coordinates matrix must be a (rows, cols, 3) array of (x, y, z), not {coordinates.shape}")
            return False

        rows, cols, _ = coordinates.shape
        self.weights = np.full((rows, cols, 8), base_cost, dtype=np.float64)

        for row in range(rows):
            for col in range(cols):
                for direction, (delta_row, delta_col) in enumerate(Constants.DIRECTIONS):
                    next_row = row + delta_row
                    next_col = col + delta_col

                    if next_row < 0 or next_row >= rows or next_col < 0 or next_col >= cols:
                        continue  # off-grid; left at base_cost since a path search will never take this edge

                    x0, y0, z0 = coordinates[row, col]
                    x1, y1, z1 = coordinates[next_row, next_col]

                    horizontal_distance = float(np.hypot(x1 - x0, y1 - y0))
                    elevation_change = z1 - z0
                    travel_cost = base_cost * horizontal_distance
                    grade_penalty = (elevation_change ** 2) / horizontal_distance if horizontal_distance > 0 else 0.0

                    if elevation_change > 0:
                        self.weights[row, col, direction] = travel_cost + climb_cost_per_meter * grade_penalty
                    else:
                        self.weights[row, col, direction] = max(0.8 * travel_cost, travel_cost - descent_cost_per_meter * grade_penalty)

        return True
