# coordinates_grid.py

from coordinates_grid.weights_grid import WeightsGrid
from dataclasses import dataclass, field
from typing import ClassVar
import numpy as np


@dataclass
class CoordinatesGrid():
    """
    Data class przechowująca graf ze współrzędnymi możliwych do odwiedzenia punktów
    oraz wartości dla każdego punktu odpowiadające prawdopodobieństwu znalezienia tam
    poszukiwanej osoby
    """

    MAX_PROBABILITY: ClassVar[float] = 1.0

    coordinates_values: np.ndarray = field()
    weights_grid: WeightsGrid = field()


    def __post_init__(self):
        if not isinstance(self.coordinates_values, np.ndarray):
            self.coordinates_values = np.array(self.coordinates_values, dtype=np.float64)

        if self.coordinates_values.ndim != 2:
            raise ValueError(f"Coordinates matrix must be two-dimensional, not {self.coordinates_values.ndim}")

        if not np.issubdtype(self.coordinates_values.dtype, np.floating):
            self.coordinates_values = self.coordinates_values.astype(np.float64)

    
    @property
    def rows(self):
        return self.coordinates_values.shape[0]
    

    @property
    def cols(self):
        return self.coordinates_values.shape[1]


    def init_empty_grid(self, x: int, y: int) -> bool:
        
        if x <= 0 or y <= 0:
            print(f"Grid size must be greater than zero, not {x}x{y}")
            return False
        
        self.coordinates_values = np.ones([x, y], dtype=np.float64)
        return True


    def init_grids(self, x: int, y: int) -> bool:
        """ Initializes coordinates_values and the nested weights_grid together """
        
        if not self.init_empty_grid(x, y):
            return False

        if not self.weights_grid.init_empty_grid(x, y):
            return False

        return True


    def set_searched_areas(self, areas_coords: np.ndarray) -> bool:

        if areas_coords.ndim != 2 or areas_coords.shape[1] != 3:
            print(f"Areas coordinates expected size is Nx3, not {areas_coords.shape}")
            return False
        
        if not np.issubdtype(areas_coords.dtype, np.floating):
            print(f"Coordinates values must be os float type, not {areas_coords.dtype}")
            return False
        
        for coords in areas_coords:

            # coords[0] -> x-position
            # coords[1] -> y-position
            # coords[2] -> probability
            
            if (coords[0] < 0 or coords[0] > self.coordinates_values.shape[0] - 1 
                or coords[1] < 0 or coords[1] > self.coordinates_values.shape[1] - 1
                or coords[2] < 0.0):
                continue
            
            if (coords[0] > 1 and coords[0] < self.coordinates_values.shape[0] - 2
                and coords[1] > 1 and coords[1] < self.coordinates_values.shape[1] - 2):

                x = int(coords[0])
                y = int(coords[1])
                probability = coords[2]

                for i in range(x - 2, x + 2):
                    for j in range(y - 2, y + 2):
                        self.coordinates_values[i, j] = probability * 0.3

                for i in range(x - 1, x + 1):
                    for j in range(y - 1, y + 1):
                        self.coordinates_values[i, j] = probability * 0.7

                self.coordinates_values[x, y] = probability
            
            else:
                return False