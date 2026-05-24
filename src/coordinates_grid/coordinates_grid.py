# coordinates_grid.py

from dataclasses import dataclass, field
import numpy as np


@dataclass
class CoordinatesGrid():
    """
    Data class przechowująca graf ze współrzędnymi możliwych do odwiedzenia punktów
    oraz wartości dla każdego punktu odpowiadające prawdopodobieństwu znalezienia tam 
    poszukiwanej osoby 
    """
    
    coordinates: np.ndarray = field()


    def __post_init__(self):
        if not isinstance(self.coordinates, np.ndarray):
            self.coordinates = np.array(self.coordinates, dtype=np.float64)

        if self.coordinates.ndim != 2:
            raise ValueError(f"Coordinates matrix must be two-dimensional, not {self.coordinates.ndim}")

        if not np.issubdtype(self.coordinates.dtype, np.floating):
            self.coordinates = self.coordinates.astype(np.float64)

    
    @property
    def rows(self):
        return self.coordinates.shape[0]
    

    @property
    def cols(self):
        return self.coordinates.shape[1]