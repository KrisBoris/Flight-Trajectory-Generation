# weights_grid.py

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
