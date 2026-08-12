# test_data_generator.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from constants import Constants
import numpy as np


def generate_random_positions(coordinates_grid: CoordinatesGrid, amount: int) -> list:
    """
    Generates `amount` random, unique (row, col) positions within
    coordinates_grid, each paired with a random probability of finding the
    searched person there, in [0.0, Constants.MAX_PROBABILITY].
    Returned as a list of (3,) float arrays (row, col, probability) - the
    same layout CoordinatesGrid.set_searched_areas expects.
    """
    total_cells = coordinates_grid.rows * coordinates_grid.cols

    if amount <= 0 or amount > total_cells:
        print(f"Amount must be between 1 and {total_cells} (rows * cols), not {amount}")
        return []

    flat_indices = np.random.choice(total_cells, size=amount, replace=False)
    rows, cols = np.unravel_index(flat_indices, (coordinates_grid.rows, coordinates_grid.cols))
    probabilities = np.random.uniform(0.0, Constants.MAX_PROBABILITY, size=amount)

    return [np.array([row, col, probability], dtype=np.float64) for row, col, probability in zip(rows, cols, probabilities)]


def _smooth_elevation(elevation: np.ndarray, iterations: int, smoothing_factor: float = 0.2) -> np.ndarray:
    """
    Repeatedly nudges each cell toward the average of its 4-connected
    neighbors - a discrete diffusion step, the same idea behind a heat
    equation or a repeated box blur. This turns cell-independent white noise
    into a field where neighboring cells are close in value, i.e. gradual
    slopes instead of random jumps. Edge cells reuse their nearest interior
    neighbor (via 'edge' padding) so borders don't erode toward zero.
    """
    smoothed = elevation.astype(np.float64)

    for _ in range(iterations):
        padded = np.pad(smoothed, pad_width=1, mode="edge")
        neighbor_average = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) / 4.0
        smoothed = smoothed + smoothing_factor * (neighbor_average - smoothed)

    return smoothed


def generate_random_terrain_coordinates(
    rows: int,
    cols: int,
    altitude_range: tuple[float, float],
    cell_size_meters: float = 1.0,
    smoothing_iterations: int = 25,
) -> np.ndarray:
    """
    Generates a rows x cols x 3 array of local (x, y, z) coordinates in
    meters, for feeding into WeightsGrid.init_from_elevation. x grows with
    column and y grows with row going up (row 0 is the "up" direction, see
    constants.Constants.DIRECTIONS), spaced cell_size_meters apart, so the grid's
    bottom-left cell - (row=rows-1, col=0) - sits at x = y = 0.

    z (altitude above sea level) starts as independent uniform noise, which
    on its own looks nothing like a mountain (every cell is unrelated to its
    neighbor, so the "terrain" is just static). It's then smoothed by
    _smooth_elevation for smoothing_iterations passes - more passes means
    gentler, more gradual slopes - and rescaled back to exactly span
    altitude_range = (min_altitude, max_altitude), since smoothing pulls
    extreme values toward the mean and would otherwise shrink the range.
    """
    min_altitude, max_altitude = altitude_range

    if rows <= 0 or cols <= 0:
        print(f"Grid size must be greater than zero, not {rows}x{cols}")
        return np.empty((0, 0, 3), dtype=np.float64)

    if min_altitude > max_altitude:
        print(f"altitude_range minimum ({min_altitude}) must not exceed its maximum ({max_altitude})")
        return np.empty((0, 0, 3), dtype=np.float64)

    x = np.arange(cols) * cell_size_meters
    y = np.arange(rows - 1, -1, -1) * cell_size_meters
    grid_x, grid_y = np.meshgrid(x, y)

    grid_z = np.random.uniform(min_altitude, max_altitude, size=(rows, cols))
    grid_z = _smooth_elevation(grid_z, iterations=smoothing_iterations)

    z_min, z_max = grid_z.min(), grid_z.max()
    if z_max > z_min:
        grid_z = min_altitude + (grid_z - z_min) / (z_max - z_min) * (max_altitude - min_altitude)

    return np.stack((grid_x, grid_y, grid_z), axis=-1)


def generate_random_blocked_mask(coordinates_grid: CoordinatesGrid, blocked_shapes: list) -> np.ndarray:
    """
    Builds a boolean blocked_mask (see TrajectoryGenerator's blocked_mask
    parameter) for coordinates_grid by randomly placing each shape in
    blocked_shapes at an in-bounds position on the grid.

    Each entry in blocked_shapes is itself a small 2D array (e.g. a storm
    cell or restricted-airspace footprint) whose truthy cells mark the
    no-fly pattern; its own shape (rows, cols) is only used to know how big
    an area to reserve when picking a random anchor. A shape larger than the
    grid in either dimension is skipped since it can never fit. The returned
    mask is the union of every placement - a cell is blocked if any shape
    ended up covering it.
    """
    rows, cols = coordinates_grid.rows, coordinates_grid.cols
    mask = np.zeros((rows, cols), dtype=bool)

    for shape in blocked_shapes:
        shape = np.asarray(shape)

        if shape.ndim != 2:
            print(f"Blocked shape must be two-dimensional, not {shape.ndim}, skipping")
            continue

        shape_rows, shape_cols = shape.shape

        if shape_rows > rows or shape_cols > cols:
            print(f"Blocked shape {shape.shape} doesn't fit in grid {rows}x{cols}, skipping")
            continue

        anchor_row = np.random.randint(0, rows - shape_rows + 1)
        anchor_col = np.random.randint(0, cols - shape_cols + 1)

        mask[anchor_row:anchor_row + shape_rows, anchor_col:anchor_col + shape_cols] |= shape.astype(bool)

    return mask