# test_data_generator.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
import numpy as np


def generate_random_positions(coordinates_grid: CoordinatesGrid, amount: int) -> np.ndarray:
    """
    Generates `amount` random, unique (row, col) positions within
    coordinates_grid, each paired with a random probability of finding the
    searched person there, in [0.0, CoordinatesGrid.MAX_PROBABILITY].
    Returned as an Nx3 float array (row, col, probability) - the same layout
    CoordinatesGrid.set_searched_areas expects.
    """
    total_cells = coordinates_grid.rows * coordinates_grid.cols

    if amount <= 0 or amount > total_cells:
        print(f"Amount must be between 1 and {total_cells} (rows * cols), not {amount}")
        return np.empty((0, 3), dtype=np.float64)

    flat_indices = np.random.choice(total_cells, size=amount, replace=False)
    rows, cols = np.unravel_index(flat_indices, (coordinates_grid.rows, coordinates_grid.cols))
    probabilities = np.random.uniform(0.0, CoordinatesGrid.MAX_PROBABILITY, size=amount)

    return np.column_stack((rows, cols, probabilities))


def generate_random_elevation_grid(rows: int, cols: int, altitude_range: tuple[float, float]) -> np.ndarray:
    """
    Generates a rows x cols elevation grid with altitudes drawn uniformly
    from altitude_range = (min_altitude, max_altitude), for feeding into
    init_weights_from_elevation.
    """
    min_altitude, max_altitude = altitude_range

    if rows <= 0 or cols <= 0:
        print(f"Grid size must be greater than zero, not {rows}x{cols}")
        return np.empty((0, 0), dtype=np.float64)

    if min_altitude > max_altitude:
        print(f"altitude_range minimum ({min_altitude}) must not exceed its maximum ({max_altitude})")
        return np.empty((0, 0), dtype=np.float64)

    return np.random.uniform(min_altitude, max_altitude, size=(rows, cols))


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