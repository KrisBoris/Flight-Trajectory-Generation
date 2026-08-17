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


def _limit_elevation_gradient(
    elevation: np.ndarray,
    cell_size_meters: float,
    max_gradient: float,
    max_iterations: int = 2000,
) -> np.ndarray:
    """
    Redistributes elevation between every pair of 4-connected neighbors so
    that no two adjacent cells end up differing in altitude by more than
    max_gradient * cell_size_meters.

    max_gradient is a slope ratio - rise over run, the same way you'd read a
    grade percentage (just not multiplied by 100). Concretely:
      - max_gradient = 1.0   -> neighbors can differ by up to a full
        cell_size_meters (a 45 degree edge - steep, cliff-like).
      - max_gradient = 0.3   -> roughly a 17 degree slope; rugged but
        walkable/flyable mountain terrain.
      - max_gradient = 0.05  -> roughly 3 degrees; gentle rolling hills.
    Since the cap is defined per meter of horizontal distance rather than
    per cell, the same max_gradient value means the same real-world
    steepness regardless of cell_size_meters or grid resolution.

    This is a thermal-erosion-style relaxation: whenever an edge exceeds the
    cap, half the excess difference slides from the higher cell to the lower
    one - like sediment sliding downhill until every slope settles at or
    below the repose angle. Repeated until every edge satisfies the cap, or
    max_iterations is reached as a safety bound (in practice this converges
    in well under a few hundred passes regardless of grid size, since the
    check below stops as soon as nothing changes).
    """
    smoothed = elevation.astype(np.float64).copy()
    max_difference = max_gradient * cell_size_meters
    tolerance = max(max_difference * 1e-6, 1e-9)

    for _ in range(max_iterations):
        changed = False

        for delta_row, delta_col in ((0, 1), (1, 0)):
            if delta_row == 0:
                a, b = smoothed[:, :-1], smoothed[:, 1:]
            else:
                a, b = smoothed[:-1, :], smoothed[1:, :]

            difference = a - b
            excess = np.abs(difference) - max_difference
            exceeding = excess > tolerance

            if not np.any(exceeding):
                continue

            changed = True
            transfer = np.where(exceeding, np.sign(difference) * excess / 2.0, 0.0)
            a -= transfer
            b += transfer

        if not changed:
            break

    return smoothed


def generate_random_terrain_coordinates(
    rows: int,
    cols: int,
    altitude_range: tuple[float, float],
    cell_size_meters: float = 1.0,
    max_gradient: float = 0.3,
) -> np.ndarray:
    """
    Generates a rows x cols x 3 array of local (x, y, z) coordinates in
    meters, for feeding into WeightsGrid.init_from_elevation. x grows with
    column and y grows with row going up (row 0 is the "up" direction, see
    constants.Constants.DIRECTIONS), spaced cell_size_meters apart, so the grid's
    bottom-left cell - (row=rows-1, col=0) - sits at x = y = 0.

    z (altitude above sea level) starts as independent uniform noise, which
    on its own looks nothing like a mountain (every cell is unrelated to its
    neighbor). _limit_elevation_gradient then redistributes it so no two
    neighboring cells differ by more than max_gradient * cell_size_meters -
    see that function's docstring for what max_gradient means and how to
    tune it - and the result is shifted (not rescaled) so its lowest point
    sits at min_altitude.

    It's a shift rather than a rescale deliberately: the redistribution
    pulls extreme values toward the mean and shrinks the elevation range, but
    stretching it back out to exactly fill altitude_range would multiply
    every gradient by the same stretch factor - directly undoing the
    max_gradient cap this function just enforced. A tight max_gradient on a
    small grid may therefore land well short of max_altitude, since only so
    much total relief fits within the cap over a limited number of cells;
    that's expected, not a bug.

    The returned coordinates are the actual altitudes used everywhere
    downstream - the cost model (WeightsGrid.init_from_elevation) and the 3D
    visualization (gui.visualizer) both see exactly this data, unmodified,
    so what you look at always matches what the drone's cost was computed
    against.
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
    grid_z = _limit_elevation_gradient(grid_z, cell_size_meters, max_gradient)
    grid_z = grid_z - grid_z.min() + min_altitude

    if grid_z.max() < max_altitude - (max_altitude - min_altitude) * 0.05:
        print(
            f"Note: max_gradient={max_gradient} kept this {rows}x{cols} terrain's peak at "
            f"{grid_z.max():.1f}, well under the requested max_altitude ({max_altitude}) - "
            f"a tighter cap needs a bigger grid (or a smaller cell_size_meters) to reach the same relief."
        )

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