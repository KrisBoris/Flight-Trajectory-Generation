# main.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from coordinates_grid.weights_grid import WeightsGrid
from coordinates_grid.test_data_generator import (
    generate_random_positions,
    generate_random_terrain_coordinates,
    generate_random_blocked_mask,
)
from gui.visualizer import launch_gui
from trajectory_generator import TrajectoryGenerator
import numpy as np


def main():
    rows, cols = 50, 50

    coordinates_grid = CoordinatesGrid(
        coordinates_values=np.ones((rows, cols)),
        weights_grid=WeightsGrid(weights=np.ones((rows, cols, 8))),
    )
    coordinates_grid.init_grids(rows, cols)

    search_areas = generate_random_positions(coordinates_grid, amount=15)
    coordinates_grid.set_searched_areas(search_areas)

    terrain_coordinates = generate_random_terrain_coordinates(
        rows, cols, altitude_range=(0.0, 50.0), cell_size_meters=10.0
    )
    coordinates_grid.weights_grid.init_from_elevation(
        terrain_coordinates, climb_cost_per_meter=0.5, descent_cost_per_meter=0.3, base_cost=1.0
    )

    blocked_mask = generate_random_blocked_mask(
        coordinates_grid, blocked_shapes=[np.ones((2, 2)), np.ones((1, 3))]
    )

    trajectory_generator = TrajectoryGenerator(grid=coordinates_grid)
    path, total_value, cost_used = trajectory_generator.find_best_path(max_cost=500, blocked_mask=blocked_mask)

    print(f"Best path found: {len(path)} steps, total value {total_value:.2f}, cost used {cost_used:.2f}")

    launch_gui(coordinates_grid, path=path, terrain_coordinates=terrain_coordinates, blocked_mask=blocked_mask)


if __name__ == "__main__":
    main()
