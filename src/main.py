# main.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from coordinates_grid.weights_grid import WeightsGrid
from coordinates_grid.test_data_generator import (
    generate_random_terrain_coordinates,
    generate_random_blocked_mask,
)
from data_loader import load_mission_data, load_drone_params
from gui.visualizer import launch_gui
from pathlib import Path
from pathfinding_algorithms.trajectory_generator import TrajectoryGenerator
import numpy as np


MISSION_DATA_FILE_NAME = "scenario.json"
MISSION_DATA = Path(__file__).resolve().parent.parent / "test_data" / MISSION_DATA_FILE_NAME
DRONE_PARAMS_FILE_NAME = "drone_config.json"
DRONE_PARAMS_DATA = Path(__file__).resolve().parent / DRONE_PARAMS_FILE_NAME



def main():

    mission_data = load_mission_data(MISSION_DATA)
    drone_params = load_drone_params(DRONE_PARAMS_DATA)
    rows, cols = mission_data["rows"], mission_data["cols"]

    coordinates_grid = CoordinatesGrid(
        coordinates_values=np.ones((rows, cols)),
        weights_grid=WeightsGrid(weights=np.ones((rows, cols, 8))),
    )
    coordinates_grid.init_grids(rows, cols)

    coordinates_grid.set_searched_areas(mission_data["search_areas"])

    terrain_coordinates = generate_random_terrain_coordinates(
        rows,
        cols,
        altitude_range=mission_data["altitude_range"],
        cell_size_meters=mission_data["cell_size_meters"],
        max_gradient=mission_data["max_gradient"],
    )
    coordinates_grid.weights_grid.init_from_elevation(
        terrain_coordinates,
        climb_cost_per_meter=drone_params["climb_cost_per_meter"],
        descent_cost_per_meter=drone_params["descent_cost_per_meter"],
        base_cost=drone_params["base_cost"],
    )

    blocked_mask = generate_random_blocked_mask(
        coordinates_grid, blocked_shapes=[np.ones((2, 2)), np.ones((1, 3))]
    )

    trajectory_generator = TrajectoryGenerator(grid=coordinates_grid)
    path, total_value, cost_used = trajectory_generator.find_best_path(
        max_cost=drone_params["max_cost"],
        require_return_to_base=drone_params["require_return_to_base"],
        blocked_mask=blocked_mask,
        algorithm="lowest_cost"
    )

    print(f"Best path found: {len(path)} steps, total value {total_value:.2f}, cost used {cost_used:.2f}")

    launch_gui(
        coordinates_grid,
        path=path,
        terrain_coordinates=terrain_coordinates,
        blocked_mask=blocked_mask,
    )



if __name__ == "__main__":
    main()
