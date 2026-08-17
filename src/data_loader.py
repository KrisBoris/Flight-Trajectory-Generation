# data_loader.py

import json
import numpy as np


def load_mission_data(file_path) -> dict:
    """
    Loads a scenario data file (JSON) describing the terrain to generate
    (rows, cols, altitude_range, cell_size_meters, max_gradient - see
    coordinates_grid.test_data_generator.generate_random_terrain_coordinates)
    and the searched person's possible locations with their probabilities.

    Expected JSON shape:
      {
        "terrain": {
          "rows": int, "cols": int,
          "altitude_range": [min_altitude, max_altitude],
          "cell_size_meters": float,        (optional, default 1.0)
          "max_gradient": float             (optional, default 0.3)
        },
        "searched_person_locations": [
          {"row": int, "col": int, "probability": float},
          ...
        ]
      }

    Returns a dict:
      {
        "rows": int, "cols": int,
        "altitude_range": (min_altitude, max_altitude),
        "cell_size_meters": float,
        "max_gradient": float,
        "search_areas": list of (3,) float ndarrays (row, col, probability) -
          the same layout CoordinatesGrid.set_searched_areas expects.
      }
    """
    with open(file_path, "r") as data_file:
        raw_data = json.load(data_file)

    terrain = raw_data["terrain"]
    locations = raw_data.get("searched_person_locations", [])

    search_areas = [
        np.array([location["row"], location["col"], location["probability"]], dtype=np.float64)
        for location in locations
    ]

    return {
        "rows": terrain["rows"],
        "cols": terrain["cols"],
        "altitude_range": tuple(terrain["altitude_range"]),
        "cell_size_meters": terrain.get("cell_size_meters", 1.0),
        "max_gradient": terrain.get("max_gradient", 0.3),
        "search_areas": search_areas,
    }


def load_drone_params(file_path) -> dict:
    """
    Loads a drone configuration file (JSON) describing the cost model used by
    WeightsGrid.init_from_elevation and the flight budget/constraints used by
    TrajectoryGenerator.find_best_path.

    Expected JSON shape:
      {
        "climb_cost_per_meter": float,
        "descent_cost_per_meter": float,
        "base_cost": float,                    (optional, default 1.0)
        "max_cost": float,
        "require_return_to_base": bool          (optional, default true)
      }
    """
    with open(file_path, "r") as data_file:
        raw_data = json.load(data_file)

    return {
        "climb_cost_per_meter": raw_data["climb_cost_per_meter"],
        "descent_cost_per_meter": raw_data["descent_cost_per_meter"],
        "base_cost": raw_data.get("base_cost", 1.0),
        "max_cost": raw_data["max_cost"],
        "require_return_to_base": raw_data.get("require_return_to_base", True),
    }
