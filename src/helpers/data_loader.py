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
          "max_gradient": float,            (optional, default 0.3)
          "seed": int                       (optional, default 0)
        },
        "start_location": {"row": int, "col": int},
        "searched_person_locations": [
          {"row": int, "col": int, "probability": float},
          ...
        ],
        "blocked_cells": [                        (optional, default [])
          {"row": int, "col": int},
          ...
        ]
      }

    start_location is where the drone launches from and returns to (e.g. the
    rescue team's base) - a fixed point dictated by the real scenario, not
    something TrajectoryGenerator gets to choose.

    blocked_cells lists individual no-fly cells (storm cells, restricted
    airspace, terrain the drone can't overfly) as part of the scenario
    itself - a fixed fact about the mission, the same way start_location and
    searched_person_locations are, rather than randomly rolled per run. See
    coordinates_grid.test_data_generator.build_blocked_mask for turning this
    into the boolean array pathfinding actually uses, and
    generate_random_blocked_mask in that same module for the still-available
    randomized alternative.

    terrain.seed is what makes a scenario's randomly generated elevation
    reproducible: passed straight through as "terrain_seed" for callers
    (see main.py and helpers.benchmark._build_grid) to hand to
    coordinates_grid.test_data_generator.generate_random_terrain_
    coordinates' own `seed` parameter, so the same scenario file always
    regenerates the exact same map instead of a fresh random one every run.
    Defaults to 0 (not None) when the field is missing, so an older
    scenario file without it still gets a fixed, reproducible seed rather
    than silently reverting to non-deterministic terrain.

    Raises ValueError if start_location is itself listed in blocked_cells -
    a self-contradictory scenario (the drone can't launch from a cell it
    isn't allowed to enter) that should fail immediately at load time,
    rather than surfacing later as an empty path with no explanation from
    whichever algorithm happens to run.

    Returns a dict:
      {
        "rows": int, "cols": int,
        "altitude_range": (min_altitude, max_altitude),
        "cell_size_meters": float,
        "max_gradient": float,
        "terrain_seed": int,
        "start_row": int, "start_col": int,
        "search_areas": list of (3,) float ndarrays (row, col, probability) -
          the same layout CoordinatesGrid.set_searched_area_values expects.
        "blocked_cells": list of (row, col) int tuples - the same layout
          coordinates_grid.test_data_generator.build_blocked_mask expects.
      }
    """
    with open(file_path, "r") as data_file:
        raw_data = json.load(data_file)

    terrain = raw_data["terrain"]
    start_location = raw_data["start_location"]
    locations = raw_data.get("searched_person_locations", [])
    start_row, start_col = start_location["row"], start_location["col"]
    blocked_cells = [(cell["row"], cell["col"]) for cell in raw_data.get("blocked_cells", [])]

    if (start_row, start_col) in blocked_cells:
        raise ValueError(
            f"{file_path}: start_location ({start_row}, {start_col}) is also listed in blocked_cells - "
            "the drone can't launch from a cell it isn't allowed to enter."
        )

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
        "terrain_seed": terrain.get("seed", 0),
        "start_row": start_row,
        "start_col": start_col,
        "search_areas": search_areas,
        "blocked_cells": blocked_cells,
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
