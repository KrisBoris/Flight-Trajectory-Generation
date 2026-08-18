# visualizer.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Patch
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 - registers the "3d" projection
from PyQt5 import QtWidgets
import matplotlib
import matplotlib.colors as mcolors
import numpy as np
import sys


# Green (low) -> yellow (medium) -> red (high) probability spectrum.
PROBABILITY_COLORMAP = "RdYlGn_r"

BLOCKED_LABEL = "blocked (no-fly)"

# Target cells-across for the 3D surface mesh - see _build_3d_view.
SURFACE_MESH_TARGET = 50


def _max_pool_2d(array: np.ndarray, block_size: int) -> np.ndarray:
    """
    Downsamples array by taking the max within each block_size x block_size
    block (a trailing partial block, if rows/cols isn't a multiple of
    block_size, still gets its own max over whatever it has).

    Used instead of plain strided sampling (array[::block_size, ::block_size])
    when downsampling probability data for the 3D mesh: probability is sparse
    and spiky - a single high-probability search cell surrounded by
    background - so naive striding can land exactly between hotspots and
    silently drop them, and averaging would dilute one into the background.
    Max-pooling guarantees every hotspot survives downsampling somewhere in
    the reduced grid.
    """
    rows, cols = array.shape
    pooled_rows = -(-rows // block_size)
    pooled_cols = -(-cols // block_size)

    pooled = np.empty((pooled_rows, pooled_cols), dtype=array.dtype)

    for i in range(pooled_rows):
        for j in range(pooled_cols):
            block = array[i * block_size:(i + 1) * block_size, j * block_size:(j + 1) * block_size]
            pooled[i, j] = block.max()

    return pooled


class TrajectoryVisualizerWindow(QtWidgets.QMainWindow):
    """
    Displays a CoordinatesGrid's probability values as a color-coded matrix -
    green for low, yellow for medium, red for high probability of finding the
    searched person, scaled between the grid's own min and max - with the
    path found by TrajectoryGenerator and any blocked_mask no-fly cells drawn
    on top. When terrain_coordinates (real-world x, y, z per cell, in meters)
    is supplied, a second tab plots the same colors, path and blocked cells
    over the actual terrain shape instead of a flat grid.
    """

    def __init__(
        self,
        coordinates_grid: CoordinatesGrid,
        path: list = None,
        terrain_coordinates: np.ndarray = None,
        blocked_mask: np.ndarray = None,
    ):
        super().__init__()

        self.setWindowTitle("Flight Trajectory Visualizer")

        values = coordinates_grid.coordinates_values
        norm = mcolors.Normalize(vmin=float(values.min()), vmax=float(values.max()))

        tabs = QtWidgets.QTabWidget()
        tabs.addTab(self._build_2d_view(values, norm, path, blocked_mask), "2D grid")

        if terrain_coordinates is not None:
            # terrain_coordinates is rendered as-is - it's already the exact
            # altitudes the cost model used (see
            # coordinates_grid.test_data_generator.generate_random_terrain_coordinates
            # and its max_gradient parameter), so what's displayed always
            # matches what the drone's cost was computed against.
            tabs.addTab(self._build_3d_view(values, norm, path, terrain_coordinates, blocked_mask), "3D terrain")

        self.setCentralWidget(tabs)
        self.resize(1400, 1000)


    def _build_2d_view(self, values: np.ndarray, norm: mcolors.Normalize, path: list, blocked_mask: np.ndarray) -> FigureCanvasQTAgg:
        figure = Figure(figsize=(10, 9))
        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        axes = figure.add_subplot(111)

        image = axes.imshow(values, cmap=PROBABILITY_COLORMAP, norm=norm, origin="upper")
        figure.colorbar(image, ax=axes, label="Probability")
        axes.set_title("Search probability")
        axes.set_xlabel("col")
        axes.set_ylabel("row")

        legend_handles = []

        if blocked_mask is not None:
            legend_handles.append(self._draw_2d_blocked_mask(axes, blocked_mask))

        if path:
            legend_handles.extend(self._draw_2d_path(axes, path))

        if legend_handles:
            axes.legend(handles=legend_handles, loc="upper right")

        canvas.draw()
        return canvas


    def _draw_2d_blocked_mask(self, axes, blocked_mask: np.ndarray) -> Patch:
        # A black, semi-transparent overlay - opaque where blocked_mask is
        # True, fully transparent elsewhere - drawn on top of the probability
        # colors so blocked cells stay visually distinct regardless of what
        # probability they'd otherwise show.
        overlay = np.zeros((*blocked_mask.shape, 4))
        overlay[blocked_mask] = (0.0, 0.0, 0.0, 0.6)
        axes.imshow(overlay, origin="upper")

        return Patch(facecolor="black", alpha=0.6, label=BLOCKED_LABEL)


    def _draw_2d_path(self, axes, path: list) -> list:
        path_rows = [cell[0] for cell in path]
        path_cols = [cell[1] for cell in path]

        path_line, = axes.plot(path_cols, path_rows, color="blue", linewidth=1.5, marker="o", markersize=3, label="path")
        start_marker, = axes.plot(path_cols[0], path_rows[0], color="black", marker="*", markersize=16, label="start")

        return [path_line, start_marker]


    def _build_3d_view(
        self,
        values: np.ndarray,
        norm: mcolors.Normalize,
        path: list,
        terrain_coordinates: np.ndarray,
        blocked_mask: np.ndarray,
    ) -> FigureCanvasQTAgg:
        figure = Figure(figsize=(10, 9))
        canvas = FigureCanvasQTAgg(figure)
        canvas.setSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        axes = figure.add_subplot(111, projection="3d")
        figure.subplots_adjust(left=0.02, right=0.95, top=0.95, bottom=0.05)

        # plot_surface draws one edge-outlined quad per 4 neighboring cells.
        # At full resolution a big grid (e.g. 100x100 = 10000 quads) packs so
        # many thin edge lines into the same screen space that they create
        # their own visual noise - a "fishnet" moire - making even genuinely
        # smooth terrain look spiky. Subsampling to roughly SURFACE_MESH_TARGET
        # cells across keeps the mesh legible regardless of grid size; a
        # SURFACE_MESH_TARGET of 50 leaves grids up to 50x50 (the size the
        # surface styling was tuned against) completely unaffected.
        rows, cols = terrain_coordinates.shape[:2]
        stride = max(1, round(max(rows, cols) / SURFACE_MESH_TARGET))

        # The mesh geometry (x, y, z) is smooth after max_gradient-limiting,
        # so simple strided sampling loses no meaningful shape detail. The
        # probability values are the opposite - sparse, isolated spikes - so
        # they're downsampled with _max_pool_2d instead, and each face takes
        # the max (not the average) of its 4 corners: averaging would dilute
        # an isolated hotspot toward the background, and either approach
        # could otherwise make a "close to maximum" cell render yellow-ish or
        # not show up at all if a plain stride happened to skip it.
        x = terrain_coordinates[::stride, ::stride, 0]
        y = terrain_coordinates[::stride, ::stride, 1]
        z = terrain_coordinates[::stride, ::stride, 2]
        pooled_values = _max_pool_2d(values, stride)

        # plot_surface connects every 4 neighboring cells into one quad face,
        # so it needs one color per face rather than per cell. shade=False
        # keeps that color exact instead of matplotlib's default lighting
        # tint, which would distort the probability spectrum.
        colormap = matplotlib.colormaps[PROBABILITY_COLORMAP]
        face_values = np.maximum(
            np.maximum(pooled_values[:-1, :-1], pooled_values[1:, :-1]),
            np.maximum(pooled_values[:-1, 1:], pooled_values[1:, 1:]),
        )
        face_colors = colormap(norm(face_values))

        axes.plot_surface(x, y, z, facecolors=face_colors, rstride=1, cstride=1, linewidth=0.2, edgecolor="dimgray", shade=False, zorder=1)

        mappable = matplotlib.cm.ScalarMappable(norm=norm, cmap=colormap)
        mappable.set_array(values)
        figure.colorbar(mappable, ax=axes, label="Probability", shrink=0.6)

        axes.set_title("Terrain with search probability")
        axes.set_xlabel("x (m)")
        axes.set_ylabel("y (m)")
        axes.set_zlabel("z (m, altitude)")

        # Now that the terrain is a solid surface rather than a wireframe, a
        # path/marker drawn exactly at ground level gets partly hidden behind
        # it - mplot3d doesn't z-sort separate artists perfectly, it only
        # approximates depth per-artist. Forcing a high zorder on the
        # path/markers - with a low zorder on the surface itself - is what
        # actually guarantees mplot3d draws them on top, regardless of how
        # small the height offset is; hover_height itself only needs to be
        # just enough to read as "hovering above" rather than "painted onto"
        # the ground, so it's kept small and close to the surface.
        hover_height = max((z.max() - z.min()) * 0.03, 0.15)

        legend_handles = []

        if blocked_mask is not None:
            legend_handles.append(self._draw_3d_blocked_mask(axes, blocked_mask, terrain_coordinates, hover_height))

        if path:
            legend_handles.extend(self._draw_3d_path(axes, path, terrain_coordinates, hover_height))

        if legend_handles:
            axes.legend(handles=legend_handles, loc="upper right")

        canvas.draw()
        return canvas


    def _draw_3d_blocked_mask(self, axes, blocked_mask: np.ndarray, terrain_coordinates: np.ndarray, hover_height: float):
        blocked_x = terrain_coordinates[:, :, 0][blocked_mask]
        blocked_y = terrain_coordinates[:, :, 1][blocked_mask]
        blocked_z = terrain_coordinates[:, :, 2][blocked_mask] + hover_height

        return axes.scatter(blocked_x, blocked_y, blocked_z, color="black", marker="x", s=80, linewidths=2, label=BLOCKED_LABEL, zorder=10)


    def _draw_3d_path(self, axes, path: list, terrain_coordinates: np.ndarray, hover_height: float) -> list:
        path_x = [terrain_coordinates[row, col, 0] for row, col in path]
        path_y = [terrain_coordinates[row, col, 1] for row, col in path]
        path_z = [terrain_coordinates[row, col, 2] + hover_height for row, col in path]

        path_line, = axes.plot(path_x, path_y, path_z, color="blue", linewidth=2.5, marker="o", markersize=4, label="path", zorder=11)
        # depthshade=False disables mplot3d's default distance-based alpha
        # fade - without it, this single black star can fade into
        # near-invisibility against the terrain depending on the current
        # view/zoom, since there's no averaging with nearby points the way a
        # dense scatter would have. magenta is used deliberately because it
        # falls outside PROBABILITY_COLORMAP's red -> yellow -> green range
        # entirely - a fill color pulled from inside that range (e.g.
        # yellow) can blend right into terrain of a similar shade, and the
        # background/default probability often lands close to yellow. The
        # black edge adds definition against light terrain, and the bigger
        # size makes it easier to spot at a glance.
        start_marker = axes.scatter(
            [path_x[0]], [path_y[0]], [path_z[0]],
            color="magenta", marker="*", s=400, edgecolors="black", linewidths=1,
            depthshade=False, label="start", zorder=20,
        )

        return [path_line, start_marker]


def launch_gui(
    coordinates_grid: CoordinatesGrid,
    path: list = None,
    terrain_coordinates: np.ndarray = None,
    blocked_mask: np.ndarray = None,
) -> None:
    """
    Opens the trajectory visualizer window and blocks until it's closed.
    """
    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication(sys.argv)
    window = TrajectoryVisualizerWindow(
        coordinates_grid,
        path=path,
        terrain_coordinates=terrain_coordinates,
        blocked_mask=blocked_mask,
    )
    window.showMaximized()
    app.exec_()
