# visualizer.py

from coordinates_grid.coordinates_grid import CoordinatesGrid
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
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

# Roughly how many direction arrows to draw along a path, regardless of its
# length - see _path_arrow_indices.
PATH_ARROW_COUNT = 60

# Roughly how many sequence-number labels to draw along a path, regardless
# of its length - see _path_label_indices. Numbering every single cell was
# tried first and looked fine on a short synthetic path, but on any real
# path from this project's algorithms (hundreds to low thousands of cells)
# the numbered dots overlap into an unreadable solid blob - this sampled
# subset keeps the numbers legible at any path length instead.
PATH_LABEL_COUNT = 40


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


def _path_end_index(path: list) -> int:
    """
    Index of the point actually worth marking as "the end" of the path - not
    necessarily path[-1]: every pathfinding algorithm appends a return leg
    (path + path[-2::-1]) when require_return_to_base is set, which makes
    path[-1] just the start cell again. Marking that would sit exactly on
    top of the start marker and show nothing new, so what's actually useful
    is the turnaround point - the farthest cell the drone reached before
    heading back.

    A return-trip path is always a palindrome (the outbound cells, then the
    same cells reversed), so its center index is that turnaround point.
    Detected here via path[0] == path[-1] (with no return leg, the path
    isn't generally a palindrome, so path[-1] is already the genuine
    endpoint) rather than threading a require_return_to_base flag through
    the whole call chain just for this.
    """
    if len(path) > 1 and path[0] == path[-1]:
        return len(path) // 2
    return len(path) - 1


def _path_arrow_indices(path_length: int) -> np.ndarray:
    """
    Indices i (0 <= i < path_length - 1) at which to draw a direction arrow
    from path[i] to path[i + 1], spaced out so a long path (which can be
    thousands of cells, especially with a return leg) gets roughly
    PATH_ARROW_COUNT arrows rather than one per edge - the latter would be
    unreadable clutter and slow to render. Returns an empty array for a path
    with fewer than 2 cells (nothing to point between).
    """
    if path_length < 2:
        return np.array([], dtype=int)

    stride = max(1, (path_length - 1) // PATH_ARROW_COUNT)
    return np.arange(0, path_length - 1, stride)


def _path_label_indices(path_length: int) -> np.ndarray:
    """
    Indices of path cells to number with their sequence position, spaced out
    so a long path gets roughly PATH_LABEL_COUNT numbers rather than one per
    cell - see PATH_LABEL_COUNT for why. Always includes the first and last
    index (0 and path_length - 1) so the numbered path still shows its true
    start and end sequence numbers, even when the stride would otherwise
    skip past them. Returns every index for a path no longer than
    PATH_LABEL_COUNT to begin with, matching the original "number every
    dot" behavior for short paths where that is actually readable.
    """
    if path_length <= 0:
        return np.array([], dtype=int)
    if path_length <= PATH_LABEL_COUNT:
        return np.arange(path_length)

    stride = max(1, path_length // PATH_LABEL_COUNT)
    indices = np.arange(0, path_length, stride)
    if indices[-1] != path_length - 1:
        indices = np.append(indices, path_length - 1)
    return indices


def _direction_chevron_offsets(dx: float, dy: float, wing_length: float, wing_angle_degrees: float = 25.0) -> tuple:
    """
    Given a horizontal direction vector (dx, dy) pointing from an arrow's
    tail toward its tip, returns the two (offset_x, offset_y) vectors - each
    wing_length long, swept wing_angle_degrees to either side of straight
    back along the shaft - that a chevron arrowhead's two wings should
    extend from the tip.

    This is a plain 2D rotation kept entirely in the horizontal plane,
    deliberately not using matplotlib's own 3D quiver arrowheads: quiver's
    arrowhead geometry is computed in raw data coordinates and then carried
    through mplot3d's per-axis independent scaling, so on a plot where axes
    are scaled very differently from each other - exactly this app's case,
    where terrain height commonly spans under a meter while x/y span
    hundreds of meters (a low max_gradient scenario) - the arrowhead wings
    get stretched by whichever axis they happen to have a component in,
    rendering as huge, wildly spiked zigzags instead of small arrowheads.
    Plain line segments between literal (x, y, z) coordinates - as used
    here, and as the path line itself already uses - don't have that
    problem, since there's no implied "correct angle" for a 2-point segment
    to preserve; it just connects its two literal endpoints.
    """
    horizontal_length = np.hypot(dx, dy)
    if horizontal_length == 0:
        return (0.0, 0.0), (0.0, 0.0)

    back_x = -dx / horizontal_length * wing_length
    back_y = -dy / horizontal_length * wing_length
    angle = np.radians(wing_angle_degrees)

    def rotate(x, y, theta):
        """Rotates the 2D vector (x, y) counterclockwise by theta radians."""
        return x * np.cos(theta) - y * np.sin(theta), x * np.sin(theta) + y * np.cos(theta)

    return rotate(back_x, back_y, angle), rotate(back_x, back_y, -angle)


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
        """
        Builds the window's tabs: a flat "2D grid" view always, plus a
        "3D terrain" view only when terrain_coordinates is given (there's
        no terrain to plot a surface over otherwise). Both views share the
        same color normalization, scaled to this coordinates_grid's own
        min/max value, so the two tabs read consistently.
        """
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
        """
        Builds the flat "2D grid" tab: values rendered as a color-coded
        image (see PROBABILITY_COLORMAP), with blocked_mask (if given) and
        path (if given) drawn on top via _draw_2d_blocked_mask/
        _draw_2d_path. Returns the finished canvas ready to add as a tab.
        """
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
        """
        Draws blocked_mask as a black, semi-transparent overlay on the 2D
        grid axes (see the overlay comment below) and returns the legend
        entry for it.
        """
        # A black, semi-transparent overlay - opaque where blocked_mask is
        # True, fully transparent elsewhere - drawn on top of the probability
        # colors so blocked cells stay visually distinct regardless of what
        # probability they'd otherwise show.
        overlay = np.zeros((*blocked_mask.shape, 4))
        overlay[blocked_mask] = (0.0, 0.0, 0.0, 0.6)
        axes.imshow(overlay, origin="upper")

        return Patch(facecolor="black", alpha=0.6, label=BLOCKED_LABEL)


    def _draw_2d_path(self, axes, path: list) -> list:
        """
        Draws the full path as a connected line on the 2D grid axes, plus
        a start marker, an end marker at the true turnaround point (see
        _path_end_index), sampled sequence-number labels (see
        _path_label_indices) and sampled direction arrows (see
        _path_arrow_indices). Returns the list of legend handles for
        whichever of these were actually drawn.
        """
        path_rows = [cell[0] for cell in path]
        path_cols = [cell[1] for cell in path]

        path_line, = axes.plot(path_cols, path_rows, color="blue", linewidth=1.5, marker="o", markersize=3, label="path", zorder=3)

        # A sampled subset of cells (see _path_label_indices/PATH_LABEL_
        # COUNT) gets a bigger white-filled marker with a checkpoint number
        # inside - numbering every single dot was tried first and looked
        # fine on a short synthetic path, but overlapped into an unreadable
        # solid blob on any real, hundreds-of-cells path from this
        # project's algorithms. The number printed is this checkpoint's own
        # 1-indexed position among the sampled subset (1, 2, 3, ...) - not
        # its raw index in `path` - since with a stride skipping most cells,
        # printing the raw path index would show large, unexplained gaps
        # (e.g. 20, 40, 60, ...) instead of a simple visiting order.
        label_indices = _path_label_indices(len(path))
        if label_indices.size > 0:
            label_x = np.asarray(path_cols)[label_indices]
            label_y = np.asarray(path_rows)[label_indices]
            axes.scatter(label_x, label_y, s=150, facecolors="white", edgecolors="blue", linewidths=1.1, zorder=4)
            for label_number, (x, y) in enumerate(zip(label_x, label_y), start=1):
                axes.annotate(str(label_number), (x, y), ha="center", va="center", fontsize=6, color="blue", zorder=4.5)

        start_marker, = axes.plot(path_cols[0], path_rows[0], color="black", marker="*", markersize=16, label="start", zorder=6)

        handles = [path_line, start_marker]
        if label_indices.size > 0:
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="white", markeredgecolor="blue", label="sequence #"))

        end_index = _path_end_index(path)
        if end_index != 0:
            end_marker, = axes.plot(
                path_cols[end_index], path_rows[end_index],
                color="red", marker="X", markersize=13, markeredgecolor="black", label="end", zorder=6,
            )
            handles.append(end_marker)

        arrow_indices = _path_arrow_indices(len(path))
        if arrow_indices.size > 0:
            arrow_x = np.asarray(path_cols)[arrow_indices]
            arrow_y = np.asarray(path_rows)[arrow_indices]
            arrow_dx = np.asarray(path_cols)[arrow_indices + 1] - arrow_x
            arrow_dy = np.asarray(path_rows)[arrow_indices + 1] - arrow_y

            axes.quiver(
                arrow_x, arrow_y, arrow_dx, arrow_dy,
                angles="xy", scale_units="xy", scale=1,
                color="black", width=0.003, headwidth=5, headlength=6, zorder=5,
            )
            handles.append(Line2D([0], [0], color="black", marker=">", linestyle="None", markersize=8, label="direction"))

        return handles


    def _build_3d_view(
        self,
        values: np.ndarray,
        norm: mcolors.Normalize,
        path: list,
        terrain_coordinates: np.ndarray,
        blocked_mask: np.ndarray,
    ) -> FigureCanvasQTAgg:
        """
        Builds the "3D terrain" tab: a colored surface mesh over
        terrain_coordinates (subsampled - see SURFACE_MESH_TARGET - and
        colored via a max-pooled, per-face version of values), with
        blocked_mask (if given) and path (if given) drawn hovering just
        above it via _draw_3d_blocked_mask/_draw_3d_path. Returns the
        finished canvas ready to add as a tab.
        """
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
        """
        Marks every blocked_mask cell with a black X, hovering
        hover_height above the terrain surface (see _build_3d_view for why
        a height offset is needed at all) so it stays visible instead of
        being drawn on/under the surface mesh. Returns the legend entry
        for it.
        """
        blocked_x = terrain_coordinates[:, :, 0][blocked_mask]
        blocked_y = terrain_coordinates[:, :, 1][blocked_mask]
        blocked_z = terrain_coordinates[:, :, 2][blocked_mask] + hover_height

        return axes.scatter(blocked_x, blocked_y, blocked_z, color="black", marker="x", s=80, linewidths=2, label=BLOCKED_LABEL, zorder=10)


    def _draw_3d_path(self, axes, path: list, terrain_coordinates: np.ndarray, hover_height: float) -> list:
        """
        The 3D counterpart of _draw_2d_path: draws the path as a connected
        line hovering hover_height above the terrain, plus a start marker,
        an end marker at the true turnaround point (see _path_end_index),
        sampled sequence-number labels (see _path_label_indices) and
        sampled direction chevrons (see _direction_chevron_offsets, used
        instead of matplotlib's own 3D arrowheads - see that function's
        docstring for why). Returns the list of legend handles for
        whichever of these were actually drawn.
        """
        path_x = [terrain_coordinates[row, col, 0] for row, col in path]
        path_y = [terrain_coordinates[row, col, 1] for row, col in path]
        path_z = [terrain_coordinates[row, col, 2] + hover_height for row, col in path]

        path_line, = axes.plot(path_x, path_y, path_z, color="blue", linewidth=2.5, marker="o", markersize=4, label="path", zorder=11)

        # A sampled subset of cells (see _path_label_indices/PATH_LABEL_
        # COUNT) gets a bigger white-filled marker with a checkpoint number
        # inside - numbering every single dot was tried first and looked
        # fine on a short synthetic path, but overlapped into an unreadable
        # solid blob on any real, hundreds-of-cells path from this
        # project's algorithms. The number printed is this checkpoint's own
        # 1-indexed position among the sampled subset (1, 2, 3, ...) - not
        # its raw index in `path` - since with a stride skipping most cells,
        # printing the raw path index would show large, unexplained gaps
        # (e.g. 20, 40, 60, ...) instead of a simple visiting order.
        label_indices = _path_label_indices(len(path))
        if label_indices.size > 0:
            label_x = np.asarray(path_x)[label_indices]
            label_y = np.asarray(path_y)[label_indices]
            label_z = np.asarray(path_z)[label_indices]
            axes.scatter(
                label_x, label_y, label_z, facecolors="white", edgecolors="blue", linewidths=1.1,
                s=110, depthshade=False, zorder=13,
            )
            for label_number, (x, y, z) in enumerate(zip(label_x, label_y, label_z), start=1):
                axes.text(x, y, z, str(label_number), ha="center", va="center", fontsize=6, color="blue", zorder=14)

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

        handles = [path_line, start_marker]
        if label_indices.size > 0:
            handles.append(Line2D([0], [0], marker="o", linestyle="None", markersize=8, markerfacecolor="white", markeredgecolor="blue", label="sequence #"))

        # See _path_end_index - with a return leg, path[-1] is just the
        # start cell again, so the real "end" worth marking is the
        # turnaround point.
        end_index = _path_end_index(path)
        if end_index != 0:
            end_marker = axes.scatter(
                [path_x[end_index]], [path_y[end_index]], [path_z[end_index]],
                color="red", marker="*", s=400, edgecolors="black", linewidths=1,
                depthshade=False, label="end", zorder=20,
            )
            handles.append(end_marker)

        arrow_indices = _path_arrow_indices(len(path))
        if arrow_indices.size > 0:
            for i in arrow_indices:
                segment_dx = path_x[i + 1] - path_x[i]
                segment_dy = path_y[i + 1] - path_y[i]
                tip_x, tip_y, tip_z = path_x[i + 1], path_y[i + 1], path_z[i + 1]
                # Wing length scales with this segment's own horizontal
                # length (a fixed grid-step distance) rather than the
                # plot's overall span, so chevrons stay a legible size
                # regardless of how large the map is.
                wing_length = np.hypot(segment_dx, segment_dy) * 0.6
                for wing_dx, wing_dy in _direction_chevron_offsets(segment_dx, segment_dy, wing_length):
                    axes.plot([tip_x, tip_x + wing_dx], [tip_y, tip_y + wing_dy], [tip_z, tip_z], color="black", linewidth=2, zorder=15)

            handles.append(Line2D([0], [0], color="black", marker=">", linestyle="None", markersize=8, label="direction"))

        return handles


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
