"""Matplotlib helpers for field, potential and trajectory figures.

Every plotting function builds a figure/axes pair and returns ``(fig,
ax)``; none of them calls ``plt.show()`` so that the caller (usually a
Jupyter notebook) controls the display backend.

The 2D helpers expect callables that accept meshgrid coordinate arrays:

- ``field_func(x, y)`` -> ``(Ex, Ey)``: two 2D grids of the vector
  field components on the ``x``/``y`` mesh.
- ``potential_func(x, y)`` -> ``V``: one 2D grid of the scalar
  potential on the ``x``/``y`` mesh.

The 3D helpers expect callables that accept an array of field points
of shape ``[N, 3]`` and return the field of shape ``[N, 3]``.

All quantities are in SI units.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3d proj)

__all__ = [
    "plot_field_lines",
    "plot_equipotentials",
    "plot_field_and_potential",
    "plot_vector_field",
    "plot_field_3d",
    "plot_trajectory_3d",
    "plot_decay",
]


def _xy_mesh(extent: float, n: int = 201) -> Tuple[np.ndarray, np.ndarray]:
    """Return a square ``x``/``y`` meshgrid over ``[-extent, extent]``."""
    axis = np.linspace(-extent, extent, n)
    return np.meshgrid(axis, axis)


def _get_ax(ax: Optional[plt.Axes], threed: bool = False) -> plt.Axes:
    """Return the given axes, or create a new spacious figure."""
    if ax is not None:
        return ax
    figsize = (9, 7) if threed else (8, 6)
    fig = plt.figure(figsize=figsize, constrained_layout=True)
    if threed:
        return fig.add_subplot(111, projection="3d")
    return fig.add_subplot(111)


def _get_fig(ax: plt.Axes) -> plt.Figure:
    """Return the figure that owns ``ax``."""
    return ax.figure


def plot_field_lines(
    field_func: Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ],
    extent: float = 2.0,
    density: float = 1.4,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Streamplot of electric field lines, colored by ``|E|``.

    Implements the visualization of ``E(x, y) = (Ex, Ey)`` as
    streamline trajectories ``dr/ds = E/|E|``; the streamline density
    is controlled by ``density`` and the color encodes the field
    magnitude ``|E| = sqrt(Ex^2 + Ey^2)``.
    """
    x, y = _xy_mesh(extent)
    ex, ey = field_func(x, y)
    ax = _get_ax(ax)
    magnitude = np.sqrt(ex**2 + ey**2)
    speed = np.log10(np.where(magnitude > 0, magnitude, np.nan))
    ax.streamplot(x, y, ex, ey, color=speed, density=density, cmap="viridis")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title if title else "Field lines (color: log10 |E|)")
    return _get_fig(ax), ax


def plot_equipotentials(
    potential_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    extent: float = 2.0,
    levels: int = 20,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Filled and line contours of the scalar potential ``V``.

    Implements the visualization of the equipotential surfaces
    ``V(x, y) = const``; each contour line is one equipotential curve
    and the filled background encodes the potential value.
    """
    x, y = _xy_mesh(extent)
    v = potential_func(x, y)
    ax = _get_ax(ax)
    finite = np.isfinite(v)
    if finite.any():
        lo = float(np.nanmin(v[finite]))
        hi = float(np.nanmax(v[finite]))
        if lo == hi:
            lo, hi = lo - 1.0, hi + 1.0
    else:
        lo, hi = -1.0, 1.0
    image = ax.contourf(
        x, y, v, levels=levels, cmap="RdBu_r", vmin=lo, vmax=hi
    )
    ax.contour(x, y, v, levels=levels, colors="k", linewidths=0.4, alpha=0.5)
    colorbar = _get_fig(ax).colorbar(image, ax=ax, shrink=0.9)
    colorbar.set_label("V (V)")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title if title else "Equipotentials of V")
    return _get_fig(ax), ax


def plot_field_and_potential(
    field_func: Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ],
    potential_func: Callable[[np.ndarray, np.ndarray], np.ndarray],
    extent: float = 2.0,
    density: float = 1.4,
    levels: int = 20,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """The classic textbook figure: equipotentials under field lines.

    Implements the combined visualization of ``E`` (streamlines) and
    ``V`` (contours) for the same source distribution; the
    equipotential contours are drawn beneath the field lines so that
    the orthogonality ``E . dV = 0`` is visible as
    lines crossing contours at right angles.
    """
    x, y = _xy_mesh(extent)
    ex, ey = field_func(x, y)
    v = potential_func(x, y)
    fig = plt.figure(figsize=(8, 6), constrained_layout=True)
    ax = fig.add_subplot(111)
    ax.contourf(x, y, v, levels=levels, cmap="RdBu_r", alpha=0.7)
    ax.contour(x, y, v, levels=levels, colors="k", linewidths=0.4, alpha=0.4)
    magnitude = np.sqrt(ex**2 + ey**2)
    speed = np.log10(np.where(magnitude > 0, magnitude, np.nan))
    ax.streamplot(x, y, ex, ey, color=speed, density=density, cmap="viridis")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title if title else "Field lines over equipotentials")
    return fig, ax


def plot_vector_field(
    field_func: Callable[
        [np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]
    ],
    extent: float = 2.0,
    n: int = 12,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Quiver plot of the field on an ``n x n`` grid, normalized arrows.

    Implements the visualization of ``E(x, y)`` as arrows of unit
    length along ``E/|E|`` (direction only; the color encodes
    ``log10 |E|`` so the magnitude information is kept).
    """
    x, y = _xy_mesh(extent, n=n)
    ex, ey = field_func(x, y)
    ax = _get_ax(ax)
    magnitude = np.sqrt(ex**2 + ey**2)
    scale = np.where(magnitude > 0, magnitude, np.inf)
    ex_unit = ex / scale
    ey_unit = ey / scale
    color = np.log10(np.where(magnitude > 0, magnitude, np.nan))
    quiver = ax.quiver(
        x, y, ex_unit, ey_unit, color, cmap="plasma", pivot="mid", clim=(-1, 2)
    )
    colorbar = _get_fig(ax).colorbar(quiver, ax=ax, shrink=0.9)
    colorbar.set_label("log10 |E| (V/m)")
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_aspect("equal")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title(title if title else "Normalized field vectors")
    return _get_fig(ax), ax


def plot_field_3d(
    field_func: Callable[[np.ndarray], np.ndarray],
    extent: float = 1.5,
    n: int = 5,
) -> Tuple[plt.Figure, plt.Axes]:
    """3D quiver of the field on an ``n^3`` grid, probing the XY plane.

    Implements the visualization of ``E(x, y, z)`` on a cubic grid of
    spacing ``2*extent/(n-1)`` centered at the origin; arrows of unit
    length show the direction of ``E/|E|``.  The initial view is the
    XY plane (``elev=0``, ``azim=-90``).
    """
    axis = np.linspace(-extent, extent, n)
    x, y, z = np.meshgrid(axis, axis, axis, indexing="ij")
    points = np.column_stack([x.ravel(), y.ravel(), z.ravel()])
    field = np.asarray(field_func(points), dtype=float)
    if field.shape != points.shape:
        raise ValueError(
            "field_func must return an array of shape [N, 3] for "
            "[N, 3] input points"
        )
    magnitude = np.linalg.norm(field, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = np.where(
            magnitude[:, None] > 0, field / magnitude[:, None], 0.0
        )
    fig = plt.figure(figsize=(9, 7), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    ax.quiver(
        points[:, 0],
        points[:, 1],
        points[:, 2],
        unit[:, 0],
        unit[:, 1],
        unit[:, 2],
        length=0.6 * (2 * extent / max(n - 1, 1)),
        normalize=False,
    )
    ax.set_xlim(-extent, extent)
    ax.set_ylim(-extent, extent)
    ax.set_zlim(-extent, extent)
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.view_init(elev=0, azim=-90)
    ax.set_title("3D field vectors (XY plane view)")
    return fig, ax


def plot_trajectory_3d(
    positions: np.ndarray,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """3D line plot of a charge trajectory ``r(t)``.

    Implements the visualization of a particle path given as an array
    of positions of shape ``[M, 3]``; the line is colored by
    increasing time (sample index) and the start/end points are
    marked.
    """
    positions = np.asarray(positions, dtype=float)
    if positions.ndim != 2 or positions.shape[1] != 3:
        raise ValueError("positions must have shape [M, 3]")
    ax = _get_ax(ax, threed=True)
    steps = np.arange(positions.shape[0])
    ax.plot(
        positions[:, 0],
        positions[:, 1],
        positions[:, 2],
        color="crimson",
        linewidth=1.5,
        label="trajectory",
    )
    if positions.shape[0] > 1:
        image = ax.scatter(
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
            c=steps,
            cmap="viridis",
            s=8,
        )
        colorbar = _get_fig(ax).colorbar(image, ax=ax, shrink=0.8, pad=0.1)
        colorbar.set_label("time step")
    ax.scatter(
        positions[0, 0],
        positions[0, 1],
        positions[0, 2],
        color="green",
        s=60,
        label="start",
    )
    ax.scatter(
        positions[-1, 0],
        positions[-1, 1],
        positions[-1, 2],
        color="black",
        s=60,
        label="end",
    )
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_zlabel("z (m)")
    ax.legend(loc="best")
    ax.set_title(title if title else "Charge trajectory r(t)")
    return _get_fig(ax), ax


def plot_decay(
    r_values: np.ndarray,
    curves: Dict[str, np.ndarray],
    loglog: bool = True,
    ax: Optional[plt.Axes] = None,
    title: str = "",
) -> Tuple[plt.Figure, plt.Axes]:
    """Plot several decay curves (e.g. 1/r vs 1/r^2) with a legend.

    Implements the visualization of how field or potential strengths
    fall off with distance ``r``; each entry of ``curves`` is one
    curve labelled by its power law.  ``loglog=True`` plots on
    log-log axes, where power laws appear as straight lines of slope
    equal to the exponent.
    """
    r_values = np.asarray(r_values, dtype=float)
    ax = _get_ax(ax)
    for label, values in curves.items():
        values = np.asarray(values, dtype=float)
        if loglog:
            ax.loglog(r_values, values, label=label)
        else:
            ax.plot(r_values, values, label=label)
    ax.set_xlabel("r (m)")
    ax.set_ylabel("field / potential strength")
    if loglog:
        ax.set_title(title if title else "Power-law decay (log-log)")
    else:
        ax.set_title(title if title else "Power-law decay")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(loc="best")
    return _get_fig(ax), ax
