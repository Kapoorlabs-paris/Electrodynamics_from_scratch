"""Magnetostatics: the Biot-Savart law, the vector potential and dipoles.

All quantities are in SI units. Field points are numpy arrays of shape
[N, 3]; the magnetic field is returned with shape [N, 3] and the vector
potential as an [N, 3] array (one column per Cartesian component).

Source currents are represented as an oriented polyline: an [M, 3] array
of vertices connected by straight segments. The Biot-Savart integral is
then evaluated segment by segment, which is exact for any curve sampled
finely enough.
"""

from __future__ import annotations

import numpy as np

from .constants import MU_0

__all__ = [
    "biot_savart",
    "vector_potential_wire",
    "circular_loop_curve",
    "solenoid_curve",
    "magnetic_dipole_moment",
    "magnetic_dipole_field",
    "lorentz_force",
]


def _segment_offsets(
    curve_points: np.ndarray, closed: bool
) -> tuple[np.ndarray, np.ndarray]:
    """Return (midpoints, dl) arrays of shape [n_seg, 3] for a polyline.

    If ``closed`` is True the polyline is wrapped so that a final
    segment joins the last vertex back to the first.
    """
    curve = np.asarray(curve_points, dtype=float)
    if curve.ndim != 2 or curve.shape[1] != 3:
        raise ValueError("curve_points must have shape [M, 3]")
    if curve.shape[0] < 2:
        raise ValueError("curve_points needs at least two vertices")
    if closed:
        start = curve
        end = np.vstack([curve[1:], curve[:1]])
    else:
        start = curve[:-1]
        end = curve[1:]
    dl = end - start
    mid = 0.5 * (start + end)
    return mid, dl


def biot_savart(
    curve_points: np.ndarray,
    current: float,
    points: np.ndarray,
    closed: bool = False,
) -> np.ndarray:
    """Magnetic field of a thin wire carrying a steady current.

    Implements the Biot-Savart law discretized over the polyline
    segments:

        B(r) = MU_0 * I / (4 pi) * sum_seg dl x (r - r')
               / |r - r'|^3

    where r' runs over the segments (taken at their midpoints) and dl
    is the segment vector oriented along the current.

    Parameters
    ----------
    curve_points : ndarray, shape [M, 3]
        Vertices of the wire, in order along the current.
    current : float
        Current I in amperes.
    points : ndarray, shape [N, 3]
        Field points r.
    closed : bool
        If True, a final segment wraps the last vertex back to the
        first, forming a closed loop.

    Returns
    -------
    ndarray, shape [N, 3]
        B at each field point in tesla.

    Raises
    ------
    ValueError
        If a field point coincides with a segment midpoint (the
        integrand diverges there and no softening is applied).
    """
    mid, dl = _segment_offsets(curve_points, closed)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    diff = pts[:, None, :] - mid[None, :, :]  # [N, n_seg, 3]
    r3 = np.linalg.norm(diff, axis=2) ** 3  # [N, n_seg]
    if np.any(r3 == 0.0):
        raise ValueError(
            "a field point coincides with a wire segment midpoint; "
            "the Biot-Savart integrand is singular there"
        )
    cross = np.cross(dl[None, :, :], diff)  # [N, n_seg, 3]
    return (MU_0 * current / (4.0 * np.pi)) * np.sum(
        cross / r3[:, :, None], axis=1
    )


def vector_potential_wire(
    curve_points: np.ndarray,
    current: float,
    points: np.ndarray,
    closed: bool = False,
) -> np.ndarray:
    """Magnetic vector potential of a thin wire in the Coulomb gauge.

    Implements

        A(r) = MU_0 * I / (4 pi) * sum_seg dl / |r - r'|

    with dl oriented along the current and r' at the segment midpoints,
    so that B = curl A reproduces biot_savart for a finely sampled
    curve.

    Parameters
    ----------
    curve_points : ndarray, shape [M, 3]
        Vertices of the wire, in order along the current.
    current : float
        Current I in amperes.
    points : ndarray, shape [N, 3]
        Field points r.
    closed : bool
        If True, a final segment wraps the last vertex back to the
        first.

    Returns
    -------
    ndarray, shape [N, 3]
        A at each field point in tesla-metre.

    Raises
    ------
    ValueError
        If a field point coincides with a segment midpoint (the
        integrand diverges there and no softening is applied).
    """
    mid, dl = _segment_offsets(curve_points, closed)
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")

    diff = pts[:, None, :] - mid[None, :, :]  # [N, n_seg, 3]
    r1 = np.linalg.norm(diff, axis=2)  # [N, n_seg]
    if np.any(r1 == 0.0):
        raise ValueError(
            "a field point coincides with a wire segment midpoint; "
            "the integrand is singular there"
        )
    return (MU_0 * current / (4.0 * np.pi)) * np.sum(
        dl[None, :, :] / r1[:, :, None], axis=1
    )


def circular_loop_curve(
    radius: float,
    center: np.ndarray,
    n: int = 200,
) -> np.ndarray:
    """Vertices of a circular loop in the xy plane.

    The loop has the given radius, is centred at ``center`` and its
    normal points along +z; the vertices run counter-clockwise when
    viewed from +z, so a positive current along them gives a +z dipole
    moment. The first vertex is not repeated at the end: pass
    ``closed=True`` to the field functions.

    Parameters
    ----------
    radius : float
        Loop radius in metres.
    center : ndarray, shape (3,)
        Loop centre.
    n : int
        Number of vertices sampled around the loop.

    Returns
    -------
    ndarray, shape [n, 3]
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if n < 2:
        raise ValueError("n must be at least 2")
    phi = 2.0 * np.pi * np.arange(n) / n
    center = np.asarray(center, dtype=float)
    if center.shape != (3,):
        raise ValueError("center must have shape (3,)")
    xy = radius * np.stack([np.cos(phi), np.sin(phi)], axis=1)
    return np.hstack([xy, np.zeros((n, 1))]) + center


def solenoid_curve(
    radius: float,
    length: float,
    turns: int,
    n_per_turn: int = 40,
) -> np.ndarray:
    """Vertices of a helical solenoid winding along the z axis.

    The winding has the given radius, runs from z = 0 to z = length and
    completes ``turns`` full revolutions (right-handed, so a positive
    current along the vertices gives a +z dipole moment per turn).
    Vertices are spaced uniformly in the winding angle.

    Parameters
    ----------
    radius : float
        Solenoid radius in metres.
    length : float
        Solenoid length in metres (z from 0 to length).
    turns : int
        Number of complete turns.
    n_per_turn : int
        Vertices sampled per turn.

    Returns
    -------
    ndarray, shape [turns * n_per_turn, 3]
    """
    if radius <= 0:
        raise ValueError("radius must be positive")
    if length <= 0:
        raise ValueError("length must be positive")
    if turns < 1:
        raise ValueError("turns must be at least 1")
    if n_per_turn < 2:
        raise ValueError("n_per_turn must be at least 2")
    n_total = turns * n_per_turn
    # Exclude the final angle of each turn so the helix does not
    # self-overlap at the seam.
    theta = 2.0 * np.pi * np.arange(n_total) / n_per_turn
    z = np.linspace(0.0, length, n_total, endpoint=False)
    xy = radius * np.stack([np.cos(theta), np.sin(theta)], axis=1)
    return np.hstack([xy, z[:, None]])


def magnetic_dipole_moment(
    curve_points: np.ndarray,
    current: float,
    closed: bool = True,
) -> np.ndarray:
    """Magnetic dipole moment of a current loop.

    Implements, with the loop centroid c subtracted from r so the
    moment is independent of the coordinate origin,

        m = (I / 2) * sum_seg (r - c) x dl

    where r is taken at the segment midpoints and dl runs along the
    current. For a planar loop this equals I times the loop area vector
    (right-hand rule).

    Parameters
    ----------
    curve_points : ndarray, shape [M, 3]
        Vertices of the loop, in order along the current.
    current : float
        Current I in amperes.
    closed : bool
        If True, a final segment wraps the last vertex back to the
        first.

    Returns
    -------
    ndarray, shape (3,)
        Dipole moment in A m^2.
    """
    mid, dl = _segment_offsets(curve_points, closed)
    centroid = mid.mean(axis=0)
    return (0.5 * current) * np.cross(mid - centroid, dl).sum(axis=0)


def magnetic_dipole_field(m: np.ndarray, points: np.ndarray) -> np.ndarray:
    """Exact field of a point magnetic dipole.

    Implements, with r the displacement from the dipole and
    rhat = r / |r|,

        B(r) = MU_0 / (4 pi) * [ 3 rhat (m . rhat) - m ] / |r|^3

    Valid in the point-dipole limit |r| >> dipole size; the field
    diverges as |r| -> 0 and no softening is applied.

    Parameters
    ----------
    m : ndarray, shape (3,)
        Dipole moment in A m^2.
    points : ndarray, shape [N, 3]
        Field points, measured from the dipole position.

    Returns
    -------
    ndarray, shape [N, 3]
        B at each field point in tesla.

    Raises
    ------
    ValueError
        If a field point coincides with the dipole.
    """
    m = np.asarray(m, dtype=float)
    if m.shape != (3,):
        raise ValueError("m must have shape (3,)")
    r = np.asarray(points, dtype=float)
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError("points must have shape [N, 3]")
    rnorm = np.linalg.norm(r, axis=1)
    if np.any(rnorm == 0.0):
        raise ValueError(
            "a field point coincides with the dipole; the point-dipole "
            "field is singular there"
        )
    rhat = r / rnorm[:, None]
    m_dot_rhat = rhat @ m
    return (
        (MU_0 / (4.0 * np.pi))
        * (3.0 * rhat * m_dot_rhat[:, None] - m[None, :])
        / (rnorm[:, None] ** 3)
    )


def lorentz_force(
    q: float,
    v: np.ndarray,
    E: np.ndarray,
    B: np.ndarray,
) -> np.ndarray:
    """Lorentz force on a charged particle.

    Implements

        F = q (E + v x B)

    Parameters
    ----------
    q : float
        Charge in coulombs (may be negative).
    v : ndarray, shape (3,) or [N, 3]
        Particle velocity in m/s.
    E : ndarray, shape [N, 3]
        Electric field in V/m.
    B : ndarray, shape [N, 3]
        Magnetic field in tesla.

    Returns
    -------
    ndarray
        Force in newtons, with the same shape as ``v``: a single (3,)
        vector if ``v`` is (3,), otherwise [N, 3].
    """
    v = np.asarray(v, dtype=float)
    E = np.asarray(E, dtype=float)
    B = np.asarray(B, dtype=float)
    if v.ndim not in (1, 2) or v.shape[-1] != 3:
        raise ValueError("v must have shape (3,) or [N, 3]")
    if E.ndim != 2 or E.shape[1] != 3 or B.ndim != 2 or B.shape[1] != 3:
        raise ValueError("E and B must have shape [N, 3]")
    if v.ndim == 1:
        return q * (E[0] + np.cross(v, B[0]))
    if E.shape[0] != v.shape[0] or B.shape[0] != v.shape[0]:
        raise ValueError(
            "E and B must have shape [N, 3] matching the number of "
            "velocities"
        )
    return q * (E + np.cross(v, B))
