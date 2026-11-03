"""Electrostatics: Coulomb fields, potentials, charge distributions, Poisson.

All quantities are in SI units. Field points are numpy arrays of shape
(N, 3); vector fields return shape (N, 3) and scalar potentials return
shape (N,). Grid-based routines take 2D or 3D numpy arrays together with
a uniform grid spacing ``dx`` and return arrays of the same shape.

The Coulomb constant is ``K_COULOMB = 1/(4 pi epsilon_0)`` from the
sibling module :mod:`electrodynamics_from_scratch.constants`.
"""

from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .constants import EPSILON_0, K_COULOMB

__all__ = [
    "point_charge_field",
    "point_charge_potential",
    "superpose",
    "line_charge_field",
    "ring_field",
    "ring_potential",
    "shell_field",
    "solid_sphere_field",
    "poisson_solve",
    "laplacian",
    "electric_field_from_potential",
    "gaussian_delta",
]


def _as_points(points: np.ndarray) -> np.ndarray:
    """Validate and reshape a field-point array to shape (N, 3)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {pts.shape!r}")
    return pts


def _as_vec3(vec: Sequence[float], name: str) -> np.ndarray:
    """Validate a single 3-vector such as a source position."""
    v = np.asarray(vec, dtype=float).reshape(-1)
    if v.size != 3:
        raise ValueError("%s must be a 3-vector; got size %d" % (name, v.size))
    return v


def point_charge_field(
    charge: float,
    position: Sequence[float],
    points: np.ndarray,
) -> np.ndarray:
    """Electric field of a single point charge.

    Implements Coulomb's law

        E(r) = K_COULOMB * q * (r - r0) / |r - r0|^3

    where r0 is the charge position.

    Parameters
    ----------
    charge : float
        Source charge q in coulombs.
    position : sequence of 3 floats
        Source position r0 in metres.
    points : ndarray, shape (N, 3)
        Field points r in metres.

    Returns
    -------
    ndarray, shape (N, 3)
        Electric field in N/C at each field point.

    Raises
    ------
    ValueError
        If a field point coincides exactly with the source position,
        where the analytic field diverges. No softening is applied.
    """
    pts = _as_points(points)
    r0 = _as_vec3(position, "position")
    disp = pts - r0
    r = np.linalg.norm(disp, axis=1)
    if np.any(r == 0.0):
        raise ValueError(
            "a field point coincides with the charge position; the "
            "Coulomb field diverges there and no softening is applied"
        )
    return (K_COULOMB * charge / r**3)[:, None] * disp


def point_charge_potential(
    charge: float,
    position: Sequence[float],
    points: np.ndarray,
) -> np.ndarray:
    """Scalar potential of a single point charge (zero at infinity).

    Implements

        V(r) = K_COULOMB * q / |r - r0|

    Parameters
    ----------
    charge : float
        Source charge q in coulombs.
    position : sequence of 3 floats
        Source position r0 in metres.
    points : ndarray, shape (N, 3)
        Field points r in metres.

    Returns
    -------
    ndarray, shape (N,)
        Potential in volts.

    Raises
    ------
    ValueError
        If a field point coincides with the source position.
    """
    pts = _as_points(points)
    r0 = _as_vec3(position, "position")
    r = np.linalg.norm(pts - r0, axis=1)
    if np.any(r == 0.0):
        raise ValueError(
            "a field point coincides with the charge position; the "
            "potential diverges there and no softening is applied"
        )
    return K_COULOMB * charge / r


def superpose(
    charges: Sequence[Tuple[float, Sequence[float]]],
    points: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Superpose the fields and potentials of many point charges.

    By linearity of Maxwell's electrostatic equations,

        E(r) = sum_i K_COULOMB * q_i * (r - r_i)/|r - r_i|^3
        V(r) = sum_i K_COULOMB * q_i / |r - r_i|

    Parameters
    ----------
    charges : sequence of (charge, position) pairs
        Each position is a 3-vector in metres.
    points : ndarray, shape (N, 3)
        Field points r in metres.

    Returns
    -------
    (E, V) : tuple of ndarrays
        E has shape (N, 3) in N/C, V has shape (N,) in volts.

    Raises
    ------
    ValueError
        If any field point coincides with any source position.
    """
    pts = _as_points(points)
    e_total = np.zeros_like(pts)
    v_total = np.zeros(pts.shape[0])
    for q, pos in charges:
        e_total += point_charge_field(q, pos, pts)
        v_total += point_charge_potential(q, pos, pts)
    return e_total, v_total


def line_charge_field(
    charge: float,
    p0: Sequence[float],
    p1: Sequence[float],
    points: np.ndarray,
    n_segments: int = 200,
) -> np.ndarray:
    """Field of a straight uniformly charged line segment.

    The segment from p0 to p1 carrying total charge q is discretized
    into ``n_segments`` equal point charges q/n_segments placed at the
    segment midpoints, and Coulomb superposition is applied:

        E(r) = sum_i K_COULOMB * (q/n) * (r - r_i)/|r - r_i|^3

    The discretization converges to the exact line-integral result as
    n_segments grows. The endpoints of the segment are excluded from
    the discretization, so field points exactly on the open segment
    still diverge.

    Parameters
    ----------
    charge : float
        Total charge q on the segment in coulombs.
    p0, p1 : sequence of 3 floats
        Segment endpoints in metres.
    points : ndarray, shape (N, 3)
        Field points r in metres.
    n_segments : int, optional
        Number of discretization elements (default 200).

    Returns
    -------
    ndarray, shape (N, 3)
        Electric field in N/C.
    """
    if n_segments < 1:
        raise ValueError("n_segments must be at least 1")
    pts = _as_points(points)
    a = _as_vec3(p0, "p0")
    b = _as_vec3(p1, "p1")
    # midpoints of n equal sub-segments, excluding both endpoints
    t = (np.arange(n_segments) + 0.5) / n_segments
    positions = a[None, :] + t[:, None] * (b - a)[None, :]
    q_each = charge / n_segments
    e_total = np.zeros_like(pts)
    for i in range(n_segments):
        e_total += point_charge_field(q_each, positions[i], pts)
    return e_total


def _ring_charge_positions(
    radius: float, center: np.ndarray, n: int
) -> Tuple[np.ndarray, float]:
    """Positions of n equal point charges representing a charged ring.

    The ring lies in the plane through ``center`` parallel to the xy
    plane (normal along z), so each element sits at angle 2 pi i / n.
    """
    angles = 2.0 * np.pi * np.arange(n) / n
    offsets = np.stack(
        [
            radius * np.cos(angles),
            radius * np.sin(angles),
            np.zeros(n),
        ],
        axis=1,
    )
    return center[None, :] + offsets, radius


def ring_field(
    charge: float,
    radius: float,
    center: Sequence[float],
    points: np.ndarray,
    n: int = 400,
) -> np.ndarray:
    """Field of a uniformly charged ring in a plane parallel to xy.

    A ring of radius R centred at ``center`` (normal along z) carrying
    total charge q is discretized into n equal point charges q/n and
    Coulomb superposition is applied. On the symmetry axis this
    reproduces the exact result

        E(z) = K_COULOMB * q * z / (z^2 + R^2)^(3/2)

    measured from the ring centre along z, because every element is
    equidistant from an axial point and the transverse components
    cancel pairwise exactly for any n >= 2.

    Parameters
    ----------
    charge : float
        Total charge q on the ring in coulombs.
    radius : float
        Ring radius R in metres.
    center : sequence of 3 floats
        Ring centre in metres.
    points : ndarray, shape (N, 3)
        Field points r in metres.
    n : int, optional
        Number of discretization elements (default 400).

    Returns
    -------
    ndarray, shape (N, 3)
        Electric field in N/C.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for a ring")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    pts = _as_points(points)
    c = _as_vec3(center, "center")
    positions, _ = _ring_charge_positions(radius, c, n)
    q_each = charge / n
    e_total = np.zeros_like(pts)
    for i in range(n):
        e_total += point_charge_field(q_each, positions[i], pts)
    return e_total


def ring_potential(
    charge: float,
    radius: float,
    center: Sequence[float],
    points: np.ndarray,
    n: int = 400,
) -> np.ndarray:
    """Potential of a uniformly charged ring in a plane parallel to xy.

    Discretizes the ring into n equal point charges q/n and sums

        V(r) = sum_i K_COULOMB * (q/n) / |r - r_i|

    On the symmetry axis this converges to the exact result
    V(z) = K_COULOMB * q / sqrt(z^2 + R^2).

    Parameters are as in :func:`ring_field`.

    Returns
    -------
    ndarray, shape (N,)
        Potential in volts.
    """
    if n < 2:
        raise ValueError("n must be at least 2 for a ring")
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    pts = _as_points(points)
    c = _as_vec3(center, "center")
    positions, _ = _ring_charge_positions(radius, c, n)
    q_each = charge / n
    v_total = np.zeros(pts.shape[0])
    for i in range(n):
        v_total += point_charge_potential(q_each, positions[i], pts)
    return v_total


def shell_field(
    charge: float,
    radius: float,
    points: np.ndarray,
) -> np.ndarray:
    """Exact field of a uniformly charged spherical shell.

    By Gauss's law the field of a thin spherical shell of radius R
    carrying total charge q is

        E(r) = 0                       for r < R  (inside)
        E(r) = K_COULOMB * q / r^2     for r >= R (outside, radial)

    directed radially outward for q > 0. The exterior field is exactly
    that of a point charge q at the centre (the shell theorem); the
    interior field vanishes exactly. The surface r = R is assigned the
    exterior limit K_COULOMB * q / R^2.

    Parameters
    ----------
    charge : float
        Total charge q on the shell in coulombs.
    radius : float
        Shell radius R in metres (must be positive).
    points : ndarray, shape (N, 3)
        Field points r in metres.

    Returns
    -------
    ndarray, shape (N, 3)
        Electric field in N/C.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    pts = _as_points(points)
    r = np.linalg.norm(pts, axis=1)
    outside = K_COULOMB * charge / np.maximum(r**2, 1e-300)
    e_mag = np.where(r >= radius, outside, 0.0)
    # guard r = 0: inside the shell the field is exactly zero, so the
    # direction is irrelevant; substitute a unit x vector there.
    safe_r = np.where(r > 0.0, r, 1.0)
    direction = pts / safe_r[:, None]
    direction[r == 0.0] = np.array([1.0, 0.0, 0.0])
    return e_mag[:, None] * direction


def solid_sphere_field(
    charge: float,
    radius: float,
    points: np.ndarray,
) -> np.ndarray:
    """Exact field of a uniformly charged solid sphere.

    For a sphere of radius R carrying total charge q distributed
    uniformly through its volume, Gauss's law gives

        E(r) = K_COULOMB * q * r / R^3      for r < R  (inside, ~ r)
        E(r) = K_COULOMB * q / r^2          for r >= R (outside, ~ 1/r^2)

    directed radially outward for q > 0. The interior field grows
    linearly with r because only the charge interior to the Gaussian
    surface contributes. The surface r = R is assigned the exterior
    limit K_COULOMB * q / R^2.

    Parameters
    ----------
    charge : float
        Total charge q in coulombs.
    radius : float
        Sphere radius R in metres (must be positive).
    points : ndarray, shape (N, 3)
        Field points r in metres.

    Returns
    -------
    ndarray, shape (N, 3)
        Electric field in N/C.
    """
    if radius <= 0.0:
        raise ValueError("radius must be positive")
    pts = _as_points(points)
    r = np.linalg.norm(pts, axis=1)
    inside = K_COULOMB * charge * r / radius**3
    outside = K_COULOMB * charge / np.maximum(r**2, 1e-300)
    e_mag = np.where(r < radius, inside, outside)
    # at r = 0 the interior field is exactly zero, direction arbitrary
    safe_r = np.where(r > 0.0, r, 1.0)
    direction = pts / safe_r[:, None]
    direction[r == 0.0] = np.array([1.0, 0.0, 0.0])
    return e_mag[:, None] * direction


def poisson_solve(rho: np.ndarray, dx: float) -> np.ndarray:
    """Solve the Poisson equation on a periodic grid via FFT.

    Solves

        -epsilon_0 * laplacian(V) = rho

    with **periodic boundary conditions** on a uniform 2D or 3D grid.
    Transforming to Fourier space, the Laplacian becomes -k^2, so

        V(k) = rho(k) / (epsilon_0 * k^2)

    with the k = 0 mode set to zero: a periodic system cannot fix the
    arbitrary additive constant of the potential, so the mean of V is
    chosen to vanish. Note that periodic boundaries are physically
    appropriate only for charge distributions that are themselves
    periodic (or neutral overall); for localized sources in a large box
    the solution is accurate away from the box edges up to O(dx^2)
    discretization error.

    Parameters
    ----------
    rho : ndarray, 2D or 3D
        Charge density in C/m^3 on a uniform grid.
    dx : float
        Uniform grid spacing in metres.

    Returns
    -------
    ndarray, same shape as rho
        Potential V in volts.
    """
    if rho.ndim not in (2, 3):
        raise ValueError("rho must be a 2D or 3D grid")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    rho_k = np.fft.fftn(rho)
    shape = rho.shape
    k_vecs = [2.0 * np.pi * np.fft.fftfreq(n, d=dx) for n in shape]
    k_grids = np.meshgrid(*k_vecs, indexing="ij")
    k2 = np.zeros(shape)
    for kg in k_grids:
        k2 += kg**2
    # k = 0 mode: zero out (mean potential fixed to zero). Avoid the
    # division by zero by only dividing where k2 > 0.
    v_k = np.zeros_like(rho_k)
    nonzero = k2 > 0.0
    v_k[nonzero] = rho_k[nonzero] / (EPSILON_0 * k2[nonzero])
    return np.real(np.fft.ifftn(v_k))


def laplacian(f: np.ndarray, dx: float) -> np.ndarray:
    """Numeric Laplacian of a scalar grid.

    Implements

        laplacian(f) = sum_i d^2 f / dx_i^2

    for a 2D or 3D array sampled on a uniform grid of spacing dx,
    using np.gradient twice along each axis (second-order central
    differences in the interior, one-sided at the boundaries).

    Note the effective interior stencil spans two grid spacings,
    (f[i+2] - 2 f[i] + f[i-2])/(4 dx^2), so it is second-order
    accurate but with a larger error constant than the compact
    three-point second derivative (f[i+1] - 2 f[i] + f[i-1])/dx^2.

    Parameters
    ----------
    f : ndarray, 2D or 3D
        Scalar field sampled on a uniform grid.
    dx : float
        Uniform grid spacing in metres.

    Returns
    -------
    ndarray, same shape as f
        The Laplacian of f.
    """
    if f.ndim not in (2, 3):
        raise ValueError("f must be a 2D or 3D grid")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    lap = np.zeros_like(f, dtype=float)
    first = np.gradient(f, dx)
    if f.ndim == 2:
        axes = (0, 1)
    else:
        axes = (0, 1, 2)
    for axis, grad_i in zip(axes, first):
        lap += np.gradient(grad_i, dx, axis=axis)
    return lap


def electric_field_from_potential(
    V: np.ndarray, dx: float
) -> Tuple[np.ndarray, ...]:
    """Electric field from a potential grid.

    Implements

        E = -grad(V)

    via np.gradient on a uniform grid of spacing dx, so the returned
    components satisfy E = -dV/dx_i to second-order accuracy.

    Parameters
    ----------
    V : ndarray, 2D or 3D
        Potential in volts sampled on a uniform grid.
    dx : float
        Uniform grid spacing in metres.

    Returns
    -------
    tuple of ndarrays
        (Ex, Ey) for a 2D grid, (Ex, Ey, Ez) for a 3D grid, each with
        the same shape as V and units of N/C (equivalently V/m).
    """
    if V.ndim not in (2, 3):
        raise ValueError("V must be a 2D or 3D grid")
    if dx <= 0.0:
        raise ValueError("dx must be positive")
    grads = np.gradient(V, dx)
    return tuple(-g for g in grads)


def gaussian_delta(x: np.ndarray, x0: float, sigma: float) -> np.ndarray:
    """Normalized Gaussian approximation of the Dirac delta.

    Implements

        delta_sigma(x - x0) = 1/(sigma sqrt(2 pi))
                              * exp(-(x - x0)^2 / (2 sigma^2))

    which integrates to 1 over the real line and tends to the Dirac
    delta as sigma -> 0. It is the standard smoothed source term for
    placing a point charge on a discrete grid (e.g. as input to
    :func:`poisson_solve`), where an exact delta cannot be represented.

    Parameters
    ----------
    x : ndarray, 1D
        Grid coordinates in metres.
    x0 : float
        Centre of the delta in metres.
    sigma : float
        Width of the Gaussian in metres (must be positive).

    Returns
    -------
    ndarray, same shape as x
        The Gaussian in units of 1/m (2D) or 1/m^3 (3D) per the
        chosen convention; the values themselves are 1/m times the
        normalization above.
    """
    if sigma <= 0.0:
        raise ValueError("sigma must be positive")
    x = np.asarray(x, dtype=float)
    norm = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    return norm * np.exp(-((x - x0) ** 2) / (2.0 * sigma**2))
