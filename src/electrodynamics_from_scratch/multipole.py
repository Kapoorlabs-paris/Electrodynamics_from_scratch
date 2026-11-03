"""Multipole expansion of localized charge distributions.

A localized cloud of point charges can be replaced, at large distances,
by an infinite series of idealized sources: the monopole (total charge),
the dipole moment, the quadrupole moment, and so on. Each successive
term falls off one power of 1/r faster, so far from the cloud the first
non-vanishing term dominates.

All functions here are static (electrostatics) and use SI units.
Field points are numpy arrays of shape (N, 3); scalar potentials
return shape (N,); vector fields return shape (N, 3).
"""

import numpy as np

from .constants import K_COULOMB

__all__ = [
    "multipole_moments",
    "exact_potential",
    "far_field_potential",
    "expansion_error_vs_r",
    "point_dipole_potential",
    "point_dipole_field",
    "quadrupole_potential",
    "power_law_fit",
]


def _validate_points(points: np.ndarray) -> np.ndarray:
    """Return *points* as a float array of shape (N, 3)."""
    pts = np.asarray(points, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError(f"points must have shape (N, 3); got {pts.shape!r}")
    return pts


def _validate_charges_positions(
    charges, positions
) -> tuple[np.ndarray, np.ndarray]:
    """Return (charges, positions) validated as (M,) and (M, 3)."""
    q = np.asarray(charges, dtype=float)
    r = np.asarray(positions, dtype=float)
    if q.ndim != 1:
        raise ValueError(f"charges must be a 1D sequence; got {q.shape!r}")
    if r.ndim != 2 or r.shape[1] != 3:
        raise ValueError(f"positions must have shape (M, 3); got {r.shape!r}")
    if q.shape[0] != r.shape[0]:
        raise ValueError(
            "charges and positions disagree in length: %d vs %d"
            % (q.shape[0], r.shape[0])
        )
    return q, r


def multipole_moments(charges, positions) -> dict:
    """Monopole, dipole and quadrupole moments of a charge distribution.

    For point charges q_a at source positions r'_a (measured from the
    expansion origin) this returns:

        Q        = sum_a q_a                                    (total charge)
        p        = sum_a q_a r'_a                               (dipole moment)
        Q_tensor = sum_a q_a (3 r'_i r'_j - r'^2 delta_ij)      (quadrupole
                   tensor, traceless by construction)

    Parameters
    ----------
    charges : sequence of float
        Point charge values in coulombs, shape (M,).
    positions : array_like
        Source positions in metres, shape (M, 3).

    Returns
    -------
    dict with keys 'Q' (float), 'p' (ndarray (3,)) and
    'Q_tensor' (ndarray (3, 3)).
    """
    q, r = _validate_charges_positions(charges, positions)
    Q = float(np.sum(q))
    p = np.sum(q[:, None] * r, axis=0)
    r_sq = np.sum(r * r, axis=1)
    Q_tensor = np.sum(
        q[:, None, None]
        * (
            3.0 * r[:, :, None] * r[:, None, :]
            - r_sq[:, None, None] * np.eye(3)
        ),
        axis=0,
    )
    return {"Q": Q, "p": p, "Q_tensor": Q_tensor}


def exact_potential(charges, positions, points) -> np.ndarray:
    """Direct Coulomb sum for a set of point charges.

        V(r) = sum_a K q_a / |r - r'_a|,   K = 1/(4 pi eps0)

    Parameters
    ----------
    charges : sequence of float
        Point charges in coulombs, shape (M,).
    positions : array_like
        Source positions in metres, shape (M, 3).
    points : array_like
        Field points in metres, shape (N, 3).

    Returns
    -------
    ndarray of shape (N,) with the potential in volts.

    Raises
    ------
    ValueError if any field point coincides with a source position
    (the potential diverges there; no softening is applied).
    """
    q, r = _validate_charges_positions(charges, positions)
    pts = _validate_points(points)
    sep = pts[:, None, :] - r[None, :, :]  # (N, M, 3)
    dist = np.sqrt(np.sum(sep * sep, axis=2))  # (N, M)
    if np.any(dist == 0.0):
        raise ValueError(
            "a field point coincides with a source position; the "
            "Coulomb potential diverges there"
        )
    return K_COULOMB * np.sum(q[None, :] / dist, axis=1)


def far_field_potential(charges, positions, points, order=2) -> np.ndarray:
    """Multipole expansion of V about the origin, to the given order.

        V(r) = K [ Q/r
                 + (p . rhat)/r^2
                 + (rhat_i Q_ij rhat_j)/(2 r^3) + ... ]

    with Q the total charge, p the dipole moment, Q_ij the (traceless)
    quadrupole tensor, and rhat = r/|r|.

    Parameters
    ----------
    charges : sequence of float
        Point charges in coulombs, shape (M,).
    positions : array_like
        Source positions in metres, shape (M, 3).
    points : array_like
        Field points in metres, shape (N, 3).
    order : int, optional
        Highest multipole order kept: 0 = monopole only, 1 = through
        the dipole term, 2 = through the quadrupole term (default).

    Returns
    -------
    ndarray of shape (N,) with the approximate potential in volts.

    Notes
    -----
    Valid only for field points with |r| much larger than the spatial
    extent of the charge distribution; the neglected higher orders are
    then small. The expansion is about the origin, so sources far from
    the origin converge slowly. Raises ValueError if any |r| is zero.
    """
    if order not in (0, 1, 2):
        raise ValueError(f"order must be 0, 1 or 2; got {order!r}")
    q, r = _validate_charges_positions(charges, positions)
    pts = _validate_points(points)
    r_mag = np.sqrt(np.sum(pts * pts, axis=1))  # (N,)
    if np.any(r_mag == 0.0):
        raise ValueError("the multipole expansion diverges at the origin")
    moments = multipole_moments(q, r)
    rhat = pts / r_mag[:, None]
    V = K_COULOMB * moments["Q"] / r_mag
    if order >= 1:
        p_dot_rhat = rhat @ moments["p"]
        V = V + K_COULOMB * p_dot_rhat / r_mag**2
    if order >= 2:
        quad = np.einsum("ni,ij,nj->n", rhat, moments["Q_tensor"], rhat)
        V = V + K_COULOMB * quad / (2.0 * r_mag**3)
    return V


def expansion_error_vs_r(charges, positions, r_values, order=2) -> np.ndarray:
    """Relative multipole-expansion error along the +x axis.

    Evaluates, at points (r, 0, 0) for each r in *r_values*:

        |V_expansion(r) - V_exact(r)| / |V_exact(r)|

    which decays as successive powers of (source extent / r) as more
    orders are kept -- the classic demonstration that the expansion
    converges.

    Parameters
    ----------
    charges : sequence of float
        Point charges in coulombs, shape (M,).
    positions : array_like
        Source positions in metres, shape (M, 3).
    r_values : array_like
        Distances along +x in metres, 1D array of shape (N,).
    order : int, optional
        Highest multipole order kept (0, 1 or 2; default 2).

    Returns
    -------
    ndarray of shape (N,) with the relative errors (dimensionless).

    Raises
    ------
    ValueError if any r equals a distance at which the exact sum
    diverges, or if any r is zero (the expansion diverges there).
    """
    r_vals = np.asarray(r_values, dtype=float)
    if r_vals.ndim != 1:
        raise ValueError(
            f"r_values must be a 1D array; got shape {r_vals.shape!r}"
        )
    pts = np.stack(
        [r_vals, np.zeros_like(r_vals), np.zeros_like(r_vals)], axis=1
    )
    V_exact = exact_potential(charges, positions, pts)
    if np.any(V_exact == 0.0):
        raise ValueError(
            "the exact potential vanishes at a sample point; the "
            "relative error is undefined there"
        )
    V_approx = far_field_potential(charges, positions, pts, order=order)
    return np.abs(V_approx - V_exact) / np.abs(V_exact)


def point_dipole_potential(p, r0, points) -> np.ndarray:
    """Potential of an ideal point dipole p at position r0.

        V(r) = K (p . (r - r0)) / |r - r0|^3,   K = 1/(4 pi eps0)

    This is the r -> r0 limit of two opposite charges +/-q separated
    by d, with p = q d held fixed.

    Parameters
    ----------
    p : array_like
        Dipole moment vector in C m, shape (3,).
    r0 : array_like
        Dipole position in metres, shape (3,).
    points : array_like
        Field points in metres, shape (N, 3).

    Returns
    -------
    ndarray of shape (N,) with the potential in volts.

    Raises
    ------
    ValueError if a field point coincides with the dipole position
    (the potential diverges there; no softening is applied).
    """
    p_vec = np.asarray(p, dtype=float).reshape(3)
    r0_vec = np.asarray(r0, dtype=float).reshape(3)
    pts = _validate_points(points)
    sep = pts - r0_vec  # (N, 3)
    dist = np.sqrt(np.sum(sep * sep, axis=1))  # (N,)
    if np.any(dist == 0.0):
        raise ValueError(
            "a field point coincides with the dipole position; the "
            "potential diverges there"
        )
    return K_COULOMB * (sep @ p_vec) / dist**3


def point_dipole_field(p, r0, points) -> np.ndarray:
    """Electric field of an ideal point dipole p at position r0.

        E(r) = K [3 rhat (p . rhat) - p] / |r - r0|^3

    with rhat = (r - r0)/|r - r0| and K = 1/(4 pi eps0). On the dipole
    axis the field is antiparallel to p with magnitude 2K|p|/r^3; in
    the equatorial plane it is antiparallel to p with magnitude
    K|p|/r^3.

    Parameters
    ----------
    p : array_like
        Dipole moment vector in C m, shape (3,).
    r0 : array_like
        Dipole position in metres, shape (3,).
    points : array_like
        Field points in metres, shape (N, 3).

    Returns
    -------
    ndarray of shape (N, 3) with the field in N/C (= V/m).

    Raises
    ------
    ValueError if a field point coincides with the dipole position
    (the field diverges there; no softening is applied).
    """
    p_vec = np.asarray(p, dtype=float).reshape(3)
    r0_vec = np.asarray(r0, dtype=float).reshape(3)
    pts = _validate_points(points)
    sep = pts - r0_vec  # (N, 3)
    dist = np.sqrt(np.sum(sep * sep, axis=1))  # (N,)
    if np.any(dist == 0.0):
        raise ValueError(
            "a field point coincides with the dipole position; the "
            "field diverges there"
        )
    rhat = sep / dist[:, None]
    p_dot_rhat = rhat @ p_vec  # (N,)
    return (
        K_COULOMB
        * (3.0 * rhat * p_dot_rhat[:, None] - p_vec[None, :])
        / dist[:, None] ** 3
    )


def quadrupole_potential(Q_tensor, r0, points) -> np.ndarray:
    """Potential of an ideal point quadrupole at position r0.

        V(r) = K/(2 r^3) * sum_ij rhat_i Q_ij rhat_j

    with rhat = (r - r0)/|r - r0|, K = 1/(4 pi eps0), and Q_ij the
    traceless quadrupole tensor returned by multipole_moments.

    Parameters
    ----------
    Q_tensor : array_like
        Quadrupole tensor in C m^2, shape (3, 3).
    r0 : array_like
        Quadrupole position in metres, shape (3,).
    points : array_like
        Field points in metres, shape (N, 3).

    Returns
    -------
    ndarray of shape (N,) with the potential in volts.

    Raises
    ------
    ValueError if a field point coincides with the quadrupole
    position (the potential diverges there; no softening is applied).
    """
    Q_mat = np.asarray(Q_tensor, dtype=float)
    if Q_mat.shape != (3, 3):
        raise ValueError(
            f"Q_tensor must have shape (3, 3); got {Q_mat.shape!r}"
        )
    r0_vec = np.asarray(r0, dtype=float).reshape(3)
    pts = _validate_points(points)
    sep = pts - r0_vec  # (N, 3)
    dist = np.sqrt(np.sum(sep * sep, axis=1))  # (N,)
    if np.any(dist == 0.0):
        raise ValueError(
            "a field point coincides with the quadrupole position; "
            "the potential diverges there"
        )
    rhat = sep / dist[:, None]
    quad = np.einsum("ni,ij,nj->n", rhat, Q_mat, rhat)
    return K_COULOMB * quad / (2.0 * dist**3)


def power_law_fit(r_values, values) -> tuple[float, float]:
    """Log-log least-squares power-law fit values ~ amplitude * r^exponent.

    Fits a straight line to (log r, log |value|) and returns
    (amplitude, exponent) so that

        |value(r)| = amplitude * r^exponent

    Useful for demonstrating how the dominant term in a multipole
    field switches between 1/r, 1/r^2 and 1/r^3 decay as the distance
    range changes.

    Parameters
    ----------
    r_values : array_like
        Distances, 1D array of shape (N,). All must be positive.
    values : array_like
        Measured quantities (e.g. |V| or |E|), 1D array of shape (N,).
        All must be nonzero; only magnitudes enter the fit.

    Returns
    -------
    (amplitude, exponent) as floats.

    Raises
    ------
    ValueError if any r is non-positive or any value is zero, or if
    fewer than two valid samples are supplied.
    """
    r_vals = np.asarray(r_values, dtype=float).ravel()
    v_vals = np.abs(np.asarray(values, dtype=float).ravel())
    if r_vals.shape != v_vals.shape:
        raise ValueError(
            "r_values and values must have the same length: %d vs %d"
            % (r_vals.shape[0], v_vals.shape[0])
        )
    if np.any(r_vals <= 0.0) or np.any(v_vals <= 0.0):
        raise ValueError(
            "log-log fit requires strictly positive r_values and "
            "nonzero values"
        )
    if r_vals.size < 2:
        raise ValueError("need at least two samples for a fit")
    log_r = np.log(r_vals)
    log_v = np.log(v_vals)
    log_r_mean = np.mean(log_r)
    log_v_mean = np.mean(log_v)
    cov = np.sum((log_r - log_r_mean) * (log_v - log_v_mean))
    var = np.sum((log_r - log_r_mean) ** 2)
    if var == 0.0:
        raise ValueError("all r_values are identical; slope undefined")
    exponent = cov / var
    amplitude = float(np.exp(log_v_mean - exponent * log_r_mean))
    return amplitude, float(exponent)
