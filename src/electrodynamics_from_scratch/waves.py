"""The wave equation, retarded time, and electromagnetic radiation.

This module collects the standard solutions of the wave equation: the
d'Alembert solution of the 1D wave equation, the retarded time that
encodes causality (no signal travels faster than light), the
Lienard-Wiechert potentials and fields of a moving point charge, the
radiation field of an oscillating electric dipole, and the Larmor
formula for the power radiated by an accelerated charge.

All quantities are in SI units. Field points are numpy arrays of shape
[N, 3]; vector fields return [N, 3]; scalar potentials return [N,].
"""

import numpy as np

from .constants import C_LIGHT, EPSILON_0, K_COULOMB

__all__ = [
    "dalembert",
    "retarded_time",
    "lieard_wiechert_potentials",
    "lieard_wiechert_fields",
    "split_velocity_acceleration",
    "dipole_radiation_fields",
    "radiation_pattern",
    "larmor_power",
]


def dalembert(f, g, x, t, c=1.0):
    """D'Alembert solution of the 1D wave equation at time t.

    Implements the solution of u_tt = c^2 u_xx with initial data
    u(x, 0) = f(x), u_t(x, 0) = g(x):

        u(x, t) = (f(x - c t) + f(x + c t)) / 2
                  + (1 / (2 c)) * Integral_{x - c t}^{x + c t} g(s) ds

    Parameters
    ----------
    f : callable
        Initial displacement, f(x) -> float.
    g : callable
        Initial velocity, g(x) -> float.
    x : numpy array, shape [M]
        1D spatial grid (assumed uniformly spaced).
    t : float
        Time at which to evaluate the solution.
    c : float, optional
        Wave speed (default 1.0).

    Returns
    -------
    numpy array, shape [M]
        u(x, t) evaluated on the grid.

    Limitations
    -----------
    The integral term is evaluated with np.trapz using only the grid
    points that fall inside the integration window [x - c t, x + c t],
    clipping the window to the grid limits. The result is therefore
    approximate when the window edges do not coincide with grid points,
    or when c t exceeds the grid extent (part of the integral is then
    silently ignored).
    """
    x = np.asarray(x, dtype=float)
    half = c * t
    left = x - half
    right = x + half
    dx = float(x[1] - x[0]) if x.size > 1 else 1.0
    u = np.empty_like(x)
    for i, (lo, hi, xi) in enumerate(zip(left, right, x)):
        lo_c = max(lo, float(x[0]))
        hi_c = min(hi, float(x[-1]))
        n_pts = int(np.floor((hi_c - lo_c) / dx)) + 1
        s = lo_c + dx * np.arange(max(n_pts, 2))
        s = np.minimum(s, hi_c)
        u[i] = 0.5 * (f(xi - half) + f(xi + half))
        u[i] += np.trapz(g(s), s) / (2.0 * c)
    return u


def retarded_time(t, r, source, c=C_LIGHT):
    """Retarded time tau = t - |r - source| / c.

    Implements causality for a signal emitted by a source at position
    `source` and observed at position r: the observation at time t is
    caused by the source state at the earlier (retarded) time

        tau = t - |r - source| / c

    Parameters
    ----------
    t : float or numpy array, shape [N]
        Observation time(s), in seconds.
    r : numpy array, shape (3,) or [N, 3]
        Observation point(s), in meters.
    source : numpy array, shape (3,)
        Source position, in meters.
    c : float, optional
        Signal speed (default the speed of light in vacuum).

    Returns
    -------
    float or numpy array
        The retarded time(s), with the same type/shape as `t`.

    Raises
    ------
    ValueError
        If an observation point coincides with the source (the light
        travel distance vanishes and the retarded time is degenerate).
    """
    r = np.asarray(r, dtype=float)
    source = np.asarray(source, dtype=float)
    if r.ndim == 1:
        dist = float(np.linalg.norm(r - source))
    else:
        dist = np.linalg.norm(r - source, axis=1)
    if np.any(dist == 0.0):
        raise ValueError(
            "observation point coincides with the source; retarded "
            "time is undefined"
        )
    if np.isscalar(t) or np.ndim(t) == 0:
        return float(t) - dist / c
    return np.asarray(t, dtype=float) - dist / c


def lieard_wiechert_potentials(q, source, velocity, r, t):
    """Lienard-Wiechert scalar and vector potentials of a point charge.

    Implements, for a charge q moving with velocity v and observed at
    field points r at time t, the potentials evaluated at the RETARDED
    source position:

        phi = K q / (R (1 - n . v / c))
        A   = v phi / c^2

    with R = |r - source| and n = (r - source) / R, K = 1/(4 pi eps0).

    IMPORTANT: `source` and `velocity` must be the source position and
    velocity evaluated AT THE RETARDED TIME
    tau = t - |r - source|/c, which the caller computes (e.g. with
    retarded_time). This function does not solve for the retarded
    state itself; it supplies the (1 - n . v / c) denominator that
    distinguishes the Lienard-Wiechert potentials from the naive
    Coulomb/Lorentz expressions.

    Parameters
    ----------
    q : float
        Charge, in coulombs.
    source : numpy array, shape (3,)
        Retarded source position, in meters.
    velocity : numpy array, shape (3,)
        Retarded source velocity, in m/s.
    r : numpy array, shape [N, 3]
        Field points, in meters.
    t : float
        Observation time (used only for the API; the retarded state is
        supplied by the caller).

    Returns
    -------
    (numpy array shape [N], numpy array shape [N, 3])
        The scalar potential phi in volts and vector potential A in
        T m (webers per meter).

    Raises
    ------
    ValueError
        If a field point coincides with the source, or if the
        denominator (1 - n . v / c) vanishes (the Cherenkov-like
        singularity of the Lienard-Wiechert potentials).
    """
    r = np.asarray(r, dtype=float)
    source = np.asarray(source, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    sep = r - source
    R = np.linalg.norm(sep, axis=1)
    if np.any(R == 0.0):
        raise ValueError("field point coincides with the source")
    n_hat = sep / R[:, None]
    denom = 1.0 - np.einsum("ij,j->i", n_hat, velocity) / C_LIGHT
    if np.any(np.isclose(denom, 0.0)):
        raise ValueError(
            "denominator (1 - n . v / c) vanishes; potentials are " "singular"
        )
    phi = K_COULOMB * q / (R * denom)
    A = (velocity[None, :] / C_LIGHT**2) * phi[:, None]
    return phi, A


def lieard_wiechert_fields(q, source, velocity, acceleration, r):
    """Full Lienard-Wiechert electric and magnetic fields.

    Implements the fields of a point charge q moving with velocity v
    and acceleration a, observed at r, with the source state taken at
    the retarded time:

        E = K q / R^2 * [(1 - beta^2)(n - beta)]
            + K q / (c R) * [n x ((n - beta) x a)]
        B = n x E / c

    where K = 1/(4 pi eps0), R = |r - source|, n = (r - source)/R and
    beta = v / c. The first (velocity) term falls off as 1/R^2 and
    reduces to the Coulomb field for v = 0; the second (acceleration)
    term falls off as 1/R and is the radiation field.

    IMPORTANT: `source`, `velocity` and `acceleration` must be
    evaluated AT THE RETARDED TIME tau = t - |r - source|/c, which the
    caller computes (e.g. with retarded_time).

    Parameters
    ----------
    q : float
        Charge, in coulombs.
    source : numpy array, shape (3,)
        Retarded source position, in meters.
    velocity : numpy array, shape (3,)
        Retarded source velocity, in m/s.
    acceleration : numpy array, shape (3,)
        Retarded source acceleration, in m/s^2.
    r : numpy array, shape [N, 3]
        Field points, in meters.

    Returns
    -------
    (numpy array shape [N, 3], numpy array shape [N, 3])
        The electric field E in V/m and magnetic field B in tesla.

    Raises
    ------
    ValueError
        If a field point coincides with the source.
    """
    r = np.asarray(r, dtype=float)
    source = np.asarray(source, dtype=float)
    velocity = np.asarray(velocity, dtype=float)
    acceleration = np.asarray(acceleration, dtype=float)
    sep = r - source
    R = np.linalg.norm(sep, axis=1)
    if np.any(R == 0.0):
        raise ValueError("field point coincides with the source")
    n_hat = sep / R[:, None]
    beta = velocity / C_LIGHT
    beta2 = float(beta @ beta)
    cross = np.cross(
        n_hat, np.cross(n_hat - beta[None, :], acceleration[None, :])
    )
    e_velocity = (
        (K_COULOMB * q / R**2)[:, None]
        * (1.0 - beta2)
        * (n_hat - beta[None, :])
    )
    e_accel = (K_COULOMB * q / (C_LIGHT * R))[:, None] * cross
    E = e_velocity + e_accel
    B = np.cross(n_hat, E) / C_LIGHT
    return E, B


def split_velocity_acceleration(E, exact_velocity_E):
    """Isolate the radiation field by subtracting the velocity piece.

    Implements the decomposition of the total Lienard-Wiechert electric
    field into its velocity (near, 1/R^2) and acceleration (radiation,
    1/R) parts:

        E_acceleration = E_total - E_velocity
        E_velocity     = exact_velocity_E (supplied)

    Parameters
    ----------
    E : numpy array, shape [N, 3]
        Total electric field at the field points, in V/m.
    exact_velocity_E : numpy array, shape [N, 3]
        The known velocity-field piece (e.g. the Coulomb field of a
        uniformly moving charge) at the same points, in V/m.

    Returns
    -------
    (numpy array shape [N, 3], numpy array shape [N, 3])
        The velocity field E_velocity and the radiation field
        E_acceleration, in V/m.
    """
    E = np.asarray(E, dtype=float)
    v_field = np.asarray(exact_velocity_E, dtype=float)
    return v_field, E - v_field


def dipole_radiation_fields(p0, omega, r, t):
    """Far-field radiation of an oscillating electric point dipole.

    Implements the radiation zone (r >> c / omega) fields of a point
    dipole p(t) = p0 cos(omega t) at the origin:

        E = K / c^2 / r * [n x (n x p_ddot(t_ret))]
        B = n x E / c

    with p_ddot(t) = -omega^2 p0 cos(omega t), the retarded time
    t_ret = t - r / c, K = 1/(4 pi eps0) and n the radial unit vector.
    Only the 1/r radiation term is kept; the near-field 1/r^2 and 1/r^3
    terms are neglected (exact only in the limit r -> infinity).

    Parameters
    ----------
    p0 : numpy array, shape (3,)
        Dipole amplitude vector, in C m.
    omega : float
        Angular frequency of oscillation, in rad/s.
    r : numpy array, shape [N, 3]
        Field points, in meters.
    t : float
        Observation time, in seconds.

    Returns
    -------
    (numpy array shape [N, 3], numpy array shape [N, 3])
        The radiative electric field E in V/m and magnetic field B in
        tesla.

    Raises
    ------
    ValueError
        If a field point is at the origin.
    """
    r = np.asarray(r, dtype=float)
    p0 = np.asarray(p0, dtype=float)
    R = np.linalg.norm(r, axis=1)
    if np.any(R == 0.0):
        raise ValueError("field point at the dipole origin")
    n_hat = r / R[:, None]
    t_ret = t - R / C_LIGHT
    p_ddot = -(omega**2) * np.cos(omega * t_ret)[:, None] * p0[None, :]
    cross = np.cross(n_hat, np.cross(n_hat, p_ddot))
    E = (K_COULOMB / C_LIGHT**2 / R**2)[:, None] * cross
    B = np.cross(n_hat, E) / C_LIGHT
    return E, B


def radiation_pattern(theta):
    """Normalized angular intensity pattern of dipole radiation.

    Implements the angular distribution of the time-averaged radiated
    intensity of an oscillating electric dipole, proportional to the
    square of the sine of the polar angle measured from the dipole
    axis:

        I(theta) proportional to sin(theta)^2

    normalized so that the peak value (at theta = pi/2, broadside to
    the dipole) is 1. The intensity vanishes along the dipole axis
    (theta = 0 or pi): a dipole does not radiate along its own axis.

    Parameters
    ----------
    theta : numpy array, shape [N]
        Polar angles from the dipole axis, in radians.

    Returns
    -------
    numpy array, shape [N]
        Normalized intensity in [0, 1].
    """
    theta = np.asarray(theta, dtype=float)
    return np.sin(theta) ** 2


def larmor_power(q, acceleration):
    """Radiated power of a non-relativistically accelerated charge.

    Implements the Larmor formula (SI units):

        P = q^2 a^2 / (6 pi eps0 c^3)

    valid for charge speeds small compared to the speed of light. The
    power is isotropic for non-relativistic motion.

    Parameters
    ----------
    q : float
        Charge, in coulombs.
    acceleration : numpy array, shape (3,)
        Acceleration vector, in m/s^2.

    Returns
    -------
    float
        Radiated power in watts.
    """
    acceleration = np.asarray(acceleration, dtype=float)
    a2 = float(acceleration @ acceleration)
    return q**2 * a2 / (6.0 * np.pi * EPSILON_0 * C_LIGHT**3)
