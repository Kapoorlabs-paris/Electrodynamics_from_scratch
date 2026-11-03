"""Scalar and vector potentials, E/B from potentials, and gauge freedom.

The electromagnetic field is described by a scalar potential V and a
vector potential A through

    B = curl(A)
    E = -grad(V) - dA/dt

The potentials are not unique: the gauge transformation

    A' = A + grad(chi)
    V' = V - d(chi)/dt

leaves E and B unchanged for any smooth scalar function chi.  This module
implements the static case only: dA/dt = 0 and d(chi)/dt = 0 are assumed
everywhere, so E = -grad(V) and V' = V.

Grid conventions
----------------
A scalar field on a grid is a numpy array of shape (ny, nx) in 2D or
(nz, ny, nx) in 3D, with the last axis always x.  A vector field is a
tuple (or list) of component grids in the order (Fx, Fy) in 2D or
(Fx, Fy, Fz) in 3D.  The grid spacing dx is a single float, the same
along every axis.  All derivatives are computed with np.gradient
(second-order central differences in the interior, one-sided at the
edges), so the grid must have at least two points per axis.
"""

from typing import Dict, List, Sequence, Tuple, Union

import numpy as np

from .constants import C_LIGHT

__all__ = [
    "fields_from_potentials",
    "gauge_transform",
    "check_gauge_invariance",
    "lorenz_residual",
    "coulomb_gauge_divergence",
]

Grid = np.ndarray
VectorField = Sequence[Grid]


def _grad(grid: Grid, dx: float) -> List[Grid]:
    """Return the partial derivatives of a scalar grid, x-component last.

    Implements grad f = (df/dx, df/dy) in 2D or (df/dx, df/dy, df/dz)
    in 3D, computed with second-order central differences via
    np.gradient(grid, dx).
    """
    if grid.ndim not in (2, 3):
        raise ValueError("grid must be 2D (ny, nx) or 3D (nz, ny, nx)")
    if dx <= 0.0:
        raise ValueError("grid spacing dx must be positive")
    # np.gradient returns derivatives along array axes in order; the
    # spatial order we publish is (x, y, z), i.e. reversed axis order.
    return [np.asarray(g) for g in np.gradient(grid, dx)][::-1]


def _div(components: Sequence[Grid], dx: float) -> Grid:
    """Return the divergence of a vector field given as component grids.

    Implements div(F) = dFx/dx + dFy/dy (+ dFz/dz in 3D).
    """
    comps = list(components)
    if len(comps) not in (2, 3):
        raise ValueError("vector field must have 2 (2D) or 3 (3D) components")
    grads = [_grad(c, dx) for c in comps]
    # grads[i][k] is d(component i)/d(axis k) with axis order (x, y, z)
    out = np.zeros_like(np.asarray(comps[0], dtype=float))
    for i, g in enumerate(grads):
        out = out + g[i]
    return out


def _curl(components: Sequence[Grid], dx: float) -> List[Grid]:
    """Return the curl of a vector field given as component grids.

    In 3D, implements curl(F) = (dFz/dy - dFy/dz, dFx/dz - dFz/dx,
    dFy/dx - dFx/dy).  In 2D only the z-component is defined and a
    one-component list [dFy/dx - dFx/dy] is returned.
    """
    comps = [np.asarray(c, dtype=float) for c in components]
    if len(comps) not in (2, 3):
        raise ValueError("vector field must have 2 (2D) or 3 (3D) components")
    grads = [_grad(c, dx) for c in comps]
    if len(comps) == 2:
        # grads[0][0] = dAx/dx, grads[1][1] = dAy/dy
        return [grads[1][0] - grads[0][1]]
    # grads[i][k] is d(component i)/d(axis k) with axis order (x, y, z)
    gx, gy, gz = grads
    return [
        gz[1] - gy[2],  # dAz/dy - dAy/dz
        gx[2] - gz[0],  # dAx/dz - dAz/dx
        gy[0] - gx[1],  # dAy/dx - dAx/dy
    ]


def fields_from_potentials(
    V: Grid, A: VectorField, dx: float
) -> Tuple[Tuple[Grid, ...], Tuple[Grid, ...]]:
    """Compute E and B from the potentials V and A (static case).

    Physics: B = curl(A) and E = -grad(V) - dA/dt.  This module treats
    the static case only, so dA/dt is taken to be zero and

        E = -grad(V),   B = curl(A).

    Parameters
    ----------
    V : ndarray
        Scalar potential on a 2D (ny, nx) or 3D (nz, ny, nx) grid.
    A : sequence of ndarray
        Vector potential components (Ax, Ay) in 2D or (Ax, Ay, Az) in
        3D, each with the same shape as V.
    dx : float
        Uniform grid spacing along every axis.

    Returns
    -------
    (E, B) : tuple of tuples of ndarray
        E = (Ex, Ey) or (Ex, Ey, Ez) and B = (Bz,) in 2D or
        (Bx, By, Bz) in 3D.

    Limitations: static fields only (dA/dt = 0); edge points use
    one-sided differences, so derivatives there are first-order.
    """
    comps = [np.asarray(c, dtype=float) for c in A]
    if len(comps) != V.ndim:
        raise ValueError(
            "A must supply one component grid per spatial dimension "
            f"({V.ndim}), got {len(comps)}"
        )
    for c in comps:
        if c.shape != V.shape:
            raise ValueError("all potential grids must have the same shape")
    grad_V = _grad(np.asarray(V, dtype=float), dx)
    E = tuple(-g for g in grad_V)
    B = tuple(_curl(comps, dx))
    return E, B


def gauge_transform(
    V: Grid, A: VectorField, chi: Grid, dx: float
) -> Tuple[Grid, Tuple[Grid, ...]]:
    """Apply a static gauge transformation to the potentials.

    Physics: A' = A + grad(chi) and V' = V - d(chi)/dt.  This module
    treats a static gauge function chi, so d(chi)/dt = 0 and

        A' = A + grad(chi),   V' = V.

    Parameters
    ----------
    V : ndarray
        Scalar potential on a 2D or 3D grid.
    A : sequence of ndarray
        Vector potential components, same shape as V.
    chi : ndarray
        Gauge function on the same grid, held static in time.
    dx : float
        Uniform grid spacing along every axis.

    Returns
    -------
    (V_new, A_new) : tuple
        V_new is the unchanged scalar potential V; A_new is the tuple
        (Ax + dchi/dx, Ay + dchi/dy) or (Ax + dchi/dx, Ay + dchi/dy,
        Az + dchi/dz).
    """
    V_arr = np.asarray(V, dtype=float)
    comps = [np.asarray(c, dtype=float) for c in A]
    chi_arr = np.asarray(chi, dtype=float)
    if chi_arr.shape != V_arr.shape:
        raise ValueError("chi must have the same shape as V")
    if len(comps) != V_arr.ndim:
        raise ValueError(
            "A must supply one component grid per spatial dimension "
            f"({V_arr.ndim}), got {len(comps)}"
        )
    grad_chi = _grad(chi_arr, dx)
    A_new = tuple(c + g for c, g in zip(comps, grad_chi))
    return V_arr, A_new


def check_gauge_invariance(
    V: Grid, A: VectorField, chi: Grid, dx: float
) -> Dict[str, Union[float, bool]]:
    """Check that E and B are unchanged by a gauge transformation.

    Physics: the gauge transformation A' = A + grad(chi), V' = V
    (static chi) leaves E = -grad(V) and B = curl(A) exactly unchanged,
    because curl(grad(chi)) = 0.

    Parameters
    ----------
    V : ndarray
        Scalar potential on a 2D or 3D grid.
    A : sequence of ndarray
        Vector potential components, same shape as V.
    chi : ndarray
        Gauge function on the same grid.
    dx : float
        Uniform grid spacing along every axis.

    Returns
    -------
    dict
        {'E_max_diff': float, 'B_max_diff': float, 'invariant': bool}
        where the diffs are the maximum absolute componentwise changes
        in E and B across the grid.  'invariant' is True when each diff
        is at most 1e-6 of the corresponding field scale (the maximum
        absolute component value before the transform), with the scale
        floored at 1.0 to keep the test relative yet well defined for
        fields near zero.
    """
    E0, B0 = fields_from_potentials(V, A, dx)
    V_new, A_new = gauge_transform(V, A, chi, dx)
    E1, B1 = fields_from_potentials(V_new, A_new, dx)
    E_max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(E0, E1))
    B_max_diff = max(float(np.max(np.abs(a - b))) for a, b in zip(B0, B1))
    e_scale = max(float(np.max(np.abs(e))) for e in E0)
    b_scale = max(float(np.max(np.abs(b))) for b in B0)
    e_scale = max(e_scale, 1.0)
    b_scale = max(b_scale, 1.0)
    tol = 1e-6
    invariant = E_max_diff <= tol * e_scale and B_max_diff <= tol * b_scale
    return {
        "E_max_diff": E_max_diff,
        "B_max_diff": B_max_diff,
        "invariant": bool(invariant),
    }


def lorenz_residual(
    V: Grid, A: VectorField, dx: float, dV_dt: Union[Grid, None] = None
) -> Grid:
    """Compute the residual of the Lorenz gauge condition on a grid.

    Physics: the Lorenz condition is

        div(A) + (1/c^2) dV/dt = 0,

    and the residual is the left-hand side evaluated pointwise.  A
    potential set satisfying the Lorenz gauge gives residual zero.  If
    dV_dt is None the static case is assumed and dV/dt is taken to be
    zero, leaving r = div(A).

    Parameters
    ----------
    V : ndarray
        Scalar potential on a 2D (ny, nx) or 3D (nz, ny, nx) grid.
        Unused in the static case (dV_dt None) but kept in the
        signature for the time-dependent form.
    A : sequence of ndarray
        Vector potential components (Ax, Ay) or (Ax, Ay, Az), same
        shape as V.
    dx : float
        Uniform grid spacing along every axis.
    dV_dt : ndarray or None
        Time derivative of V on the same grid, or None for the static
        case.

    Returns
    -------
    ndarray
        The residual grid r = div(A) + (1/c^2) dV/dt, same shape as V,
        in SI units of T (equivalently V s / m^2 for the div term).
    """
    comps = [np.asarray(c, dtype=float) for c in A]
    V_arr = np.asarray(V, dtype=float)
    if len(comps) != V_arr.ndim:
        raise ValueError(
            "A must supply one component grid per spatial dimension "
            f"({V_arr.ndim}), got {len(comps)}"
        )
    residual = _div(comps, dx)
    if dV_dt is not None:
        dV = np.asarray(dV_dt, dtype=float)
        if dV.shape != V_arr.shape:
            raise ValueError("dV_dt must have the same shape as V")
        residual = residual + dV / C_LIGHT**2
    return residual


def coulomb_gauge_divergence(A: VectorField, dx: float) -> Grid:
    """Compute div(A), the quantity the Coulomb gauge sets to zero.

    Physics: the Coulomb gauge is the condition

        div(A) = 0.

    This function returns the left-hand side pointwise; a potential set
    in the Coulomb gauge gives a zero grid.

    Parameters
    ----------
    A : sequence of ndarray
        Vector potential components (Ax, Ay) on a 2D grid or
        (Ax, Ay, Az) on a 3D grid.
    dx : float
        Uniform grid spacing along every axis.

    Returns
    -------
    ndarray
        The divergence grid div(A) = dAx/dx + dAy/dy (+ dAz/dz in 3D),
        in SI units of T.
    """
    comps = [np.asarray(c, dtype=float) for c in A]
    if len(comps) not in (2, 3):
        raise ValueError("A must have 2 (2D) or 3 (3D) components")
    return _div(comps, dx)
