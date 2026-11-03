"""Electrodynamics from scratch.

Source code for the "Electrodynamics from Scratch" lecture series: the
functions used by the course notebooks to compute and visualize fields,
potentials, equipotential surfaces, multipoles, and radiation.
"""

try:
    from ._version import version as __version__
except ImportError:
    __version__ = "unknown"

from .constants import (
    C_LIGHT,
    ELEMENTARY_CHARGE,
    EPSILON_0,
    K_COULOMB,
    MU_0,
)
from .electrostatics import (
    electric_field_from_potential,
    gaussian_delta,
    laplacian,
    line_charge_field,
    point_charge_field,
    point_charge_potential,
    poisson_solve,
    ring_field,
    ring_potential,
    shell_field,
    solid_sphere_field,
    superpose,
)
from .gauge import (
    check_gauge_invariance,
    coulomb_gauge_divergence,
    fields_from_potentials,
    gauge_transform,
    lorenz_residual,
)
from .magnetostatics import (
    biot_savart,
    circular_loop_curve,
    lorentz_force,
    magnetic_dipole_field,
    magnetic_dipole_moment,
    solenoid_curve,
    vector_potential_wire,
)
from .multipole import (
    exact_potential,
    expansion_error_vs_r,
    far_field_potential,
    multipole_moments,
    point_dipole_field,
    point_dipole_potential,
    power_law_fit,
    quadrupole_potential,
)
from .visualize import (
    plot_decay,
    plot_equipotentials,
    plot_field_3d,
    plot_field_and_potential,
    plot_field_lines,
    plot_trajectory_3d,
    plot_vector_field,
)
from .waves import (
    dalembert,
    dipole_radiation_fields,
    larmor_power,
    lieard_wiechert_fields,
    lieard_wiechert_potentials,
    radiation_pattern,
    retarded_time,
    split_velocity_acceleration,
)

__all__ = [
    "__version__",
    # constants
    "C_LIGHT",
    "ELEMENTARY_CHARGE",
    "EPSILON_0",
    "K_COULOMB",
    "MU_0",
    # electrostatics
    "electric_field_from_potential",
    "gaussian_delta",
    "laplacian",
    "line_charge_field",
    "point_charge_field",
    "point_charge_potential",
    "poisson_solve",
    "ring_field",
    "ring_potential",
    "shell_field",
    "solid_sphere_field",
    "superpose",
    # gauge
    "check_gauge_invariance",
    "coulomb_gauge_divergence",
    "fields_from_potentials",
    "gauge_transform",
    "lorenz_residual",
    # magnetostatics
    "biot_savart",
    "circular_loop_curve",
    "lorentz_force",
    "magnetic_dipole_field",
    "magnetic_dipole_moment",
    "solenoid_curve",
    "vector_potential_wire",
    # multipole
    "expansion_error_vs_r",
    "exact_potential",
    "far_field_potential",
    "multipole_moments",
    "point_dipole_field",
    "point_dipole_potential",
    "power_law_fit",
    "quadrupole_potential",
    # waves
    "dalembert",
    "dipole_radiation_fields",
    "larmor_power",
    "lieard_wiechert_fields",
    "lieard_wiechert_potentials",
    "radiation_pattern",
    "retarded_time",
    "split_velocity_acceleration",
    # visualize
    "plot_decay",
    "plot_equipotentials",
    "plot_field_3d",
    "plot_field_and_potential",
    "plot_field_lines",
    "plot_trajectory_3d",
    "plot_vector_field",
]
