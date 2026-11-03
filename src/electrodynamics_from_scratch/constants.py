"""Physical constants in SI units.

The 2019 SI definitions: the speed of light c and the elementary charge e
are exact, and the vacuum permeability follows from the vacuum permittivity
through the exact relation mu_0 * epsilon_0 = 1/c^2.
"""

import numpy as np

# vacuum permittivity, F/m
EPSILON_0 = 8.8541878128e-12
# speed of light in vacuum, m/s (exact)
C_LIGHT = 299792458.0
# vacuum permeability, H/m — from the exact relation mu_0 = 1/(epsilon_0 c^2)
MU_0 = 1.0 / (EPSILON_0 * C_LIGHT**2)
# elementary charge, C (exact)
ELEMENTARY_CHARGE = 1.602176634e-19
# Coulomb constant 1/(4 pi epsilon_0), N m^2 / C^2
K_COULOMB = 1.0 / (4.0 * np.pi * EPSILON_0)

__all__ = [
    "EPSILON_0",
    "MU_0",
    "C_LIGHT",
    "ELEMENTARY_CHARGE",
    "K_COULOMB",
]
