import numpy as np
from scipy.interpolate import interp1d

# Load natO data [Energy (eV), Cross Section (b)]
natO = np.loadtxt("data/O.ntot")
# Load natSi data [Energy (eV), Cross Section (b)]
natSi = np.loadtxt("data/Si.ntot")

# Create linear interpolation functions (or log-log interpolation for low E if preferred)
f_O = interp1d(natO[:, 0], natO[:, 1], bounds_error=False, fill_value="extrapolate")
f_Si = interp1d(natSi[:, 0], natSi[:, 1], bounds_error=False, fill_value="extrapolate")

# Define energy range (combining unique points from both grids or using a fine mesh)
energy_grid = np.unique(np.sort(np.concatenate((natO[:, 0], natSi[:, 0]))))

# Calculate cross section: \sigma(SiO2) = \sigma(Si) + 2 * \sigma(O)
sigma_SiO2 = f_Si(energy_grid) + 2.0 * f_O(energy_grid)

# Save result to file formatted in standard ENDF/pointwise E-format
with open("data/SiO2.ntot", "w") as f:
    for e, xs in zip(energy_grid, sigma_SiO2):
        f.write(f"{e:15.8E} {xs:15.8E}\n")
