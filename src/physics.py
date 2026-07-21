"""
physics.py
==========
First-principles PEMFC polarization and degradation model.

Reproduces the exact model used in the Summer of Science end-term report
(Sections 6 and 11). Used both to draw the physics figures and to generate
the synthetic datasets the ML models in this repo are trained on.

Single cell:
    V(j)      = E_thermo - eta_act - eta_ohmic - eta_conc
    eta_act   = (R T / (alpha n F)) * ln(j / j0)          # Tafel form, j > j0
    eta_ohmic = j * ASR
    eta_conc  = c * ln(j_L / (j_L - j)),  c = (R T / n F)(1 + 1/alpha)
    E_thermo  = E0 + (dS / nF)(T - T0)                    # temperature-corrected OCV

Degradation (Section 11):
    V(t) = V0 - k1 t - k2 t^2      (rated current density)
    End-of-life = 10 % drop from V0
"""

import numpy as np

# ---- physical constants ----
F = 96485.0          # C/mol
R = 8.3145           # J/(mol K)
N_E = 2              # electrons per H2
E0 = 1.229           # V, standard reversible potential
dS = -44.34          # J/(mol K), reaction entropy change
T0 = 298.15          # K, reference temperature


def e_thermo(T):
    """Temperature-corrected reversible (thermodynamic) voltage."""
    return E0 + (dS / (N_E * F)) * (T - T0)


def cell_voltage(j, T=353.15, j0=1e-3, alpha=0.5, ASR=0.15, jL=1.5):
    """Single-cell voltage (V) at current density j (A/cm^2). Vectorised over j."""
    j = np.asarray(j, dtype=float)
    Eth = e_thermo(T)
    eta_act = (R * T / (alpha * N_E * F)) * np.log(j / j0)
    eta_ohmic = j * ASR
    c = (R * T / (N_E * F)) * (1.0 + 1.0 / alpha)
    eta_conc = c * np.log(jL / np.clip(jL - j, 1e-6, None))
    return Eth - eta_act - eta_ohmic - eta_conc


def stack_power(j, n_cells=45, area=150.0, **kw):
    """Total stack power (W) = V_cell * current * n_cells."""
    return cell_voltage(j, **kw) * j * area * n_cells


def degradation(t, V0=0.72, k1=1.4e-5, k2=1.4e-9):
    """Rated-current-density voltage as a function of operating hours t."""
    t = np.asarray(t, dtype=float)
    return V0 - k1 * t - k2 * t ** 2


def time_to_eol(V0=0.72, k1=1.4e-5, k2=1.4e-9, drop=0.10):
    """Hours until the cell drops `drop` below beginning-of-life voltage."""
    target = drop * V0
    if k2 <= 0:
        return target / k1
    disc = k1 ** 2 + 4 * k2 * target
    return (-k1 + np.sqrt(disc)) / (2 * k2)


if __name__ == "__main__":
    print("Self-check against report Table 4 (T=80C, j0=1e-3, ASR=0.15, jL=1.5):")
    for j in [0.02, 0.20, 1.00, 1.48]:
        print(f"  j={j:>4}  V_cell={cell_voltage(j):.4f} V   P_stack={stack_power(j):8.1f} W")
