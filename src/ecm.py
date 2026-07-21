"""
ecm.py
======
Equivalent-Circuit Modelling (ECM) for PEM fuel-cell cathode impedance.

This is the "model-based analysis" half of the project. Electrochemical
Impedance Spectroscopy (EIS) measures the complex impedance Z(w) of a cell
over a range of frequencies. Fitting an equivalent circuit to that spectrum
gives a small set of *physically explainable* parameters, the most important
of which is the cathode charge-transfer resistance R_ct. R_ct rises as the
catalyst layer degrades, so it is the feature we later predict with XGBoost.

Two candidate circuits are provided and the better-fitting (yet still
physically meaningful) one is selected per spectrum via an AIC criterion:

  M1  Rs - (Rct | CPE)                 no mass transport
  M2  Rs - ( (Rct + Warburg) | CPE )   Randles cell with Warburg diffusion

CPE (constant-phase element) is used instead of an ideal capacitor because the
porous electrode gives a depressed semicircle (n < 1).
"""

import numpy as np
from scipy.optimize import least_squares


# ---------- element impedances ----------
def z_cpe(w, Q, n):
    """Constant-phase element."""
    return 1.0 / (Q * (1j * w) ** n)


def z_warburg(w, sigma):
    """Semi-infinite Warburg (mass-transport) impedance."""
    return sigma * (1 - 1j) / np.sqrt(w)


def z_m1(w, Rs, Rct, Q, n):
    """M1: Rs in series with (Rct parallel CPE)."""
    z_par = 1.0 / (1.0 / Rct + 1.0 / z_cpe(w, Q, n))
    return Rs + z_par


def z_m2(w, Rs, Rct, Q, n, sigma):
    """M2: Randles cell, Rs + ((Rct + Warburg) | CPE)."""
    z_far = Rct + z_warburg(w, sigma)
    z_par = 1.0 / (1.0 / z_far + 1.0 / z_cpe(w, Q, n))
    return Rs + z_par


MODELS = {
    "M1_Rs_RctCPE":        {"fn": z_m1, "p0names": ["Rs", "Rct", "Q", "n"]},
    "M2_Rs_RctWarburgCPE": {"fn": z_m2, "p0names": ["Rs", "Rct", "Q", "n", "sigma"]},
}


# ---------- fitting ----------
def _residuals(params, w, z_data, fn):
    z = fn(w, *params)
    # modulus-weighted residuals keep high- and low-|Z| points comparable
    wgt = 1.0 / np.abs(z_data)
    return np.concatenate([(z.real - z_data.real) * wgt,
                           (z.imag - z_data.imag) * wgt])


def _initial_guess(w, z_data, with_warburg):
    Rs0 = max(z_data.real.min(), 1e-3)
    Rct0 = max(z_data.real.max() - Rs0, 1e-3)
    # characteristic frequency ~ peak of -Im(Z); Q ~ 1/(Rct * w_peak)
    w_peak = w[np.argmax(-z_data.imag)]
    Q0 = 1.0 / (Rct0 * w_peak ** 0.85)
    p0 = [Rs0, Rct0 * 0.7, Q0, 0.85]
    lo = [1e-4, 1e-4, 1e-8, 0.5]
    hi = [1e2, 1e3, 1e2, 1.0]
    if with_warburg:
        p0 += [Rct0 * 0.3 * np.sqrt(w_peak)]
        lo += [1e-6]
        hi += [1e3]
    return np.array(p0), (np.array(lo), np.array(hi))


def fit_circuit(w, z_data, model="M2_Rs_RctWarburgCPE"):
    """Fit one candidate circuit to a spectrum. Returns dict of params + AIC."""
    fn = MODELS[model]["fn"]
    names = MODELS[model]["p0names"]
    p0, bounds = _initial_guess(w, z_data, with_warburg=(len(names) == 5))
    sol = least_squares(_residuals, p0, bounds=bounds, args=(w, z_data, fn),
                        method="trf", max_nfev=6000)
    z_fit = fn(w, *sol.x)
    rss = float(np.sum(np.abs(z_fit - z_data) ** 2))
    n_obs = 2 * len(w)                      # real + imag components
    k = len(names)
    aic = n_obs * np.log(rss / n_obs + 1e-30) + 2 * k
    return {"model": model, "params": dict(zip(names, sol.x)),
            "Rct": float(dict(zip(names, sol.x))["Rct"]),
            "rss": rss, "aic": aic, "z_fit": z_fit}


def select_best(w, z_data):
    """Fit both candidate circuits and return the one with the lower AIC."""
    fits = [fit_circuit(w, z_data, m) for m in MODELS]
    fits.sort(key=lambda d: d["aic"])
    return fits[0], fits


# ---------- synthetic spectrum generator (ground truth known) ----------
def synth_spectrum(Rs, Rct, Q, n, sigma, freqs=None, noise=0.0, rng=None):
    """Generate a noisy Randles (M2) spectrum with known parameters."""
    if freqs is None:
        freqs = np.logspace(-1, 4, 50)      # 0.1 Hz .. 10 kHz
    w = 2 * np.pi * freqs
    z = z_m2(w, Rs, Rct, Q, n, sigma)
    if noise > 0:
        rng = rng or np.random.default_rng()
        z = z + (rng.normal(0, noise * np.abs(z)) +
                 1j * rng.normal(0, noise * np.abs(z)))
    return w, z


if __name__ == "__main__":
    rng = np.random.default_rng(0)
    w, z = synth_spectrum(0.05, 0.25, 0.05, 0.85, 0.03, noise=0.01, rng=rng)
    best, allf = select_best(w, z)
    print("true Rct = 0.25 ohm.cm^2")
    for f in allf:
        print(f"  {f['model']:22s} AIC={f['aic']:8.1f}  Rct={f['Rct']:.4f}")
    print("selected:", best["model"], "Rct=%.4f" % best["Rct"])
