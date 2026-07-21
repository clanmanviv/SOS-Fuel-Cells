# PEMFC-ML — Cathode Degradation from EIS via Equivalent-Circuit Modelling + XGBoost

A small, reproducible study that predicts **cathode catalyst-layer degradation**
in a PEM fuel cell from its operating conditions and material composition. It
follows the standard *combined electrochemical + model-based* workflow used in
fuel-cell durability research:

1. **Model-based feature extraction (ECM).** An equivalent circuit is fitted to
   each Electrochemical Impedance Spectroscopy (EIS) spectrum to recover a few
   physically meaningful parameters. The key one is the **charge-transfer
   resistance `R_ct`**, which rises as the catalyst layer loses active surface
   area. Two candidate circuits are compared and the better-fitting, still
   physically explainable one is selected per spectrum (AIC):
   - `Rs – (Rct | CPE)`
   - `Rs – ((Rct + Warburg) | CPE)` &nbsp;(Randles cell with diffusion)

2. **Data-driven prediction (XGBoost).** Gradient-boosted trees predict the
   `R_ct`-vs-time curve from operating conditions (temperature, humidity,
   current density, upper potential, start-stop rate) and composition (Pt
   loading, ionomer/carbon ratio, carbon-support type). Splitting is by cell,
   so every test cell is unseen.

## Results (held-out cells)

| Stage | Result |
|---|---|
| ECM `R_ct` recovery on synthetic spectra | **0.44 %** mean abs error |
| Linear regression (baseline) | RMSE 0.024, R² 0.82 |
| Random forest (baseline) | RMSE 0.009, R² 0.98 |
| **XGBoost** | **RMSE 0.006, R² 0.99** |

Top degradation drivers found by XGBoost: **Pt loading, ageing time,
carbon-support type, start-stop rate** — all consistent with the known
Pt-dissolution and carbon-corrosion mechanisms.

## Run it

```bash
pip install -r requirements.txt
bash run_all.sh          # generates data, runs ECM + XGBoost, writes figures
```

Outputs land in `results/`: the EIS/ECM fit (Nyquist), predicted-vs-true `R_ct`
curves, an XGBoost parity plot, feature importances, and `metrics.json`.

## Layout

```
src/physics.py        PEMFC polarization + degradation physics (report Sections 6, 11)
src/ecm.py            equivalent-circuit impedance models, fitting, model selection
src/generate_data.py  synthetic cathode-degradation dataset (R_ct vs time)
src/train.py          Stage 1 ECM feature extraction + Stage 2 XGBoost prediction
```

## Honest note on the data

The data here is **synthetic**, generated from a physics-based degradation model,
so the study demonstrates the full ECM + ML workflow rather than a finding about
real hardware. The natural next step is to run the same pipeline on measured EIS
spectra (e.g. a real PEMFC ageing campaign) — the code already accepts external
spectra and `R_ct` tables. See the report, Section 12.4.

MIT licensed. Author: Vivaan Bhatia (Summer of Science 2026, IIT Bombay).
