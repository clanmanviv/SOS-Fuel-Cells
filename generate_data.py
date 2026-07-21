"""
generate_data.py
================
Build the synthetic cathode-degradation dataset.

Each virtual cell has a material composition and an operating condition. Its
cathode charge-transfer resistance R_ct grows with operating time as the
catalyst layer degrades (loss of electrochemically active surface area from
Pt dissolution and carbon corrosion). The growth rate depends on the known
degradation drivers: upper cell potential, start-stop frequency, temperature,
carbon-support type, humidity and Pt loading.

We record R_ct at a series of ageing times, giving one R_ct-vs-time curve per
cell. This is the quantity the XGBoost model learns to predict in train.py.
Data is synthetic and clearly labelled as such; see the report, Section 12.4.
"""

import os
import numpy as np
import pandas as pd

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
TIMES = np.arange(0, 2001, 250, dtype=float)      # 0 .. 2000 h, 9 points


def initial_rct(pt, T_C, RH):
    """Beginning-of-life R_ct (ohm.cm^2): lower with more Pt and higher T."""
    return 0.05 * (0.25 / pt) * np.exp(-0.015 * (T_C - 70)) * (1 + 0.30 * (1 - RH / 100))


def growth_per_1000h(V_upper, ss_rate, T_C, graphitized, RH):
    """Fractional R_ct growth rate per 1000 h from the degradation drivers."""
    beta = 0.18
    beta *= np.exp(1.8 * (V_upper - 0.85))          # Pt dissolution at high potential
    beta *= (1 + 0.06 * ss_rate)                    # carbon corrosion on start-stop
    beta *= np.exp(0.35 * (T_C - 70) / 10)          # thermal acceleration
    beta *= (1.0 if graphitized else 1.6)           # graphitised carbon resists corrosion
    beta *= (1 + 0.15 * (1 - RH / 100))             # dry operation mildly worse
    return beta


def make_dataset(n_cells=700, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for cid in range(n_cells):
        pt = rng.uniform(0.10, 0.40)
        io_c = rng.uniform(0.6, 1.0)
        graph = int(rng.integers(0, 2))
        T_C = rng.uniform(60, 90)
        RH = rng.uniform(40, 100)
        j_op = rng.uniform(0.4, 1.4)
        V_upper = rng.uniform(0.85, 0.98)
        ss_rate = rng.uniform(0, 20)                # start-stop events / 100 h

        Rct0 = initial_rct(pt, T_C, RH)
        beta = growth_per_1000h(V_upper, ss_rate, T_C, graph, RH)
        gamma = 0.25 * beta                         # mild quadratic acceleration
        for t in TIMES:
            tau = t / 1000.0
            Rct = Rct0 * (1 + beta * tau + gamma * tau ** 2)
            Rct *= (1 + rng.normal(0, 0.02))        # measurement/cell noise
            rows.append(dict(cell_id=cid, pt_loading=pt, io_c_ratio=io_c,
                             carbon_graphitized=graph, T_C=T_C, RH=RH,
                             j_op=j_op, V_upper=V_upper, ss_rate=ss_rate,
                             time_h=t, R_ct=Rct))
    return pd.DataFrame(rows)


if __name__ == "__main__":
    os.makedirs(DATA, exist_ok=True)
    df = make_dataset()
    df.to_csv(os.path.join(DATA, "aging_rct.csv"), index=False)
    print(f"wrote aging_rct.csv: {len(df)} rows, {df.cell_id.nunique()} cells")
    print("R_ct range: %.3f .. %.3f ohm.cm^2" % (df.R_ct.min(), df.R_ct.max()))
    print("mean R_ct at t=0:   %.3f" % df[df.time_h == 0].R_ct.mean())
    print("mean R_ct at t=2000: %.3f" % df[df.time_h == 2000].R_ct.mean())
