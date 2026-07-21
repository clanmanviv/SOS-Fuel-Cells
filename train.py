"""
train.py
========
Cathode-degradation study, in two stages that mirror a real EIS + ML pipeline.

STAGE 1  Model-based feature extraction (ecm.py)
    Fit an equivalent circuit to each impedance spectrum and read off the
    charge-transfer resistance R_ct. We verify on synthetic spectra (whose true
    R_ct is known) that the ECM step recovers R_ct accurately.

STAGE 2  Data-driven prediction (XGBoost)
    Predict R_ct from the operating conditions, the material composition and the
    ageing time, so the model reproduces a cell's R_ct-vs-time curve. A linear
    model and a random forest are kept as baselines. Splitting is done by cell,
    so every test cell is completely unseen during training.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

import ecm

HERE = os.path.dirname(__file__)
DATA = os.path.join(HERE, "..", "data")
RES = os.path.join(HERE, "..", "results")
os.makedirs(RES, exist_ok=True)
NAVY = "#1F3A5F"; AMBER = "#E8B23A"
rng = np.random.default_rng(0)


def metrics(y, p):
    return (float(np.sqrt(mean_squared_error(y, p))),
            float(mean_absolute_error(y, p)), float(r2_score(y, p)))


# ============================================================
# STAGE 1 -- ECM feature extraction: does it recover R_ct?
# ============================================================
def stage1_ecm(df):
    sample = df.sample(40, random_state=1)
    true, rec = [], []
    example = None
    for _, r in sample.iterrows():
        Rct = r.R_ct
        # plausible remaining cathode-circuit parameters
        Rs, Q, n, sigma = 0.05, 0.05, 0.85, 0.03
        w, z = ecm.synth_spectrum(Rs, Rct, Q, n, sigma, noise=0.01, rng=rng)
        best, _ = ecm.select_best(w, z)
        true.append(Rct); rec.append(best["Rct"])
        if example is None:
            example = (w, z, best)
    true, rec = np.array(true), np.array(rec)
    mape = float(np.mean(np.abs(rec - true) / true) * 100)

    # --- Nyquist figure for one spectrum + its ECM fit ---
    w, z, best = example
    plt.figure(figsize=(6.2, 4.6))
    plt.plot(z.real, -z.imag, "o", ms=5, color=NAVY, label="synthetic EIS data")
    plt.plot(best["z_fit"].real, -best["z_fit"].imag, "-", lw=2, color=AMBER,
             label="fitted equivalent circuit (%s)" % best["model"].split("_")[0])
    plt.axvline(best["params"]["Rs"], ls=":", color="gray", lw=1)
    plt.annotate("R$_s$", (best["params"]["Rs"], 0.002), fontsize=9, color="gray")
    plt.annotate("R$_{ct}$ = %.3f $\\Omega\\cdot$cm$^2$" % best["Rct"],
                 (best["params"]["Rs"] + best["Rct"] * 0.5, max(-z.imag) * 0.8),
                 fontsize=10)
    plt.xlabel("Z$_{real}$ ($\\Omega\\cdot$cm$^2$)")
    plt.ylabel("$-$Z$_{imag}$ ($\\Omega\\cdot$cm$^2$)")
    plt.title("Stage 1 -- EIS spectrum and equivalent-circuit fit")
    plt.legend(fontsize=9); plt.axis("equal"); plt.grid(alpha=0.3)
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_eis_nyquist.png"), dpi=150)
    plt.close()
    return {"n_spectra": len(true), "recovery_mape_pct": mape}


# ============================================================
# STAGE 2 -- XGBoost predicts R_ct(t)
# ============================================================
FEATURES = ["pt_loading", "io_c_ratio", "carbon_graphitized", "T_C", "RH",
            "j_op", "V_upper", "ss_rate", "time_h"]
PRETTY = {"pt_loading": "Pt loading", "io_c_ratio": "ionomer/carbon",
          "carbon_graphitized": "graphitised C", "T_C": "temperature",
          "RH": "humidity", "j_op": "current density", "V_upper": "upper potential",
          "ss_rate": "start-stop rate", "time_h": "ageing time"}


def stage2_ml(df):
    cells = df.cell_id.unique()
    tr_cells, te_cells = train_test_split(cells, test_size=0.2, random_state=42)
    tr = df[df.cell_id.isin(tr_cells)]; te = df[df.cell_id.isin(te_cells)]
    Xtr, ytr = tr[FEATURES].values, tr.R_ct.values
    Xte, yte = te[FEATURES].values, te.R_ct.values

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=300, max_depth=12,
                                               random_state=0, n_jobs=-1),
        "XGBoost": XGBRegressor(n_estimators=500, max_depth=4, learning_rate=0.05,
                                subsample=0.9, colsample_bytree=0.9,
                                random_state=0, n_jobs=-1),
    }
    res = {}
    for name, m in models.items():
        m.fit(Xtr, ytr)
        res[name] = {"train": metrics(ytr, m.predict(Xtr)),
                     "test": metrics(yte, m.predict(Xte))}
    xgb = models["XGBoost"]

    # --- parity plot (XGBoost, test cells) ---
    pte = xgb.predict(Xte)
    plt.figure(figsize=(5.6, 5.2))
    lim = [0, max(yte.max(), pte.max()) * 1.05]
    plt.plot(lim, lim, "--", color=AMBER, label="perfect prediction")
    plt.scatter(yte, pte, s=12, alpha=0.4, color=NAVY, edgecolor="none")
    r = res["XGBoost"]["test"]
    plt.xlabel("true R$_{ct}$ ($\\Omega\\cdot$cm$^2$)")
    plt.ylabel("predicted R$_{ct}$ ($\\Omega\\cdot$cm$^2$)")
    plt.title("Stage 2 -- XGBoost R$_{ct}$ prediction (unseen cells)\n"
              "RMSE=%.4f, R$^2$=%.3f" % (r[0], r[2]))
    plt.legend(fontsize=9); plt.xlim(lim); plt.ylim(lim)
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_xgb_parity.png"), dpi=150); plt.close()

    # --- feature importance ---
    imp = xgb.feature_importances_
    order = np.argsort(imp)
    plt.figure(figsize=(6.4, 4.4))
    plt.barh([PRETTY[FEATURES[i]] for i in order], imp[order], color=NAVY)
    plt.xlabel("XGBoost feature importance (gain)")
    plt.title("Stage 2 -- what drives cathode R$_{ct}$ degradation")
    plt.tight_layout(); plt.savefig(os.path.join(RES, "fig_xgb_importance.png"), dpi=150)
    plt.close()

    # --- example R_ct(t) curves: predicted vs true for a few test cells ---
    plt.figure(figsize=(6.4, 4.6))
    for k, cid in enumerate(te_cells[:5]):
        c = te[te.cell_id == cid].sort_values("time_h")
        plt.plot(c.time_h, c.R_ct, "o", color=plt.cm.viridis(k / 5), ms=5)
        plt.plot(c.time_h, xgb.predict(c[FEATURES].values), "-",
                 color=plt.cm.viridis(k / 5), lw=1.8)
    plt.plot([], [], "ko", label="true R$_{ct}$"); plt.plot([], [], "k-", label="XGBoost")
    plt.xlabel("operating time (h)"); plt.ylabel("R$_{ct}$ ($\\Omega\\cdot$cm$^2$)")
    plt.title("Stage 2 -- predicted vs true R$_{ct}$ curves (test cells)")
    plt.legend(fontsize=9); plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(os.path.join(RES, "fig_rct_curves.png"), dpi=150); plt.close()

    imp_named = sorted(zip([PRETTY[f] for f in FEATURES], imp.tolist()),
                       key=lambda x: -x[1])
    return res, imp_named


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA, "aging_rct.csv"))
    print("Stage 1: ECM feature extraction ...")
    s1 = stage1_ecm(df)
    print("  recovered R_ct on %d spectra, mean abs error %.2f%%"
          % (s1["n_spectra"], s1["recovery_mape_pct"]))

    print("Stage 2: XGBoost R_ct prediction ...")
    s2, imp = stage2_ml(df)
    for name, d in s2.items():
        rmse, mae, r2 = d["test"]
        print(f"  {name:18s} test  RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.3f}")
    print("  top drivers:", ", ".join(f"{n} ({v:.2f})" for n, v in imp[:4]))

    out = {"stage1_ecm": s1,
           "stage2_models": {k: v for k, v in s2.items()},
           "feature_importance": imp}
    with open(os.path.join(RES, "metrics.json"), "w") as f:
        json.dump(out, f, indent=2)
    print("wrote results/metrics.json and 4 figures to results/")
