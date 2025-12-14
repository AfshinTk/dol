\
from __future__ import annotations
import os
import json
import numpy as np

# single-thread safety for BLAS backends (important in restricted environments)
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

from model import make_case
from solvers.centralized import solve_centralized
from solvers.admm import solve_admm
from solvers.dual import solve_dual_decomposition
from solvers.primal import solve_primal_penalty
from solvers.primal_dual import solve_primal_dual
from solvers.decomposition import solve_primal_decomposition
from solvers.common import total_cost, consensus_violation
from plots import line_plot, voltage_profile_plot, bar_plot

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(BASE_DIR)
FIG_DIR = os.path.join(ROOT, "figures")
OUT_DIR = os.path.join(ROOT, "outputs")
REP_DIR = os.path.join(ROOT, "report")

def _pad_series(series_list):
    maxlen = max(len(s) for s in series_list)
    out = []
    for s in series_list:
        s = list(s)
        if len(s) < maxlen:
            s = s + [s[-1]]*(maxlen-len(s))
        out.append(s)
    return np.array(out).T  # shape (K,S)

def run():
    os.makedirs(FIG_DIR, exist_ok=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(REP_DIR, exist_ok=True)

    # Use small T for speed but still meaningful convergence plots
    case = make_case(T=2, M=2, seed=7)

    # Centralized benchmark
    cent = solve_centralized(case, verbose=False)
    J_star = cent["objective"]

    # Distributed methods
    admm = solve_admm(case, rho=5.0, alpha=1.6, max_iter=10, tol=1e-3, adaptive=False)
    dual = solve_dual_decomposition(case, step=0.8, max_iter=10, tol=1e-3)
    prim = solve_primal_penalty(case, beta=8.0, max_iter=10, tol=1e-3)
    pd   = solve_primal_dual(case, tau=0.4, sigma=0.4, gamma=1.0, max_iter=10, tol=1e-3)
    pdec = solve_primal_decomposition(case, step=0.6, max_iter=8, tol=1e-3)

    methods = [("Centralized", cent, None),
               ("ADMM", admm, "r_primal"),
               ("Dual", dual, "r"),
               ("Primal", prim, "r"),
               ("PrimalDual", pd, "r"),
               ("PrimalDecomp", pdec, "r")]

    # Plot objective trajectories
    obj_series = []
    labels = []
    for name, res, _ in methods[1:]:
        obj_series.append(res["history"]["objective"])
        labels.append(name)
    Y = _pad_series(obj_series)
    line_plot(Y, title="Objective vs Iteration (lower is better)",
              xlabel="iteration", ylabel="objective", path=os.path.join(FIG_DIR, "objective_vs_iter.png"))

    # Plot residuals for each method in separate figures
    # ADMM residuals
    line_plot(_pad_series([admm["history"]["r_primal"], admm["history"]["r_dual"]]),
              title="ADMM residuals", xlabel="iteration", ylabel="residual",
              path=os.path.join(FIG_DIR, "admm_residuals.png"))
    # Others
    for name, res, key in methods[2:]:
        if key is None:
            continue
        line_plot(np.array(res["history"][key]), title=f"{name} residual (consensus mismatch)",
                  xlabel="iteration", ylabel="residual", path=os.path.join(FIG_DIR, f"{name.lower()}_residual.png"))

    # Engineering figure: voltage profile from ADMM final network (last time step)
    V_last = admm["net"]["V"][-1]
    voltage_profile_plot(V_last, title="Voltage profile at final iteration (ADMM, last time)", path=os.path.join(FIG_DIR, "voltage_profile.png"))

    # Engineering figure: bus active power mismatch check (should be near zero inside network QP)
    # We compute mismatch from the network equality constraints quickly
    # Here we use net result and recompute mismatch with loads and injections
    # For simplicity: use sum of absolute injection mismatch for MGs as a proxy
    mismatch = []
    for m in range(len(case.mgs)):
        mismatch.append(np.mean(np.abs(admm["net"]["pmg"][:,m] - admm["mgs"][m]["pin"])))
    bar_plot(np.array(mismatch), title="Mean |P_injection mismatch| per microgrid (ADMM)",
             xlabel="microgrid index", ylabel="mean abs mismatch", path=os.path.join(FIG_DIR, "mg_p_mismatch.png"))

    # Sensitivity: ADMM with different rho
    sens_rhos = [1.0, 10.0]
    sens_obj = []
    for r in sens_rhos:
        rr = solve_admm(case, rho=r, alpha=1.6, max_iter=6, tol=1e-3, adaptive=False)
        sens_obj.append(rr["history"]["objective"])
    Y2 = _pad_series(sens_obj)
    line_plot(Y2, title="ADMM sensitivity: objective vs iter for different rho",
              xlabel="iteration", ylabel="objective", path=os.path.join(FIG_DIR, "admm_rho_sensitivity.png"))

    # Summary table
    summary = {}
    for name, res, _ in methods:
        if name == "Centralized":
            final_obj = res["objective"]
            iters = res["iters"]
            cons = 0.0
        else:
            final_obj = res["history"]["objective"][-1]
            iters = len(res["history"]["objective"])
            cons = res["consensus_violation"]
        summary[name] = dict(final_objective=float(final_obj),
                             relative_gap=float((final_obj - J_star)/max(1e-9, abs(J_star))),
                             iters=int(iters),
                             consensus_violation=float(cons))

    # Save outputs
    out = dict(J_star=float(J_star), methods=summary)
    with open(os.path.join(OUT_DIR, "results.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)

    # LaTeX table snippet
    rows = []
    for name in ["Centralized","ADMM","Dual","Primal","PrimalDual","PrimalDecomp"]:
        r = summary[name]
        rows.append(f"{name} & {r['final_objective']:.4f} & {r['relative_gap']:.4e} & {r['iters']} & {r['consensus_violation']:.3e} \\\\")
    table = r"""\begin{table}[t]
\centering
\caption{مقایسه روش ها نسبت به حل متمرکز}
\label{tab:compare}
\begin{tabular}{lcccc}
\hline
روش & تابع هدف نهایی & شکاف نسبی & تعداد تکرار & نقض اجماع \\
\hline
""" + "\n".join(rows) + r"""
\hline
\end{tabular}
\end{table}
"""
    with open(os.path.join(REP_DIR, "results_table.tex"), "w", encoding="utf-8") as f:
        f.write(table)

    print("Done. Outputs saved to outputs/ and figures/. Table saved to report/results_table.tex")

if __name__ == "__main__":
    run()
