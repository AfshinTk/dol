\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import unpack_network_vars, unpack_mg_vars, total_cost, consensus_violation
from model import Case, build_network_qp, build_microgrid_qp

def solve_dual_decomposition(case: Case,
                             step: float = 0.8,
                             max_iter: int = 60,
                             tol: float = 1e-3,
                             verbose: bool = False) -> Dict:
    """
    Dual decomposition / dual ascent on coupling constraints:
        p_mg_network = p_inj_mg
        q_mg_network = q_inj_mg
    Lambda plays the role of price signals (Wang 2017 interpretation).
    """
    M = len(case.mgs); T = case.T
    lam = np.zeros((M, T, 2))

    history = {"objective": [], "r": [], "step": []}

    net0 = build_network_qp(case)
    net_meta = net0["meta"]

    mg_vars = None
    net_vars = None

    for k in range(max_iter):
        # Microgrids: minimize local + lambda^T x
        mg_vars = []
        x_mg = np.zeros((M, T, 2))
        for m in range(M):
            qp = build_microgrid_qp(case, m=m, lam=lam[m])
            res = solve_qp_eq_box(qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"], max_iter=20, tol=1e-7)
            mv = unpack_mg_vars(res["x"], qp["meta"])
            mg_vars.append(mv)
            x_mg[m, :, 0] = mv["pin"]
            x_mg[m, :, 1] = mv["qin"]

        # Network: minimize network - lambda^T z
        net_qp = build_network_qp(case, lam=lam)
        net_res = solve_qp_eq_box(net_qp["H"], net_qp["f"], net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"], max_iter=25, tol=1e-7)
        net_vars = unpack_network_vars(net_res["x"], net_meta)
        z = np.zeros((M, T, 2))
        z[:, :, 0] = net_vars["pmg"].T
        z[:, :, 1] = net_vars["qmg"].T

        # Dual update
        diff = (z - x_mg)
        r = np.linalg.norm(diff.reshape(-1))
        # diminishing step optional
        eta = step / np.sqrt(k+1)
        lam = lam + eta * diff

        obj = total_cost(case, net_vars, mg_vars)
        history["objective"].append(obj)
        history["r"].append(float(r))
        history["step"].append(float(eta))

        if verbose:
            print(f"Dual iter {k}: obj={obj:.4f} r={r:.3e} eta={eta:.3e}")

        if r < tol:
            break

    return dict(method="dual", history=history, net=net_vars, mgs=mg_vars,
                consensus_violation=consensus_violation(net_vars, mg_vars))
