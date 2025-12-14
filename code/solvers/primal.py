\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import unpack_network_vars, unpack_mg_vars, total_cost, consensus_violation
from model import Case, build_network_qp, build_microgrid_qp

def solve_primal_penalty(case: Case,
                         beta: float = 8.0,
                         max_iter: int = 40,
                         tol: float = 1e-3,
                         verbose: bool = False) -> Dict:
    """
    Primal penalty / alternating minimization on the coupling:
      minimize f(x_mg) + g(z_net) + (beta/2)||x_mg - z||^2
    No dual variables. Often slower or may stall, but must be reported if it fails (per audio).
    """
    M = len(case.mgs); T = case.T
    z = np.zeros((M, T, 2))

    history = {"objective": [], "r": [], "beta": []}

    net0 = build_network_qp(case)
    net_meta = net0["meta"]

    mg_vars = None
    net_vars = None

    for k in range(max_iter):
        # MG updates given z
        x = np.zeros((M, T, 2))
        mg_vars = []
        for m in range(M):
            qp = build_microgrid_qp(case, m=m, z=z[m], u=np.zeros_like(z[m]), rho=beta)
            res = solve_qp_eq_box(qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"], max_iter=20, tol=1e-7)
            mv = unpack_mg_vars(res["x"], qp["meta"])
            mg_vars.append(mv)
            x[m,:,0] = mv["pin"]
            x[m,:,1] = mv["qin"]

        # Network update given x
        net_qp = build_network_qp(case, x_mg=x, u_mg=np.zeros_like(x), rho=beta)
        net_res = solve_qp_eq_box(net_qp["H"], net_qp["f"], net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"], max_iter=25, tol=1e-7)
        net_vars = unpack_network_vars(net_res["x"], net_meta)
        z = np.zeros_like(z)
        z[:,:,0] = net_vars["pmg"].T
        z[:,:,1] = net_vars["qmg"].T

        r = np.linalg.norm((x - z).reshape(-1))
        obj = total_cost(case, net_vars, mg_vars)
        history["objective"].append(obj)
        history["r"].append(float(r))
        history["beta"].append(float(beta))

        if verbose:
            print(f"Primal iter {k}: obj={obj:.4f} r={r:.3e}")

        if r < tol:
            break

    return dict(method="primal", history=history, net=net_vars, mgs=mg_vars,
                consensus_violation=consensus_violation(net_vars, mg_vars))
