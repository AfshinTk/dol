\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import unpack_network_vars, unpack_mg_vars, total_cost, consensus_violation
from model import Case, build_network_qp, build_microgrid_qp

def solve_admm(case: Case,
               rho: float = 5.0,
               alpha: float = 1.6,
               max_iter: int = 40,
               tol: float = 1e-3,
               adaptive: bool = True,
               verbose: bool = False) -> Dict:
    """
    Consensus-style ADMM aligned with Chen 2023:
    - Microgrids (MC) solve local energy hub QPs with penalty to match PCC injections.
    - Coordinator (DMS) solves the network QP with penalty to match the microgrids.
    Coupling variables: (p_mg, q_mg) for each MG and each time.
    """
    M = len(case.mgs)
    T = case.T

    # Initialize
    z = np.zeros((M, T, 2))  # coordinator injections
    u = np.zeros((M, T, 2))  # scaled dual

    history = {"objective": [], "r_primal": [], "r_dual": [], "rho": []}

    # Prebuild base QPs for parsing meta shapes
    net0 = build_network_qp(case)
    net_meta = net0["meta"]
    mg0 = [build_microgrid_qp(case, m=i) for i in range(M)]
    mg_metas = [q["meta"] for q in mg0]

    # Keep last z for dual residual
    z_prev = z.copy()

    for k in range(max_iter):
        # x-update: microgrids
        x = np.zeros((M, T, 2))
        mg_vars = []
        for m in range(M):
            qp = build_microgrid_qp(case, m=m, z=z[m], u=u[m], rho=rho)
            res = solve_qp_eq_box(qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"], max_iter=20, tol=1e-7)
            mv = unpack_mg_vars(res["x"], qp["meta"])
            mg_vars.append(mv)
            x[m, :, 0] = mv["pin"]
            x[m, :, 1] = mv["qin"]

        # over-relaxation
        x_hat = alpha*x + (1.0-alpha)*z

        # z-update: network
        net_qp = build_network_qp(case, x_mg=x_hat, u_mg=u, rho=rho)
        net_res = solve_qp_eq_box(net_qp["H"], net_qp["f"], net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"], max_iter=25, tol=1e-7)
        net_vars = unpack_network_vars(net_res["x"], net_meta)
        z = np.zeros_like(z)
        z[:, :, 0] = net_vars["pmg"].T
        z[:, :, 1] = net_vars["qmg"].T

        # u-update
        u = u + (x_hat - z)

        # residuals
        r_pr = np.linalg.norm((x_hat - z).reshape(-1))
        r_du = np.linalg.norm((rho*(z - z_prev)).reshape(-1))
        z_prev = z.copy()

        obj = total_cost(case, net_vars, mg_vars)
        history["objective"].append(obj)
        history["r_primal"].append(float(r_pr))
        history["r_dual"].append(float(r_du))
        history["rho"].append(float(rho))

        if verbose:
            print(f"ADMM iter {k}: obj={obj:.4f} r_pr={r_pr:.3e} r_du={r_du:.3e} rho={rho:.2f}")

        # adaptive rho (scaled dual update)
        if adaptive:
            mu = 10.0
            tau = 2.0
            if r_pr > mu*r_du and r_du > 0:
                rho *= tau
                u /= tau
            elif r_du > mu*r_pr and r_pr > 0:
                rho /= tau
                u *= tau

        if r_pr < tol and r_du < tol:
            break

    return dict(method="admm", history=history, net=net_vars, mgs=mg_vars,
                consensus_violation=consensus_violation(net_vars, mg_vars))
