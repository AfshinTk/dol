\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import unpack_network_vars, unpack_mg_vars, total_cost, consensus_violation
from model import Case, build_network_qp, build_microgrid_qp

def solve_primal_dual(case: Case,
                      tau: float = 0.4,
                      sigma: float = 0.4,
                      gamma: float = 1.0,
                      max_iter: int = 60,
                      tol: float = 1e-3,
                      verbose: bool = False) -> Dict:
    """
    Proximal primal-dual method for coupling equality:
        x_mg = z_net
    Uses price-like dual variable lambda and proximal stabilization.
    """
    M = len(case.mgs); T = case.T
    lam = np.zeros((M, T, 2))
    x_prev = np.zeros((M, T, 2))
    z_prev = np.zeros((M, T, 2))

    history = {"objective": [], "r": [], "tau": [], "sigma": []}

    net0 = build_network_qp(case)
    net_meta = net0["meta"]

    mg_vars = None
    net_vars = None

    for k in range(max_iter):
        # MG update with proximal on injections
        x = np.zeros((M, T, 2))
        mg_vars = []
        for m in range(M):
            qp = build_microgrid_qp(case, m=m, lam=lam[m])
            # add prox on p_inj and q_inj: (1/(2tau))||x - x_prev||^2
            meta = qp["meta"]
            dim_t = meta["dim_t"]
            off = meta["offsets"]
            H = qp["H"].copy()
            f = qp["f"].copy()
            for t in range(T):
                ip = t*dim_t + off["pin"]
                iq = t*dim_t + off["qin"]
                H[ip, ip] += 1.0/tau
                H[iq, iq] += 1.0/tau
                f[ip] += -(1.0/tau)*x_prev[m, t, 0]
                f[iq] += -(1.0/tau)*x_prev[m, t, 1]
            res = solve_qp_eq_box(H, f, qp["A"], qp["b"], qp["l"], qp["u"], max_iter=20, tol=1e-7)
            mv = unpack_mg_vars(res["x"], meta)
            mg_vars.append(mv)
            x[m,:,0] = mv["pin"]
            x[m,:,1] = mv["qin"]

        # Network update with proximal on pmg/qmg
        net_qp = build_network_qp(case, lam=lam)
        metaN = net_qp["meta"]
        dim_tN = metaN["dim_t"]
        offN = metaN["offsets"]
        Hn = net_qp["H"].copy()
        fn = net_qp["f"].copy()
        for t in range(T):
            for m in range(M):
                ip = t*dim_tN + offN["pmg"] + m
                iq = t*dim_tN + offN["qmg"] + m
                Hn[ip, ip] += 1.0/sigma
                Hn[iq, iq] += 1.0/sigma
                fn[ip] += -(1.0/sigma)*z_prev[m, t, 0]
                fn[iq] += -(1.0/sigma)*z_prev[m, t, 1]
        net_res = solve_qp_eq_box(Hn, fn, net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"], max_iter=25, tol=1e-7)
        net_vars = unpack_network_vars(net_res["x"], net_meta)
        z = np.zeros((M, T, 2))
        z[:,:,0] = net_vars["pmg"].T
        z[:,:,1] = net_vars["qmg"].T

        # Dual update
        diff = (z - x)
        r = np.linalg.norm(diff.reshape(-1))
        lam = lam + gamma*diff

        x_prev = x.copy()
        z_prev = z.copy()

        obj = total_cost(case, net_vars, mg_vars)
        history["objective"].append(obj)
        history["r"].append(float(r))
        history["tau"].append(float(tau))
        history["sigma"].append(float(sigma))

        if verbose:
            print(f"Primal-Dual iter {k}: obj={obj:.4f} r={r:.3e}")

        if r < tol:
            break

    return dict(method="primal_dual", history=history, net=net_vars, mgs=mg_vars,
                consensus_violation=consensus_violation(net_vars, mg_vars))
