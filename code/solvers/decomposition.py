\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import unpack_network_vars, unpack_mg_vars, total_cost, consensus_violation
from model import Case, build_network_qp, build_microgrid_qp

def _network_qp_with_fixed_injection(case: Case, s: np.ndarray):
    """
    Build network QP and add equality constraints fixing pmg/qmg to s (shape M,T,2).
    Returns qp dict and row mapping for multipliers.
    """
    qp = build_network_qp(case)
    H,f,A,b,l,u = qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"]
    meta = qp["meta"]
    T = meta["T"]; M = meta["M"]; dim_t = meta["dim_t"]; off = meta["offsets"]

    m0 = A.shape[0]
    m_add = T*M*2
    A2 = np.zeros((m0+m_add, A.shape[1]))
    b2 = np.zeros(m0+m_add)
    A2[:m0] = A
    b2[:m0] = b

    # rows for fixed constraints start here
    row = m0
    rows_p = np.zeros((M,T), dtype=int)
    rows_q = np.zeros((M,T), dtype=int)

    def net_index(t, local_off, k):
        return t*dim_t + local_off + k

    for m in range(M):
        for t in range(T):
            A2[row, net_index(t, off["pmg"], m)] = 1.0
            b2[row] = s[m,t,0]
            rows_p[m,t] = row
            row += 1
            A2[row, net_index(t, off["qmg"], m)] = 1.0
            b2[row] = s[m,t,1]
            rows_q[m,t] = row
            row += 1

    return dict(H=H, f=f, A=A2, b=b2, l=l, u=u, meta=meta, rows_p=rows_p, rows_q=rows_q)

def solve_primal_decomposition(case: Case,
                               step: float = 0.6,
                               max_iter: int = 40,
                               tol: float = 1e-3,
                               verbose: bool = False) -> Dict:
    """
    Primal decomposition aligned with Boyd:
    - Master variable s fixes coupling injections.
    - Subproblems: each MG and the network are solved independently given s.
    - Master updates s by (sub)gradient of the value functions via KKT multipliers.
    """
    M = len(case.mgs); T = case.T
    # master injection targets
    s = np.zeros((M, T, 2))
    history = {"objective": [], "r": [], "step": []}

    mg_vars = None
    net_vars = None

    for k in range(max_iter):
        # Microgrid subproblems with fixed injection = s
        mg_vars = []
        grad_mg = np.zeros((M, T, 2))
        for m in range(M):
            qp = build_microgrid_qp(case, m=m, enforce_target=True, target=s[m])
            res = solve_qp_eq_box(qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"], max_iter=25, tol=1e-7)
            mv = unpack_mg_vars(res["x"], qp["meta"])
            mg_vars.append(mv)
            # multipliers: per t rows (heat, pin-rel, pin-target, q-target)
            nu = res["nu"]
            m_t = 4
            for t in range(T):
                row_base = t*m_t
                nu_p = nu[row_base+2]
                nu_q = nu[row_base+3]
                grad_mg[m,t,0] = -nu_p
                grad_mg[m,t,1] = -nu_q

        # Network subproblem with fixed injection = s
        net_qp = _network_qp_with_fixed_injection(case, s)
        net_res = solve_qp_eq_box(net_qp["H"], net_qp["f"], net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"], max_iter=25, tol=1e-7)
        net_vars = unpack_network_vars(net_res["x"], net_qp["meta"])

        # Gradient from fixed injection constraints multipliers
        nuN = net_res["nu"]
        grad_net = np.zeros((M, T, 2))
        for m in range(M):
            for t in range(T):
                grad_net[m,t,0] = -nuN[net_qp["rows_p"][m,t]]
                grad_net[m,t,1] = -nuN[net_qp["rows_q"][m,t]]

        grad = grad_mg + grad_net

        # Step size schedule (diminishing)
        eta = step / np.sqrt(k+1)
        s_new = s - eta * grad

        # Project to feasible box bounds for injections
        s_new[:,:,0] = np.clip(s_new[:,:,0], -0.40, 0.40)
        for m, mg in enumerate(case.mgs):
            s_new[m,:,1] = np.clip(s_new[m,:,1], -mg.q_max, mg.q_max)

        s = s_new

        # Define "residual" as magnitude of gradient (stationarity proxy)
        r = float(np.linalg.norm(grad.reshape(-1)))

        obj = total_cost(case, net_vars, mg_vars)
        history["objective"].append(obj)
        history["r"].append(r)
        history["step"].append(float(eta))

        if verbose:
            print(f"PrimalDecomp iter {k}: obj={obj:.4f} ||grad||={r:.3e}")

        if r < tol:
            break

    return dict(method="primal_decomposition", history=history, net=net_vars, mgs=mg_vars,
                consensus_violation=consensus_violation(net_vars, mg_vars))
