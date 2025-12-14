\
from __future__ import annotations
import numpy as np
from typing import Dict
from .qp_box_eq import solve_qp_eq_box
from .common import qp_objective, unpack_network_vars, unpack_mg_vars
from model import Case, build_network_qp, build_microgrid_qp

def build_centralized_qp(case: Case) -> Dict:
    """
    Assemble one QP that includes:
    - network (Coordinator) variables
    - all microgrid internal variables
    - coupling equalities: (p_mg_network, q_mg_network) = (p_inj_mg, q_inj_mg)
    """
    net_qp = build_network_qp(case)
    mg_qps = [build_microgrid_qp(case, m=i) for i in range(len(case.mgs))]

    Hn, fn, An, bn, ln, un = net_qp["H"], net_qp["f"], net_qp["A"], net_qp["b"], net_qp["l"], net_qp["u"]
    n_net = Hn.shape[0]

    n_mg = sum(q["H"].shape[0] for q in mg_qps)
    n = n_net + n_mg

    # Block-diagonal H
    H = np.zeros((n, n))
    f = np.zeros(n)
    H[:n_net, :n_net] = Hn
    f[:n_net] = fn

    # Equality constraints stacking
    m_net = An.shape[0]
    m_mg = sum(q["A"].shape[0] for q in mg_qps)
    # Coupling constraints: for each mg and each t, p_mg_net - p_inj_mg = 0, q_mg_net - q_inj_mg = 0
    T = case.T
    M = len(case.mgs)
    m_cpl = T * M * 2
    m = m_net + m_mg + m_cpl

    A = np.zeros((m, n))
    b = np.zeros(m)

    # network part
    A[:m_net, :n_net] = An
    b[:m_net] = bn

    # microgrid parts
    offset_var = n_net
    offset_row = m_net
    mg_var_offsets = []
    for q in mg_qps:
        nm = q["H"].shape[0]
        mm = q["A"].shape[0]
        H[offset_var:offset_var+nm, offset_var:offset_var+nm] = q["H"]
        f[offset_var:offset_var+nm] = q["f"]
        A[offset_row:offset_row+mm, offset_var:offset_var+nm] = q["A"]
        b[offset_row:offset_row+mm] = q["b"]
        mg_var_offsets.append(offset_var)
        offset_var += nm
        offset_row += mm

    # bounds
    l = np.zeros(n); u = np.zeros(n)
    l[:n_net] = ln; u[:n_net] = un
    offset_var = n_net
    for q in mg_qps:
        nm = q["H"].shape[0]
        l[offset_var:offset_var+nm] = q["l"]
        u[offset_var:offset_var+nm] = q["u"]
        offset_var += nm

    # coupling rows
    meta = net_qp["meta"]
    dim_t = meta["dim_t"]
    off = meta["offsets"]
    # each time block in network vector
    def net_index(t, local_off, k):
        return t*dim_t + local_off + k

    row0 = m_net + m_mg
    for mgi in range(M):
        mg_meta = mg_qps[mgi]["meta"]
        mg_dim_t = mg_meta["dim_t"]
        mg_off = mg_meta["offsets"]
        mg_base = mg_var_offsets[mgi]

        for t in range(T):
            # p coupling
            r = row0
            A[r, net_index(t, off["pmg"], mgi)] = 1.0
            A[r, mg_base + t*mg_dim_t + mg_off["pin"]] = -1.0
            b[r] = 0.0
            row0 += 1
            # q coupling
            r = row0
            A[r, net_index(t, off["qmg"], mgi)] = 1.0
            A[r, mg_base + t*mg_dim_t + mg_off["qin"]] = -1.0
            b[r] = 0.0
            row0 += 1

    return dict(H=H, f=f, A=A, b=b, l=l, u=u, meta=dict(net=net_qp["meta"], mg_metas=[q["meta"] for q in mg_qps], n_net=n_net))

def solve_centralized(case: Case, verbose: bool = False) -> Dict:
    qp = build_centralized_qp(case)
    res = solve_qp_eq_box(qp["H"], qp["f"], qp["A"], qp["b"], qp["l"], qp["u"], max_iter=25, tol=1e-7, verbose=verbose)
    x = res["x"]
    obj = qp_objective(qp["H"], qp["f"], x)

    n_net = qp["meta"]["n_net"]
    x_net = x[:n_net]
    x_mg_all = x[n_net:]

    net_vars = unpack_network_vars(x_net, qp["meta"]["net"])

    mg_vars = []
    offset = 0
    for meta in qp["meta"]["mg_metas"]:
        nm = meta["T"]*meta["dim_t"]
        mg_vars.append(unpack_mg_vars(x_mg_all[offset:offset+nm], meta))
        offset += nm

    return dict(status=res["status"], iters=res["iters"], kkt=res["kkt_residual"],
                objective=obj, net=net_vars, mgs=mg_vars)
