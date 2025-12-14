\
from __future__ import annotations
import numpy as np
from typing import Dict, Tuple

def qp_objective(H, f, x):
    return 0.5*float(x @ (H @ x)) + float(f @ x)

def unpack_network_vars(x: np.ndarray, meta: Dict):
    """
    Extract structured arrays from network decision vector.
    Returns dict of arrays with shapes:
      theta: (T,N), V:(T,N), P:(T,L), Q:(T,L), pmg:(T,M), qmg:(T,M), pimp:(T,), pexp:(T,), qsl:(T,)
    """
    T = meta["T"]; N = meta["N"]; L = meta["L"]; M = meta["M"]
    dim_t = meta["dim_t"]
    off = meta["offsets"]
    def sl(t, off0, n0):
        s = t*dim_t + off0
        return slice(s, s+n0)
    theta = np.zeros((T,N))
    V = np.zeros((T,N))
    P = np.zeros((T,L))
    Q = np.zeros((T,L))
    pmg = np.zeros((T,M))
    qmg = np.zeros((T,M))
    pimp = np.zeros(T); pexp = np.zeros(T); qsl = np.zeros(T)
    for t in range(T):
        theta[t] = x[sl(t, off["theta"], N)]
        V[t]     = x[sl(t, off["V"], N)]
        P[t]     = x[sl(t, off["P"], L)]
        Q[t]     = x[sl(t, off["Q"], L)]
        pmg[t]   = x[sl(t, off["pmg"], M)]
        qmg[t]   = x[sl(t, off["qmg"], M)]
        pimp[t]  = x[sl(t, off["pimp"], 1)]
        pexp[t]  = x[sl(t, off["pexp"], 1)]
        qsl[t]   = x[sl(t, off["qsl"], 1)]
    return dict(theta=theta, V=V, P=P, Q=Q, pmg=pmg, qmg=qmg, pimp=pimp, pexp=pexp, qsl=qsl)

def unpack_mg_vars(x: np.ndarray, meta: Dict):
    T = meta["T"]; dim_t = meta["dim_t"]; off = meta["offsets"]
    def sl(t, o): return t*dim_t + o
    p = np.array([x[sl(t, off["p"])] for t in range(T)])
    hb = np.array([x[sl(t, off["hb"])] for t in range(T)])
    dr = np.array([x[sl(t, off["dr"])] for t in range(T)])
    pin = np.array([x[sl(t, off["pin"])] for t in range(T)])
    qin = np.array([x[sl(t, off["qin"])] for t in range(T)])
    return dict(p=p, hb=hb, dr=dr, pin=pin, qin=qin)


def total_cost(case, net_vars, mg_vars):
    """
    Compute comparable objective value (without algorithmic penalty terms).
    Matches model.py costs:
      Network: import/export + voltage deviation + loss proxy + qsl penalty
      MG: CHP quadratic+linear + DR quadratic + boiler linear + q quadratic
    """
    c_import = 1.2
    c_export = 0.6
    w_v = 5.0
    w_loss = 0.2
    w_qsl = 0.05

    T = case.T
    # Network
    cost = 0.0
    for t in range(T):
        V = net_vars["V"][t]
        P = net_vars["P"][t]
        Q = net_vars["Q"][t]
        pimp = net_vars["pimp"][t]
        pexp = net_vars["pexp"][t]
        qsl = net_vars["qsl"][t]
        cost += c_import*pimp - c_export*pexp
        cost += w_v*np.sum((V - 1.0)**2)
        cost += w_loss*(np.sum(P**2) + np.sum(Q**2))
        cost += w_qsl*(qsl**2)

    # Microgrids
    for m, mg in enumerate(case.mgs):
        mv = mg_vars[m]
        p = mv["p"]
        dr = mv["dr"]
        hb = mv["hb"]
        q = mv["qin"]
        cost += np.sum(0.5*mg.c2_p*p**2 + mg.c1_p*p + 0.5*mg.c_dr*dr**2 + mg.c_boiler*hb + 0.5*mg.c_q*q**2)
    return float(cost)

def consensus_violation(net_vars, mg_vars):
    """|| (pmg,qmg)_net - (pin,qin)_mg ||_2 aggregated over t,m."""
    pmg = net_vars["pmg"]; qmg = net_vars["qmg"]
    M = pmg.shape[1]
    T = pmg.shape[0]
    v = 0.0
    for m in range(M):
        pin = mg_vars[m]["pin"]
        qin = mg_vars[m]["qin"]
        v += np.sum((pmg[:,m] - pin)**2) + np.sum((qmg[:,m] - qin)**2)
    return float(np.sqrt(v))
