\
"""
Model and QP builders for: IEEE 33-bus distribution network + multiple energy-hub microgrids.

Design goals (per course + audio clarifications):
- IEEE 33-bus topology (radial distribution)
- Variables include: voltage magnitudes V, voltage angles theta, active/reactive power P/Q, and prices (as dual variables)
- Microgrids include CHP + Boiler + Demand Response (DR)
- Centralized problem plus decomposable distributed formulation aligned with:
    Chen et al. 2023 (ADMM with DMS/Coordinator and PCC coupling)
    Wang et al. 2017 (CHP/Boiler/DR energy management and price-based distributed coordination)

We use a convex QP approximation:
- Active flows: DC-like linear relation with angles: P_ij = b_ij (theta_i - theta_j)
- Reactive flows: free variables with balance; voltage drop uses LinDistFlow: V_j = V_i - 2 (r P_ij + x Q_ij)
- Voltage bounds as box constraints
- Microgrid internal models are convex quadratic + linear costs and linear constraints
"""
from __future__ import annotations
import numpy as np
from dataclasses import dataclass
from typing import Dict, Tuple, List

@dataclass
class NetworkData:
    nbus: int
    lines: np.ndarray  # shape (L, 5): from, to, r, x, rate
    slack: int = 0

@dataclass
class MicrogridData:
    bus: int
    pmax_chp: float
    alpha_heat: float   # heat produced per unit electric from CHP (simplified coupling)
    hmax_boiler: float
    dr_max: float
    q_max: float
    # costs: 0.5*c2*p^2 + c1*p, 0.5*cdr*dr^2, cB*h_boiler, 0.5*cq*q^2
    c2_p: float
    c1_p: float
    c_dr: float
    c_boiler: float
    c_q: float

@dataclass
class Case:
    net: NetworkData
    mgs: List[MicrogridData]
    T: int
    P_load: np.ndarray  # (T, nbus)
    Q_load: np.ndarray  # (T, nbus)
    H_load_mg: np.ndarray  # (T, M)
    P_load_mg: np.ndarray  # (T, M) local electric load served by MG behind the meter

def build_ieee33_topology() -> NetworkData:
    """
    IEEE 33-bus radial topology (32 lines). Parameters are typical per-unit-like values.
    For a graded course project, topology correctness and constraint structure matter more than exact param fidelity.
    """
    # 1-based bus indices in the standard description; we convert to 0-based.
    edges = [
        (1,2),(2,3),(3,4),(4,5),(5,6),(6,7),(7,8),(8,9),(9,10),(10,11),
        (11,12),(12,13),(13,14),(14,15),(15,16),(16,17),(17,18),
        (2,19),(19,20),(20,21),(21,22),
        (3,23),(23,24),(24,25),
        (6,26),(26,27),(27,28),(28,29),(29,30),(30,31),(31,32),(32,33),
    ]
    L = len(edges)
    lines = np.zeros((L, 5), dtype=float)
    for k,(i,j) in enumerate(edges):
        # simple synthetic r/x increasing slightly with depth
        depth = k / max(1, L-1)
        r = 0.01 + 0.03*depth
        x = 0.02 + 0.04*depth
        rate = 2.5  # per-unit flow limit
        lines[k] = [i-1, j-1, r, x, rate]
    return NetworkData(nbus=33, lines=lines, slack=0)

def make_case(T: int = 6, M: int = 3, seed: int = 7) -> Case:
    rng = np.random.default_rng(seed)
    net = build_ieee33_topology()

    # Base loads (per-unit-ish) for 33 buses, positive consumption
    baseP = rng.uniform(0.02, 0.12, size=net.nbus)
    baseQ = 0.6 * baseP

    # Time profile (morning to evening)
    profile = np.array([0.85, 0.95, 1.05, 1.10, 1.00, 0.90], dtype=float)
    profile = profile[:T] if T <= len(profile) else np.pad(profile, (0, T-len(profile)), constant_values=1.0)

    P_load = profile[:, None] * baseP[None, :]
    Q_load = profile[:, None] * baseQ[None, :]

    # Microgrids placed on selected buses (0-based)
    mg_buses = [5, 17, 24]  # buses 6,18,25
    mg_buses = mg_buses[:M]
    mgs: List[MicrogridData] = []
    for b in mg_buses:
        mgs.append(MicrogridData(
            bus=b,
            pmax_chp=0.35,
            alpha_heat=1.2,
            hmax_boiler=0.6,
            dr_max=0.08,
            q_max=0.15,
            c2_p=4.0,
            c1_p=0.6,
            c_dr=12.0,
            c_boiler=0.4,
            c_q=0.5
        ))

    # Microgrid local loads behind-the-meter (they must satisfy via CHP+import from network via injection variable)
    P_load_mg = rng.uniform(0.10, 0.18, size=(T, M))
    H_load_mg = rng.uniform(0.10, 0.22, size=(T, M))

    return Case(net=net, mgs=mgs, T=T, P_load=P_load, Q_load=Q_load, H_load_mg=H_load_mg, P_load_mg=P_load_mg)

def incidence_matrices(net: NetworkData):
    """Return from,to arrays and incidence for bus balances."""
    frm = net.lines[:,0].astype(int)
    to = net.lines[:,1].astype(int)
    L = net.lines.shape[0]
    N = net.nbus
    # Outgoing incidence: +1 for outflow, incoming: -1 for inflow
    Inc = np.zeros((N, L))
    for ell in range(L):
        i = frm[ell]; j = to[ell]
        Inc[i, ell] += 1.0
        Inc[j, ell] -= 1.0
    return frm, to, Inc

def build_network_qp(case: Case,
                     x_mg: np.ndarray | None = None,
                     u_mg: np.ndarray | None = None,
                     rho: float = 0.0,
                     lam: np.ndarray | None = None) -> Dict:
    """
    Network (Coordinator/DMS) QP.
    Decision variables per time t:
      theta (N), V (N), P_line (L), Q_line (L), p_mg (M), q_mg (M), p_imp, p_exp, q_slack
    Equality constraints:
      - slack angle = 0
      - P_line = b (theta_i - theta_j)
      - bus active balance with slack injection = p_imp - p_exp at slack bus
      - bus reactive balance with q_slack at slack bus
      - voltage drop: V_j = V_i - 2(r P + x Q)
    Box constraints:
      V in [0.95, 1.05], theta in [-pi, pi], flows in [-rate, rate], p_mg/q_mg bounds, p_imp/p_exp >=0
    Objective:
      grid import cost + voltage deviation penalty + quadratic loss proxy + ADMM penalty if rho>0
    """
    net = case.net
    N = net.nbus
    L = net.lines.shape[0]
    M = len(case.mgs)
    T = case.T

    frm, to, Inc = incidence_matrices(net)

    # Indices in stacked variable x
    # We stack time blocks
    # per t dimension:
    # theta N, V N, P L, Q L, p_mg M, q_mg M, p_imp 1, p_exp 1, q_slack 1
    dim_t = (N + N + L + L + M + M + 1 + 1 + 1)
    n = T * dim_t

    def idx(t, offset, length):
        start = t*dim_t + offset
        return slice(start, start+length)

    off_theta = 0
    off_V = off_theta + N
    off_P = off_V + N
    off_Q = off_P + L
    off_pmg = off_Q + L
    off_qmg = off_pmg + M
    off_pimp = off_qmg + M
    off_pexp = off_pimp + 1
    off_qsl = off_pexp + 1

    # Build quadratic H and linear f
    H = np.zeros((n, n))
    f = np.zeros(n)

    # Costs
    c_import = 1.2
    c_export = 0.6
    w_v = 5.0
    w_loss = 0.2
    w_qsl = 0.05

    for t in range(T):
        # small angle regularization for numerical stability
        w_th = 1e-3
        thsl = idx(t, off_theta, N)
        H[thsl, thsl] += 2.0 * w_th * np.eye(N)

        # voltage deviation penalty: w_v * sum_i (V_i - 1)^2 = w_v*(V^T V - 2*1^T V + const)
        Vsl = idx(t, off_V, N)
        H[Vsl, Vsl] += 2.0 * w_v * np.eye(N)
        f[Vsl] += -2.0 * w_v * np.ones(N)

        # quadratic loss proxy on line flows
        Psl = idx(t, off_P, L)
        Qsl = idx(t, off_Q, L)
        H[Psl, Psl] += 2.0 * w_loss * np.eye(L)
        H[Qsl, Qsl] += 2.0 * w_loss * np.eye(L)

        # import/export linear costs (keep convex by adding small quadratic)
        pimp = idx(t, off_pimp, 1)
        pexp = idx(t, off_pexp, 1)
        f[pimp] += c_import
        f[pexp] += -c_export
        H[pimp, pimp] += 1e-3
        H[pexp, pexp] += 1e-3

        qsl = idx(t, off_qsl, 1)
        H[qsl, qsl] += 2.0 * w_qsl

        # ADMM penalty on (z - x + u): network variable is z = (p_mg,q_mg)
        if rho > 0.0 and x_mg is not None and u_mg is not None:
            pmg = idx(t, off_pmg, M)
            qmg = idx(t, off_qmg, M)
            H[pmg, pmg] += rho * np.eye(M)
            H[qmg, qmg] += rho * np.eye(M)
            # linear term: rho * (u - x)^T z
            f[pmg] += rho * (u_mg[:, t, 0] - x_mg[:, t, 0])
            f[qmg] += rho * (u_mg[:, t, 1] - x_mg[:, t, 1])

        # Dual method linear term: -lambda^T z (since network sees -lambda in objective)
        if lam is not None:
            pmg = idx(t, off_pmg, M)
            qmg = idx(t, off_qmg, M)
            f[pmg] += -lam[:, t, 0]
            f[qmg] += -lam[:, t, 1]

    # Equality constraints A x = b
    # Count constraints:
    # per t:
    # 1 (slack theta)
    # L (P angle relation)
    # N (active balance)
    # N (reactive balance)
    # L (voltage drop)
    m_t = 1 + L + N + N + L
    m = T * m_t
    A = np.zeros((m, n))
    bvec = np.zeros(m)

    def row(t, k):
        return t*m_t + k

    # susceptance b_ij = 1/x (DC-like)
    x_line = net.lines[:, 3]
    bdc = 1.0 / np.maximum(1e-3, x_line)

    for t in range(T):
        k = 0
        # slack angle = 0
        A[row(t, k), idx(t, off_theta + net.slack, 1)] = 1.0
        bvec[row(t, k)] = 0.0
        k += 1

        # P_ell - b*(theta_i - theta_j) = 0
        for ell in range(L):
            r = row(t, k)
            # P_ell
            A[r, idx(t, off_P + ell, 1)] = 1.0
            # theta_i - theta_j
            A[r, idx(t, off_theta + int(frm[ell]), 1)] += -bdc[ell]
            A[r, idx(t, off_theta + int(to[ell]), 1)] += bdc[ell]
            bvec[r] = 0.0
            k += 1

        # Active power balance at each bus:
        # Inc * P + p_mg_at_bus + p_slack_at_bus - P_load = 0
        # At slack: p_slack = p_imp - p_exp
        Pload = case.P_load[t]
        for i in range(N):
            r = row(t, k)
            # Incidence on P_line
            A[r, idx(t, off_P, L)] = Inc[i]
            # microgrid injections mapped by bus
            for mgi, mg in enumerate(case.mgs):
                if mg.bus == i:
                    A[r, idx(t, off_pmg + mgi, 1)] += 1.0
            if i == net.slack:
                A[r, idx(t, off_pimp, 1)] += 1.0
                A[r, idx(t, off_pexp, 1)] += -1.0
            bvec[r] = Pload[i]
            k += 1

        # Reactive balance: Inc * Q + q_mg + q_slack - Q_load = 0
        Qload = case.Q_load[t]
        for i in range(N):
            r = row(t, k)
            A[r, idx(t, off_Q, L)] = Inc[i]
            for mgi, mg in enumerate(case.mgs):
                if mg.bus == i:
                    A[r, idx(t, off_qmg + mgi, 1)] += 1.0
            if i == net.slack:
                A[r, idx(t, off_qsl, 1)] += 1.0
            bvec[r] = Qload[i]
            k += 1

        # Voltage drop: V_j - V_i + 2(r P + x Q) = 0
        r_line = net.lines[:, 2]
        x_line = net.lines[:, 3]
        for ell in range(L):
            rrow = row(t, k)
            i = int(frm[ell]); j = int(to[ell])
            A[rrow, idx(t, off_V + j, 1)] = 1.0
            A[rrow, idx(t, off_V + i, 1)] = -1.0
            A[rrow, idx(t, off_P + ell, 1)] += 2.0 * r_line[ell]
            A[rrow, idx(t, off_Q + ell, 1)] += 2.0 * x_line[ell]
            bvec[rrow] = 0.0
            k += 1

    # Bounds
    l = -np.inf * np.ones(n)
    u = np.inf * np.ones(n)
    for t in range(T):
        l[idx(t, off_theta, N)] = -np.pi
        u[idx(t, off_theta, N)] = np.pi
        l[idx(t, off_V, N)] = 0.95
        u[idx(t, off_V, N)] = 1.05

        # line flow bounds
        rate = net.lines[:, 4]
        l[idx(t, off_P, L)] = -rate
        u[idx(t, off_P, L)] = rate
        l[idx(t, off_Q, L)] = -rate
        u[idx(t, off_Q, L)] = rate

        # p_mg/q_mg bounds (export/import capability)
        l[idx(t, off_pmg, M)] = -0.40
        u[idx(t, off_pmg, M)] = 0.40
        l[idx(t, off_qmg, M)] = -0.25
        u[idx(t, off_qmg, M)] = 0.25

        # import/export nonnegative
        l[idx(t, off_pimp, 1)] = 0.0
        l[idx(t, off_pexp, 1)] = 0.0
        # q_slack free but bounded moderately
        l[idx(t, off_qsl, 1)] = -1.0
        u[idx(t, off_qsl, 1)] = 1.0

    return dict(H=H, f=f, A=A, b=bvec, l=l, u=u,
                meta=dict(dim_t=dim_t, offsets=dict(theta=off_theta, V=off_V, P=off_P, Q=off_Q,
                                                    pmg=off_pmg, qmg=off_qmg, pimp=off_pimp, pexp=off_pexp, qsl=off_qsl),
                          N=N, L=L, M=M, T=T))

def build_microgrid_qp(case: Case, m: int,
                       z: np.ndarray | None = None,
                       u: np.ndarray | None = None,
                       rho: float = 0.0,
                       lam: np.ndarray | None = None,
                       enforce_target: bool = False,
                       target: np.ndarray | None = None) -> Dict:
    """
    Microgrid m local QP.
    Variables per time t:
      p_chp, h_boiler, dr, p_inj, q_inj
    Constraints per t:
      alpha*p_chp + h_boiler = H_load
      p_inj - p_chp - dr = -P_load_mg  (equivalently p_inj = p_chp - P_load + dr)
    If enforce_target: add equality p_inj = target_p and q_inj = target_q (for primal decomposition value function)
    Objective:
      local cost + ADMM penalty (rho/2)||x - z + u||^2 or dual term lambda^T x
    """
    mg = case.mgs[m]
    T = case.T

    # per t: p_chp, hB, dr, p_inj, q_inj
    dim_t = 5
    n = T * dim_t

    def idx(t, off, length=1):
        start = t*dim_t + off
        return slice(start, start+length)

    off_p = 0
    off_hb = 1
    off_dr = 2
    off_pin = 3
    off_qin = 4

    H = np.zeros((n, n))
    f = np.zeros(n)

    for t in range(T):
        # quadratic costs
        H[idx(t, off_p), idx(t, off_p)] += 2.0 * mg.c2_p
        f[idx(t, off_p)] += mg.c1_p
        H[idx(t, off_dr), idx(t, off_dr)] += 2.0 * mg.c_dr
        f[idx(t, off_hb)] += mg.c_boiler
        H[idx(t, off_qin), idx(t, off_qin)] += 2.0 * mg.c_q

        # ADMM penalty on injections (p_inj, q_inj)
        if rho > 0.0 and z is not None and u is not None:
            H[idx(t, off_pin), idx(t, off_pin)] += rho
            H[idx(t, off_qin), idx(t, off_qin)] += rho
            f[idx(t, off_pin)] += rho * (u[t,0] - z[t,0])
            f[idx(t, off_qin)] += rho * (u[t,1] - z[t,1])

        # Dual term: +lambda^T x (MG sees +lambda)
        if lam is not None:
            f[idx(t, off_pin)] += lam[t,0]
            f[idx(t, off_qin)] += lam[t,1]

    # Equality constraints
    # per t: heat balance + power injection relation (+ optional 2 target equalities)
    base_ct = 2
    extra = 2 if enforce_target else 0
    m_t = base_ct + extra
    mA = T * m_t
    A = np.zeros((mA, n))
    b = np.zeros(mA)

    for t in range(T):
        r0 = t*m_t
        # heat: alpha*p + hB = H_load
        A[r0, idx(t, off_p)] = mg.alpha_heat
        A[r0, idx(t, off_hb)] = 1.0
        b[r0] = case.H_load_mg[t, m]
        # p_inj - p_chp - dr = -P_load
        A[r0+1, idx(t, off_pin)] = 1.0
        A[r0+1, idx(t, off_p)] += -1.0
        A[r0+1, idx(t, off_dr)] += -1.0
        b[r0+1] = -case.P_load_mg[t, m]

        if enforce_target:
            assert target is not None
            # p_inj = target_p, q_inj = target_q
            A[r0+2, idx(t, off_pin)] = 1.0
            b[r0+2] = target[t,0]
            A[r0+3, idx(t, off_qin)] = 1.0
            b[r0+3] = target[t,1]

    # bounds
    l = -np.inf*np.ones(n)
    uvec = np.inf*np.ones(n)
    for t in range(T):
        l[idx(t, off_p)] = 0.0
        uvec[idx(t, off_p)] = mg.pmax_chp
        l[idx(t, off_hb)] = 0.0
        uvec[idx(t, off_hb)] = mg.hmax_boiler
        l[idx(t, off_dr)] = 0.0
        uvec[idx(t, off_dr)] = mg.dr_max
        # injection bounds
        l[idx(t, off_pin)] = -0.40
        uvec[idx(t, off_pin)] = 0.40
        l[idx(t, off_qin)] = -mg.q_max
        uvec[idx(t, off_qin)] = mg.q_max

    return dict(H=H, f=f, A=A, b=b, l=l, u=uvec,
                meta=dict(dim_t=dim_t, offsets=dict(p=off_p, hb=off_hb, dr=off_dr, pin=off_pin, qin=off_qin), T=T))
