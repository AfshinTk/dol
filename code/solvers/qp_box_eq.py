\
"""
Quadratic program solver for:
    minimize 0.5 x^T H x + f^T x
    subject to A x = b
               l <= x <= u

Assumptions:
- H is symmetric positive definite (or at least SPD on the nullspace of A)
- Constraints are only equality + box, which matches our course project formulation

We use a primal-dual active-set for bound constraints (box QP) with equality constraints.
This is deterministic, fast for medium-size QPs, and avoids slow general NLP solvers.
"""
from __future__ import annotations
import numpy as np

class QPResult(dict):
    pass

def _solve_kkt(H, f, A, b):
    """Solve equality-constrained QP via (regularized) KKT."""
    n = H.shape[0]
    m = A.shape[0]
    eps = 1e-9
    K = np.block([[H, A.T],
                  [A, -eps*np.eye(m)]])
    rhs = np.concatenate([-f, b])
    try:
        sol = np.linalg.solve(K, rhs)
    except np.linalg.LinAlgError:
        sol, *_ = np.linalg.lstsq(K, rhs, rcond=None)
    x = sol[:n]
    nu = sol[n:]
    return x, nu

def solve_qp_eq_box(H, f, A, b, l, u, max_iter=60, tol=1e-8, verbose=False):
    """
    Active-set method for box constraints.
    Returns dict with x, nu, status, iters, kkt_residual.
    """
    H = np.asarray(H, float)
    f = np.asarray(f, float).reshape(-1)
    A = np.asarray(A, float)
    b = np.asarray(b, float).reshape(-1)
    l = np.asarray(l, float).reshape(-1)
    u = np.asarray(u, float).reshape(-1)
    n = H.shape[0]
    assert f.shape[0] == n and l.shape[0] == n and u.shape[0] == n

    # Start with no active bounds
    active = np.zeros(n, dtype=int)  # 0 free, -1 at lower, +1 at upper
    x = np.zeros(n)
    nu = np.zeros(A.shape[0])

    # Heuristic: solve equality QP and then build an initial active set from violations
    x0, nu0 = _solve_kkt(H, f, A, b)
    x = x0.copy()
    viol_low = x < l
    viol_up = x > u
    active[viol_low] = -1
    active[viol_up] = +1
    x[viol_low] = l[viol_low]
    x[viol_up] = u[viol_up]

    def stationarity(x, nu):
        return H @ x + f + A.T @ nu

    for it in range(max_iter):
        W = np.where(active != 0)[0]
        F = np.where(active == 0)[0]

        xW = x[W].copy()
        # Force exact bound values
        for idx in W:
            if active[idx] == -1:
                x[idx] = l[idx]
            else:
                x[idx] = u[idx]
        xW = x[W].copy()

        # Reduced QP in free variables
        if F.size > 0:
            H_FF = H[np.ix_(F, F)]
            f_F = f[F]
            if W.size > 0:
                H_FW = H[np.ix_(F, W)]
                f_red = f_F + H_FW @ xW
                A_F = A[:, F]
                A_W = A[:, W]
                b_red = b - A_W @ xW
            else:
                f_red = f_F
                A_F = A[:, F]
                b_red = b

            # Solve reduced KKT
            xF, nu = _solve_kkt(H_FF, f_red, A_F, b_red)
            x[F] = xF
        else:
            # No free variables: just solve for nu from Ax=b consistency (already forced via xW)
            # We compute least-squares for nu in stationarity, but keep it simple
            # nu from KKT: A x = b is satisfied by construction if feasible
            # We'll keep previous nu
            pass

        # Check bound violations for free vars
        add_idx = None
        add_type = 0
        if F.size > 0:
            vlow = x[F] - l[F]
            vup = u[F] - x[F]
            min_vlow = vlow.min()
            min_vup = vup.min()
            if min_vlow < -tol or min_vup < -tol:
                # Add most violated
                if min_vlow < min_vup:
                    j = F[np.argmin(vlow)]
                    add_idx = j
                    add_type = -1
                    x[j] = l[j]
                else:
                    j = F[np.argmin(vup)]
                    add_idx = j
                    add_type = +1
                    x[j] = u[j]
                active[add_idx] = add_type
                if verbose:
                    print("Add bound", add_idx, add_type)
                continue

        # Check KKT sign conditions on active bounds, possibly remove one
        s = stationarity(x, nu)
        remove_idx = None
        for j in W:
            if active[j] == -1 and s[j] > tol:  # should have s<=0 at lower
                remove_idx = j
                break
            if active[j] == +1 and s[j] < -tol:  # should have s>=0 at upper
                remove_idx = j
                break
        if remove_idx is not None:
            active[remove_idx] = 0
            if verbose:
                print("Remove bound", remove_idx)
            continue

        # Converged if equality satisfied and stationarity holds approximately
        eq_res = np.linalg.norm(A @ x - b, ord=np.inf)
        # stationarity on free vars near 0, and sign on active enforced
        s_free = np.linalg.norm(s[F], ord=np.inf) if F.size > 0 else 0.0
        kkt = max(eq_res, s_free)
        if kkt < 10*tol:
            return QPResult(x=x, nu=nu, status="optimal", iters=it+1, kkt_residual=float(kkt))

    # If max iters reached
    eq_res = np.linalg.norm(A @ x - b, ord=np.inf)
    s = stationarity(x, nu)
    F = np.where(active == 0)[0]
    s_free = np.linalg.norm(s[F], ord=np.inf) if F.size > 0 else 0.0
    kkt = max(eq_res, s_free)
    return QPResult(x=x, nu=nu, status="max_iter", iters=max_iter, kkt_residual=float(kkt))
