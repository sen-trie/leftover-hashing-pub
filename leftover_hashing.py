import csv

import cvxpy as cp
import numpy as np
from scipy.optimize import brentq, minimize_scalar
from scipy.special import gammaln, logsumexp
from toqito.state_props import sandwiched_renyi_conditional_entropy

P_ERR = 0.01
EPS = 1e-6
MU = EPS / 2
SDP_SOLVER = cp.MOSEK

N_SDP = [1, 2, 3]
N_TABLE = [100, 200, 500, 1000]
CSV_PATH = "key_lengths.csv"


def main():
    eps_low = np.sqrt(EPS - MU)

    print("SDP versus closed form")
    print(f"{'n':>3} {'H_sdp':>12} {'H_closed_form':>14} {'difference':>11}  status")
    for n in N_SDP:
        H_sdp, status = smooth_min_entropy_sdp(P_ERR, n, eps_low)
        H_cf, _ = smooth_min_entropy(P_ERR, n, eps_low)
        diff = "" if H_sdp is None else f"{H_sdp - H_cf:.2e}"
        print(f"{n:>3} {fmt(H_sdp, 6):>12} {H_cf:>14.6f} {diff:>11}  {status}")

    rows = [table_row(P_ERR, n, EPS, MU) for n in N_TABLE]
    header = ["n", "H_low", "H_up", "ell_low", "ell_up", "ell_alpha", "alpha_star"]
    digits = [0, 3, 3, 2, 2, 2, 2]

    print()
    print(f"Key lengths for p={P_ERR:g}, eps={EPS:g}, mu=eps/2")
    print(" ".join(f"{h:>10}" for h in header))
    for row in rows:
        print(" ".join(f"{fmt(v, d):>10}" for v, d in zip(row, digits)))

    with open(CSV_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for row in rows:
            writer.writerow(
                [row[0]] + ["" if v is None else repr(float(v)) for v in row[1:]]
            )
    print(f"\nwritten to {CSV_PATH}")


def smooth_min_entropy_sdp(p, n, eps_smooth, solver=SDP_SOLVER, verbose=False):
    d = 2**n
    c = np.sqrt(1.0 - eps_smooth**2)
    V = conditional_vectors(p, n)

    S = cp.Variable((d, d), hermitian=True)
    taus = [cp.Variable((d, d), hermitian=True) for _ in range(d)]
    Ls = [cp.Variable((d, d), complex=True) for _ in range(d)]

    constraints = [S >> 0]
    for x in range(d):
        constraints += [
            S - taus[x] >> 0,
            cp.bmat([[np.outer(V[x], V[x]), Ls[x]], [Ls[x].H, taus[x]]]) >> 0,
        ]
    constraints += [
        sum(cp.real(cp.trace(t)) for t in taus) <= 1,
        sum(cp.real(cp.trace(L)) for L in Ls) >= c,
    ]

    problem = cp.Problem(cp.Minimize(cp.real(cp.trace(S))), constraints)
    problem.solve(solver=solver, verbose=verbose)
    if problem.status not in (cp.OPTIMAL, cp.OPTIMAL_INACCURATE) or not problem.value:
        return None, problem.status
    return float(-np.log2(problem.value)), problem.status


def smooth_min_entropy(p, n, eps_smooth):
    c = np.sqrt(1.0 - eps_smooth**2)
    logm, logpi, logr = binomial_terms(p, n)
    logqg = 0.5 * logpi + 0.5 * logm
    pi = np.exp(logpi - logsumexp(logpi))
    log_eps2 = 2.0 * np.log(eps_smooth)

    def gap(log_lam):
        support = logr < log_lam
        x = np.exp(logr[support] - log_lam)
        p_s = pi[support]
        tail = pi[~support].sum()
        m = np.dot(p_s, -np.expm1(logr[support] - log_lam))
        dev = tail + np.dot(p_s, x) - x
        var = np.dot(p_s, dev**2) + tail * m**2
        e2 = np.dot(p_s, np.expm1(logr[support] - log_lam) ** 2)
        with np.errstate(divide="ignore"):
            return np.log(var) - np.log(e2) - log_eps2

    lo = logr.min() + 1e-12
    hi = logr.max() + 40.0

    if gap(lo) <= 0.0:
        return float(n - 2.0 * (np.log(c) + logr.min()) / np.log(2)), "vertex"

    try:
        log_lam = brentq(gap, lo, hi, xtol=1e-14, rtol=1e-14, maxiter=500)
    except ValueError:
        return None, "failed"

    t = np.clip(-np.expm1(np.minimum(logr - log_lam, 0.0)), 0.0, None)
    mask = t > 0
    log_e2 = logsumexp(logpi, b=t**2)
    log_ert = logsumexp(logqg[mask], b=t[mask])
    H = float(n - 2.0 * (log_ert - 0.5 * log_e2) / np.log(2))
    status = "optimal" if abs(gap(log_lam)) < 1e-9 else "inaccurate"
    return H, status


def renyi_key_length(p, n, eps=EPS, alpha_bounds=(1.001, 2.0)):
    def ell(alpha):
        return (
            n * renyi_entropy(p, alpha)
            - alpha / (alpha - 1.0) * np.log2(1.0 / eps)
            + 2.0
        )

    res = minimize_scalar(lambda a: -ell(a), bounds=alpha_bounds, method="bounded")
    return float(-res.fun), float(res.x)


def optimal_mu(p, n, eps=EPS):
    def neg(t):
        ell = key_length_low(p, n, eps, eps / (1.0 + np.exp(-t)))
        return np.inf if ell is None else -ell

    res = minimize_scalar(
        neg, bounds=(-8.0, 16.0), method="bounded", options={"xatol": 1e-6}
    )
    return 1.0 / (1.0 + np.exp(-res.x)), -res.fun


def table_row(p, n, eps, mu):
    H_low, _ = smooth_min_entropy(p, n, np.sqrt(eps - mu))
    H_up, _ = smooth_min_entropy(p, n, np.sqrt(eps))
    ell_alpha, alpha = renyi_key_length(p, n, eps)
    return [
        n,
        H_low,
        H_up,
        key_length_low(p, n, eps, mu),
        key_length_up(p, n, eps),
        ell_alpha,
        alpha,
    ]


def key_length_low(p, n, eps=EPS, mu=MU):
    H, _ = smooth_min_entropy(p, n, np.sqrt(eps - mu))
    return None if H is None else H + np.log2(4 * mu**3)


def key_length_up(p, n, eps=EPS):
    H, _ = smooth_min_entropy(p, n, np.sqrt(eps))
    return None if H is None else H + np.log2(1.0 / (1.0 - eps))


def renyi_entropy(p, alpha):
    return sandwiched_renyi_conditional_entropy(
        phase_error_state(p), alpha, dim=[2, 2], variant="uparrow"
    )


def phase_error_state(p):
    phis = [np.array([np.sqrt(p), (-1) ** x * np.sqrt(1 - p)]) for x in range(2)]
    rho = np.zeros((4, 4), dtype=complex)
    for x in range(2):
        rho[2 * x : 2 * x + 2, 2 * x : 2 * x + 2] = 0.5 * np.outer(phis[x], phis[x])
    return rho


def conditional_vectors(p, n):
    d = 2**n
    idx = np.arange(d)
    parity = np.zeros((d, d), dtype=int)
    for k in range(n):
        parity ^= ((idx[:, None] >> k) & 1) & ((idx[None, :] >> k) & 1)
    return 2.0 ** (-n / 2) * phi_zero(p, n)[None, :] * (1 - 2 * parity)


def phi_zero(p, n):
    phi0 = np.ones(1)
    for _ in range(n):
        phi0 = np.kron(phi0, np.array([np.sqrt(p), np.sqrt(1 - p)]))
    return phi0


def binomial_terms(p, n):
    h = np.arange(n + 1)
    logm = gammaln(n + 1) - gammaln(h + 1) - gammaln(n - h + 1)
    logpi = logm + (n - h) * np.log(p) + h * np.log1p(-p)
    logr = -0.5 * ((n - h) * np.log(p) + h * np.log1p(-p))
    return logm, logpi, logr


def fmt(v, digits):
    if v is None:
        return "-"
    if digits == 0:
        return str(int(v))
    return f"{float(v):.{digits}f}"


if __name__ == "__main__":
    main()
