# %%
"""One-sided Jacobi SVD reference implementations and B-ASIC SFG generator test."""

import numpy as np

from b_asic.core_operations import (
    Absolute,
    AddSub,
    Multiplication,
    Reciprocal,
    SquareRoot,
)
from b_asic.schedule import Schedule
from b_asic.scheduler import ASAPScheduler
from b_asic.sfg_generators import svd_one_sided_jacobi
from b_asic.svd_operations import Sign

# %%


def svd_one_sided_jacobi_hw(A, n_sweeps=30):
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_one_sided_jacobi_hw(A.T, n_sweeps=n_sweeps)
        return Vt_t.T, s, U_t.T

    G = A.astype(float, copy=True)
    V = np.eye(n)

    for _ in range(n_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                alpha = G[:, p] @ G[:, p]
                beta = G[:, q] @ G[:, q]
                gamma = G[:, p] @ G[:, q]

                if gamma == 0.0:
                    c, s = 1.0, 0.0
                else:
                    zeta = (beta - alpha) / (2.0 * gamma)
                    sgn = 1.0 if zeta >= 0.0 else -1.0
                    t = 1 / (abs(zeta) + np.sqrt(1.0 + zeta**2))
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = c * t * sgn

                col_p = G[:, p].copy()
                col_q = G[:, q].copy()
                G[:, p] = c * col_p - s * col_q
                G[:, q] = s * col_p + c * col_q

                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q

    s = np.array([np.sqrt(G[:, i] @ G[:, i]) for i in range(n)])

    U = G / s

    return U, s, V.T


def svd_one_sided_jacobi_hw_loops(A, n_sweeps=30):
    """
    Fully scalar loop rewrite of svd_one_sided_jacobi_hw.

    This is a faithful, line-by-line translation of the same algorithm:
    every dot product, column copy, and column update is expressed as an
    explicit index loop instead of NumPy vector/matrix notation. It
    reproduces the original's behavior exactly, bugs included (e.g. no
    near-zero fallback for U = G / s), so it is just as "badly broken" as
    svd_one_sided_jacobi_hw.
    """
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_one_sided_jacobi_hw_loops(A.T, n_sweeps=n_sweeps)
        return Vt_t.T, s, U_t.T

    G = A.astype(float, copy=True)
    V = np.eye(n)

    for _ in range(n_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                for i in range(m):
                    if i == 0:
                        alpha = G[i, p] * G[i, p]
                        beta = G[i, q] * G[i, q]
                        gamma = G[i, p] * G[i, q]
                    else:
                        alpha += G[i, p] * G[i, p]
                        beta += G[i, q] * G[i, q]
                        gamma += G[i, p] * G[i, q]

                if gamma == 0.0:
                    c, s = 1.0, 0.0
                else:
                    zeta = (beta - alpha) / (2.0 * gamma)
                    sgn = 1.0 if zeta >= 0.0 else -1.0
                    t = 1 / (abs(zeta) + np.sqrt(1.0 + zeta**2))
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = c * t * sgn

                for i in range(m):
                    g_ip = G[i, p]
                    g_iq = G[i, q]
                    G[i, p] = c * g_ip - s * g_iq
                    G[i, q] = s * g_ip + c * g_iq

                for i in range(n):
                    v_ip = V[i, p]
                    v_iq = V[i, q]
                    V[i, p] = c * v_ip - s * v_iq
                    V[i, q] = s * v_ip + c * v_iq

    s = np.zeros(n)
    for i in range(n):
        total = 0.0
        for k in range(m):
            total += G[k, i] * G[k, i]
        s[i] = np.sqrt(total)

    U = np.zeros((m, n))
    for i in range(m):
        for j in range(n):
            U[i, j] = G[i, j] / s[j]

    return U, s, V.T


# %%
if __name__ == "__main__":
    rng = np.random.default_rng(0)

    test_cases = [
        ("Random 4x4 (well-conditioned)", rng.standard_normal((4, 4))),
        (
            "Rank-deficient 4x4 (rank 1)",
            np.outer([1.0, 2.0, 3.0, 4.0], [4.0, 3.0, 2.0, 1.0]),
        ),
        (
            "Ill-conditioned 5x5",
            np.dot(
                np.dot(rng.standard_normal((5, 5)), np.diag([1e5, 1e2, 1, 1e-2, 1e-5])),
                rng.standard_normal((5, 5)),
            ),
        ),
        ("Rectangular 5x3 (tall)", rng.standard_normal((5, 3))),
    ]

    methods = [
        ("HW (vectorized)", svd_one_sided_jacobi_hw),
        ("HW (loops)", svd_one_sided_jacobi_hw_loops),
    ]

    for case_name, A in test_cases:
        print("=" * 80)
        print(case_name)
        print("=" * 80)

        U_ref, s_ref, Vt_ref = np.linalg.svd(A, full_matrices=True)
        print("Singular values (built-in):", s_ref)

        for name, method_fn in methods:
            U_hw, s_hw, Vt_hw = method_fn(A)

            print(f"\n--- {name} ---")
            print("Singular values:", s_hw)

            S_matrix = np.zeros((U_hw.shape[1], Vt_hw.shape[0]))
            np.fill_diagonal(S_matrix, s_hw)
            A_recon = U_hw @ S_matrix @ Vt_hw
            recon_err = np.linalg.norm(A - A_recon)
            print("Reconstruction error ||A - U*S*Vt||:", recon_err)
            print("Reconstruction close to A:", np.allclose(A_recon, A, atol=1e-8))

            U_ortho_err = np.linalg.norm(U_hw.T @ U_hw - np.eye(U_hw.shape[1]))
            V_ortho_err = np.linalg.norm(Vt_hw @ Vt_hw.T - np.eye(Vt_hw.shape[0]))
            print("U orthogonality error ||U.T @ U - I||:", U_ortho_err)
            print("V orthogonality error ||V @ V.T - I||:", V_ortho_err)
        print()

# %%
# Build a small test SFG for the one-sided Jacobi SVD generator and schedule it.
test_sfg = svd_one_sided_jacobi(m=3, n=2, n_sweeps=1)

test_sfg.set_latency_of_type(AddSub, 1)
test_sfg.set_latency_of_type(Multiplication, 2)
test_sfg.set_latency_of_type(Reciprocal, 4)
test_sfg.set_latency_of_type(SquareRoot, 4)
test_sfg.set_latency_of_type(Absolute, 1)
test_sfg.set_latency_of_type(Sign, 1)

test_sfg.set_execution_time_of_type(AddSub, 1)
test_sfg.set_execution_time_of_type(Multiplication, 1)
test_sfg.set_execution_time_of_type(Reciprocal, 1)
test_sfg.set_execution_time_of_type(SquareRoot, 1)
test_sfg.set_execution_time_of_type(Absolute, 1)
test_sfg.set_execution_time_of_type(Sign, 1)

schedule = Schedule(test_sfg, scheduler=ASAPScheduler())
print("Scheduling time:", schedule.schedule_time)

# %%
test_sfg
schedule

# %%
