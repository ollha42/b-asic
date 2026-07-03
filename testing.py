# %%
"""Pure NumPy implementations of SVD algorithms and benchmarks against the built-in reference."""

import time

import numpy as np


def householder_vector(x):
    """
    Return (v, beta) such that H = I - beta * v * v.T is a Householder reflection
    mapping vector x to [sgn * ||x||, 0, ..., 0]^T.
    Normalizes v so that v[0] = 1.0.
    """
    n = len(x)
    if n == 0:
        return np.array([]), 0.0

    v = np.array(x, dtype=float)
    sigma = np.dot(v[1:], v[1:])
    v[0] = 1.0

    if sigma == 0:
        beta = 0.0 if x[0] >= 0 else 2.0
    else:
        mu = np.sqrt(x[0] ** 2 + sigma)
        if x[0] <= 0:
            v[0] = x[0] - mu
        else:
            v[0] = -sigma / (x[0] + mu)
        beta = 2.0 * v[0] ** 2 / (sigma + v[0] ** 2)
        v = v / v[0]

    return v, beta


def householder_bidiagonalization(A):
    """
    Reduces an m x n matrix A (with m >= n) to upper bidiagonal form B
    using Householder reflections, such that A = U_0 * B * V_0^T.

    Returns:
        U_0: orthogonal matrix of size m x m
        B: upper bidiagonal matrix of size m x n
        V_0: orthogonal matrix of size n x n
    """
    m, n = A.shape
    B = A.astype(float, copy=True)
    U = np.eye(m)
    V = np.eye(n)

    for i in range(n):
        # 1. Eliminate below diagonal in column i
        x_col = B[i:, i]
        v_col, beta_col = householder_vector(x_col)

        # Apply reflection from left to B[i:, i:]
        B[i:, i:] -= beta_col * np.outer(v_col, v_col @ B[i:, i:])
        # Update U
        U[:, i:] -= beta_col * np.outer(U[:, i:] @ v_col, v_col)

        # 2. Eliminate to the right of superdiagonal in row i
        if i < n - 2:
            x_row = B[i, i + 1 :]
            v_row, beta_row = householder_vector(x_row)

            # Apply reflection from right to B[i:, i+1:]
            B[i:, i + 1 :] -= beta_row * np.outer(B[i:, i + 1 :] @ v_row, v_row)
            # Update V
            V[:, i + 1 :] -= beta_row * np.outer(V[:, i + 1 :] @ v_row, v_row)

    return U, B, V


def householder_qr(A):
    """
    Reduces an m x n matrix A (with m >= n) to upper triangular form R
    using Householder reflections, such that A = Q * R.

    Returns:
        Q: orthogonal matrix of size m x m
        R: upper triangular matrix of size m x n
    """
    m, n = A.shape
    R = A.astype(float, copy=True)
    Q = np.eye(m)

    for i in range(n):
        # Eliminate below diagonal in column i
        x_col = R[i:, i]
        v_col, beta_col = householder_vector(x_col)

        # Apply reflection from left to R[i:, i:]
        R[i:, i:] -= beta_col * np.outer(v_col, v_col @ R[i:, i:])
        # Update Q
        Q[:, i:] -= beta_col * np.outer(Q[:, i:] @ v_col, v_col)

    return Q, R


def givens_rotation(a, b):
    """
    Given scalars a and b, compute c, s such that:
    [ c   s ] [ a ] = [ r ]
    [-s   c ] [ b ] = [ 0 ].
    """
    if b == 0:
        c = 1.0
        s = 0.0
    else:
        if abs(b) > abs(a):
            t = a / b
            s = 1.0 / np.sqrt(1.0 + t**2)
            c = s * t
        else:
            t = b / a
            c = 1.0 / np.sqrt(1.0 + t**2)
            s = c * t
    return c, s


def svd_golub_reinsch(A, tol=1e-15, max_iter=1000):
    """
    Compute SVD of A using pure NumPy Golub-Reinsch algorithm:
    A = U * diag(S) * Vt.
    """
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_golub_reinsch(A.T, tol=tol, max_iter=max_iter)
        return Vt_t.T, s, U_t.T

    # Step 1: Householder Bidiagonalization
    U, B, V = householder_bidiagonalization(A)

    # Extract diagonal and superdiagonal of B
    d = np.zeros(n)
    e = np.zeros(n - 1)
    for i in range(n):
        d[i] = B[i, i]
        if i < n - 1:
            e[i] = B[i, i + 1]

    # Step 2: Implicit QR Iteration on the bidiagonal system
    iteration = 0
    while iteration < max_iter:
        # Check for small elements in superdiagonal to split/deflate the matrix
        for i in range(n - 1):
            threshold = tol * (abs(d[i]) + abs(d[i + 1]))
            if abs(e[i]) <= threshold:
                e[i] = 0.0

        # Find active block B[p:n-q, p:n-q]
        # q is the largest index such that e[n-q-1:] are all 0
        q = 0
        while q < n - 1 and abs(e[n - 2 - q]) == 0.0:
            q += 1

        if q == n - 1:
            # All superdiagonal values converged to 0!
            break

        # p is the smallest index such that e[p:n-q-1] has no zeros
        p = n - q - 2
        while p >= 0 and abs(e[p]) != 0.0:
            p -= 1
        p += 1

        # Now active unreduced submatrix is between p and n - q - 1
        # Check if any diagonal element d[i] in the active block is close to 0
        zero_found = False
        for i in range(p, n - q):
            # If a diagonal element is close to 0, zero out the corresponding superdiagonal
            if abs(d[i]) <= tol * np.linalg.norm(d[p : n - q]):
                d[i] = 0.0
                if i < n - q - 1:
                    # Cancel e[i] using left rotations
                    g = e[i]
                    e[i] = 0.0
                    for k in range(i + 1, n - q):
                        c, s = givens_rotation(d[k], g)
                        d[k] = c * d[k] + s * g
                        if k < n - q - 1:
                            g = -s * e[k]
                            e[k] = c * e[k]
                        # Update U
                        col_k = U[:, k].copy()
                        col_i = U[:, i].copy()
                        U[:, k] = c * col_k + s * col_i
                        U[:, i] = -s * col_k + c * col_i
                else:
                    # Cancel e[i-1] using right rotations
                    g = e[i - 1]
                    e[i - 1] = 0.0
                    for j in range(i - 1, p - 1, -1):
                        c, s = givens_rotation(d[j], g)
                        d[j] = c * d[j] + s * g
                        if j > p:
                            g = -s * e[j - 1]
                            e[j - 1] = c * e[j - 1]
                        # Update V
                        col_j = V[:, j].copy()
                        col_last = V[:, i].copy()
                        V[:, j] = c * col_j + s * col_last
                        V[:, i] = -s * col_j + c * col_last
                zero_found = True
                break

        if zero_found:
            continue

        # No diagonal element was zero, perform an implicitly shifted QR step
        N = n - q
        # Trailing 2x2 of active B^T * B
        d_nm2 = d[N - 2]
        d_nm1 = d[N - 1]
        e_nm2 = e[N - 2]
        e_nm3 = e[N - 3] if p <= N - 3 else 0.0

        a = d_nm2**2 + e_nm3**2
        b = d_nm2 * e_nm2
        c = d_nm1**2 + e_nm2**2

        # Wilkinson shift
        delta = (a - c) / 2.0
        sgn = 1.0 if delta >= 0.0 else -1.0
        mu = c - (sgn * b**2) / (abs(delta) + np.sqrt(delta**2 + b**2))

        # First Givens rotation parameters from first column of B^T * B - mu * I
        x = d[p] ** 2 - mu
        z = d[p] * e[p]

        c, s = givens_rotation(x, z)

        # Apply first column rotation on columns p and p+1
        d_p = d[p]
        e_p = e[p]
        d[p] = c * d_p + s * e_p
        e[p] = -s * d_p + c * e_p

        d_pp1 = d[p + 1]
        plus_bulge = s * d_pp1
        d[p + 1] = c * d_pp1

        # Update V
        col_p = V[:, p].copy()
        col_pp1 = V[:, p + 1].copy()
        V[:, p] = c * col_p + s * col_pp1
        V[:, p + 1] = -s * col_p + c * col_pp1

        # Bulge chasing
        for i in range(p, N - 1):
            # 1. Zero out plus_bulge using row rotation on rows i and i+1
            c_u, s_u = givens_rotation(d[i], plus_bulge)
            d[i] = c_u * d[i] + s_u * plus_bulge

            e_i = e[i]
            d_ip1 = d[i + 1]
            e[i] = c_u * e_i + s_u * d_ip1
            d[i + 1] = -s_u * e_i + c_u * d_ip1

            if i < N - 2:
                e_ip1 = e[i + 1]
                right_bulge = s_u * e_ip1
                e[i + 1] = c_u * e_ip1

            # Update U
            col_i = U[:, i].copy()
            col_ip1 = U[:, i + 1].copy()
            U[:, i] = c_u * col_i + s_u * col_ip1
            U[:, i + 1] = -s_u * col_i + c_u * col_ip1

            # 2. Zero out right_bulge using column rotation on columns i+1 and i+2
            if i < N - 2:
                c_v, s_v = givens_rotation(e[i], right_bulge)
                e[i] = c_v * e[i] + s_v * right_bulge

                d_ip1 = d[i + 1]
                e_ip1 = e[i + 1]
                d[i + 1] = c_v * d_ip1 + s_v * e_ip1
                e[i + 1] = -s_v * d_ip1 + c_v * e_ip1

                d_ip2 = d[i + 2]
                plus_bulge = s_v * d_ip2
                d[i + 2] = c_v * d_ip2

                # Update V
                col_ip1 = V[:, i + 1].copy()
                col_ip2 = V[:, i + 2].copy()
                V[:, i + 1] = c_v * col_ip1 + s_v * col_ip2
                V[:, i + 2] = -s_v * col_ip1 + c_v * col_ip2

        iteration += 1

    # Post-processing: ensure singular values are non-negative
    for i in range(n):
        if d[i] < 0:
            d[i] = -d[i]
            U[:, i] = -U[:, i]

    # Post-processing: sort in descending order
    idx = np.argsort(d)[::-1]
    d = d[idx]
    U = U[:, idx]
    V = V[:, idx]

    return U, d, V.T


def svd_builtin(A):
    """
    Compute SVD using built-in NumPy linalg function.
    """
    U, s, Vt = np.linalg.svd(A, full_matrices=True)
    return U, s, Vt


def _jacobi_symmetric_2x2(s11, s12, s22):
    """
    Closed-form Jacobi rotation (c, s) diagonalizing the symmetric 2x2
    matrix [[s11, s12], [s12, s22]] via V^T S V = diag. Returns (c, s) with
    c = 1.0, s = 0.0 if s12 is already zero.
    """
    if s12 == 0.0:
        return 1.0, 0.0
    zeta = (s22 - s11) / (2.0 * s12)
    sgn = 1.0 if zeta >= 0.0 else -1.0
    t = sgn / (abs(zeta) + np.sqrt(1.0 + zeta**2))
    c = 1.0 / np.sqrt(1.0 + t**2)
    s = c * t
    return c, s


def _two_sided_jacobi_square(A, tol=1e-15, max_iter=1000, verbose=False):
    """
    Core two-sided (Kogbetliantz) Jacobi SVD sweep for a square n x n
    matrix A: A = U * diag(s) * V.T.

    At each pivot (p, q), the 2x2 block [[a, b], [c, d]] = A[[p,q]][:,[p,q]]
    is diagonalized analytically by first finding a left rotation angle
    phi that symmetrizes the block (making it equal to its transpose), then
    diagonalizing the resulting symmetric 2x2 block with the standard
    closed-form Jacobi angle psi. The overall left rotation angle is then
    phi + psi (composition of two rotations) and the right rotation angle
    is psi. Both rotations are applied simultaneously (left to rows p, q;
    right to columns p, q), so, unlike one-sided Jacobi, A never needs to
    be reduced via A.T @ A and both A[p, q] and A[q, p] are annihilated in
    a single step.
    """
    n = A.shape[0]
    G = A.astype(float, copy=True)
    U = np.eye(n)
    V = np.eye(n)

    for iteration in range(max_iter):
        if verbose:
            print(f"\n--- Sweep {iteration} ---")
        converged = True
        for p in range(n):
            for q in range(p + 1, n):
                a = G[p, p]
                b = G[p, q]
                c = G[q, p]
                d = G[q, q]

                threshold = tol * (abs(a) + abs(d))
                if abs(b) <= threshold and abs(c) <= threshold:
                    if verbose:
                        print(f"  (p={p}, q={q}): skip, already diagonal")
                    continue
                converged = False

                # Left rotation angle phi that symmetrizes [[a, b], [c, d]].
                phi = np.arctan2(c - b, a + d)
                cf = np.cos(phi)
                sf = np.sin(phi)

                # Symmetric part S = R(phi).T @ [[a, b], [c, d]].
                s11 = cf * a + sf * c
                s12 = cf * b + sf * d
                s22 = -sf * b + cf * d

                # Diagonalize the symmetric block: angle psi.
                cv, sv = _jacobi_symmetric_2x2(s11, s12, s22)
                psi = np.arctan2(sv, cv)

                # Overall left rotation angle is phi + psi.
                theta = phi + psi
                cu = np.cos(theta)
                su = np.sin(theta)

                if verbose:
                    print(
                        f"  (p={p}, q={q}): rotate, a={a:.6g}, b={b:.6g}, "
                        f"c={c:.6g}, d={d:.6g}, phi={phi:.6g}, psi={psi:.6g}"
                    )

                # Apply left rotation to rows p, q of G (all columns).
                row_p = G[p, :].copy()
                row_q = G[q, :].copy()
                G[p, :] = cu * row_p + su * row_q
                G[q, :] = -su * row_p + cu * row_q

                # Accumulate the left singular vectors. G's rows are rotated
                # by L = [[cu, su], [-su, cu]], so U must accumulate L.T
                # (U <- U * L.T) to keep the invariant G = U.T @ A @ V.
                col_p = U[:, p].copy()
                col_q = U[:, q].copy()
                U[:, p] = cu * col_p + su * col_q
                U[:, q] = -su * col_p + cu * col_q

                # Apply right rotation to columns p, q of G (all rows).
                col_p = G[:, p].copy()
                col_q = G[:, q].copy()
                G[:, p] = cv * col_p - sv * col_q
                G[:, q] = sv * col_p + cv * col_q

                # Accumulate the right singular vectors (V <- V * R(psi)).
                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = cv * v_p - sv * v_q
                V[:, q] = sv * v_p + cv * v_q

        if converged:
            if verbose:
                print(f"\nConverged after {iteration + 1} sweep(s); breaking.")
            break
        if verbose:
            print("  Not yet converged; continuing to next sweep.")
    else:
        if verbose:
            print(f"\nReached max_iter={max_iter} without converging.")

    s = np.diag(G).copy()

    # Post-processing: ensure singular values are non-negative.
    for i in range(n):
        if s[i] < 0:
            s[i] = -s[i]
            U[:, i] = -U[:, i]

    # Post-processing: sort in descending order.
    idx = np.argsort(s)[::-1]
    s = s[idx]
    U = U[:, idx]
    V = V[:, idx]

    if verbose:
        print(f"\nSingular values (descending): {s}")

    return U, s, V.T


def svd_two_sided_jacobi(A, tol=1e-15, max_iter=1000, verbose=False):
    """
    Compute the SVD of A using the two-sided (Kogbetliantz) Jacobi
    algorithm: A = U * diag(S) * Vt.

    Unlike svd_one_sided_jacobi, which only rotates columns of A from the
    right, this method applies a pair of rotations at each pivot (p, q):
    one from the left (rows) and one from the right (columns), chosen
    together to annihilate both A[p, q] and A[q, p] in a single step. This
    drives A directly toward a diagonal matrix whose entries are the
    singular values. Because U and V are pure rotation accumulations
    starting from the identity, they remain orthogonal by construction even
    for rank-deficient input, unlike svd_one_sided_jacobi which needs an
    explicit Gram-Schmidt completion step for zero singular values.

    The two-sided rotations only make sense for a square matrix (there is
    no "mirror" column/row to pair with entries below row n), so a
    rectangular A with m > n is first reduced to a square upper triangular
    matrix R via Householder QR (A = Q * R), the core two-sided Jacobi
    sweep is applied to R, and the result is combined with Q.

    If verbose is True, prints the sweep number, the (p, q) pivot pairs
    visited within each sweep, whether each pivot is skipped (already
    diagonal) or rotated (with the symmetrizing angle phi and
    diagonalizing angle psi), and whether the algorithm continues to
    another sweep or breaks (converged).
    """
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_two_sided_jacobi(
            A.T, tol=tol, max_iter=max_iter, verbose=verbose
        )
        return Vt_t.T, s, U_t.T

    if m == n:
        return _two_sided_jacobi_square(A, tol=tol, max_iter=max_iter, verbose=verbose)

    # m > n: reduce to a square upper triangular matrix via Householder QR.
    Q_full, R_full = householder_qr(A)
    Q = Q_full[:, :n]
    R = R_full[:n, :n]

    U_r, s, Vt = _two_sided_jacobi_square(
        R, tol=tol, max_iter=max_iter, verbose=verbose
    )
    U = Q @ U_r

    return U, s, Vt


def svd_one_sided_jacobi(A, tol=1e-15, max_iter=1000, verbose=False):
    """
    Compute the SVD of A using the one-sided (Hestenes) Jacobi algorithm:
    A = U * diag(S) * Vt.

    Unlike svd_eigen, this method never forms A.T @ A. It applies plane
    rotations directly to the columns of A until they become mutually
    orthogonal; the resulting column norms are the singular values, the
    normalized columns are the left singular vectors U, and the accumulated
    rotations form the right singular vectors V. Avoiding A.T @ A preserves
    the full numerical accuracy of the data, which makes this the preferred
    Jacobi variant for ill-conditioned matrices and for systolic hardware
    SVD arrays.

    If verbose is True, prints the sweep number, the (p, q) column pair
    order visited within each sweep, whether each pair is skipped (zero
    column / already orthogonal) or rotated, and whether the algorithm
    continues to another sweep or breaks (converged).
    """
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_one_sided_jacobi(
            A.T, tol=tol, max_iter=max_iter, verbose=verbose
        )
        return Vt_t.T, s, U_t.T

    # Work on a copy whose columns will be orthogonalized in place.
    G = A.astype(float, copy=True)
    V = np.eye(n)

    # A column whose squared norm falls below this is treated as numerically
    # zero (it corresponds to a zero singular value and needs no rotation).
    col_zero = (tol * np.sqrt(np.sum(G * G))) ** 2

    for iteration in range(max_iter):
        if verbose:
            print(f"\n--- Sweep {iteration} ---")
        converged = True
        for p in range(n):
            for q in range(p + 1, n):
                alpha = G[:, p] @ G[:, p]
                beta = G[:, q] @ G[:, q]

                # Skip pairs involving a (numerically) zero column.
                if alpha <= col_zero or beta <= col_zero:
                    if verbose:
                        print(f"  (p={p}, q={q}): skip, zero column")
                    continue

                gamma = G[:, p] @ G[:, q]

                # Skip if columns p and q are already orthogonal.
                if abs(gamma) <= tol * np.sqrt(alpha * beta):
                    if verbose:
                        print(f"  (p={p}, q={q}): skip, already orthogonal")
                    continue
                converged = False

                # Jacobi rotation that diagonalizes the 2x2 Gram matrix
                # [[alpha, gamma], [gamma, beta]] of columns p and q.
                zeta = (beta - alpha) / (2.0 * gamma)
                sgn = 1.0 if zeta >= 0.0 else -1.0
                t = sgn / (abs(zeta) + np.sqrt(1.0 + zeta**2))
                c = 1.0 / np.sqrt(1.0 + t**2)
                s = c * t

                if verbose:
                    print(
                        f"  (p={p}, q={q}): rotate, alpha={alpha:.6g}, "
                        f"beta={beta:.6g}, gamma={gamma:.6g}, "
                        f"c={c:.6g}, s={s:.6g}"
                    )

                # Rotate columns p and q of G (G <- G J).
                col_p = G[:, p].copy()
                col_q = G[:, q].copy()
                G[:, p] = c * col_p - s * col_q
                G[:, q] = s * col_p + c * col_q

                # Accumulate the right singular vectors (V <- V J).
                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q

        if converged:
            if verbose:
                print(f"\nConverged after {iteration + 1} sweep(s); breaking.")
            break
        if verbose:
            print("  Not yet converged; continuing to next sweep.")
    else:
        if verbose:
            print(f"\nReached max_iter={max_iter} without converging.")

    # Singular values are the norms of the orthogonalized columns.
    s = np.array([np.sqrt(G[:, i] @ G[:, i]) for i in range(n)])

    # Sort in descending order.
    idx = np.argsort(s)[::-1]
    s = s[idx]
    G = G[:, idx]
    V = V[:, idx]

    if verbose:
        print(f"\nSingular values (descending): {s}")

    # Left singular vectors: normalized columns. Columns with a (numerically)
    # zero singular value are completed with an orthonormal basis via
    # Gram-Schmidt so that the thin U stays orthogonal.
    s_max = s[0] if n > 0 else 0.0
    rank_tol = s_max * 1e-12
    U = np.zeros((m, n))
    for i in range(n):
        if s[i] > rank_tol:
            U[:, i] = G[:, i] / s[i]
            if verbose:
                print(f"  U[:, {i}]: normalized column, s[{i}]={s[i]:.6g}")
        else:
            w = np.zeros(m)
            for k in range(m):
                cand = np.zeros(m)
                cand[k] = 1.0
                for j in range(i):
                    cand -= (U[:, j] @ cand) * U[:, j]
                norm = np.sqrt(cand @ cand)
                if norm > 1e-12:
                    w = cand / norm
                    break
            U[:, i] = w
            if verbose:
                print(
                    f"  U[:, {i}]: s[{i}]={s[i]:.6g} <= rank_tol={rank_tol:.6g}, "
                    "completed via Gram-Schmidt"
                )

    return U, s, V.T


def svd_one_sided_jacobi_hw(A, n_sweeps=30):
    """
    Hardware-friendly variant of the one-sided Jacobi SVD.

    Compared to svd_one_sided_jacobi, this version:
      - Always runs a fixed number of sweeps (n_sweeps), with no
        convergence check and no early exit.
      - Never skips a (p, q) column pair: every pair is rotated on every
        sweep, regardless of whether a column is (numerically) zero or the
        pair is already orthogonal.
      - Has no tolerance parameter and no tolerance-based decisions.
      - Does not sort the results; sort_svd_descending() should be applied
        by the caller as a separate post-processing step.

    This is expected to be slower per call (no skipping) and numerically
    worse on rank-deficient/ill-conditioned inputs (no tolerance-based rank
    handling), which is the point: it is a closer model of a fixed-schedule
    hardware systolic Jacobi array.
    """
    m, n = A.shape
    if m < n:
        U_t, s, Vt_t = svd_one_sided_jacobi_hw(A.T, n_sweeps=n_sweeps)
        return Vt_t.T, s, U_t.T

    # Work on a copy whose columns will be orthogonalized in place.
    G = A.astype(float, copy=True)
    V = np.eye(n)

    for _ in range(n_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                alpha = G[:, p] @ G[:, p]
                beta = G[:, q] @ G[:, q]
                gamma = G[:, p] @ G[:, q]

                # Jacobi rotation that diagonalizes the 2x2 Gram matrix
                # [[alpha, gamma], [gamma, beta]] of columns p and q.
                # Guard the exact gamma == 0 case (no rotation needed) so
                # the zeta formula never divides by zero.
                if gamma == 0.0:
                    c, s = 1.0, 0.0
                else:
                    zeta = (beta - alpha) / (2.0 * gamma)
                    sgn = 1.0 if zeta >= 0.0 else -1.0
                    t = sgn / (abs(zeta) + np.sqrt(1.0 + zeta**2))
                    c = 1.0 / np.sqrt(1.0 + t**2)
                    s = c * t

                # Rotate columns p and q of G (G <- G J).
                col_p = G[:, p].copy()
                col_q = G[:, q].copy()
                G[:, p] = c * col_p - s * col_q
                G[:, q] = s * col_p + c * col_q

                # Accumulate the right singular vectors (V <- V J).
                v_p = V[:, p].copy()
                v_q = V[:, q].copy()
                V[:, p] = c * v_p - s * v_q
                V[:, q] = s * v_p + c * v_q

    # Singular values are the norms of the orthogonalized columns.
    s = np.array([np.sqrt(G[:, i] @ G[:, i]) for i in range(n)])

    # Left singular vectors: normalized columns, divided directly with no
    # near-zero fallback. Rank-deficient inputs may yield inf/nan/huge
    # columns here; that degradation is expected and left unhandled.
    U = G / s

    return U, s, V.T


def sort_svd_descending(U, s, Vt):
    """
    Post-processing helper: reorders an (U, s, Vt) SVD triplet so that the
    singular values in s are in descending order.
    """
    idx = np.argsort(s)[::-1]
    return U[:, idx], s[idx], Vt[idx, :]


# %%


def run_tests():
    # Set random seed for reproducibility
    rng = np.random.default_rng(42)

    test_cases = [
        {"name": "3x3 Square Matrix", "matrix": rng.standard_normal((3, 3))},
        {
            "name": "5x3 Rectangular Matrix (Tall)",
            "matrix": rng.standard_normal((5, 3)),
        },
        {
            "name": "4x6 Rectangular Matrix (Wide)",
            "matrix": rng.standard_normal((4, 6)),
        },
        {
            "name": "Rank Deficient 4x4 Matrix",
            "matrix": np.outer([1, 2, 3, 4], [4, 3, 2, 1]),
        },
        {
            "name": "Ill-conditioned 5x5 Matrix",
            "matrix": np.dot(
                np.dot(rng.standard_normal((5, 5)), np.diag([1e5, 1e2, 1, 1e-2, 1e-5])),
                rng.standard_normal((5, 5)),
            ),
        },
    ]

    # Custom methods to compare against the built-in reference.
    methods = [
        ("Golub-Reinsch", svd_golub_reinsch),
        ("One-Sided Jacobi", svd_one_sided_jacobi),
        (
            "One-Sided Jacobi (HW)",
            lambda A: sort_svd_descending(*svd_one_sided_jacobi_hw(A)),
        ),
        ("Two-Sided Jacobi", svd_two_sided_jacobi),
    ]

    print("=" * 80)
    print("                    SVD IMPLEMENTATIONS COMPARISON REPORT")
    print("=" * 80)

    for case in test_cases:
        name = case["name"]
        A = case["matrix"]
        m, n = A.shape

        print(f"\n--- Test Case: {name} ({m}x{n}) ---")

        # Built-in SVD as the reference for singular values.
        t0 = time.perf_counter()
        _U_built, s_built, _Vt_built = svd_builtin(A)
        t_built = time.perf_counter() - t0

        header = (
            f"{'Method':<22} | {'Time (s)':<10} | {'SV Close':<8} | "
            f"{'Recon Close':<11} | {'U Ortho':<8} | {'V Ortho':<8}"
        )
        print(header)
        print("-" * len(header))

        for method_name, method_fn in methods:
            t0 = time.perf_counter()
            U, s, Vt = method_fn(A)
            elapsed = time.perf_counter() - t0

            # Reconstruction: ||A - U * S * Vt||_F
            S_matrix = np.zeros((U.shape[1], Vt.shape[0]))
            np.fill_diagonal(S_matrix, s)
            A_recon = U @ S_matrix @ Vt

            sv_close = np.allclose(s, s_built, atol=1e-8)
            recon_close = np.allclose(A_recon, A, atol=1e-8)
            u_ortho = np.allclose(U.T @ U, np.eye(U.shape[1]), atol=1e-8)
            v_ortho = np.allclose(Vt @ Vt.T, np.eye(Vt.shape[0]), atol=1e-8)

            print(
                f"{method_name:<22} | {elapsed:<10.6f} | {sv_close!s:<8} | "
                f"{recon_close!s:<11} | {u_ortho!s:<8} | {v_ortho!s:<8}"
            )

        # Built-in reference row.
        print(
            f"{'linalg.svd (Built-in)':<22} | {t_built:<10.6f} | {'True':<8} | "
            f"{'True':<11} | {'True':<8} | {'True':<8}"
        )

    print("\n" + "=" * 80)
    print("Verification complete.")
    print("=" * 80)


if __name__ == "__main__":
    run_tests()
