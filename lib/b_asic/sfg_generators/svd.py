"""
B-ASIC signal flow graph generators.

This module contains a number of functions generating SFGs for singular value
decomposition (SVD).
"""

from b_asic.core_operations import (
    Absolute,
    AddSub,
    Constant,
    Multiplication,
    Reciprocal,
    SquareRoot,
)
from b_asic.graph_component import Name
from b_asic.sfg import SFG
from b_asic.special_operations import Input, Output
from b_asic.svd_operations import Sign


def _dot_product(col0, col1):
    """
    Sum of elementwise products of col0 and col1, using only Multiplication
    and AddSub.
    """
    acc = Multiplication(col0[0], col1[0])
    for a, b in zip(col0[1:], col1[1:], strict=True):
        acc = AddSub(is_add=True, src0=acc, src1=Multiplication(a, b))
    return acc


def svd_one_sided_jacobi(
    m: int, n: int, n_sweeps: int = 30, name: str | None = None
) -> SFG:
    """
    Generate an SFG for the one-sided Jacobi SVD algorithm.

    Fully unrolled, static B-ASIC implementation of the one-sided Jacobi SVD
    algorithm, built only from AddSub, Multiplication, Reciprocal, SquareRoot,
    and Absolute operations, plus a custom :class:`~b_asic.svd_operations.Sign`
    operation. The gamma == 0.0 special case (present in the reference
    algorithm to avoid unnecessary rotations once converged) is skipped; the
    general-case rotation formula is always used.

    Parameters
    ----------
    m : int
        Number of rows of the input matrix A. Must satisfy m >= n.
    n : int
        Number of columns of the input matrix A.
    n_sweeps : int, default: 30
        Number of full Jacobi sweeps to unroll.
    name : str, optional
        The name of the SFG. If None, "One-sided Jacobi SVD".

    Returns
    -------
    SFG
        Signal Flow Graph with inputs A[i,j] (m x n) and outputs U[i,j] (m x n),
        s[i] (length n), and Vt[i,j] (n x n).

    Notes
    -----
    Does not implement the m < n transpose branch of the reference
    implementation. Callers with m < n should transpose the input matrix and
    swap the roles of U and V themselves.
    """
    if m < n:
        raise ValueError("m must be >= n; transpose A and swap U/V for m < n.")
    if name is None:
        name = "One-sided Jacobi SVD"

    A = [[Input(f"A[{i},{j}]") for j in range(n)] for i in range(m)]

    G = [row[:] for row in A]
    V = [[Constant(1) if i == j else Constant(0) for j in range(n)] for i in range(n)]

    for _ in range(n_sweeps):
        for p in range(n):
            for q in range(p + 1, n):
                g_col_p = [G[i][p] for i in range(m)]
                g_col_q = [G[i][q] for i in range(m)]

                alpha = _dot_product(g_col_p, g_col_p)
                beta = _dot_product(g_col_q, g_col_q)
                gamma = _dot_product(g_col_p, g_col_q)

                # gamma == 0.0 special case skipped; always use the general
                # rotation formula.
                diff = AddSub(is_add=False, src0=beta, src1=alpha)
                two_gamma = AddSub(is_add=True, src0=gamma, src1=gamma)
                zeta = Multiplication(diff, Reciprocal(two_gamma))

                sgn = Sign(zeta)

                zeta_sq = Multiplication(zeta, zeta)
                abs_zeta = Absolute(zeta)
                one_plus_zeta_sq = AddSub(is_add=True, src0=Constant(1), src1=zeta_sq)
                denom = AddSub(
                    is_add=True, src0=abs_zeta, src1=SquareRoot(one_plus_zeta_sq)
                )
                t = Reciprocal(denom)

                one_plus_t_sq = AddSub(
                    is_add=True, src0=Constant(1), src1=Multiplication(t, t)
                )
                c = Reciprocal(SquareRoot(one_plus_t_sq))

                s_val = Multiplication(Multiplication(c, t), sgn)

                for i in range(m):
                    g_ip = G[i][p]
                    g_iq = G[i][q]
                    new_gip = AddSub(
                        is_add=False,
                        src0=Multiplication(c, g_ip),
                        src1=Multiplication(s_val, g_iq),
                    )
                    new_giq = AddSub(
                        is_add=True,
                        src0=Multiplication(s_val, g_ip),
                        src1=Multiplication(c, g_iq),
                    )
                    G[i][p], G[i][q] = new_gip, new_giq

                for i in range(n):
                    v_ip = V[i][p]
                    v_iq = V[i][q]
                    new_vip = AddSub(
                        is_add=False,
                        src0=Multiplication(c, v_ip),
                        src1=Multiplication(s_val, v_iq),
                    )
                    new_viq = AddSub(
                        is_add=True,
                        src0=Multiplication(s_val, v_ip),
                        src1=Multiplication(c, v_iq),
                    )
                    V[i][p], V[i][q] = new_vip, new_viq

    s_ops = []
    s_inv_ops = []
    for j in range(n):
        col = [G[i][j] for i in range(m)]
        s_j = SquareRoot(_dot_product(col, col))
        s_ops.append(s_j)
        s_inv_ops.append(Reciprocal(s_j))

    U = [[Multiplication(G[i][j], s_inv_ops[j]) for j in range(n)] for i in range(m)]

    outputs = [Output(U[i][j], f"U[{i},{j}]") for i in range(m) for j in range(n)]
    outputs += [Output(s_ops[j], f"s[{j}]") for j in range(n)]
    outputs += [Output(V[i][j], f"Vt[{j},{i}]") for i in range(n) for j in range(n)]

    inputs = [A[i][j] for i in range(m) for j in range(n)]

    return SFG(inputs=inputs, outputs=outputs, name=Name(name))
