"""Optional full-state DenseDDA Jacobian assembly via receiver adjoints.

Only NumPy is imported. The caller supplies an ORIGINAL, already-created full
DDAState and its physical Tangent. This module imports no historical runner,
constructs no physical model, changes no kernel, and performs no work at import.
Calling the assembly function DOES spend m complex full-adjoint RHS. Its output
is full-model/offline-reference information and must not leak into a selector.
"""
from __future__ import annotations

import time
from typing import Callable

import numpy as np

__all__ = ["assemble_full_material_jacobian"]


def _counter_totals(state) -> dict[str, int]:
    return {key: int(value) for key, value in state.counters.totals.items()}


def assemble_full_material_jacobian(
    state,
    tangent,
    scale: float = 1.0,
    *,
    charge_rhs: Callable[[int], None] | None = None,
    ledger=None,
) -> tuple[np.ndarray, dict]:
    """Return the scaled real full material Jacobian and accounting metadata.

    Parameters
    ----------
    state
        Original full ``a9_engine.DDAState`` (``basis is None``), whose existing
        LU is reused by its original counted ``model.solver.solve`` method.
    tangent
        Original ``experiment_support.Tangent``: voxel ``Q=None`` or real
        volume-orthonormal spatial ``Q`` such as its Gaussian basis. The real
        material coordinates are ``[z_R(q); z_I(q)]``.
    scale
        Same positive data scale passed to original ``operators.make_problem``.
    charge_rhs
        Optional caller-owned resource admission/charge callback, called with
        ``m`` BEFORE solving. It may raise to prevent the solve. It is the
        caller's responsibility to pass it when the active budget requires it.
    ledger
        Optional original PhaseLedger (e.g. ``ctx.ledger``). Its
        ``full_adjoint_rhs`` counter is charged before solving. This is a
        second accounting VIEW of the original model counters, not an extra
        set of physical solves to add to those counters.

    Returns
    -------
    J : float64, shape (P*2*m, 2*q)
        Each source contributes ``[Re(data_p); Im(data_p)]`` in source-major
        order, identical to original ``EvaluationContext.j`` and its
        ``stack_real_illuminations`` packing. No global-all-real/global-all-imag
        reordering is used.
    metadata : dict
        JSON-compatible dimensions, actual receiver-adjoint RHS count, original
        counter deltas, and wall time. No hashes or reference truth are used.

    Notes
    -----
    The original solver charges failed numerical solves before LAPACK/torch is
    entered; a raised exception is propagated. There is no retry, jitter, new
    factorization, or cross-state receiver-adjoint cache here. The parent must
    establish equivalence against original JVP/VJP and finite differences before
    using this optional adapter for a physical trial.
    """
    model = state.model
    if bool(state.reduced) or state.basis is not None:
        raise ValueError("receiver-adjoint assembly requires an original full state")
    scale = float(scale)
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError("scale must be positive finite")

    N, P, m, n = int(model.N), int(model.P), int(model.m), int(model.n)
    if N <= 0 or P <= 0 or m <= 0 or n != 3 * N:
        raise ValueError("incompatible DenseDDA cell/component dimensions")
    volume = float(model.volume)
    if not np.isfinite(volume) or volume <= 0.0:
        raise ValueError("model volume must be positive finite")
    if not np.isclose(float(tangent.volume), volume, rtol=1e-12, atol=0.0):
        raise ValueError("tangent and model must use the same cell volume")
    sqrt_v = float(state.sqrt_v)
    if not np.isclose(sqrt_v, np.sqrt(volume), rtol=1e-12, atol=0.0):
        raise ValueError("state sqrt_v does not match the physical material metric")
    if int(tangent.n) != N:
        raise ValueError("tangent cell count does not match the model")

    Q = getattr(tangent, "Q", None)
    if Q is None:
        q = N
        material_basis = "voxel: delta_chi=(z_R+i*z_I)/sqrt(volume)"
    else:
        Q = np.asarray(Q)
        if np.iscomplexobj(Q):
            raise ValueError("original Tangent spatial Q must be real")
        Q = np.asarray(Q, dtype=np.float64)
        if Q.ndim != 2 or Q.shape[0] != N or Q.shape[1] == 0:
            raise ValueError("spatial Q must have shape (N,q), q>0")
        if not np.all(np.isfinite(Q)):
            raise ValueError("spatial Q contains non-finite values")
        q = int(Q.shape[1])
        if np.linalg.norm(volume * (Q.T @ Q) - np.eye(q)) > 1e-10:
            raise ValueError("spatial Q must be volume-orthonormal, as in original make_problem")
        material_basis = "real spatial Q: delta_chi=Q@(z_R+i*z_I)"
    if int(tangent.d) != 2 * q:
        raise ValueError("tangent real material dimension must equal 2*q")

    GS = np.asarray(model.GS, dtype=np.complex128)
    exciting = np.asarray(state.exciting, dtype=np.complex128)
    da = np.asarray(state.da, dtype=np.complex128)
    if GS.shape != (m, n) or exciting.shape != (P, n) or da.shape != (N,):
        raise ValueError("original GS, exciting-field, or polarizability-derivative layout mismatch")
    if not all(np.all(np.isfinite(a)) for a in (GS, exciting, da)):
        raise ValueError("non-finite state/observation inputs")

    before = _counter_totals(state)
    started = time.perf_counter()
    if charge_rhs is not None:
        charge_rhs(m)
    if ledger is not None:
        ledger.add("full_adjoint_rhs", m)

    # This is the ORIGINAL lower-level counted solve and existing full LU.
    # U=L^{-H}GS^H implies U^H=GS L^{-1}; it is common to every source.
    U = model.solver.solve(
        state._L_factor,
        GS.conj().T,
        trans=2,
        matrix=state.L,
        label="full_adjoint",
    )
    U = np.asarray(U, dtype=np.complex128)
    if U.shape != (n, m) or not np.all(np.isfinite(U)):
        raise ValueError("original receiver-adjoint solve returned invalid output")
    adjoint_solve_wall_s = time.perf_counter() - started

    # B_p(delta_chi)_{ic}=E_exc[p,ic] da[i] delta_chi[i]/sqrt(v).
    # Assemble one source at a time rather than storing a (P,m,N) complex bank.
    U_conj = U.conj().reshape(N, 3, m)
    injection = exciting.reshape(P, N, 3) * (da[None, :, None] / sqrt_v)
    J_blocks = np.empty((P, 2 * m, 2 * q), dtype=np.float64)
    for p in range(P):
        raw_chi = np.einsum("ncr,nc->rn", U_conj, injection[p], optimize=True)
        T = raw_chi / (sqrt_v * scale) if Q is None else (raw_chi @ Q) / scale
        # T(z_R+i*z_I): Re=[Re(T),-Im(T)], Im=[Im(T),Re(T)].
        J_blocks[p, :m, :q] = T.real
        J_blocks[p, :m, q:] = -T.imag
        J_blocks[p, m:, :q] = T.imag
        J_blocks[p, m:, q:] = T.real
    J = J_blocks.reshape(P * 2 * m, 2 * q)
    if not np.all(np.isfinite(J)):
        raise ValueError("assembled full material Jacobian contains non-finite entries")

    after = _counter_totals(state)
    delta = {
        key: after.get(key, 0) - before.get(key, 0)
        for key in before.keys() | after.keys()
        if after.get(key, 0) != before.get(key, 0)
    }
    metadata = {
        "method": "original_DenseDDA_receiver_adjoint_assembly",
        "scope": "full-model offline reference; never selector input",
        "N": N,
        "P": P,
        "receiver_channels_m": m,
        "complex_material_coordinates_q": q,
        "real_material_dimension": 2 * q,
        "jacobian_shape": [P * 2 * m, 2 * q],
        "dtype": str(J.dtype),
        "data_layout": "source-major: per source [Re(m); Im(m)]",
        "material_layout": "[Re(q); Im(q)]",
        "material_basis": material_basis,
        "scale": scale,
        "receiver_adjoint_rhs": m,
        "physical_full_solver_rhs": m,
        "physical_full_tangent_rhs": 0,
        "physical_new_lu_factorizations": delta.get("lu_factorizations", 0),
        "original_counter_totals_delta": delta,
        "context_ledger_full_adjoint_rhs_added": m if ledger is not None else 0,
        "caller_charge_rhs_invoked": charge_rhs is not None,
        "counter_views": "original counters and optional context ledger are not additive",
        "adjoint_solve_and_admission_wall_s": adjoint_solve_wall_s,
        "assembly_total_wall_s": time.perf_counter() - started,
        "numerical_equivalence_status": "PARENT_CHECK_REQUIRED_BEFORE_USE",
    }
    return J, metadata
