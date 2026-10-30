"""Production adapter: realified guarded paired-TSOM projection on frozen DDA states.

Bounded-worker deliverable for ``Gaussian/A14``. It implements the operator
interface of ``Gaussian/A14/paired_tsom_research`` (see ``paired_core.py`` and
``THEORY_AND_CODEX_SPEC.md``) for the existing A9/A10 dense-DDA / A12 Galerkin ROM
machinery, with these explicit design rules:

* Everything in the normal-equation layer is REAL. A complex-linear physical map
  ``R^{2q} -> C^{P x m}`` is applied as ``[Re; Im]`` per illumination and stacked
  into ``(P*2m)`` data coordinates. Material coordinates are ``2q = tangent.d``.
* ``J``/``J^T`` (full) and ``J0``/``J0^T`` (base ROM) are exact Euclidean real
  adjoints; ``H0 = J0^T J0 + lam I``, ``M0 = H0^{-1}``, ``F0 = M0^{1/2}``.
* The current factors ``Y Gamma N -> realified illumination-stacked (Yr, Tr)``
  are built directly from the ROM geometry. ``CurrentUpdate`` is NOT instantiated
  (no QR/SVD of the big stacked transfer, no unused work).
* ``M0``'s square root reuses the cheap compressed ROM Jacobian ``M0.Z``; the full
  material Jacobian is never formed online.
* Every physical action is charged to a phase-specific counter (columns and
  illumination-RHS). Block calls still incur their per-column physical cost.

Reproducibility: no hidden jitter, no tolerance relaxation, no silent rank edits.
Failures raise. This module makes no acceptance/novelty/speedup claim.
"""
from __future__ import annotations

import sys
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
from scipy import linalg as la
from scipy.linalg import cho_factor, cho_solve

Array = np.ndarray

# --------------------------------------------------------------------------- #
# Path injection: reuse the frozen A12 / A10 / A9 / A13 machinery verbatim.
# --------------------------------------------------------------------------- #
HERE = Path(__file__).resolve().parent
GAUSSIAN = HERE.parents[1]  # .../Research/Inv_SLAM/Gaussian
for _p in (GAUSSIAN / "A12" / "code", GAUSSIAN / "A10" / "code",
           GAUSSIAN / "A9" / "code", GAUSSIAN / "A13" / "src"):
    if _p.exists() and str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

from campaign_common import (  # noqa: E402
    DenseDDA, Tangent, grid, geometry, physical_object, Ops, cost, costdiff,
    sync, build_current_basis,
)
from galerkin_rom import GalerkinROM  # noqa: E402
from reduced_normal import ReducedNormalInverse  # noqa: E402


# --------------------------------------------------------------------------- #
# Realification helpers
# --------------------------------------------------------------------------- #
def realify(a: Array) -> Array:
    """Complex-linear map written on real+imag input/output coordinates."""
    a = np.asarray(a)
    return np.block([[a.real, -a.imag], [a.imag, a.real]])


def stack_real_illuminations(data: Array) -> Array:
    """(P, m, [k]) complex -> (P*2m, [k]) real with per-illumination [Re; Im]."""
    data = np.asarray(data)
    squeeze = data.ndim == 2
    d = data[:, :, None] if squeeze else data
    P, m, k = d.shape
    out = np.empty((P, 2 * m, k), dtype=float)
    out[:, :m] = d.real
    out[:, m:] = d.imag
    out = out.reshape(P * 2 * m, k)
    return out[:, 0] if squeeze else out


def unstack_real_illuminations(w: Array, P: int, m: int) -> Array:
    """(P*2m, [k]) real -> (P, m, [k]) complex."""
    w = np.asarray(w)
    squeeze = w.ndim == 1
    w2 = w[:, None] if squeeze else w
    if w2.shape[0] != P * 2 * m:
        raise ValueError("stacked data has incompatible leading dimension")
    blocks = w2.reshape(P, 2 * m, w2.shape[1])
    out = blocks[:, :m] + 1j * blocks[:, m:]
    return out[:, :, 0] if squeeze else out


def tangent_pack(c: Array) -> Array:
    c = np.asarray(c)
    return np.concatenate([c.real, c.imag], axis=0)


def tangent_adjoint(tangent: Tangent, g: Array) -> Array:
    """Exact real adjoint of ``Tangent.expand`` for (N,) or (N, k)."""
    g = np.asarray(g, dtype=complex)
    c = g / np.sqrt(tangent.volume) if tangent.Q is None else tangent.Q.T @ g
    return tangent_pack(c)


def project_orth(C: Array, V: Array | None = None,
                 rank: int | None = None, tol: float = 1e-12,
                 atol: float = 0.0) -> Array:
    """Orthonormalise columns of C after (twice) projecting outside V.

    ``keep = s > max(atol, tol * s[0])`` so a caller projecting a normalised
    pool can additionally reject roundoff-only columns absolutely.
    """
    C = np.asarray(C, dtype=complex)
    if C.ndim == 1:
        C = C[:, None]
    if V is not None and V.shape[1]:
        for _ in range(2):
            C = C - V @ (V.conj().T @ C)
    if C.shape[1] == 0:
        return C
    u, s, _ = la.svd(C, full_matrices=False, check_finite=False)
    keep = (int(np.sum(s > max(float(atol), tol * s[0])))
            if s.size and s[0] > 0 else 0)
    if rank is not None:
        keep = min(keep, int(rank))
    return np.ascontiguousarray(u[:, :keep])


def orth(a: Array, rtol: float = 1e-11, atol: float = 0.0) -> Array:
    """Reference orth from paired_core (unchanged)."""
    a = np.asarray(a)
    if a.ndim != 2:
        raise ValueError("orth expects a matrix")
    if a.shape[1] == 0:
        return a.copy()
    u, s, _ = np.linalg.svd(a, full_matrices=False)
    if len(s) == 0:
        return a[:, :0]
    keep = s > max(atol, rtol * s[0])
    return u[:, keep]


def sym(a: Array) -> Array:
    return (a + a.conj().T) / 2


def spd_power(a: Array, power: float) -> Array:
    """Dense diagnostic only (small ranks); never an online full-material solve."""
    w, v = np.linalg.eigh(sym(a))
    if np.any(w <= 0):
        raise ValueError("Matrix is not positive definite")
    return (v * w**power) @ v.conj().T


# --------------------------------------------------------------------------- #
# Phase ledger and counted physical callbacks
# --------------------------------------------------------------------------- #
class PhaseLedger:
    """Accumulate physical action/cost counts keyed by explicit phase name."""

    def __init__(self) -> None:
        self._records: dict[str, dict[str, float]] = {}
        self._stack: list[str] = []

    @contextmanager
    def phase(self, name: str):
        self._stack.append(str(name))
        try:
            yield
        finally:
            self._stack.pop()

    @property
    def current(self) -> str:
        return self._stack[-1] if self._stack else "unassigned"

    def add(self, key: str, value: float = 1.0) -> None:
        rec = self._records.setdefault(self.current, {})
        rec[key] = rec.get(key, 0.0) + float(value)

    def as_dict(self) -> dict:
        return {ph: {k: (round(v, 6) if isinstance(v, float) else v)
                     for k, v in rec.items()}
                for ph, rec in self._records.items()}


class PhysicalAction:
    """Wrap a real material/data callback and charge the active ledger phase."""

    def __init__(self, fn: Callable[[Array], Array], kind: str, ledger: PhaseLedger):
        self.fn = fn
        self.kind = kind
        self.ledger = ledger

    def __call__(self, x: Array) -> Array:
        cols = 1 if np.asarray(x).ndim == 1 else int(np.asarray(x).shape[1])
        t0 = time.perf_counter()
        out = self.fn(x)
        ledger = self.ledger
        ledger.add(f"{self.kind}_calls", 1)
        ledger.add(f"{self.kind}_columns", cols)
        ledger.add(f"{self.kind}_wall_s", time.perf_counter() - t0)
        return out


# --------------------------------------------------------------------------- #
# Reference kernels copied unchanged from paired_tsom_research/paired_core.py
# --------------------------------------------------------------------------- #
@dataclass
class CGResult:
    x: Array
    converged: bool
    iterations: int
    true_residual: float
    tolerance: float
    operator_actions: int
    iterates: list


def pcg(A: Callable[[Array], Array], b: Array, M: Callable[[Array], Array],
        x0: Array | None = None, rtol: float = 1e-10, atol: float = 0.0,
        maxiter: int = 300, keep_history: bool = False,
        initial_residual: Array | None = None) -> CGResult:
    """Fixed SPD PCG; an optional paid/cached initial residual saves one A action."""
    b = np.asarray(b, dtype=float)
    if b.ndim != 1 or not np.all(np.isfinite(b)):
        raise ValueError("b must be a finite real vector")
    if rtol < 0 or atol < 0 or maxiter < 0:
        raise ValueError("Invalid stopping settings")
    x = np.zeros_like(b) if x0 is None else np.array(x0, dtype=float, copy=True)
    tol = max(atol, rtol * np.linalg.norm(b))
    if initial_residual is None:
        r = b - A(x)
        actions = 1
    else:
        r = np.asarray(initial_residual, dtype=float)
        if r.shape != b.shape or not np.all(np.isfinite(r)):
            raise ValueError("initial_residual must match b and be finite")
        r = r.copy()
        actions = 0
    hist = [x.copy()] if keep_history else []
    if np.linalg.norm(r) <= tol:
        # A cached residual may be wrong or stale; never accept it alone.
        if initial_residual is not None:
            r = b - A(x)
            actions += 1
        if np.linalg.norm(r) <= tol:
            return CGResult(x, True, 0, float(np.linalg.norm(r)), tol, actions, hist)
    z = M(r)
    rz = float(r @ z)
    if not np.isfinite(rz) or rz <= 0:
        raise ValueError("Preconditioner is not positive on residual")
    p = z.copy()
    k = 0
    for k in range(1, maxiter + 1):
        ap = A(p)
        actions += 1
        pap = float(p @ ap)
        if not np.isfinite(pap) or pap <= 0:
            raise ValueError("Operator is not positive on search direction")
        alpha = rz / pap
        x += alpha * p
        r -= alpha * ap
        if keep_history:
            hist.append(x.copy())
        if np.linalg.norm(r) <= tol:
            true_r = b - A(x)
            actions += 1
            if np.linalg.norm(true_r) <= tol:
                return CGResult(x, True, k, float(np.linalg.norm(true_r)),
                                tol, actions, hist)
            r = true_r
            z = M(r)
            rz = float(r @ z)
            if rz <= 0:
                raise ValueError("Nonpositive preconditioned true residual")
            p = z.copy()
            continue
        z = M(r)
        new_rz = float(r @ z)
        if not np.isfinite(new_rz) or new_rz <= 0:
            raise ValueError("Preconditioner lost positivity")
        p = z + (new_rz / rz) * p
        rz = new_rz
    true_r = b - A(x)
    actions += 1
    return CGResult(x, np.linalg.norm(true_r) <= tol, k,
                    float(np.linalg.norm(true_r)), tol, actions, hist)


def guarded_projection(J, JT, Lambda, M0, r, ell, s0, Z,
                       rtol: float = 1e-9, atol: float = 1e-12,
                       maxiter: int = 300) -> dict:
    """Reference single-phase guarded projected solve (paired_core)."""
    Z = orth(Z)
    h = JT(r) + ell
    js0 = J(s0)
    if Z.shape[1]:
        JZ = J(Z)
        G = sym(JZ.T @ JZ + Z.T @ Lambda(Z))
        bproj = JZ.T @ (r + js0) + Z.T @ (Lambda(s0) + ell)
        cf = cho_factor(G, lower=True, check_finite=True)
        a = -cho_solve(cf, bproj)
        s = s0 + Z @ a
        js = js0 + JZ @ a
    else:
        G, JZ = np.zeros((0, 0)), np.zeros((len(r), 0))
        s, js = s0.copy(), js0
    g = JT(r + js) + Lambda(s) + ell
    tol = max(atol, rtol * np.linalg.norm(h))
    guard = np.linalg.norm(g) <= tol
    if guard:
        cg = None
        result = s
    else:
        cg = pcg(lambda x: JT(J(x)) + Lambda(x), -h, M0, x0=s,
                 rtol=rtol, atol=atol, maxiter=maxiter)
        result = cg.x
    return {"x": result, "projected_x": s, "projection_accepted": bool(guard),
            "projected_true_residual": float(np.linalg.norm(g)),
            "tolerance": float(tol), "fallback": cg, "projected_curvature": G,
            "JZ": JZ, "converged": bool(guard or (cg is not None and cg.converged))}


def two_level_inverse(Z: Array, HZ: Array, M0) -> Callable[[Array], Array]:
    """Standard balanced SPD two-level inverse (paired_core, unchanged)."""
    if Z.shape != HZ.shape:
        raise ValueError("Z and HZ shapes differ")
    if Z.shape[1] == 0:
        return M0
    fac = cho_factor(sym(Z.T @ HZ), lower=True)

    def action(x: Array) -> Array:
        ztx = Z.T @ x
        ptx = x - HZ @ cho_solve(fac, ztx)
        tmp = M0(ptx)
        return Z @ cho_solve(fac, ztx) + tmp - Z @ cho_solve(fac, HZ.T @ tmp)

    return action


def paired_from_gramians(Y, T, J0T, M0, qmax=None, rtol=1e-10, H0=None):
    """Reference Gramian route (paired_core, unchanged)."""
    if np.iscomplexobj(Y) or np.iscomplexobj(T):
        raise ValueError("Use realified arrays and real material callbacks")
    if qmax is not None and qmax < 0:
        raise ValueError("qmax must be nonnegative")
    MT = M0(T.T)
    Gin = sym(T @ MT)
    Qy, Ry = np.linalg.qr(Y, mode="reduced")
    small = sym(Ry @ Gin @ Ry.T)
    ev, vec = np.linalg.eigh(small)
    idx = np.argsort(ev)[::-1]
    ev, vec = ev[idx], vec[:, idx]
    scale = max(float(np.max(np.abs(ev), initial=0.0)), np.finfo(float).tiny)
    if ev.min(initial=0.0) < -1e-9 * scale:
        raise ValueError("Indefinite input Gramian: check M0/adjoint/precision")
    cutoff = max(rtol**2, 100 * np.finfo(float).eps) * scale
    valid = np.flatnonzero(ev > cutoff)
    if qmax is not None:
        valid = valid[:qmax]
    sig = np.sqrt(ev[valid])
    left = Qy @ vec[:, valid]
    obs = M0(J0T(left))
    inj = (MT @ (Y.T @ left)) / sig[None, :] if len(sig) else obs[:, :0]
    Z = orth(np.column_stack([obs, inj]))
    if H0 is not None and Z.shape[1]:
        Z = Z @ spd_power(sym(Z.T @ H0(Z)), -0.5)
    return Z, {"joint_singular_values": sig, "material_rank": Z.shape[1],
               "left_modes": left, "candidate_coefficient_count": T.shape[0],
               "gramian_cutoff": float(cutoff)}


def paired_from_root_factors(Y, T, J0T, F0, F0T, qmax=None, rtol=1e-12, H0=None):
    """Reference QR-SVD route (paired_core, unchanged)."""
    if np.iscomplexobj(Y) or np.iscomplexobj(T):
        raise ValueError("Use realified arrays")
    weighted_T = F0T(T.T).T
    Qy, Ry = np.linalg.qr(Y, mode="reduced")
    Qf, Rf = np.linalg.qr(weighted_T.T, mode="reduced")
    us, sig, vhs = np.linalg.svd(Ry @ Rf.T, full_matrices=False)
    keep = np.flatnonzero(sig > rtol * sig.max(initial=0.0))
    if qmax is not None:
        if qmax < 0:
            raise ValueError("qmax must be nonnegative")
        keep = keep[:qmax]
    left = Qy @ us[:, keep]
    right = Qf @ vhs[keep].T
    obs = F0(F0T(J0T(left)))
    inj = F0(right)
    Z = orth(np.column_stack([obs, inj]))
    if H0 is not None and Z.shape[1]:
        Z = Z @ spd_power(sym(Z.T @ H0(Z)), -0.5)
    return Z, {"joint_singular_values": sig[keep], "material_rank": Z.shape[1],
               "left_modes": left, "candidate_coefficient_count": T.shape[0]}


# --------------------------------------------------------------------------- #
# Frozen physical state
# --------------------------------------------------------------------------- #
@dataclass
class FrozenState:
    """Frozen anchor: realified full/base operators, common r/ell/s0, lambda.

    Build with :func:`build_frozen_state` or :func:`frozen_state_from_config`.
    All callbacks are phase-charged through :attr:`ledger`.
    """

    model: object
    state: object
    tangent: Tangent
    data: Array
    scale: float
    lam: float
    rank: int
    prior: float = 0.0
    device: str = "cpu"
    diagnostics: bool = True

    def __post_init__(self) -> None:
        self.P = int(self.model.P)
        self.m = int(self.model.m)
        self.q = int(self.tangent.d) // 2
        self.dim_material = int(self.tangent.d)
        self.dim_data = self.P * 2 * self.m
        self.ledger = PhaseLedger()
        self.op = Ops(self.state, self.tangent, self.scale)

        # Base ROM from the frozen retained current basis.
        self.Vs = self.model.gs_modes()["V"]
        self.V = build_current_basis(self.state, self.Vs, self.rank)
        self.rom = GalerkinROM(self.state, self.V)
        self.M0 = ReducedNormalInverse(self.rom, self.tangent, self.scale, self.lam)
        self._f0_V = None
        self.f0_rank = 0

        with self.ledger.phase("factory"):
            # residual is a realified illumination-stacked data vector (P*2m,)
            self.r = stack_real_illuminations(
                (self.state.field - self.data) / self.scale)
            self.ell = self.prior * self.tangent.project(
                self.state.chi - (0.1 + 0.04j))
            self.h0 = self.j0t_real(self.r) + self.ell
            self.s0 = -self.M0.solve(self.h0)
            probe = np.random.default_rng(20260929).normal(size=self.dim_material)
            probe /= max(np.linalg.norm(probe), 1e-300)
            self.checks = self._cheap_checks(probe) if self.diagnostics else {}
        if self.diagnostics:
            # Optional full-J adjoint probes, charged to a dedicated phase so the
            # timed online setup is never silently mixed with diagnostics.
            with self.ledger.phase("factory_diagnostics"):
                self.checks.update(self._full_adjoint_checks())

    # -- construction helpers --------------------------------------------- #
    @staticmethod
    def _apply_sqrt(x: Array, V: Array, w: Array, lam: float) -> Array:
        x = np.asarray(x, dtype=float)
        shape = x.shape
        X = x.reshape(V.shape[0], -1)
        out = X / np.sqrt(lam) + V @ (w[:, None] * (V.T @ X))
        return out.reshape(shape)

    def _build_f0(self) -> None:
        """Cheap symmetric square root of M0 from the compressed ROM Jacobian."""
        A = realify(self.M0.Z)  # (2*rows, 2q) real, M0^-1 = lam I + A^T A
        if A.shape[1] != self.dim_material:
            raise RuntimeError("M0.Z columns do not match the real material dimension")
        _, s, vh = la.svd(A, full_matrices=False, check_finite=False)
        V = vh.conj().T  # right singular vectors, (2q, k)
        w = 1.0 / np.sqrt(self.lam + s**2) - 1.0 / np.sqrt(self.lam)
        self._f0_V = np.ascontiguousarray(V)
        self._f0_w = np.ascontiguousarray(w)
        self.f0_rank = int(np.count_nonzero(s > 0))

    def _cheap_checks(self, probe: Array) -> dict:
        """Convention checks that do not touch the full physical Jacobian."""
        out = {}
        h0 = self.h0_real(probe)
        out["M0_times_H0_relative"] = float(
            la.norm(self.M0.solve(h0) - probe) / max(la.norm(probe), 1e-300))
        out["F0_times_F0T_relative"] = float(
            la.norm(self.f0_action(self.f0_action(probe)) - self.M0.solve(probe))
            / max(la.norm(probe), 1e-300))
        w = np.random.default_rng(7).normal(size=self.dim_data)
        z = np.random.default_rng(11).normal(size=self.dim_material)
        out["base_adjoint_relative"] = float(
            abs(float(self.j0_real(z) @ w) - float(z @ self.j0t_real(w)))
            / max(abs(float(self.j0_real(z) @ w)), 1e-300))
        return out

    def _full_adjoint_checks(self) -> dict:
        """One-vector full-J adjoint residual (diagnostic-only phase)."""
        w = np.random.default_rng(7).normal(size=self.dim_data)
        z = np.random.default_rng(11).normal(size=self.dim_material)
        return {"full_adjoint_relative": float(
            abs(float(self.j_real(z) @ w) - float(z @ self.jt_real(w)))
            / max(abs(float(self.j_real(z) @ w)), 1e-300))}

    # -- complex physical maps -------------------------------------------- #
    def j_complex(self, z: Array) -> Array:
        """Full measured perturbation ``op.j``; counts J material columns."""
        z = np.asarray(z, dtype=float)
        cols = 1 if z.ndim == 1 else z.shape[1]
        t0 = time.perf_counter()
        out = self.op.j(z)
        self.ledger.add("J_calls", 1)
        self.ledger.add("J_material_columns", cols)
        self.ledger.add("J_illumination_rhs", cols * self.P)
        self.ledger.add("J_wall_s", time.perf_counter() - t0)
        return out

    def jt_complex(self, w: Array) -> Array:
        """Full real adjoint of the realified J; counts JT illumination RHS."""
        w = np.asarray(w, dtype=complex)
        batched = w.ndim == 3
        cols = w.shape[2] if batched else 1
        t0 = time.perf_counter()
        if batched:
            if w.shape[:2] != (self.P, self.m):
                raise ValueError("batched cotangent must have shape (P, m, columns)")
            if cols == 0:
                out = np.empty((self.dim_material, 0), dtype=float)
            else:
                # A9's scalar vjp, with one RHS column per (illumination,
                # cotangent) pair. Keep its conjugations and material metric;
                # only the adjoint LU call is batched.
                state = self.state
                model = self.model
                tv = time.perf_counter()
                state.counters.bump("vjp_calls")
                y = np.einsum("oc,pob->pcb", model.GS.conj(), w / self.scale)
                rhs = y.transpose(1, 0, 2).reshape(model.n, self.P * cols)
                if state.reduced:
                    V = state.basis
                    red = V.conj().T @ rhs
                    coeff = model.solver.solve(
                        state._Lr_factor, red, trans=2, matrix=state.Lr,
                        label="reduced_adjoint")
                    u = (V @ coeff).reshape(model.n, self.P, cols).transpose(1, 0, 2)
                else:
                    solved = model.solver.solve(
                        state._L_factor, rhs, trans=2, matrix=state.L,
                        label="full_adjoint")
                    u = solved.reshape(model.n, self.P, cols).transpose(1, 0, 2)
                inner = np.einsum(
                    "pnc,pncb->nb",
                    state.exciting.reshape(self.P, model.N, 3).conj(),
                    u.reshape(self.P, model.N, 3, cols),
                )
                grad = state.da.conj()[:, None] * inner / state.sqrt_v
                state.counters.add_time("vjp_s", time.perf_counter() - tv)
                out = tangent_adjoint(self.tangent, grad)
        else:
            out = self.op.v(w)
        self.ledger.add("JT_calls", 1)
        self.ledger.add("JT_cotangent_columns", cols)
        self.ledger.add("JT_illumination_rhs", cols * self.P)
        self.ledger.add("JT_wall_s", time.perf_counter() - t0)
        return out

    def j0_complex(self, z: Array) -> Array:
        z = np.asarray(z, dtype=float)
        cols = 1 if z.ndim == 1 else z.shape[1]
        t0 = time.perf_counter()
        out = self.rom.jvp(self.tangent.expand(z)) / self.scale
        self.ledger.add("J0_calls", 1)
        self.ledger.add("J0_material_columns", cols)
        self.ledger.add("J0_illumination_rhs", cols * self.P)
        self.ledger.add("J0_wall_s", time.perf_counter() - t0)
        return out

    def j0t_complex(self, w: Array) -> Array:
        w = np.asarray(w, dtype=complex)
        batched = w.ndim == 3
        cols = w.shape[2] if batched else 1
        t0 = time.perf_counter()
        if batched:
            out = tangent_adjoint(self.tangent, self.rom.vjp(w) / self.scale)
        else:
            out = tangent_adjoint(self.tangent, self.rom.vjp(w) / self.scale)
        self.ledger.add("J0T_calls", 1)
        self.ledger.add("J0T_cotangent_columns", cols)
        self.ledger.add("J0T_illumination_rhs", cols * self.P)
        self.ledger.add("J0T_wall_s", time.perf_counter() - t0)
        return out

    # -- realified physical actions --------------------------------------- #
    def j_real(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        out = self.j_complex(Z)
        return stack_real_illuminations(out)

    def jt_real(self, W: Array) -> Array:
        W = np.asarray(W, dtype=float)
        one = W.ndim == 1
        W2 = W[:, None] if one else W
        wc = unstack_real_illuminations(W2, self.P, self.m)
        g = self.jt_complex(wc)
        return g[:, 0] if one else g

    def j0_real(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        out = self.j0_complex(Z)
        return stack_real_illuminations(out)

    def j0t_real(self, W: Array) -> Array:
        W = np.asarray(W, dtype=float)
        one = W.ndim == 1
        W2 = W[:, None] if one else W
        wc = unstack_real_illuminations(W2, self.P, self.m)
        g = self.j0t_complex(wc)
        return g[:, 0] if one else g

    def H_real(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        return self.jt_real(self.j_real(Z)) + self.lam * Z

    def h0_real(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        return self.j0t_real(self.j0_real(Z)) + self.lam * Z

    def lambda_action(self, Z: Array) -> Array:
        Z = np.asarray(Z, dtype=float)
        cols = 1 if Z.ndim == 1 else Z.shape[1]
        self.ledger.add("lambda_calls", 1)
        self.ledger.add("lambda_columns", cols)
        return self.lam * Z

    def m0_action(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        cols = 1 if X.ndim == 1 else X.shape[1]
        t0 = time.perf_counter()
        out = self.M0.solve(X)
        self.ledger.add("M0_apps", cols)
        self.ledger.add("M0_wall_s", time.perf_counter() - t0)
        return out

    def f0_action(self, X: Array) -> Array:
        X = np.asarray(X, dtype=float)
        cols = 1 if X.ndim == 1 else X.shape[1]
        if self._f0_V is None:
            t = time.perf_counter()
            self._build_f0()
            self.ledger.add("F0_setup_s", time.perf_counter()-t)
        out = self._apply_sqrt(X, self._f0_V, self._f0_w, self.lam)
        self.ledger.add("F0_apps", cols)
        return out

    f0t_action = f0_action  # symmetric square-root factor

    # -- phase helpers ----------------------------------------------------- #
    def phase(self, name: str):
        return self.ledger.phase(name)

    def callback(self, kind: str, fn: Callable[[Array], Array]) -> PhysicalAction:
        return PhysicalAction(fn, kind, self.ledger)

    def info(self) -> dict:
        return {
            "device": self.device,
            "P": self.P, "m": self.m, "R": int(self.model.R),
            "N": int(self.model.N), "volume": float(self.model.volume),
            "k": float(self.model.k),
            "complex_material_dim": self.q,
            "real_material_dim": self.dim_material,
            "real_data_dim": self.dim_data,
            "lam": float(self.lam), "scale": float(self.scale),
            "rank_requested": int(self.rank),
            "retained_current_rank": int(self.rom.rstate),
            "M0_branch": self.M0.branch,
            "F0_rank": self.f0_rank,
            "consistency": self.checks,
        }


def build_frozen_state(model, state, tangent: Tangent, data: Array, scale: float,
                       lam: float, rank: int, prior: float = 0.0,
                       device: str = "cpu", diagnostics: bool = True) -> FrozenState:
    """Assemble a :class:`FrozenState` from an existing full anchor state."""
    if state.reduced:
        raise ValueError("build_frozen_state requires a full anchor state")
    return FrozenState(model=model, state=state, tangent=tangent,
                       data=np.asarray(data, dtype=complex), scale=float(scale),
                       lam=float(lam), rank=int(rank), prior=float(prior),
                       device=str(device), diagnostics=bool(diagnostics))


def frozen_state_from_config(config: dict, anchor: dict | str | Path, iteration: int,
                             device: str | None = None, rank: int | None = None,
                             lam: float | None = None,
                             prior: float | None = None,
                             diagnostics: bool = True) -> FrozenState:
    """Parent gate-runner factory from the A12 config/anchor convention.

    ``config`` has ``case`` (scene) and optional ``policy``/``prior`` keys.
    ``anchor`` is an ``npz`` path or a loaded mapping exposing ``chi_{i}``,
    optional ``lambda_{i}`` and ``data``. No frozen development state is built
    or solved by this module; the caller decides which anchors to load.
    """
    if isinstance(anchor, (str, Path)):
        archive = np.load(Path(anchor), allow_pickle=False)
    else:
        archive = anchor
    scene = config["case"]
    policy = config.get("policy", {})
    chi = np.asarray(archive[f"chi_{iteration}"]).copy()
    lam_final = float(lam if lam is not None else
                      (archive[f"lambda_{iteration}"]
                       if f"lambda_{iteration}" in archive else policy["lam"]))
    rank_final = int(rank if rank is not None else policy["rank"])
    prior_final = float(prior if prior is not None else config.get("prior", 0.0))
    dev = str(device if device is not None else policy.get("device", "cpu")).lower()
    points, volume = grid(int(scene["n"]))
    dirs, pols, receivers, obs = geometry(scene)
    model = DenseDDA(points, volume, float(scene["k"]), dirs, pols, receivers, obs,
                     device=dev)
    tangent = Tangent(points, volume, scene.get("representation", "voxel"))
    state = model.state(chi)
    data = np.asarray(archive["data"], dtype=complex)
    scale = float(np.linalg.norm(np.asarray(archive["data0"])))
    return build_frozen_state(model, state, tangent, data, scale, lam_final,
                              rank_final, prior_final, device=dev,
                              diagnostics=diagnostics)


# --------------------------------------------------------------------------- #
# Current candidate screening and Y / Gamma / N factors
# --------------------------------------------------------------------------- #
def current_candidates(fs: FrozenState, kind: str = "residual",
                       rank: int = 6, seed: int = 130001) -> tuple[Array, dict]:
    """Candidate current columns ``C`` (complex, n x b) outside the ROM basis."""
    with fs.phase("current_screening"):
        if kind == "residual":
            K = fs.op.k(fs.s0) - fs.rom.current_jvp(fs.tangent.expand(fs.s0))
            # op.k solves the *reduced* current equation in the frozen anchor;
            # charge its physical illumination RHS explicitly in this phase.
            fs.ledger.add("current_solve_calls", 1)
            fs.ledger.add("current_solve_illumination_rhs", fs.P)
            from a10_common import current_columns
            C = project_orth(current_columns(K), fs.V, rank=rank)
            meta = {"family": "tangent_current_correction", "new_full_jvp": True,
                    "current_solve_illumination_rhs": fs.P}
        elif kind == "gs":
            # Discarded receiver modes only: the leading Vs[:, :rank] columns are
            # already inside the retained basis V, so start beyond them and drop
            # projected roundoff-only columns by an absolute cutoff.
            pool = fs.Vs[:, int(fs.rank):]
            atol = 1e-10 * max(1.0, float(la.norm(pool))) if pool.size else 0.0
            C = project_orth(pool, fs.V, rank=rank, atol=atol)
            meta = {"family": "discarded_receiver_modes",
                    "pool_start_column": int(fs.rank),
                    "absolute_cutoff": float(atol)}
        elif kind == "fourier":
            waves = sorted([(i, j, k) for i in range(-2, 3) for j in range(-2, 3)
                            for k in range(-2, 3)],
                           key=lambda v: (sum(x * x for x in v), v))
            cols = []
            for w in waves:
                e = np.exp(2j * np.pi * (fs.model.points @ np.array(w)) / 1.5)
                for dxyz in range(3):
                    c = np.zeros((fs.model.N, 3), complex)
                    c[:, dxyz] = e
                    cols.append(c.ravel())
                if len(cols) >= rank + int(fs.rom.rstate):
                    break
            C = project_orth(np.column_stack(cols), fs.V, rank=rank)
            meta = {"family": "centered_frequency_then_xyz"}
        elif kind == "random":
            rng = np.random.default_rng(seed)
            C = project_orth(rng.normal(size=(fs.model.n, rank))
                             + 1j * rng.normal(size=(fs.model.n, rank)), fs.V)
            meta = {"family": "deterministic_random", "seed": seed}
        else:
            raise ValueError(f"unknown candidate kind {kind!r}")
    meta.update(requested_rank=int(rank), actual_rank=int(C.shape[1]),
                kind=kind)
    if C.shape[1] == 0:
        raise ValueError("candidate bank is empty after rank revelation")
    return C, meta


def current_transfer_factors(fs: FrozenState, C: Array,
                             orth_tol: float = 1e-12) -> dict:
    """Build realified illumination-stacked ``(Yr, Tr)`` with shared material.

    ``D_C = Yr @ Tr`` is the exact first-order difference map of the enriched
    Galerkin model with the same fixed B. Cross-illumination blocks of
    ``G_i = Tr M0 Tr^T`` are preserved (common material, P data blocks).
    """
    C = project_orth(C, fs.V, tol=orth_tol)
    b = int(C.shape[1])
    if b == 0:
        raise ValueError("empty current candidate bank")
    V, rom, tan, model = fs.V, fs.rom, fs.tangent, fs.model
    with fs.phase("current_factor"):
        LC = rom.apply_L(C)
        F = C.conj().T @ rom.LV
        E = V.conj().T @ LC
        AE = la.lu_solve(rom._factor, E, check_finite=False)
        Cbar = C - V @ AE
        Gamma = C.conj().T @ LC - F @ AE
        gamma_s = la.svdvals(Gamma)
        if gamma_s[-1] <= orth_tol * gamma_s[0]:
            raise ValueError("Schur matrix numerically singular; no jitter added")
        Y = model.GS @ Cbar / fs.scale  # (m, b), illumination-independent
        field = (fs.state.exciting.reshape(model.P, model.N, 3)
                 * fs.state.da[None, :, None] / fs.state.sqrt_v)
        CB = np.einsum("bnc,pnc->pbn", C.conj().T.reshape(b, model.N, 3), field,
                       optimize=True)
        N = CB - np.einsum("br,prn->pbn", F, rom.F, optimize=True)
        Q = getattr(tan, "Q", None)
        # Coordinate convention verified against A13 CurrentUpdate:
        # voxel expansion carries 1/sqrt(volume); a tangent Q basis does not.
        if Q is None:
            Nq = N / np.sqrt(tan.volume)
        else:
            Nq = np.einsum("pbn,nq->pbq", N, Q, optimize=True)
        Tp = np.stack([la.solve(Gamma, Nq[p], check_finite=False)
                       for p in range(model.P)], axis=0)  # (P, b, q)
        Yr = la.block_diag(*([realify(Y)] * model.P)) if model.P > 1 else realify(Y)
        Tr = np.vstack([realify(Tp[p]) for p in range(model.P)])  # (P*2b, 2q)
    if Yr.shape != (fs.dim_data, model.P * 2 * b):
        raise RuntimeError("realified Y has incompatible shape")
    if Tr.shape != (model.P * 2 * b, fs.dim_material):
        raise RuntimeError("realified T has incompatible shape")
    return {"Yr": Yr, "Tr": Tr, "Y": Y, "T": Tp, "Gamma": Gamma,
            "gamma_s": gamma_s, "current_rank": b, "P": model.P,
            "schur_condition": float(gamma_s[0] / gamma_s[-1]),
            "C": C}


def current_factor_map(fs: FrozenState, transfer: dict, Z: Array) -> Array:
    """Apply the realified model difference ``D_C`` to material columns."""
    Z = np.asarray(Z, dtype=float)
    return transfer["Yr"] @ (transfer["Tr"] @ Z)


# --------------------------------------------------------------------------- #
# Paired material basis and the two algorithms
# --------------------------------------------------------------------------- #
def paired_material_basis(fs: FrozenState, transfer: dict,
                          qmax: int | None = None, method: str = "qr_svd",
                          rtol: float | None = None,
                          h0_normalize: bool = True) -> tuple[Array, dict]:
    """Reference paired basis ``Z`` (real material columns) and diagnostics."""
    with fs.phase(f"paired_basis_{method}"):
        H0 = fs.h0_real if h0_normalize else None
        J0T = fs.callback("paired_J0T", fs.j0t_real)
        if method == "qr_svd":
            rt = 1e-12 if rtol is None else rtol
            Z, meta = paired_from_root_factors(
                transfer["Yr"], transfer["Tr"], J0T,
                fs.callback("paired_F0", fs.f0_action),
                fs.callback("paired_F0T", fs.f0t_action),
                qmax=qmax, rtol=rt, H0=H0)
        elif method == "gramian":
            rt = 1e-10 if rtol is None else rtol
            Z, meta = paired_from_gramians(
                transfer["Yr"], transfer["Tr"], J0T,
                fs.callback("paired_M0", fs.m0_action),
                qmax=qmax, rtol=rt, H0=H0)
        else:
            raise ValueError("method must be 'qr_svd' or 'gramian'")
    meta.update(method=method, qmax=qmax, h0_normalized=bool(h0_normalize),
                current_rank=transfer["current_rank"],
                material_rank=int(Z.shape[1]),
                transfer_rank=int(len(meta["joint_singular_values"])))
    return Z, meta


def full_normal_residual(fs: FrozenState, s: Array, phase: str = "full_check") -> float:
    """``||J^T(r + Js) + lam s + ell||`` under an explicit phase."""
    with fs.phase(phase):
        g = fs.jt_real(fs.r + fs.j_real(s)) + fs.lambda_action(s) + fs.ell
    return float(np.linalg.norm(g))


def algorithm1_guarded_projected(fs: FrozenState, Z: Array, rtol: float = 1e-9,
                                 atol: float = 1e-12, maxiter: int = 300,
                                 cached_h: Array | None = None) -> dict:
    """Guarded projected full-quadratic solve with fixed-M0 full-PCG escape.

    Phase-tagged equivalent of ``paired_core.guarded_projection``: one J(s0),
    one J(Z) block, a cached or paid JT(r), and the guard JT, then the fallback. The coarse
    residual is never the stopping test.
    """
    cols = Z.shape[1] if Z.ndim == 2 else int(Z.ndim and 1)
    if cached_h is not None:
        cached_h = np.asarray(cached_h, dtype=float)
        if cached_h.shape != (fs.dim_material,):
            raise ValueError("cached_h shape must match real material dimension")
    with fs.phase("A1_base"):
        js0 = fs.j_real(fs.s0)
        h = fs.jt_real(fs.r) + fs.ell if cached_h is None else cached_h
    if cols:
        with fs.phase("A1_AZ"):
            JZ = fs.j_real(Z)
            G = sym(JZ.T @ JZ + Z.T @ fs.lambda_action(Z))
            bproj = (JZ.T @ (fs.r + js0)
                     + Z.T @ (fs.lambda_action(fs.s0) + fs.ell))
        cf = cho_factor(G, lower=True, check_finite=True)
        a = -cho_solve(cf, bproj)
        s = fs.s0 + Z @ a
        js = js0 + JZ @ a
    else:
        G, JZ = np.zeros((0, 0)), np.zeros((fs.dim_data, 0))
        s, js = fs.s0.copy(), js0
    with fs.phase("A1_guard"):
        g = fs.jt_real(fs.r + js) + fs.lambda_action(s) + fs.ell
    tol = max(atol, rtol * np.linalg.norm(h))
    guard = bool(np.linalg.norm(g) <= tol)
    if guard:
        cg = None
        x = s
    else:
        with fs.phase("A1_fallback"):
            cg = pcg(fs.H_real, -h,
                     fs.callback("fallback_M0", fs.m0_action), x0=s,
                     initial_residual=-g,
                     rtol=rtol, atol=atol, maxiter=maxiter)
        x = cg.x
    return {"x": x, "projected_x": s, "projection_accepted": guard,
            "projected_true_residual": float(np.linalg.norm(g)),
            "tolerance": float(tol), "fallback": cg, "projected_curvature": G,
            "JZ": JZ, "material_rank": int(cols),
            "fallback_initial_point": "projected_x" if not guard else None,
            "fallback_initial_residual_source": "paid_full_guard" if not guard else None,
            "converged": bool(guard or (cg is not None and cg.converged))}


def algorithm2_two_level(fs: FrozenState, Z: Array, rtol: float = 1e-9,
                         atol: float = 1e-12, maxiter: int = 300,
                         cached_HZ: Array | None = None,
                         cached_h: Array | None = None) -> dict:
    """Standard paired-sourced two-level PCG (M2 SPD). Charges every HZ column."""
    cols = Z.shape[1] if Z.ndim == 2 else 0
    if cached_HZ is None:
        with fs.phase("A2_HZ"):
            HZ = fs.H_real(Z) if cols else np.zeros((fs.dim_material, 0))
    else:
        HZ = np.asarray(cached_HZ, dtype=float)
        if HZ.shape != Z.shape:
            raise ValueError("cached_HZ shape must match material basis Z")
    M2 = two_level_inverse(Z, HZ, fs.callback("A2_M0", fs.m0_action))
    if cached_h is not None:
        cached_h = np.asarray(cached_h, dtype=float)
        if cached_h.shape != (fs.dim_material,):
            raise ValueError("cached_h shape must match real material dimension")
    with fs.phase("A2_base"):
        h = fs.jt_real(fs.r) + fs.ell if cached_h is None else cached_h
    with fs.phase("A2_pcg"):
        cg = pcg(fs.H_real, -h, M2, x0=fs.s0, rtol=rtol, atol=atol,
                 maxiter=maxiter)
    return {"x": cg.x, "fallback": cg, "HZ": HZ, "material_rank": int(cols),
            "converged": bool(cg.converged)}


def api_signatures() -> dict:
    """Machine-readable API surface for the parent gate runner."""
    return {
        "factory": {
            "frozen_state_from_config": "(config, anchor, iteration, "
                                        "device=None, rank=None, lam=None, "
                                        "prior=None, diagnostics=True) -> "
                                        "FrozenState",
            "build_frozen_state": "(model, state, tangent, data, scale, lam, "
                                  "rank, prior=0.0, device='cpu', "
                                  "diagnostics=True) -> FrozenState",
        },
        "frozen_state": {
            "j_real": "(Z:(2q,[k])) -> (P*2m,[k])",
            "jt_real": "(W:(P*2m,[k])) -> (2q,[k])",
            "j0_real": "(Z) -> (P*2m,[k])",
            "j0t_real": "(W) -> (2q,[k])",
            "H_real": "J^T J + lam I",
            "h0_real": "J0^T J0 + lam I",
            "lambda_action": "lam * X",
            "m0_action": "H0^{-1} X (ReducedNormalInverse.solve)",
            "f0_action/f0t_action": "M0^{1/2} X from M0.Z",
            "full_normal_residual": "(s) -> ||J^T(r+Js)+lam s+ell||",
        },
        "pipeline": {
            "current_candidates": "(fs, kind='residual', rank=6, seed=...) -> "
                                  "(C, meta)",
            "current_transfer_factors": "(fs, C) -> {Yr, Tr, Gamma, gamma_s, ...}",
            "current_factor_map": "(fs, transfer, Z) -> Yr Tr Z",
            "paired_material_basis": "(fs, transfer, qmax=None, "
                                     "method='qr_svd'|'gramian', rtol=None, "
                                     "h0_normalize=True) -> (Z, meta)",
            "algorithm1_guarded_projected": "(fs, Z, rtol=1e-9, atol=1e-12, "
                                            "maxiter=300) -> dict",
            "algorithm2_two_level": "(fs, Z, rtol=1e-9, atol=1e-12, "
                                    "maxiter=300) -> dict",
        },
        "conventions": {
            "material_coordinates": "real 2q = tangent.d",
            "data_coordinates": "real P*2m stacked per illumination [Re; Im]",
            "lambda": "scalar positive Tikhonov/LM curvature, shared by J0 and J",
        },
    }


__all__ = [
    "FrozenState", "PhaseLedger", "PhysicalAction", "CGResult",
    "build_frozen_state", "frozen_state_from_config",
    "current_candidates", "current_transfer_factors", "current_factor_map",
    "paired_material_basis", "algorithm1_guarded_projected",
    "algorithm2_two_level", "full_normal_residual", "api_signatures",
    "guarded_projection", "two_level_inverse", "pcg",
    "paired_from_gramians", "paired_from_root_factors",
    "realify", "orth", "sym", "spd_power", "project_orth",
]
