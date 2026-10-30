"""Strict anchored Galerkin current ROM for the A12 algebraic contract.

Shapes
------
V: (n_current, r) complex, V^H V = I
material perturbation dc: (N,) or (N, batch) complex
current_jvp(dc): (P, n_current) or (P, n_current, batch)
jvp(dc): (P, n_data) or (P, n_data, batch)
vjp(w): w=(P, n_data[, batch]) complex -> (N[, batch]) complex
F: (P, r, N), CV: (n_data, r)

The reduced inverse is exactly R0 = V (V^H L V)^-1 V^H.  This is not
the A10 anchored minimum-residual inverse V (LV)^dagger.
"""
from __future__ import annotations

import time
import numpy as np
from scipy import linalg as la


class GalerkinROM:
    """Frozen-full-state injection with a strict Galerkin current inverse."""

    def __init__(self, state, V):
        started = time.perf_counter()
        if state.reduced:
            raise ValueError("GalerkinROM requires a full anchor state")
        V = np.asarray(V, dtype=complex)
        if V.ndim != 2 or V.shape[0] != state.model.n:
            raise ValueError("V must have shape (n_current, r)")
        r = V.shape[1]
        if r == 0:
            raise ValueError("V must contain at least one column")
        orth_error = la.norm(V.conj().T @ V - np.eye(r))
        if orth_error > 1e-8:
            raise ValueError(f"V is not orthonormal: {orth_error:g}")

        self.state = state
        self.model = state.model
        self.counters = state.counters
        self.basis = V.copy()
        self.rstate = r
        self.LV = self.apply_L(self.basis)
        self.Lr = self.basis.conj().T @ self.LV
        singular = la.svdvals(self.Lr)
        if singular[-1] <= 1e-13 * max(singular[0], 1.0):
            raise ValueError("V^H L V is singular or numerically rank deficient")
        self._factor = la.lu_factor(self.Lr, check_finite=False)

        # T = (V^H L V)^-1 V^H maps a current RHS to Galerkin coefficients.
        T = la.lu_solve(self._factor, self.basis.conj().T, check_finite=False)
        E = (
            state.exciting.reshape(self.model.P, self.model.N, 3)
            * state.da[None, :, None]
            / state.sqrt_v
        )
        self.F = np.einsum(
            "rnc,pnc->prn", T.reshape(r, self.model.N, 3), E, optimize=True
        )
        self.CV = self.model.GS @ self.basis
        self.build_s = time.perf_counter() - started
        self.counters.bump("a12_galerkin_rom_build")
        self.counters.add_time("a12_galerkin_rom_build_s", self.build_s)

    @property
    def field(self):
        return self.state.field

    @property
    def current(self):
        return self.state.current

    def apply_R(self, rhs):
        """Apply R0 to rhs=(n_current,) or (n_current, batch)."""
        rhs = np.asarray(rhs, dtype=complex)
        one = rhs.ndim == 1
        rhs2 = rhs[:, None] if one else rhs
        if rhs2.ndim != 2 or rhs2.shape[0] != self.model.n:
            raise ValueError("rhs must have leading dimension n_current")
        coeff = la.lu_solve(
            self._factor, self.basis.conj().T @ rhs2, check_finite=False
        )
        out = self.basis @ coeff
        return out[:, 0] if one else out

    def apply_R_adjoint(self, rhs):
        """Apply R0^H = V (V^H L^H V)^-1 V^H."""
        rhs = np.asarray(rhs, dtype=complex)
        one = rhs.ndim == 1
        rhs2 = rhs[:, None] if one else rhs
        if rhs2.ndim != 2 or rhs2.shape[0] != self.model.n:
            raise ValueError("rhs must have leading dimension n_current")
        coeff = la.lu_solve(
            self._factor,
            self.basis.conj().T @ rhs2,
            trans=2,
            check_finite=False,
        )
        out = self.basis @ coeff
        return out[:, 0] if one else out

    def _apply_L(self, rhs, adjoint=False):
        """Apply the anchor-local L or L^H, batching all RHS columns.

        On CUDA the dense matrix is uploaded once and cached on this exact
        ``state`` object.  The cache is therefore impossible to reuse across
        material anchors.  Results remain NumPy arrays for compatibility with
        the existing A9/A10 interfaces.
        """
        rhs = np.asarray(rhs, dtype=complex)
        one = rhs.ndim == 1
        rhs2 = rhs[:, None] if one else rhs
        if rhs2.ndim != 2 or rhs2.shape[0] != self.model.n:
            raise ValueError("L action rhs must have leading dimension n_current")
        started = time.perf_counter()
        if self.model.device == "cuda":
            torch = self.model.solver.torch
            device = self.model.solver.torch_device
            cache_name = "_a12_L_cuda"
            matrix = getattr(self.state, cache_name, None)
            if matrix is None:
                matrix = torch.as_tensor(
                    self.state.L, dtype=torch.complex128, device=device
                )
                setattr(self.state, cache_name, matrix)
                self.counters.bump("a12_L_cuda_uploads")
            source = torch.as_tensor(rhs2, dtype=torch.complex128, device=device)
            result = ((matrix.mH if adjoint else matrix) @ source).detach().cpu().numpy()
        else:
            matrix = self.state.L.conj().T if adjoint else self.state.L
            result = matrix @ rhs2
        label = "a12_L_adjoint_matvec_rhs" if adjoint else "a12_L_matvec_rhs"
        timing = "a12_L_adjoint_matvec_s" if adjoint else "a12_L_matvec_s"
        self.counters.bump(label, rhs2.shape[1])
        self.counters.add_time(timing, time.perf_counter() - started)
        return result[:, 0] if one else result

    def apply_L(self, rhs):
        return self._apply_L(rhs, adjoint=False)

    def apply_L_adjoint(self, rhs):
        return self._apply_L(rhs, adjoint=True)

    def coeff(self, dc):
        dc = np.asarray(dc)
        if dc.ndim not in (1, 2):
            raise ValueError("dc must have shape (N,) or (N, batch)")
        self.counters.bump(
            "a12_reduced_tangent_rhs",
            self.model.P * (1 if dc.ndim == 1 else dc.shape[1]),
        )
        return np.einsum(
            "prn,nb->prb", self.F, dc[:, None] if dc.ndim == 1 else dc, optimize=True
        )

    def current_jvp(self, dc):
        z = self.coeff(dc)
        out = np.einsum("nr,prb->pnb", self.basis, z, optimize=True)
        return out[:, :, 0] if np.asarray(dc).ndim == 1 else out

    def jvp(self, dc):
        z = self.coeff(dc)
        out = np.einsum("mr,prb->pmb", self.CV, z, optimize=True)
        return out[:, :, 0] if np.asarray(dc).ndim == 1 else out

    def vjp(self, w):
        w = np.asarray(w, dtype=complex)
        if w.ndim == 2:
            if w.shape != (self.model.P, self.model.m):
                raise ValueError("w must have shape (P, n_data)")
            self.counters.bump("a12_reduced_adjoint_rhs", self.model.P)
            c = np.einsum("mr,pm->pr", self.CV.conj(), w, optimize=True)
            return np.einsum("prn,pr->n", self.F.conj(), c, optimize=True)
        if w.ndim == 3:
            if w.shape[:2] != (self.model.P, self.model.m):
                raise ValueError("batched w must have shape (P, n_data, batch)")
            self.counters.bump("a12_reduced_adjoint_rhs", self.model.P * w.shape[2])
            c = np.einsum("mr,pmb->prb", self.CV.conj(), w, optimize=True)
            return np.einsum("prn,prb->nb", self.F.conj(), c, optimize=True)
        raise ValueError("w must have shape (P, n_data) or (P, n_data, batch)")

    def tangent_rhs(self, dc):
        rhs, _, single = self.state._tangent_rhs(dc)
        return rhs[:, :, 0] if single else rhs

    def tangent_residual(self, dc):
        k = self.current_jvp(dc)
        rhs = self.tangent_rhs(dc)
        if k.ndim == 2:
            return rhs - self.apply_L(k.T).T
        P, n, batch = k.shape
        Lk = self.apply_L(k.transpose(1, 0, 2).reshape(n, P * batch))
        return rhs - Lk.reshape(n, P, batch).transpose(1, 0, 2)
