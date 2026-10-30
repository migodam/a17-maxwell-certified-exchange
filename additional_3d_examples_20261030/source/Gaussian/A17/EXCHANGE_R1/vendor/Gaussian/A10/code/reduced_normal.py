"""Exact reusable normal inverse for an A10 anchored minimum-residual ROM.

The anchored derivative is complex-linear in the material perturbation

    c = x + i y,                     z = [x, y],
    Y_p(c) = CV F_p Q c / scale.

If ``CV = Uc Rc`` is an economic QR factorisation, ``Uc`` is an isometry and
the reduced data Hessian is represented exactly by

    Z = stack_p (Rc F_p Q) / scale,
    Hc = Z^H Z + lambda I.

This module never forms the full measurement-by-real-material Jacobian.  It
factors ``Hc`` directly for small material bases (for example the 27-complex-
coordinate Gaussian tangent), or applies Woodbury through the smaller
``lambda I + Z Z^H`` system for voxel tangents.
"""
from __future__ import annotations

import time

import numpy as np
from scipy import linalg as la


class ReducedNormalInverse:
    """Factor and apply ``(J_rom^* J_rom + lambda I)^{-1}`` in real coordinates.

    Parameters
    ----------
    rom
        Anchored ROM exposing ``F`` with shape ``(P, r, N)`` and ``CV`` with
        shape ``(m, r)``.
    tangent
        Real physical tangent exposing ``d`` and either a real matrix ``Q``
        (``N x q``), scalar cell ``volume`` (voxel expansion ``I/sqrt(v)``),
        or scalar grid spacing ``h`` (2-D pixel expansion ``I/h``).
    scale
        Positive data-whitening scale used by ``Ops``.
    lam
        Positive isotropic LM/Tikhonov curvature.
    branch
        ``"direct"``, ``"woodbury"``, or ``"auto"``.  Auto factors the
        smaller of the complex material and compressed measurement systems.
    """

    def __init__(self, rom, tangent, scale, lam, branch="auto"):
        started = time.perf_counter()
        self.scale = float(scale)
        self.lam = float(lam)
        if not np.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("scale must be finite and positive")
        if not np.isfinite(self.lam) or self.lam <= 0:
            raise ValueError("lam must be finite and positive")

        self.q = int(tangent.d) // 2
        if int(tangent.d) != 2 * self.q:
            raise ValueError("tangent.d must contain equal real and imaginary blocks")
        F = np.asarray(rom.F)
        CV = np.asarray(rom.CV)
        if F.ndim != 3 or CV.ndim != 2 or F.shape[1] != CV.shape[1]:
            raise ValueError("expected F=(P,r,N), CV=(m,r)")

        # ||CV a||_2 == ||Rc a||_2.  Qc need not be retained.
        _, Rc = la.qr(CV, mode="economic", check_finite=False)
        Q = getattr(tangent, "Q", None)
        if Q is not None:
            Q = np.asarray(Q)
            if Q.shape != (F.shape[2], self.q):
                raise ValueError("tangent.Q has incompatible shape")
            if np.iscomplexobj(Q) and np.max(np.abs(Q.imag)) > 1e-13:
                raise ValueError("complex tangent.Q is not supported by real Re/Im packing")
            blocks = [Rc @ (Fp @ Q.real) for Fp in F]
        else:
            if hasattr(tangent, "volume"):
                coordinate_scale = np.sqrt(float(tangent.volume))
            elif hasattr(tangent, "h"):
                coordinate_scale = float(tangent.h)
            else:
                raise ValueError("tangent needs Q, volume, or h")
            if not np.isfinite(coordinate_scale) or coordinate_scale <= 0:
                raise ValueError("physical coordinate scale must be finite and positive")
            if F.shape[2] != self.q:
                raise ValueError("identity physical tangent has incompatible material size")
            blocks = [Rc @ Fp / coordinate_scale for Fp in F]
        self.Z = np.concatenate(blocks, axis=0) / self.scale
        self.rows = int(self.Z.shape[0])
        if self.Z.shape[1] != self.q:
            raise RuntimeError("internal compressed Jacobian shape mismatch")

        if branch == "auto":
            branch = "direct" if self.q <= self.rows else "woodbury"
        if branch not in ("direct", "woodbury"):
            raise ValueError("branch must be 'auto', 'direct', or 'woodbury'")
        self.branch = branch
        if branch == "direct":
            Hc = self.Z.conj().T @ self.Z
            Hc.flat[:: self.q + 1] += self.lam
            self._factor = la.cho_factor(Hc, lower=True, check_finite=False)
            self.factor_dimension = self.q
        else:
            W = self.Z @ self.Z.conj().T
            W.flat[:: self.rows + 1] += self.lam
            self._factor = la.cho_factor(W, lower=True, check_finite=False)
            self.factor_dimension = self.rows
        self.build_s = time.perf_counter() - started

    def _pack(self, z):
        z = np.asarray(z, dtype=float)
        if z.shape[0] != 2 * self.q:
            raise ValueError("real vector has incompatible leading dimension")
        return z[: self.q] + 1j * z[self.q :]

    def _unpack(self, c):
        return np.concatenate([c.real, c.imag], axis=0)

    def matvec(self, z):
        """Apply the exact reduced real normal Hessian."""
        c = self._pack(z)
        return self._unpack(self.Z.conj().T @ (self.Z @ c) + self.lam * c)

    def solve(self, rhs):
        """Solve the reduced normal system for one or more real RHS columns."""
        b = self._pack(rhs)
        if self.branch == "direct":
            x = la.cho_solve(self._factor, b, check_finite=False)
        else:
            correction = self.Z.conj().T @ la.cho_solve(
                self._factor, self.Z @ b, check_finite=False
            )
            x = (b - correction) / self.lam
        return self._unpack(x)

    def check_action(self, op, trials=3, seed=0):
        """Compare against ``op.v(op.j(z)) + lambda*z`` on seeded probes."""
        rng = np.random.default_rng(seed)
        errors = []
        for _ in range(int(trials)):
            z = rng.normal(size=2 * self.q)
            reference = op.v(op.j(z)) + self.lam * z
            errors.append(
                float(la.norm(self.matvec(z) - reference) / max(la.norm(reference), 1e-300))
            )
        return {
            "trials": int(trials),
            "max_relative": max(errors, default=0.0),
            "relative": errors,
            "branch": self.branch,
            "complex_material_dim": self.q,
            "compressed_measurement_dim": self.rows,
            "factor_dimension": self.factor_dimension,
            "build_s": self.build_s,
        }

