"""Online-only FP64 numerical significance, not a Maxwell error certificate.

The API accepts residuals, paid directional outputs and the common prior only.
It has no optimum, truth, evaluator, reference-floor or teacher argument.
The scale includes gain terms and the energy of the current base point, so a
near-zero difference is not assigned a spurious zero rounding tolerance.
"""
import numpy as np

SCHEMA = 'a17.online.numerical_significance.v1'
SAFETY = 32.0


def directional_check(r, u, z, x, d, ell, lam):
    """Return signed gain, online scale and threshold for columns of d/z.

    The experiment uses realified shared-material variables and scalar lam.
    32*gamma_N is a declared rounding-significance rule, not a propagated
    current-solve/output-error bound. Solver quality is checked separately.
    """
    if np.iscomplexobj(r):
        raise ValueError('residual must use the declared realified layout')
    r = np.asarray(r, dtype=np.float64)
    u = np.asarray(u)
    z = np.asarray(z)
    x = np.asarray(x)
    d = np.asarray(d)
    ell = np.asarray(ell)
    if any(np.iscomplexobj(v) for v in (x, d, ell)):
        raise ValueError('material coordinates must be real')
    if d.ndim not in (1,2) or z.ndim not in (1,2):
        raise ValueError('directions and outputs must be vectors or column blocks')
    if d.ndim == 1:
        d = d[:, None]
    if z.ndim == 1:
        z = z[:, None]
    if r.ndim != 1 or u.shape != r.shape or x.ndim != 1 or ell.shape != x.shape:
        raise ValueError('base residual/material shape mismatch')
    if d.shape[0] != len(x) or z.shape != (len(u), d.shape[1]):
        raise ValueError('direction shape mismatch')
    if not np.isfinite(lam) or lam <= 0 or not all(np.all(np.isfinite(v)) for v in (r,u,z,x,d,ell)):
        raise ValueError('finite online inputs and positive common lam required')
    f = np.finfo(np.float64)
    n = len(r) + 3*len(x) + 16
    gamma = (n*f.eps)/(1.0-n*f.eps)
    # Realification makes the original complex receiver inner product a real
    # dot product. The complex form is also accepted for output-only unit tests.
    obs = np.real(np.conj(u) @ z)
    prior_linear = (ell + lam*x) @ d
    data_cost = .5*np.sum(np.abs(z)**2, axis=0)
    prior_cost = .5*lam*np.sum(d*d, axis=0)
    gain = -obs-prior_linear-data_cost-prior_cost
    abs_terms = (np.abs(u) @ np.abs(z) +
                 np.abs(ell+lam*x) @ np.abs(d) + data_cost + prior_cost)
    base_scale = (float(np.vdot(r,r).real) + float(np.vdot(u,u).real) +
                  abs(float(ell @ x)) + lam*float(x @ x))
    scale = abs_terms + base_scale
    tau_abs = SAFETY*f.tiny
    tau_rel = SAFETY*gamma
    tau = tau_abs + tau_rel*scale
    return dict(schema=SCHEMA, gain=gain, scale=scale, tau=tau,
                tau_abs=tau_abs, tau_rel=tau_rel, accumulation_length=n,
                observation_term=obs, prior_linear_term=prior_linear,
                data_curvature_cost=data_cost, prior_curvature_cost=prior_cost,
                deterministic_certificate=False)
