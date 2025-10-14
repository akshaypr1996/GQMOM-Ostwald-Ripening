#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lognormal–GQMOM Pt Ostwald-Ripening (OR) — SIMPLE SCRIPT
========================================================



------------
• Evolves the first five raw moments M0..M4 of f(r,t) on r ∈ [R_MIN, ∞)
  under a PBE with growth G(r, Cpt) and a boundary flux at r = R_MIN.
• Closes interior integrals with a Gauss quadrature (GQMOM) on R+ using
  a Stieltjes–Wigert (lognormal) extension.
• Estimates f(R_MIN) with a lognormal-kernel smoother tied to
  η = sqrt(M2*M0)/M1 = exp(σ_ln^2/2).
• Plots: M0(t), Cpt(t), σ_ln(t), r̄_N(t), S_N(t), M_N(t), node snapshots,
  and reconstructed number density (initial vs final).


References
----------
• Fox, Laurent, Passalacqua (2023): GQMOM on R+ (Stieltjes–Wigert)
• Lage (2011); Yuan et al. (2012): moment transport + boundary terms
• Baroody, H. (2018). Dynamic modelling of platinum degradation in polymer electrolyte fuel cells.
  PhD Thesis, Simon Fraser University. https://summit.sfu.ca/item/18647
  — MSAC initial PRD treated as log-normal; we use μ = 0.612 and σ = 0.460 to set M0..M4.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

# ===================== User inputs & constants =====================

# Initial raw moments (M0..M4)
# Initial raw moments (M0..M4) are set from the MSAC log-normal PRD parameters
# (μ = 0.612, σ = 0.460) reported in:
# Baroody, H. (2018). Dynamic modelling of platinum degradation in polymer
# electrolyte fuel cells. PhD Thesis, Simon Fraser University. https://summit.sfu.ca/item/18647

M_init = [1.0, 2.0499, 5.1924, 16.2517, 62.8531]


# Electrolyte: initial dissolved Pt (mol m^-3)
CPT0 = 0.0

# Physical / kinetic constants
Vm = 9.09e-6            # m^3 mol^-1
krdp = 1e-12             # m s^-1
kdis = 6.309e-12        # m s^-1
Cpt_ref = 1.0           # mol m^-3
R0 = 14.1545268876485   # nm (characteristic radius parameter)

# Catalyst-layer / coupling constant (units → see thesis notes)
m_u = 0.004             # kg m^-2 (0.4 mg cm^-2)
L_CL = 10e-6            # m
m_V = m_u / L_CL        # kg m^-3
M_Pt = 195.084e-3       # kg mol^-1
I_v = 0.3               # –
IV_MU_over_MPT = I_v * m_V / M_Pt  # coupling for Cpt(t)

# GQMOM controls
N_GQMOM = 10                 # number of quadrature nodes (keep small/safe)
R_MIN, R_MAX = 1.4, 30.0     # nm (R_MAX only for plotting)

# Time integration
T_FINAL_H = 4.0              # hours
LOG_EVERY_H = 0.001          # hours

# For normalization
M3_INITIAL = M_init[3]

# ===================== Model building blocks =====================

def growth_function(r, c_pt):
    """Growth law G(r, Cpt) in nm/s. +G: growth; −G: shrinkage."""
    r = np.asarray(r, dtype=float)
    return Vm * 1e9 * (
        krdp * c_pt * np.exp(-R0 / r)     # redeposition
        - kdis * Cpt_ref * np.exp(R0 / r) # dissolution
    )


def hankel2(M):
    """Hankel 2×2 determinant for realizability on R+."""
    M0, M1, M2 = M[:3]
    return M0 * M2 - M1 * M1


def hankel4(M):
    """Hankel 3×3 determinant for realizability on R+."""
    M0, M1, M2, M3, M4 = M[:5]
    a = M0 * (M2 * M4 - M3 * M3)
    b = -M1 * (M1 * M4 - M2 * M3)
    c = M2 * (M1 * M3 - M2 * M2)
    return a + b + c


def check_realizable_Rplus(M, eps=1e-14):
    """Loose realizability check (allows tiny negatives from roundoff)."""
    H2 = hankel2(M)
    H4 = hankel4(M)
    return (M[0] > eps) and (H2 > -1e-12) and (H4 > -1e-12)


def base_recurrence_from_moments(M):
    """Compute base monic 3-term recurrence (a0, a1, b1, b2) from M0..M4 for R+."""
    M0, M1, M2, M3, M4 = M
    if M0 <= 0 or not np.all(np.isfinite(M)):
        raise RuntimeError("Invalid moments: M0<=0 or NaN/Inf")

    H2_raw = hankel2(M)
    H4_raw = hankel4(M)
    H2 = max(H2_raw, 1e-18)
    H4 = max(H4_raw, 1e-18)

    a0 = M1 / M0
    b1 = max(H2 / (M0 * M0), 1e-18)

    den = M2 - 2.0 * a0 * M1 + (a0 ** 2) * M0
    num = M3 - 2.0 * a0 * M2 + (a0 ** 2) * M1
    a1 = a0 if den <= 1e-18 else num / den

    b2 = (H4 * M0) / (H2 * H2)
    b2 = max(b2 if np.isfinite(b2) else 0.0, 1e-18)
    return a0, a1, b1, b2


def ab_to_zeta4(a0, a1, b1, b2):
    """Map (a,b) → positive ζ parameters (R+ support)."""
    z1 = a0
    if z1 <= 0:
        raise RuntimeError("ζ1 <= 0; a0 must be > 0 for R+.")
    z2 = b1 / z1
    z3 = a1 - z2
    if z3 <= 0:
        z3 = 1e-12 if z3 > -1e-12 else (_ for _ in ()).throw(RuntimeError("ζ3 <= 0"))
    z4 = b2 / z3
    if z4 <= 0:
        z4 = 1e-12 if z4 > -1e-12 else (_ for _ in ()).throw(RuntimeError("ζ4 <= 0"))
    return z1, z2, z3, z4


def eta_from_moments(M):
    """η = sqrt(M2*M0)/M1 = exp(σ_ln^2/2) > 1."""
    M0, M1, M2 = M[:3]
    if M1 <= 0:
        raise RuntimeError("M1 must be > 0 to compute η.")
    eta = np.sqrt((M2 * M0) / (M1 * M1))
    return max(eta, 1.0 + 1e-12)


def extend_log_gqmom_zeta(z1, z2, z3, z4, eta, N):
    """Stieltjes–Wigert (lognormal) extension; produces ζ up to ζ_{2N−1}."""
    Z = {1: z1, 2: z2, 3: z3, 4: z4}
    denom = eta**4 - 1.0
    near_unity = abs(denom) < 1e-12
    for i in range(3, N):
        Z[2*i - 1] = max((eta**(4*(i-2))) * Z[3], 1e-18)
        ratio = (i/2.0) if near_unity else (eta**(2*i) - 1.0)/denom
        Z[2*i]     = max((eta**(2*(i-2))) * ratio * Z[4], 1e-18)
    Z[2*N - 1] = max((eta**(4*(N-2))) * Z[3], 1e-18)
    return Z


def recurrence_from_zeta(Z, N, a0, a1, b1, b2):
    """Build full a[i], b[i] (i=0..N-1) from ζ and base coefficients."""
    a = np.zeros(N)
    b = np.zeros(N)
    a[0] = a0
    a[1] = a1
    b[1] = b1
    b[2] = b2
    for i in range(2, N):
        zi_even = Z[2*i]
        zi_odd_next = Z.get(2*i + 1, 1e-18)
        a[i] = zi_even + zi_odd_next
        b[i] = max(Z[2*i - 1] * Z[2*i], 1e-18)
    return a, b


def quadrature_from_recurrence(a, b, M0):
    """Jacobi matrix eigen-decomposition → abscissas x, weights w."""
    d = a.copy()
    e = np.sqrt(b[1:])
    J = np.diag(d) + np.diag(e, k=1) + np.diag(e, k=-1)
    evals, evecs = np.linalg.eigh(J)
    x = evals.astype(float)
    w = (evecs[0, :] ** 2) * M0
    return x, w


def gqmom_nodes_weights(M, N):
    """moments → (x,w,η) via lognormal–GQMOM on R+."""
    a0, a1, b1, b2 = base_recurrence_from_moments(M)
    z1, z2, z3, z4 = ab_to_zeta4(a0, a1, b1, b2)
    eta = eta_from_moments(M)
    Z = extend_log_gqmom_zeta(z1, z2, z3, z4, eta, N)
    a, b = recurrence_from_zeta(Z, N, a0, a1, b1, b2)
    x, w = quadrature_from_recurrence(a, b, M[0])
    return x, w, eta, (a, b, Z)


def mixture_pdf_lognormal_at_point(r, x_nodes, w_nodes, eta):
    """Estimate f(r) using a lognormal kernel with σ_ln from η (for boundary term)."""
    r = float(r)
    tiny = 1e-300
    sigma = np.sqrt(2.0 * np.log(eta))
    if not np.isfinite(sigma) or sigma <= 0:
        return 0.0
    ln_r = np.log(r)
    ln_norm = -(ln_r + np.log(sigma) + 0.5 * np.log(2.0 * np.pi))
    ln_x = np.log(np.maximum(x_nodes, tiny))
    z = - (ln_r - ln_x) ** 2 / (2.0 * sigma * sigma) + np.log(np.maximum(w_nodes, tiny))
    m = np.max(z)
    s = np.exp(z - m).sum()
    return float(np.exp(ln_norm + m) * s)

# ===================== ODE RHS (moments + Cpt) =====================

def rhs(t, Y):
    """
    Right-hand side for [M0..M4, Cpt].
      dM_k/dt = + R_MIN^k G(R_MIN) f(R_MIN)
                + k Σ w_j x_j^{k-1} G(x_j, Cpt)   (x_j ≥ R_MIN)
      dCpt/dt = - (IV_MU_over_MPT) * (dM_3/dt) / M3_INITIAL
    """
    M = np.array(Y[:5], dtype=float)
    c_pt = float(Y[5])

    if not check_realizable_Rplus(M):
        M = 0.99999999 * M + 1.0e-8 * np.array(M_init, dtype=float)

    x, w, eta, _ = gqmom_nodes_weights(M, N_GQMOM)

    mask = x >= R_MIN
    x_eff = x[mask]
    w_eff = w[mask]

    f_lo = mixture_pdf_lognormal_at_point(R_MIN, x, w, eta)
    G_min = growth_function(R_MIN, c_pt)

    dM = []
    Gx = growth_function(x_eff, c_pt) if x_eff.size else np.array([])
    for k in range(5):
        interior = np.sum(w_eff * (x_eff ** (k - 1)) * Gx) if (x_eff.size and k > 0) else 0.0
        flux_k = (R_MIN ** k) * f_lo * G_min
        dM.append(k * interior + flux_k)

    dMdis_dt = dM[3] / M3_INITIAL
    dCpt_dt = -IV_MU_over_MPT * dMdis_dt
    return dM + [dCpt_dt]

# ===================== Driver & plotting =====================

def main():
    state0 = M_init + [CPT0]
    t_eval = np.arange(0.0, T_FINAL_H * 3600.0 + 1.0, LOG_EVERY_H * 3600.0)

    sol = solve_ivp(
        rhs,
        (0.0, T_FINAL_H * 3600.0),
        state0,
        method="BDF",
        rtol=1e-6, atol=1e-6,
        max_step=60,
        t_eval=t_eval
    )
    if not sol.success:
        print("Solver stopped:", sol.message)
        return

    times_s = sol.t
    t_hr = times_s / 3600.0
    M_hist = sol.y[:5]
    Cpt_hist = sol.y[5]

    # Reconstruct nodes/weights for plotting
    eta_hist, sigma_hist = [], []
    X_hist = np.full((N_GQMOM, len(t_hr)), np.nan)
    W_hist = np.full((N_GQMOM, len(t_hr)), np.nan)
    for i in range(len(t_hr)):
        Mi = M_hist[:, i]
        try:
            x, w, eta, _ = gqmom_nodes_weights(Mi, N_GQMOM)
            X_hist[:, i] = x
            W_hist[:, i] = w
            eta_hist.append(eta)
            sigma_hist.append(np.sqrt(2.0 * np.log(eta)))
        except Exception:
            eta_hist.append(np.nan)
            sigma_hist.append(np.nan)
    eta_hist = np.array(eta_hist)
    sigma_hist = np.array(sigma_hist)

    # Normalized diagnostics
    rN = (M_hist[1] / M_hist[0]) / (M_hist[1, 0] / M_hist[0, 0])
    SN = M_hist[2] / M_hist[2, 0]
    MN = M_hist[3] / M_hist[3, 0]

    # === Time-series plots
    plt.figure(); plt.plot(t_hr, M_hist[0], lw=2)
    plt.title("Zeroth moment $M_0(t)$"); plt.xlabel("time [h]"); plt.ylabel("–"); plt.grid()

    plt.figure(); plt.plot(t_hr, Cpt_hist, lw=2)
    plt.title("Dissolved Pt concentration $C_{Pt}(t)$"); plt.xlabel("time [h]"); plt.ylabel("mol m$^{-3}$"); plt.grid()

    plt.figure(); plt.plot(t_hr, sigma_hist, marker='.')
    plt.title("Implied log-normal width $\\sigma_{\\ln}(t)$ from $\\eta$")
    plt.xlabel("time [h]"); plt.ylabel("$\\sigma_{\\ln}$"); plt.grid()

    plt.figure(); plt.plot(t_hr, rN, lw=2)
    plt.title(r"Normalised mean radius $\bar{r}_N(t)$"); plt.xlabel("time [h]"); plt.ylabel("–"); plt.grid()

    plt.figure(); plt.plot(t_hr, 100 * SN, lw=2)
    plt.title("Normalised surface area $S_N(t)$"); plt.xlabel("time [h]"); plt.ylabel("% of initial"); plt.grid()

    plt.figure(); plt.plot(t_hr, 100 * MN, lw=2)
    plt.title("Normalised mass $M_N(t)$"); plt.xlabel("time [h]"); plt.ylabel("% of initial"); plt.grid()

    # === Node snapshots (log–log)
    snapshot_times = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]

    plt.figure()
    for t_snap in snapshot_times:
        idx = np.argmin(np.abs(t_hr - t_snap))
        x_snap = X_hist[:, idx]; w_snap = W_hist[:, idx]
        m = (x_snap >= R_MIN) & np.isfinite(x_snap) & np.isfinite(w_snap) & (w_snap > 1e-18)
        plt.scatter(x_snap[m], w_snap[m], s=60, label=f"{t_snap:.1f} h", alpha=0.85)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r'abscissa $x_j$ [nm] (log)'); plt.ylabel(r'node weight $w_j$ (log)')
    plt.title("Snapshots of GQMOM nodes (raw weights, log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5); plt.legend()

    plt.figure()
    for t_snap in snapshot_times:
        idx = np.argmin(np.abs(t_hr - t_snap))
        x_snap = X_hist[:, idx]; w_snap = W_hist[:, idx]
        m = (x_snap >= R_MIN) & np.isfinite(x_snap) & np.isfinite(w_snap) & (w_snap > 1e-18)
        plt.scatter(x_snap[m], (w_snap[m] / M_hist[0, idx]), s=60, label=f"{t_snap:.1f} h", alpha=0.85)
    plt.xscale("log"); plt.yscale("log")
    plt.xlabel(r'abscissa $x_j$ [nm] (log)'); plt.ylabel(r'weight fraction $w_j/M_0$ (log)')
    plt.title("Snapshots of GQMOM nodes (weight fractions, log–log)")
    plt.grid(True, which="both", ls="--", lw=0.5); plt.legend()

    # === Reconstructed pdf snapshots (initial vs final)
    def mixture_pdf_stable_many(r, x_nodes, w_nodes, eta):
        r = np.asarray(r, dtype=float)
        tiny = 1e-300
        sigma = np.sqrt(2.0 * np.log(eta))
        if not np.isfinite(sigma) or sigma <= 0:
            return np.zeros_like(r)
        ln_r = np.log(np.maximum(r, tiny))
        ln_x = np.log(np.maximum(x_nodes, tiny))[:, None]
        ln_norm = -(ln_r + np.log(sigma) + 0.5 * np.log(2.0 * np.pi))
        Z = - (ln_r[None, :] - ln_x) ** 2 / (2.0 * sigma * sigma) + np.log(np.maximum(w_nodes, tiny))[:, None]
        m = np.max(Z, axis=0)
        s = np.exp(Z - m).sum(axis=0)
        return np.exp(ln_norm + m) * s

    try:
        i0, iT = 0, len(t_hr) - 1
        x0, w0, eta0, _ = gqmom_nodes_weights(M_hist[:, i0], N_GQMOM)
        xT, wT, etaT, _ = gqmom_nodes_weights(M_hist[:, iT], N_GQMOM)
        r_grid = np.linspace(max(1e-6, R_MIN * 1e-3), 10.0, 1000)
        f0 = mixture_pdf_stable_many(r_grid, x0, w0, eta0)
        fT = mixture_pdf_stable_many(r_grid, xT, wT, etaT)
        plt.figure()
        plt.plot(r_grid, f0, lw=2, label=fr"initial ($t={t_hr[i0]:.2f}\,h$)")
        plt.plot(r_grid, fT, lw=2, label=fr"final ($t={t_hr[iT]:.2f}\,h$)")
        plt.axvline(R_MIN, ls='--', lw=1, label=r"$R_{\min}$")
        plt.title("Number density (smoothed): initial vs final")
        plt.xlabel("radius r [nm]"); plt.ylabel(r"$f(r)$ [# per nm]")
        plt.xlim(0.0, 10.0)
        plt.grid(True); plt.legend()
    except Exception:
        pass

    plt.show()


if __name__ == "__main__":
    main()
