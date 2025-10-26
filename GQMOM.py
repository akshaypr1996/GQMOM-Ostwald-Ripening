#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Lognormal–GQMOM Pt Ostwald-Ripening (OR)
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
from doctest import debug
from operator import le
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import momentInversion as momentInversion

# ===================== User inputs & constants =====================

# Initial raw moments (M0..M4)
# Initial raw moments (M0..M4) are set from the MSAC log-normal PRD parameters
# (μ = 0.612, σ = 0.460) reported in:
# Baroody, H. (2018). Dynamic modelling of platinum degradation in polymer
# electrolyte fuel cells. PhD Thesis, Simon Fraser University. https://summit.sfu.ca/item/18647

#M_init = [1.0, 2.0499, 5.1924, 16.2517, 62.8531]

# AP: Replaced with direct calculation of lognormal moments from the moments' formula
# Define an array of the first 5 lognormal moments with mu = 0.612 and sigma = 0.460
m = 0.612
s = 0.460
N_Initial_Moments = 5
M_init = [np.exp(k * m + 0.5 * k * k * s * s) for k in range(N_Initial_Moments)]
M_init[0] = 1.0

# Electrolyte: initial dissolved Pt (mol m^-3)
CPT0 = 0.0

# Physical / kinetic constants
Vm = 9.09e-6            # m^3 mol^-1
krdp = 1e-12            # m s^-1
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

# AP: Added controls for GQMOM inversion on R+
# GQMOM controls

# Type of distribution for GQMOM on R+: gamma or lognormal
distributionType = "lognormal"

# AP: Number of generalized quadrature nodes: the number of nodes is used in the
# GQMOM inversion. The number of moments used in the base QMOM is found from
# the realizability criterion, and automatically determined in 
# momentsToZetaWheeler, defined in momentInversion.py
N_GQMOM = 10

# AP: Minimum zeta value for to establish positivity of the zeta_k quantities.
# This must be small and positive. Setting to zero is acceptable if no 
# realizability issues are observed. Negative values are not acceptable.
# Below this value, zeta_k is set to zero, and the number of moments
# in the interior of the moment space is set accordingly.
smallZeta = 1.0e-14

# AP: Flag to enable or disable printing debug information in GQMOM
debugGQMOM = True   # debug prints for GQMOM inversion

R_MIN = 0.5         # nm
R_MAX = 10.0        # nm (for plotting)

# Time integration
T_FINAL_H = 4            # hours
LOG_EVERY_H = 0.001      # hours

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

def calculate_moments_from_quadrature(weights, abscissae, max_moment_order):
    """Calculate raw moments up to max_moment_order from GQMOM nodes."""
    moments = []
    for k in range(max_moment_order + 1):
        mk = np.sum(weights * (abscissae ** k))
        moments.append(mk)
    return moments

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

    w, x, eta, _ = momentInversion.WheelerGQMOMRPlus(M, N_GQMOM, type=distributionType, smallZeta=smallZeta, debug=debugGQMOM)

    mask = x >= R_MIN
    x_eff = x[mask]
    w_eff = w[mask]

    # AP: Recalculate moments from effective nodes to keep consistency with
    #     the quadrature after applying the mask
    M = calculate_moments_from_quadrature(w_eff, x_eff, len(M) - 1)

    # Print moments
    print(f"t = {t/3600:.3f} h: Moments = {M}")
    # Print quadrature nodes
    print(f"  Nodes (x_j, w_j):")
    for wj, xj in zip(w_eff, x_eff):
        print(f"    ({wj:.6e}, {xj:.6f} nm)")

    # AP: Masked weights and abscissae need to be used also here
    f_lo = mixture_pdf_lognormal_at_point(R_MIN, x_eff, w_eff, eta)
    G_min = growth_function(R_MIN, c_pt)

    dM = []
    Gx = growth_function(x_eff, c_pt) if x_eff.size else np.array([])

    # The flux used here should be computed with the quadrature-based
    # distribution, instead of assuming a lognormal, which works around the 
    # QBMM approach. I.e., replace the delta function approximation in the
    # integrals and obtain the expression of the flux in terms of weights and
    # abscissae
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
        max_step=10,
        t_eval=t_eval
    )

    if not sol.success:
        print("Solver stopped:", sol.message)
        return

    times_s = sol.t
    t_hr = times_s / 3600.0
    M_hist = sol.y[:5]
    Cpt_hist = sol.y[5]

    # === Time-series plots

    # Plot all moment histories in one graph
    plt.figure()
    for k in range(5):
        plt.plot(t_hr, M_hist[k], lw=2, label=f"$M_{k}$")
    plt.title("Moment histories $M_0$ to $M_4$"); 
    plt.legend(); plt.xlabel("time [h]"); plt.ylabel("–"); plt.grid()

    # Reconstruct nodes/weights for plotting
    eta_hist, sigma_hist = [], []
    X_hist = np.full((N_GQMOM, len(t_hr)), np.nan)
    W_hist = np.full((N_GQMOM, len(t_hr)), np.nan)

    for i in range(len(t_hr)):
        Mi = M_hist[:, i]
        try:
            w, x, eta, _ = momentInversion.WheelerGQMOMRPlus(Mi, N_GQMOM, type=distributionType)
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

    #2
    plt.figure(); 
    plt.plot(t_hr, Cpt_hist, lw=2)
    plt.title("Dissolved Pt concentration $C_{Pt}(t)$"); 
    plt.xlabel("time [h]"); 
    plt.ylabel("mol m$^{-3}$"); 
    plt.grid()

    #3
    plt.figure(); 
    plt.plot(t_hr, sigma_hist, marker='.')
    plt.title("Implied log-normal width $\\sigma_{\\ln}(t)$ from $\\eta$")
    plt.xlabel("time [h]"); 
    plt.ylabel("$\\sigma_{\\ln}$"); 
    plt.grid()

    #4
    plt.figure(); 
    plt.plot(t_hr, rN, lw=2)
    plt.title(r"Normalised mean radius $\bar{r}_N(t)$"); 
    plt.xlabel("time [h]"); 
    plt.ylabel("–"); 
    plt.grid()

    #5
    plt.figure(); 
    plt.plot(t_hr, 100 * SN, lw=2)
    plt.title("Normalised surface area $S_N(t)$"); 
    plt.xlabel("time [h]"); 
    plt.ylabel("% of initial"); 
    plt.grid()

    #6
    plt.figure(); 
    plt.plot(t_hr, 100 * MN, lw=2)
    plt.title("Normalised mass $M_N(t)$"); 
    plt.xlabel("time [h]"); 
    plt.ylabel("% of initial"); 
    plt.grid()

    # === Node snapshots (log–log)
    snapshot_times = [0, 0.5, 1, 1.5, 2, 2.5, 3, 3.5]

    #7
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

    #8
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
        w0, x0, eta0, _ = momentInversion.WheelerGQMOMRPlus(M_hist[:, i0], N_GQMOM, type=distributionType)
        wT, xT, etaT, _ = momentInversion.WheelerGQMOMRPlus(M_hist[:, iT], N_GQMOM, type=distributionType)
        r_grid = np.linspace(max(1e-6, R_MIN * 1e-3), 10.0, 1000)
        f0 = mixture_pdf_stable_many(r_grid, x0, w0, eta0)
        fT = mixture_pdf_stable_many(r_grid, xT, wT, etaT)

        #9
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
