# Lognormal–GQMOM Pt Ostwald-Ripening (Simple)

This repo contains a single Python script (`GQMOM.py`) that simulates Pt nanoparticle Ostwald ripening by evolving the first five raw moments **M₀…M₄** of the size distribution
on \( r \in [R_{\min}, \infty) \). Interior integrals are closed with **GQMOM on \( \mathbb{R}^+ \)** using a **Stieltjes–Wigert (lognormal) extension**. A boundary flux at \( r=R_{\min} \)
uses a lognormal-kernel estimate for \( f(R_{\min}) \).

## Requirements
- Python 3.9+
- `numpy`, `scipy`, `matplotlib`


