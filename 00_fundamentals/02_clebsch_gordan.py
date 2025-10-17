"""
02 – Clebsch–Gordan Coefficients
================================

This script demonstrates how **angular momenta** combine in quantum mechanics
and how **Clebsch–Gordan coefficients (CGCs)** arise from this coupling.

In the context of the **Atomic Cluster Expansion (ACE)** and **MACE**, CGCs are
used to combine multiple spherical harmonics into rotationally equivariant
tensors.

Concepts covered:
-----------------
- Coupling of angular momenta
- Computation of Clebsch–Gordan coefficients
- Construction of coupled basis Y_{l1} ⊗ Y_{l2}
- Visualization of the resulting pattern
"""

import numpy as np
import matplotlib.pyplot as plt
from sympy import S
from sympy.physics.wigner import clebsch_gordan
from scipy.special import sph_harm


# -----------------------------------------------------
# 1. Clebsch–Gordan coefficients
# -----------------------------------------------------
def cg(l1, l2, L, m1, m2, M):
    """
    Return Clebsch–Gordan coefficient ⟨l1,m1; l2,m2 | L,M⟩
    """
    return float(clebsch_gordan(S(l1), S(l2), S(L), S(m1), S(m2), S(M)))


# Example table for l1 = l2 = 1 → L = 0,1,2
print("\nExample: Coupling l1 = 1, l2 = 1 → L = 0, 1, 2")
for L in [0, 1, 2]:
    for m1 in range(-1, 2):
        for m2 in range(-1, 2):
            val = cg(1, 1, L, m1, m2, 0)
            if abs(val) > 1e-10:
                print(f"<1,{m1}; 1,{m2} | {L},0> = {val:+.4f}")


# -----------------------------------------------------
# 2. Combine two spherical harmonics
# -----------------------------------------------------
def coupled_spherical_harmonic(l1, l2, L, M, theta, phi):
    """
    Construct the coupled spherical harmonic:
        Φ_LM(θ, φ) = Σ_{m1,m2} ⟨l1,m1; l2,m2 | L,M⟩ Y_l1,m1(θ,φ) Y_l2,m2(θ,φ)
    """
    total = np.zeros_like(theta, dtype=complex)
    for m1 in range(-l1, l1 + 1):
        for m2 in range(-l2, l2 + 1):
            if m1 + m2 == M:
                coeff = cg(l1, l2, L, m1, m2, M)
                total += coeff * sph_harm(m1, l1, phi, theta) * sph_harm(m2, l2, phi, theta)
    return total


# -----------------------------------------------------
# 3. Visualization
# -----------------------------------------------------
def plot_coupled_Y(l1, l2, L, M):
    theta = np.linspace(0, np.pi, 200)
    phi = np.linspace(0, 2*np.pi, 200)
    theta, phi = np.meshgrid(theta, phi)

    Φ = coupled_spherical_harmonic(l1, l2, L, M, theta, phi)
    Φ_abs = np.abs(Φ)

    # Convert to Cartesian coordinates for 3D surface
    x = Φ_abs * np.sin(theta) * np.cos(phi)
    y = Φ_abs * np.sin(theta) * np.sin(phi)
    z = Φ_abs * np.cos(theta)

    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        x, y, z, facecolors=plt.cm.plasma(Φ_abs / Φ_abs.max()),
        rstride=3, cstride=3, linewidth=0, antialiased=False
    )
    ax.set_title(f"Coupled Spherical Harmonic Φ_{L}^{M} (l₁={l1}, l₂={l2})", fontsize=13)
    ax.axis("off")
    plt.show()


# Example visualization: combine Y_{1m} and Y_{1m'} → L=2
plot_coupled_Y(l1=1, l2=1, L=2, M=0)


# -----------------------------------------------------
# 4. Orthogonality check of coupled basis
# -----------------------------------------------------
def check_coupled_orthogonality(l1, l2, L1, L2, M=0, N=200):
    """
    Verify orthogonality between two coupled harmonics:
        ∫ Φ_L1M* Φ_L2M sinθ dθ dφ = δ_L1,L2
    """
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    TH, PH = np.meshgrid(theta, phi)

    Φ1 = coupled_spherical_harmonic(l1, l2, L1, M, TH, PH)
    Φ2 = coupled_spherical_harmonic(l1, l2, L2, M, TH, PH)
    integrand = np.conjugate(Φ1) * Φ2 * np.sin(TH)
    integral = np.sum(integrand) * dtheta * dphi
    return integral


val_same = check_coupled_orthogonality(1, 1, 2, 2)
val_diff = check_coupled_orthogonality(1, 1, 2, 1)
print(f"\n⟨Φ₂⁰ | Φ₂⁰⟩ = {np.real(val_same):.5f}")
print(f"⟨Φ₂⁰ | Φ₁⁰⟩ = {np.real(val_diff):.5e}")


# -----------------------------------------------------
# 5. Physical interpretation
# -----------------------------------------------------
print("""
Physical Interpretation
-----------------------
- Clebsch–Gordan coefficients describe how two angular momenta (l₁, l₂)
  combine to form a total angular momentum L.

- In ACE and equivariant MLIPs, this allows combining multiple
  spherical harmonics into a *rotationally consistent* tensor basis.

- The coupled spherical harmonics Φ_LM(θ,φ) serve as building blocks
  for higher-order body terms (3-body, 4-body) that remain equivariant
  under 3D rotations.

Example:
--------
Y_{1m} ⊗ Y_{1m'}  →  L = 0, 1, 2  (s, p, d channels)
These are directly analogous to s–p–d orbital couplings in atoms.
""")
