"""
01 – Spherical Harmonics
========================

This script explores the mathematical structure and visualization
of spherical harmonics Y_{l m}(θ, φ), which describe angular patterns
on the unit sphere.

These functions form the foundation of the angular basis used in
the Atomic Cluster Expansion (ACE) and equivariant neural networks
like MACE and NequIP.

Dependencies:
-------------
numpy, scipy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm


# -----------------------------------------------------
# 1. Definition of spherical harmonics
# -----------------------------------------------------
def Ylm(l, m, theta, phi):
    """
    Compute the spherical harmonic Y_lm(θ, φ).

    Parameters
    ----------
    l : int
        Angular momentum quantum number (≥ 0)
    m : int
        Magnetic quantum number (-l ≤ m ≤ l)
    theta : ndarray
        Polar angle [0, π]
    phi : ndarray
        Azimuthal angle [0, 2π]
    """
    return sph_harm(m, l, phi, theta)


# -----------------------------------------------------
# 2. Visualization of |Y_lm|
# -----------------------------------------------------
def plot_Ylm(l=3, m=2, cmap="viridis"):
    """
    Plot the magnitude |Y_lm| as a 3D surface on the unit sphere.
    """
    theta = np.linspace(0, np.pi, 200)
    phi = np.linspace(0, 2*np.pi, 200)
    theta, phi = np.meshgrid(theta, phi)

    Y = Ylm(l, m, theta, phi)
    Yabs = np.abs(Y)

    # Convert to Cartesian coordinates
    x = Yabs * np.sin(theta) * np.cos(phi)
    y = Yabs * np.sin(theta) * np.sin(phi)
    z = Yabs * np.cos(theta)

    # Plot 3D surface
    fig = plt.figure(figsize=(7, 6))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(x, y, z, facecolors=plt.cm.get_cmap(cmap)(Yabs / Yabs.max()),
                           rstride=3, cstride=3, linewidth=0, antialiased=False, shade=True)
    ax.set_box_aspect([1, 1, 1])
    ax.set_title(f"Spherical Harmonic |Y({l},{m})|", fontsize=14)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


# Example: visualize a few harmonics
plot_Ylm(l=2, m=0)
plot_Ylm(l=3, m=2)


# -----------------------------------------------------
# 3. Real and Imaginary parts
# -----------------------------------------------------
def plot_real_imag(l=3, m=2):
    """
    Plot real and imaginary parts of Y_lm(θ, φ) in polar projection.
    """
    theta = np.linspace(0, np.pi, 200)
    phi = np.linspace(0, 2*np.pi, 200)
    theta, phi = np.meshgrid(theta, phi)
    Y = Ylm(l, m, theta, phi)

    Y_real = np.real(Y)
    Y_imag = np.imag(Y)

    fig, axs = plt.subplots(1, 2, figsize=(10, 4), subplot_kw={"projection": "polar"})
    axs[0].contourf(phi, theta, Y_real, 100, cmap="coolwarm")
    axs[0].set_title(f"Re[Y({l},{m})]", fontsize=12)
    axs[1].contourf(phi, theta, Y_imag, 100, cmap="coolwarm")
    axs[1].set_title(f"Im[Y({l},{m})]", fontsize=12)
    plt.suptitle(f"Real and Imaginary Parts of Y({l},{m})")
    plt.show()


plot_real_imag(l=3, m=2)


# -----------------------------------------------------
# 4. Orthogonality check
# -----------------------------------------------------
def check_orthogonality(l1, m1, l2, m2, N=300):
    """
    Verify ∫ Y_l1,m1* Y_l2,m2 sinθ dθ dφ = δ_l1,l2 δ_m1,m2
    """
    theta = np.linspace(0, np.pi, N)
    phi = np.linspace(0, 2*np.pi, N)
    dtheta = theta[1] - theta[0]
    dphi = phi[1] - phi[0]
    TH, PH = np.meshgrid(theta, phi)

    Y1 = Ylm(l1, m1, TH, PH)
    Y2 = Ylm(l2, m2, TH, PH)
    integrand = np.conjugate(Y1) * Y2 * np.sin(TH)
    integral = np.sum(integrand) * dtheta * dphi
    return integral


print("\nOrthogonality check:")
same = check_orthogonality(2, 1, 2, 1)
diff = check_orthogonality(2, 1, 3, 0)
print(f"<Y(2,1)|Y(2,1)> = {np.real(same):.5f}")
print(f"<Y(2,1)|Y(3,0)> = {np.real(diff):.5e}")


# -----------------------------------------------------
# 5. Physical interpretation
# -----------------------------------------------------
print("""
Physical Interpretation
-----------------------
- Each Y_{l,m} corresponds to an angular pattern with 2l lobes (approx).
- l controls angular complexity; m controls azimuthal variation.
- Under rotation (SO(3)), Y_{l,m} transforms linearly → key to equivariance.
- In ACE, the product R_n(r) Y_{l,m}(θ, φ) describes how atomic densities
  vary with direction.

Spherical harmonics are the "angular fingerprints" of an atom's environment.
""")
