# 00 – Fundamentals (Vectors, Rotations, Yℓm, and CG Coupling)

This module builds the mathematical foundation used throughout ACE, MACE, and the Multi-ACE framework:
- Rotations (SO(3)), how vectors transform, and why this matters
- Spherical harmonics \(Y_{\ell m}\) as an angular basis
- Clebsch–Gordan (CG) coefficients to **couple** angular channels
- Equivariance vs invariance (why MLIP features must transform consistently)

> **Run any file** with:  
> `python 00_vectors_and_rotations.py` (or the others)  
> Make sure you have the dependencies installed: `numpy`, `scipy`, `sympy`, `matplotlib`.

---

## File Guide

### 1) `00_vectors_and_rotations.py` — *Vectors & Rotations (SO(3) intuition)*

**What it does**
- Implements a **stable rotation matrix** generator using the quaternion/Rodrigues form:
  - `rotation_matrix(axis, theta)` → returns a 3×3 matrix.
- Rotates a unit vector by **45° about the z-axis**, prints original & rotated coordinates.
- Plots two 3D arrows (original vs rotated) so you **see** the rotation.
- Demonstrates **sequential rotations** (z then y) → highlights **non-commutativity** of 3D rotations.
- Prints a short **physical interpretation**: rotations preserve norms/angles and define how geometric features transform.

**Why it matters for ACE/MACE**
- ACE/MACE operate on 3D atomic coordinates. Knowing how **rotations act** on vectors is the seed of **equivariance**.
- Later, tensors (built from \(Y_{\ell m}\)) must rotate consistently; this script gives the geometric intuition.

**Edit & explore**
- Change the axis/angle in `rotation_matrix([0,0,1], np.pi/4)` to see different rotations.
- Stack rotations in different orders to observe non-commutativity.

**Expected output**
- Two 3D figures (vector before/after rotation).
- Printed coordinates confirming the rotation.

---

### 2) `01_spherical_harmonics.py` — *Spherical Harmonics \(Y_{\ell m}\)*

**What it does**
- Wraps SciPy’s `sph_harm` into a helper `Ylm(l, m, theta, phi)`.
- **3D visualization** of \(|Y_{\ell m}|\) as a surface on the unit sphere:
  - `plot_Ylm(l=3, m=2)` shows the familiar **lobed** shapes.
- **Real & Imaginary** parts plotted in **polar coordinates**:
  - `plot_real_imag(l=3, m=2)` → contour plots for intuition.
- **Orthogonality check**:
  - Numerically integrates \( \int Y_{\ell m}^* Y_{\ell' m'} \sin\theta\, d\theta\, d\phi \)  
    to confirm \(\delta_{\ell\ell'}\delta_{mm'}\).

**Why it matters for ACE/MACE**
- \(Y_{\ell m}\) are the **angular basis** in ACE descriptors: features look like \(R_n(r)\,Y_{\ell m}(\hat r)\).
- In equivariant GNNs (e.g., MACE/NequIP), channels carry **irreps** labeled by \(\ell\); these transform under SO(3) exactly like \(Y_{\ell m}\).

**Edit & explore**
- Try different `(l, m)` in `plot_Ylm` and `plot_real_imag`.
- Increase grid resolution to tighten the orthogonality integral.

**Expected output**
- 3D surface plots of \(|Y_{\ell m}|\).  
- Two polar contour plots (Re/Im).  
- Printout showing `<Y(2,1)|Y(2,1)> ≈ 1` and `<Y(2,1)|Y(3,0)> ≈ 0`.

---

### 3) `02_clebsch_gordan.py` — *Clebsch–Gordan Coupling \((\ell_1,m_1)\otimes(\ell_2,m_2)\to(L,M)\)*

**What it does**
- Uses SymPy’s `clebsch_gordan` to compute **CG coefficients**:
  - `cg(l1, l2, L, m1, m2, M)` returns \(\langle \ell_1 m_1; \ell_2 m_2 | L M\rangle\).
- Prints a **mini-table** for \(\ell_1=\ell_2=1\) → \(L=0,1,2\).
- Builds a **coupled spherical harmonic**:
  \[
    \Phi_{LM}(\theta,\phi) \;=\; \sum_{m_1,m_2} 
    \langle \ell_1 m_1; \ell_2 m_2 \mid L M\rangle
    Y_{\ell_1 m_1}(\theta,\phi)\,Y_{\ell_2 m_2}(\theta,\phi)
  \]
  via `coupled_spherical_harmonic(...)`.
- Visualizes \(|\Phi_{LM}|\) as a **3D surface**.
- Checks **orthogonality** between coupled bases:
  \(\int \Phi_{L_1M}^*\,\Phi_{L_2M}\,\sin\theta\,d\theta\,d\phi \;=\; \delta_{L_1L_2}\).

**Why it matters for ACE/MACE**
- ACE builds **higher-body** invariants by **coupling** angular channels —
  this is exactly what CG coefficients do.
- Equivariant message passing (MACE) internally performs **tensor products of irreps** that mirror these CG couplings.

**Edit & explore**
- Change `(l1, l2, L, M)` in `plot_coupled_Y(...)`.  
- Try `l1=2, l2=1` and explore `L ∈ {|l1−l2|,…,l1+l2}`.  
- Compare orthogonality for different `L`.

**Expected output**
- Console table of CGCs for chosen example.  
- 3D surface plot of the **coupled** basis \(\Phi_{LM}\).  
- Orthogonality printout (same \(L\) ≈ 1, different \(L\) ≈ 0).

---

### 4) `03_group_equivariance.py` — *Equivariance vs Invariance (why symmetry matters)*

**What it does**
- Defines a **2D rotation** operator and rotates a simple **triangle** shape → visual SO(2) action.
- Demonstrates:
  - **Invariant** scalar function: \(f(\mathbf{x})=\|\mathbf{x}\|\) ⇒ \(f(R\mathbf{x})=f(\mathbf{x})\).
  - **Equivariant** vector function: \(g(\mathbf{x})=\mathbf{x}\) ⇒ \(g(R\mathbf{x})=R\,g(\mathbf{x})\).
- Plots input/rotated vectors with quivers to **see** equivariance.
- Prints a **SO(3) generalization** note: how this extends to 3D and why equivariance is essential for physical MLIPs.

**Why it matters for ACE/MACE**
- **E(3)-equivariance** guarantees that if you rotate the atomic system,  
  the **features/forces rotate accordingly** (physics-consistent).
- ACE uses **invariants** for energies; MACE carries **equivariant** tensors internally then reduces to invariants for scalar outputs.

**Edit & explore**
- Change rotation angle and shape points; try multiple rotations in different orders.
- Replace the scalar/vector functions to design your own invariants/equivariants.

**Expected output**
- 2D plots showing the rotated shape and vectors.  
- Console output illustrating invariance/equivariance numerically.

---

## Installation & Quick Start

```bash
# from repo root
pip install numpy scipy sympy matplotlib

# run any script in this folder
cd 00_fundamentals
python 00_vectors_and_rotations.py
python 01_spherical_harmonics.py
python 02_clebsch_gordan.py
python 03_group_equivariance.py
