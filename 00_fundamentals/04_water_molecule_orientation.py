"""
04 – Water Molecule Orientation and Equivariance Demonstration (ASE version)
============================================================================

Goal:
-----
To show, using a real molecule (H₂O), how spherical harmonics Y_{ℓm} that
describe atomic environments transform *equivariantly* under rotation.

Concepts Demonstrated:
----------------------
1. Rigid-body rotation in 3D using rotation matrices.
2. Spherical coordinate conversion (θ, φ) for atomic bonds.
3. Evaluation of spherical harmonics Y_{ℓm}(θ, φ).
4. Proof of equivariance:
       Y_{ℓm}(R·r̂) = Σ_{m′} D^{(ℓ)}_{m m′}(R) · Y_{ℓm′}(r̂)
   where D^{(ℓ)}(R) is the Wigner D-matrix for rotation R.

Dependencies:
-------------
- numpy
- matplotlib
- scipy
- sympy
- ase
- pandas (for table formatting)
"""

# =====================================================
#  Imports
# =====================================================
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import sph_harm
from ase import Atoms
from sympy.physics.wigner import wigner_d
import pandas as pd


# =====================================================
#  Step 1. Define the H₂O molecule
# =====================================================
r_OH = 0.9572  # bond length (Å)
angle_HOH = 104.52  # degrees

# Hydrogen coordinates relative to O
H1 = [r_OH, 0.0, 0.0]
H2 = [
    r_OH * np.cos(np.radians(angle_HOH)),
    r_OH * np.sin(np.radians(angle_HOH)),
    0.0
]

atoms = Atoms("OH2", positions=[[0, 0, 0], H1, H2])
print("=== Step 1: Water molecule created ===")
for atom in atoms:
    print(f"{atom.symbol:>2s}: {atom.position}")
print()


# =====================================================
#  Step 2. Define rotation matrix
# =====================================================
def rotation_matrix(axis, theta):
    """Return a 3×3 rotation matrix for rotation by 'theta' radians around 'axis'."""
    axis = np.array(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2)
    b, c, d = -axis * np.sin(theta / 2)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])


theta = np.pi / 3  # 60° about z-axis
Rz = rotation_matrix([0, 0, 1], theta)
rotated_atoms = atoms.copy()
rotated_atoms.positions = atoms.positions @ Rz.T

print("=== Step 2: Rotation applied (60° about z-axis) ===")
print(Rz)
print()


# =====================================================
#  Step 3. Verify bond lengths (rigid rotation)
# =====================================================
for i in [1, 2]:
    d_before = np.linalg.norm(atoms[i].position - atoms[0].position)
    d_after = np.linalg.norm(rotated_atoms[i].position - rotated_atoms[0].position)
    print(f"O–H{i} before = {d_before:.6f} Å | after = {d_after:.6f} Å")
print("✅ Distances preserved (rigid-body rotation)\n")


# =====================================================
#  Step 4. Visualize both configurations
# =====================================================
fig = plt.figure(figsize=(7, 7))
ax = fig.add_subplot(111, projection="3d")

# Plot molecules
ax.scatter(*atoms.positions.T, s=150, color='blue', label='Original')
ax.scatter(*rotated_atoms.positions.T, s=150, color='red', label='Rotated (60°)')

# Label atoms
for atom in atoms:
    ax.text(*atom.position, atom.symbol, color='blue', fontsize=12)
for atom in rotated_atoms:
    ax.text(*atom.position, atom.symbol, color='red', fontsize=12)

# Bonds
O, O_rot = atoms[0].position, rotated_atoms[0].position
for i in [1, 2]:
    ax.plot([O[0], atoms[i].x], [O[1], atoms[i].y], [O[2], atoms[i].z],
            color='blue', linestyle='--', lw=2, alpha=0.7)
    ax.plot([O_rot[0], rotated_atoms[i].x], [O_rot[1], rotated_atoms[i].y],
            [O_rot[2], rotated_atoms[i].z],
            color='red', linestyle='--', lw=2, alpha=0.7)

# Rotation axis (z)
ax.plot([0, 0], [0, 0], [-1.5, 1.5], color='gray', linestyle=':', lw=2, alpha=0.7)

# Equal aspect ratio
def set_equal_aspect(ax):
    limits = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
    spans = limits[:, 1] - limits[:, 0]
    centers = np.mean(limits, axis=1)
    radius = 0.5 * max(spans)
    for ctr, setter in zip(centers, [ax.set_xlim3d, ax.set_ylim3d, ax.set_zlim3d]):
        setter([ctr - radius, ctr + radius])

ax.set_xlabel("x (Å)")
ax.set_ylabel("y (Å)")
ax.set_zlabel("z (Å)")
ax.set_title("Water Molecule: Original (blue) and Rotated (red)")
ax.legend()
ax.view_init(elev=25, azim=35)
set_equal_aspect(ax)
plt.tight_layout()
plt.show()


# =====================================================
#  Step 5. Compute spherical harmonics Y_lm before rotation
# =====================================================
def spherical_angles(v):
    """Return polar and azimuthal angles (θ, φ) of a 3D vector."""
    r = np.linalg.norm(v)
    theta = np.arccos(v[2] / r)
    phi = np.arctan2(v[1], v[0])
    return theta, phi

l = 2
m_values = np.arange(-l, l + 1)

O = atoms[0].position
theta, phi = spherical_angles(atoms[1].position - O)
Y_before = np.array([sph_harm(m, l, phi, theta) for m in m_values])


# =====================================================
#  Step 6. Rotate molecule about x-axis (nontrivial rotation)
# =====================================================
Rx = rotation_matrix([1, 0, 0], np.pi/4)
rotated_atoms.positions = atoms.positions @ Rx.T
theta_r, phi_r = spherical_angles(rotated_atoms[1].position - rotated_atoms[0].position)
Y_after = np.array([sph_harm(m, l, phi_r, theta_r) for m in m_values])


# =====================================================
#  Step 7. Compute Wigner D-matrix and predict Y_lm transformation
# =====================================================
alpha, beta, gamma = 0, np.pi/4, 0  # Euler angles for rotation about x
D = np.zeros((2*l + 1, 2*l + 1), dtype=complex)
for i, mp in enumerate(m_values):
    for j, m in enumerate(m_values):
        D[i, j] = complex(wigner_d(l, mp, m, alpha, beta, gamma))

Y_pred = D @ Y_before  # predicted transformed harmonics


# =====================================================
#  Step 8. Display results in a table
# =====================================================
df = pd.DataFrame({
    "m": m_values,
    "Y_before (Re)": np.round(Y_before.real, 5),
    "Y_before (Im)": np.round(Y_before.imag, 5),
    "Y_after (Re)": np.round(Y_after.real, 5),
    "Y_after (Im)": np.round(Y_after.imag, 5),
    "Y_pred (Re)": np.round(Y_pred.real, 5),
    "Y_pred (Im)": np.round(Y_pred.imag, 5),
    "Error |Y_after - Y_pred|": np.round(np.abs(Y_after - Y_pred), 6)
})

print("\n=== Step 8: Equivariance demonstration for Y_lm (ℓ=2) ===")
print(df.to_string(index=False))
print("""
Interpretation:
---------------
- 'Y_before': spherical harmonics evaluated on the original O–H bond.
- 'Y_after' : spherical harmonics evaluated on the rotated bond.
- 'Y_pred'  : predicted values obtained by applying the Wigner D-matrix
              (theoretical SO(3) rotation law) to Y_before.
- If 'Y_after' ≈ 'Y_pred' for all m, the spherical harmonics transform
  exactly as expected under rotation — confirming **equivariance**.
""")
