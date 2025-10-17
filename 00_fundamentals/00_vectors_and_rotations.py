"""
00 – Vectors and Rotations
==========================

This script introduces 3D vector rotations — a foundation for understanding
symmetry operations in ACE and equivariant neural networks.

Concepts covered:
- Rotation matrices
- Visualization of rotated vectors
- Sequential rotations
- Physical interpretation of SO(3) transformations
"""

import numpy as np
import matplotlib.pyplot as plt

# -----------------------------
# 1. Rotation Matrix Definition
# -----------------------------
def rotation_matrix(axis, theta):
    """
    Create a 3x3 rotation matrix for rotation by angle theta (radians)
    about an arbitrary axis (x, y, z).

    Parameters
    ----------
    axis : array-like of shape (3,)
        Axis of rotation (e.g. [0,0,1] for z-axis).
    theta : float
        Rotation angle in radians.
    """
    axis = np.asarray(axis, dtype=float)
    axis /= np.linalg.norm(axis)
    a = np.cos(theta / 2.0)
    b, c, d = -axis * np.sin(theta / 2.0)
    return np.array([
        [a*a + b*b - c*c - d*d, 2*(b*c - a*d), 2*(b*d + a*c)],
        [2*(b*c + a*d), a*a + c*c - b*b - d*d, 2*(c*d - a*b)],
        [2*(b*d - a*c), 2*(c*d + a*b), a*a + d*d - b*b - c*c]
    ])


# -----------------------------
# 2. Example: Rotate a vector
# -----------------------------
v = np.array([1, 0, 0])  # unit vector along x-axis
theta = np.pi / 4         # 45° rotation
Rz = rotation_matrix([0, 0, 1], theta)
v_rot = Rz @ v

print("Original vector:", v)
print("Rotated vector :", v_rot)


# -----------------------------
# 3. Visualization
# -----------------------------
fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, *v, color='blue', lw=3, label='original')
ax.quiver(0, 0, 0, *v_rot, color='red', lw=3, label='rotated (45° about z-axis)')

ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.set_zlabel("z")
ax.legend()
ax.set_title("Rotation of a vector around z-axis (45°)")

plt.tight_layout()
plt.show()


# -----------------------------
# 4. Sequential Rotations
# -----------------------------
# Rotate around z, then y
Rz = rotation_matrix([0, 0, 1], np.pi / 4)
Ry = rotation_matrix([0, 1, 0], np.pi / 3)
R_combined = Ry @ Rz

v2 = R_combined @ v

print("\nAfter sequential rotations (z then y):")
print("Rotated vector:", v2)

fig = plt.figure(figsize=(6, 6))
ax = fig.add_subplot(111, projection='3d')

ax.quiver(0, 0, 0, *v, color='blue', lw=3, label='original')
ax.quiver(0, 0, 0, *v2, color='green', lw=3, label='rotated (z→y)')
ax.set_xlim([-1, 1])
ax.set_ylim([-1, 1])
ax.set_zlim([-1, 1])
ax.legend()
ax.set_title("Sequential Rotations: z (45°) → y (60°)")
plt.show()


# -----------------------------
# 5. Physical Interpretation
# -----------------------------
print("""
Summary:
--------
- Rotations in 3D belong to the group SO(3), preserving vector length and angles.
- This operation defines how geometric features (e.g., atomic neighbor directions)
  transform under coordinate changes.
- Later, spherical harmonics Y_lm(θ,φ) will represent these angular dependencies.
""")
