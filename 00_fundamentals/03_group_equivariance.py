"""
03 – Group Equivariance
=======================

This script illustrates the fundamental distinction between
**invariance** and **equivariance** under geometric transformations.

Concepts covered:
-----------------
- Rotation as a group operation (SO(2), SO(3))
- Function equivariance and invariance
- Visualization of how shapes, vectors, and features transform
- Connection to equivariant neural networks (MACE, NequIP)

Dependencies:
-------------
numpy, matplotlib
"""

import numpy as np
import matplotlib.pyplot as plt


# -----------------------------------------------------
# 1. Rotation operator in 2D
# -----------------------------------------------------
def rotate(points, theta):
    """
    Rotate a set of 2D points by an angle theta (radians).
    """
    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    return points @ R.T


# Define a simple geometric shape (triangle)
shape = np.array([
    [0.0, 0.5],
    [0.5, -0.5],
    [-0.5, -0.5]
])

theta = np.pi / 3  # 60° rotation
rotated_shape = rotate(shape, theta)

# -----------------------------------------------------
# 2. Visualize rotation (SO(2) action)
# -----------------------------------------------------
plt.figure(figsize=(6, 6))
plt.plot(*shape.T, 'bo-', lw=2, label='original')
plt.plot(*rotated_shape.T, 'ro-', lw=2, label='rotated 60°')
plt.axis('equal')
plt.title("Rotation of 2D Shape – SO(2) Group Action")
plt.legend()
plt.grid(True)
plt.show()


# -----------------------------------------------------
# 3. Invariance vs Equivariance
# -----------------------------------------------------
# A scalar property f(x) = ||x|| is rotation *invariant*:
x = np.array([1, 0])
x_rot = rotate(x[None, :], np.pi / 4)[0]
f = lambda x: np.linalg.norm(x)

print("Invariant scalar function:")
print(f"‖x‖ before rotation = {f(x):.2f}")
print(f"‖x‖ after rotation  = {f(x_rot):.2f}")

# A vector-valued function g(x) = x is rotation *equivariant*:
g = lambda x: x
print("\nEquivariant vector function:")
print("g(x) before rotation =", g(x))
print("g(x) after rotation  =", g(x_rot))


# -----------------------------------------------------
# 4. Visualize equivariance with arrows
# -----------------------------------------------------
def plot_vector_equivariance(theta=np.pi/3):
    """
    Show how a vector-valued function transforms equivariantly.
    """
    x = np.array([1, 0])
    g_x = g(x)

    R = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    x_rot = R @ x
    g_rot = R @ g_x

    plt.figure(figsize=(6, 6))
    plt.quiver(0, 0, *x, angles='xy', scale_units='xy', scale=1, color='blue', label='input x')
    plt.quiver(0, 0, *x_rot, angles='xy', scale_units='xy', scale=1, color='red', label='rotated x')
    plt.quiver(0, 0, *g_x, angles='xy', scale_units='xy', scale=1, color='blue', alpha=0.5, linestyle='--', label='g(x)')
    plt.quiver(0, 0, *g_rot, angles='xy', scale_units='xy', scale=1, color='red', alpha=0.5, linestyle='--', label='g(Rx)')
    plt.axis('equal')
    plt.legend()
    plt.title("Equivariance of g(x) under rotation")
    plt.show()

plot_vector_equivariance()


# -----------------------------------------------------
# 5. SO(3) intuition (conceptual, not 3D plotting)
# -----------------------------------------------------
print("""
SO(3) Generalization
--------------------
In 3D, rotation matrices act on vectors the same way.

For a function g: ℝ³ → ℝ³, equivariance means:
    g(Rx) = R g(x)

For a scalar function f: ℝ³ → ℝ,
    f(Rx) = f(x)  (invariance)

In ACE and MACE:
----------------
- Atomic positions transform by R (rotation)
- Spherical harmonics Yₗₘ(θ,φ) transform as basis functions of SO(3)
- Equivariant message passing ensures physical consistency
  (forces, dipoles, local features all rotate correctly)
""")
