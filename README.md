# 🧠 Multi-ACE Learning Roadmap

**A structured roadmap and code repository to learn the theory behind the Multi-ACE framework**  
*Batatia et al., Nature Machine Intelligence (2025)*

This project connects the dots between:

- 🧩 **Atomic Cluster Expansion (ACE)** – physics-based, symmetry-adapted descriptors  
- 🧠 **Equivariant Neural Networks (E(3)-NNs)** – message-passing generalizations of ACE  
- ⚙️ **MACE and Multi-ACE** – unified models combining body-order and network depth  

---

## 📚 Learning Route Overview

| Phase | Folder | Focus |
|:--:|:--|:--|
| 00 | [`00_fundamentals/`](./00_fundamentals) | Geometry, spherical harmonics, Clebsch–Gordan, SO(3)/E(3) basics |
| 01 | [`01_atomic_cluster_expansion/`](./01_atomic_cluster_expansion) | Atomic Cluster Expansion (ACE): neighbor density, one-particle basis |
| 02 | [`02_equivariant_neural_networks/`](./02_equivariant_neural_networks) | E(3)-equivariant message passing, tensor products (`e3nn`) |
| 03 | [`03_mace_architecture/`](./03_mace_architecture) | Multiplicative ACE (MACE): learnable equivariant model |
| 04 | [`04_multi_ace_framework/`](./04_multi_ace_framework) | Multi-ACE 2025: unified design space for all equivariant IPs |
| — | [`figures/`](./figures) | Generated plots and visual diagrams |

---

## 🧩 Learning Goals

1. Understand **rotational symmetry** through spherical harmonics and tensor algebra.  
2. Build ACE-style **invariant descriptors** from neighbor densities.  
3. Implement simple **E(3)-equivariant layers** using `e3nn`.  
4. Explore the **MACE architecture** and how it extends ACE.  
5. Re-derive the **Multi-ACE equations (1–4)** and map them to MPNN frameworks.  

Each phase includes Jupyter notebooks (`.ipynb`) and Python scripts (`.py`) for hands-on learning.

---

## 🧭 Study Order

1. `00_fundamentals/` → mathematical background  
2. `01_atomic_cluster_expansion/` → ACE theory and coding  
3. `02_equivariant_neural_networks/` → symmetry & equivariance  
4. `03_mace_architecture/` → practical MACE model  
5. `04_multi_ace_framework/` → unification and advanced topics  
---

## 📦 Setup

```bash
git clone https://github.com/<your-username>/multi-ace-learning.git
cd multi-ace-learning
pip install -r requirements.txt
```

---

## 🧩 Recommended Dependencies

```
numpy
scipy
ase
matplotlib
sympy
e3nn
torch
mace-torch
```

---

## 📖 Study Order

1. `01_spherical_harmonics/`
2. `02_atomic_cluster_expansion/`
3. `03_equivariant_networks/`
4. `04_multi_ace_framework/`

---

🧑‍🔬 **Author:  *Harry  Halim  
📚 **Purpose:** Research and teaching reference for understanding Multi-ACE, MACE, and ACE.
