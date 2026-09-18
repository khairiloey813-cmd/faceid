# Face ID via Spectral Decomposition (PCA from Scratch)

An end-to-end biometric facial recognition engine implemented from scratch in NumPy using the **Turk & Pentland Gram matrix trick** and **1-NN classification**, evaluated on the Olivetti Faces dataset.

---

## Technical Overview

* **Dimensionality Reduction:** Circumvents the high-dimensional bottleneck ($d \gg N$, where $d = 4096$ pixels and $N = 320$ training samples) by diagonalizing the smaller Gram matrix $S = X_c^T X_c \in \mathbb{R}^{N \times N}$ instead of the full covariance matrix $\Sigma \in \mathbb{R}^{d \times d}$.
* **Biometric Signatures:** Reconstructs the top $k$ spatial eigenvectors (Eigenfaces) $U_k$ and projects centered facial vectors to extract compact coordinate signatures $w \in \mathbb{R}^k$.
* **Classification:** Uses minimum Euclidean distance (1-Nearest Neighbor) to match test signatures against the gallery.
* **Empirical Performance:** Achieves **86.25% accuracy** on unseen test poses ($k = 40$).

---

## Grayscale vs. RGB Dimension Constraint

> **Important Limitation:** This implementation natively operates on **grayscale (single-channel) images**.

* **Grayscale representation:** Each image is a 2D scalar field $I \in \mathbb{R}^{H \times W}$. Vectorization produces a 1D column vector $x \in \mathbb{R}^d$ where $d = H \times W$ (e.g., $64 \times 64 = 4096$).
* **RGB representation:** Color images introduce a 3rd tensor axis (channels: Red, Green, Blue), yielding $I_{\text{color}} \in \mathbb{R}^{H \times W \times 3}$.
* **Handling color:** To process RGB data directly without converting to luminance, the feature space triples ($d = 3 \times H \times W$), requiring either:
  1. Flattening concatenated color channels into a higher-dimensional vector space.
  2. Converting input frames to grayscale via standard luminance weighting ($Y = 0.299R + 0.587G + 0.114B$).
  3. Formulating the projection via multilinear algebra (Tensor PCA / HOSVD).

---

## Quickstart

### 1. Setup Environment
```bash
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
pip install numpy matplotlib scikit-learn