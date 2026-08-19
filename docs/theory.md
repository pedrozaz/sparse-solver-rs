# Mathematical Foundations — sparse-solver-rs

This document provides the mathematical background for the algorithms implemented in `sparse-solver-rs`.

---

## 1. Compressed Sparse Row (CSR) Format

### Motivation

In numerical simulations (FEM, CFD), matrices are typically very large ($N \gg 10^4$) but extremely sparse — over 99% of entries are zero. Storing such matrices in dense format would require $O(N^2)$ memory, which is infeasible.

### Representation

A CSR matrix stores only the non-zero entries using three contiguous arrays:

- **`values`** $\in \mathbb{R}^{\text{nnz}}$: Non-zero values in row-major order.
- **`col_indices`** $\in \mathbb{N}^{\text{nnz}}$: Column index for each non-zero value.
- **`row_ptr`** $\in \mathbb{N}^{N+1}$: `row_ptr[i]` indicates where row $i$ starts in `values` and `col_indices`. Row $i$ spans indices `row_ptr[i]..row_ptr[i+1]`.

### Sparse Matrix-Vector Multiplication (SpMV)

Given $A \in \mathbb{R}^{N \times N}$ in CSR format and $x \in \mathbb{R}^N$, the product $y = Ax$ is computed as:

$$y_i = \sum_{k=\text{row\_ptr}[i]}^{\text{row\_ptr}[i+1]-1} \text{values}[k] \cdot x[\text{col\_indices}[k]]$$

**Complexity**: $O(\text{nnz})$ — linear in the number of non-zeros, with sequential memory access (cache-friendly).

---

## 2. Conjugate Gradient (CG) Method

### Problem Statement

Given a symmetric positive-definite (SPD) matrix $A$ and a right-hand side vector $b$, find $x$ such that $Ax = b$.

### Variational Formulation

The solution $x^*$ minimizes the quadratic form:

$$f(x) = \frac{1}{2} x^T A x - b^T x$$

The gradient of $f$ at $x$ is $\nabla f(x) = Ax - b = -r$, where $r = b - Ax$ is the residual.

### Algorithm

Starting from $x_0 = 0$, $r_0 = b$, $p_0 = r_0$:

For $k = 0, 1, 2, \ldots$ until convergence:

$$\alpha_k = \frac{r_k^T r_k}{p_k^T A p_k}$$

$$x_{k+1} = x_k + \alpha_k p_k$$

$$r_{k+1} = r_k - \alpha_k A p_k$$

$$\beta_k = \frac{r_{k+1}^T r_{k+1}}{r_k^T r_k}$$

$$p_{k+1} = r_{k+1} + \beta_k p_k$$

### Convergence

CG converges in at most $N$ iterations (exact arithmetic). The convergence rate depends on the condition number $\kappa(A) = \lambda_{\max} / \lambda_{\min}$:

$$\|e_k\|_A \leq 2 \left(\frac{\sqrt{\kappa} - 1}{\sqrt{\kappa} + 1}\right)^k \|e_0\|_A$$

For the 2D Poisson equation, $\kappa(A) = O(N^2)$, so CG requires $O(N)$ iterations — total cost $O(N^3)$ for an $N \times N$ grid ($N^2$ unknowns).

### Stopping Criterion

We use the relative residual norm:

$$\frac{\|r_k\|_2}{\|b\|_2} < \text{tol}$$

This makes the criterion scale-invariant.

---

## 3. Geometric Multigrid (V-Cycle)

### Motivation: Why Smoothers Alone Are Not Enough

Iterative methods like Jacobi or Gauss-Seidel are effective at eliminating **high-frequency** (oscillatory) error components but stall on **low-frequency** (smooth) error components. This is because smooth errors look nearly constant to local stencil operations — each update sees almost no local residual.

The key insight of multigrid: **a smooth error on a fine grid appears oscillatory on a coarser grid**, where it can be efficiently eliminated by the same smoother.

### Transfer Operators

#### Restriction ($R$: Fine $\to$ Coarse)

Full-Weighting 2D restriction uses a 9-point stencil centered at the coarse grid point $(I, J)$ corresponding to fine grid point $(2I+1, 2J+1)$:

$$v^c(I, J) = \frac{1}{16}\left[4 v^f(i,j) + 2\big(v^f(i\pm1,j) + v^f(i,j\pm1)\big) + 1\big(v^f(i\pm1,j\pm1)\big)\right]$$

#### Prolongation ($P$: Coarse $\to$ Fine)

Bilinear interpolation maps coarse grid values to the fine grid:

- **Coincident nodes** $(2I+1, 2J+1)$: direct injection $v^f = v^c(I, J)$
- **Horizontal edges** $(2I+2, 2J+1)$: average of two horizontal neighbors
- **Vertical edges** $(2I+1, 2J+2)$: average of two vertical neighbors
- **Cell centers** $(2I+2, 2J+2)$: average of four surrounding coarse nodes

### Weighted Jacobi Smoother

The weighted Jacobi iteration with relaxation factor $\omega$ is:

$$x_i^{(k+1)} = x_i^{(k)} + \frac{\omega}{a_{ii}} \left(b_i - \sum_j a_{ij} x_j^{(k)}\right)$$

For the 5-point Poisson stencil, $a_{ii} = 4$. The optimal damping factor for 2D Poisson is $\omega = 2/3$.

### V-Cycle Algorithm

Given $A^h x^h = b^h$ on a grid with spacing $h$:

1. **Pre-smoothing**: Apply $\nu_1$ iterations of weighted Jacobi.
2. **Compute residual**: $r^h = b^h - A^h x^h$
3. **Restrict**: $r^{2h} = R \cdot r^h$ (scaled by $(h_c/h_f)^2 = 4$)
4. **Coarse grid solve**: Solve $A^{2h} e^{2h} = r^{2h}$ recursively (or directly at coarsest level).
5. **Prolongate**: $e^h = P \cdot e^{2h}$
6. **Correct**: $x^h \leftarrow x^h + e^h$
7. **Post-smoothing**: Apply $\nu_2$ iterations of weighted Jacobi.

### Convergence

The V-cycle convergence rate is **independent of the grid size** $N$. The number of iterations to reach a given tolerance remains bounded as $N \to \infty$.

Combined with the $O(N)$ cost per V-cycle (each level has $1/4$ the unknowns of the previous), the total cost is $O(N)$ — **optimal complexity**.

---

## 4. 2D Poisson Equation (Validation Case)

### Continuous Problem

$$-\nabla^2 u = -\left(\frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2}\right) = f(x, y) \quad \text{on } \Omega = [0, 1]^2$$

with homogeneous Dirichlet boundary conditions $u = 0$ on $\partial\Omega$.

### Finite Difference Discretization

On a uniform grid with $N$ interior points per direction and spacing $h = 1/(N+1)$, the 5-point stencil approximation yields:

$$\frac{4u_{i,j} - u_{i-1,j} - u_{i+1,j} - u_{i,j-1} - u_{i,j+1}}{h^2} = f(x_i, y_j)$$

This produces a linear system $Au = b$ where $A \in \mathbb{R}^{N^2 \times N^2}$ is SPD with at most 5 non-zeros per row.

### Manufactured Solution

We choose $f(x,y) = 2\pi^2 \sin(\pi x) \sin(\pi y)$, which yields the exact solution:

$$u_{\text{exact}}(x, y) = \sin(\pi x) \sin(\pi y)$$

### Spatial Accuracy

The 5-point stencil is second-order accurate: $\|u_h - u_{\text{exact}}\|_{L_2} = O(h^2)$. When $N$ doubles (halving $h$), the $L_2$ error decreases by a factor of $\approx 4$.

---

## References

1. Saad, Y. (2003). *Iterative Methods for Sparse Linear Systems*, 2nd Edition. SIAM.
2. Briggs, W. L., Henson, V. E., & McCormick, S. F. (2000). *A Multigrid Tutorial*, 2nd Edition. SIAM.
3. Trottenberg, U., Oosterlee, C. W., & Schüller, A. (2001). *Multigrid*. Academic Press.
