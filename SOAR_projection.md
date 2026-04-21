# SOAR: A Second-Order Arnoldi Reduction for the Frequency-Independent BGT System

A moment-matching model order reduction strategy that operates directly on the Quadratic Eigenvalue Problem associated with Eq. (17), without linearizing to twice the dimension.

---

## 1. Context and motivation

As in Section 2 of the companion note on complex modal projection, the frequency-independent BGT formulation of Eq. (17) reads

$$
\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\mathbf{P}_q = \mathbf{F}_q,
$$

with $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3 \in \mathbb{C}^{(n_p+n_q) \times (n_p+n_q)}$ the frequency-independent block matrices of Eq. (16). Setting $\lambda = jk$ turns this into the second-order parametric system

$$
\left[\mathbf{D}_1 + \lambda\,\mathbf{D}_2 + \lambda^2\mathbf{D}_3\right]\mathbf{P}_q(\lambda) = \mathbf{F}_q.
$$

Two model-reduction paradigms are natural here:

- **Modal projection** (QEP complex modes): builds a physically meaningful basis from eigenpairs of the pencil. Expensive offline cost; basis size driven by the spectral content of the band.
- **Moment matching** (WCAWE, Padé-via-Lanczos, SOAR): builds a Krylov-like basis tailored to reproducing the transfer function around a shift $\sigma$. Basis size driven by the required order of accuracy at $\sigma$ and the effective damping of the operator.

WCAWE is already used in the paper as a moment-matching reference. **SOAR** (Second-Order ARnoldi Reduction, introduced by Bai and Su, 2005) occupies a different niche within the moment-matching family: it is specifically designed for **quadratic** parametric systems and preserves the second-order structure without explicit linearization. It is therefore the natural moment-matching counterpart to the QEP-based complex modal projection.

The motivation for adding SOAR alongside WCAWE is threefold:

- it preserves the **second-order structure** of Eq. (17) internally, avoiding the $2(n_p + n_q)$ dimension blow-up inherent to the linearized Arnoldi method;
- it shares the moment-matching guarantees of WCAWE at the shift $\sigma$, with a different orthogonalization policy that is sometimes more robust on heavily damped, non-symmetric systems such as the BGT pencil;
- it is compatible with the **block-structure-preserving (BSP) expansion** of Section 3.4 of the paper.

---

## 2. Moment matching for the quadratic pencil

Let $\sigma$ be a complex shift (typically $\sigma = j\,2\pi f_0 / c$ for a central frequency $f_0$ in the band of interest). Expand $\lambda = \sigma + \mu$ and rewrite Eq. (17) as

$$
\left[\mathbf{A}_0 + \mu\,\mathbf{A}_1 + \mu^2\mathbf{A}_2\right]\mathbf{P}_q(\mu) = \mathbf{F}_q,
$$

where

$$
\mathbf{A}_0 = \mathbf{D}_1 + \sigma\,\mathbf{D}_2 + \sigma^2\,\mathbf{D}_3, \qquad
\mathbf{A}_1 = \mathbf{D}_2 + 2\sigma\,\mathbf{D}_3, \qquad
\mathbf{A}_2 = \mathbf{D}_3.
$$

Assuming $\mathbf{A}_0$ is non-singular (which holds as long as $\sigma$ is not itself a resonance of the QEP), the solution admits the local power series

$$
\mathbf{P}_q(\mu) = \sum_{i \geq 0} \mathbf{r}_i\,\mu^i,
$$

whose coefficients $\mathbf{r}_i$ satisfy the three-term recurrence

$$
\mathbf{r}_0 = \mathbf{A}_0^{-1}\mathbf{F}_q, \qquad
\mathbf{r}_1 = -\mathbf{A}_0^{-1}\mathbf{A}_1\mathbf{r}_0, \qquad
\mathbf{r}_i = -\mathbf{A}_0^{-1}\left(\mathbf{A}_1\mathbf{r}_{i-1} + \mathbf{A}_2\mathbf{r}_{i-2}\right),\ i \geq 2.
$$

The ideal moment-matching subspace of order $n$ is therefore

$$
\mathcal{G}_n(\sigma) = \mathrm{span}\{\mathbf{r}_0,\ \mathbf{r}_1,\ \ldots,\ \mathbf{r}_{n-1}\}.
$$

Projecting Eq. (17) onto any basis whose span contains $\mathcal{G}_n(\sigma)$ yields a reduced transfer function matching the first $n$ moments of the full-order transfer function at $\sigma$. The power iterates $\mathbf{r}_i$ are, however, a numerically unstable basis: they collapse towards the dominant mode. A stable, orthonormal basis for $\mathcal{G}_n(\sigma)$ is required, and this is precisely what SOAR provides.

---

## 3. The SOAR recurrence

SOAR (Bai and Su, 2005) generates an orthonormal basis $\mathbf{Q}_n = [\mathbf{q}_1, \ldots, \mathbf{q}_n]$ of $\mathcal{G}_n(\sigma)$ via a two-vector recurrence that mimics Arnoldi on the linearized pencil but operates on vectors of the original size $(n_p + n_q)$.

Define the two operators

$$
\mathbf{B}_1 = -\mathbf{A}_0^{-1}\mathbf{A}_1, \qquad
\mathbf{B}_2 = -\mathbf{A}_0^{-1}\mathbf{A}_2.
$$

The algorithm maintains, in addition to $\mathbf{Q}_n$, an auxiliary vector sequence $\mathbf{P}_n = [\mathbf{p}_1, \ldots, \mathbf{p}_n]$ that plays the role of the "second row" of the companion iterate without ever being stored at the doubled size. A compact statement of the method is:

**Algorithm (SOAR).**

1. Compute $\mathbf{r}_0 = \mathbf{A}_0^{-1}\mathbf{F}_q$, set $\mathbf{q}_1 = \mathbf{r}_0 / \lVert \mathbf{r}_0\rVert$, $\mathbf{p}_1 = \mathbf{0}$.
2. For $j = 1, \ldots, n-1$:
   1. $\mathbf{t} = \mathbf{B}_1\mathbf{q}_j + \mathbf{B}_2\mathbf{p}_j$.
   2. **Modified Gram–Schmidt:** for $i = 1, \ldots, j$, set $h_{ij} = \mathbf{q}_i^H\mathbf{t}$ and $\mathbf{t} \leftarrow \mathbf{t} - h_{ij}\mathbf{q}_i$.
   3. $h_{j+1,j} = \lVert \mathbf{t}\rVert$.
   4. If $h_{j+1,j} > \text{tol}$: $\mathbf{q}_{j+1} = \mathbf{t} / h_{j+1,j}$; else trigger a **deflation** step (see §4).
   5. Update the auxiliary vector: $\mathbf{p}_{j+1} = \mathbf{q}_j - \sum_{i=1}^{j} h_{ij}\mathbf{p}_i$, rescaled by $1/h_{j+1,j}$.

Each iteration requires one linear solve with $\mathbf{A}_0$ plus two sparse matrix–vector products. The vectors $\mathbf{p}_j$ are not orthonormal — they are auxiliary quantities that make the recurrence closed.

After $n$ steps, $\mathbf{Q}_n$ spans $\mathcal{G}_n(\sigma)$ (up to breakdown), the columns are orthonormal, and the upper Hessenberg matrix $\mathbf{H}_n = (h_{ij})$ satisfies a second-order Arnoldi relation that is the direct analogue of the classical Arnoldi identity for linear problems.

---

## 4. Deflation and breakdown

Two breakdown modes must be handled.

- **Lucky breakdown** ($h_{j+1,j} \approx 0$ and $\mathbf{p}_{j+1} \neq 0$): the new Krylov vector has been exhausted but the auxiliary sequence still carries information. A common remedy is to set $\mathbf{q}_{j+1}$ to a fresh random vector orthogonalized against $\mathbf{Q}_j$ and continue.
- **Hard breakdown** ($h_{j+1,j} \approx 0$ and $\mathbf{p}_{j+1} \approx 0$): the subspace has saturated. The iteration terminates and the current $\mathbf{Q}_j$ is used as-is.

In practice, for the BGT pencil, deflation events are rare when the shift is well chosen inside the band and the number of moments $n$ remains below a few hundred, but the check must be present in any robust implementation.

---

## 5. Shift and order selection

Two parameters drive the quality of the reduced model:

- The **shift** $\sigma = j\,2\pi f_0 / c$ controls *where* in the frequency band the moments are matched. A single shift at the geometric mean of $[f_{\min}, f_{\max}]$ is usually sufficient for moderate bandwidths. For wider bands, a **multi-point SOAR** variant concatenates bases built at several shifts $\{\sigma_1, \ldots, \sigma_m\}$, followed by a global orthogonalization and optional SVD-based rank truncation.
- The **order** $n$ sets the dimension of the unreduced SOAR basis, and therefore the number of matched moments at each shift. As with WCAWE, a practical rule of thumb on damped radiating problems is to aim for $n$ on the order of twice the number of resonances falling inside the band $[f_{\min}, f_{\max}]$, with a conservative margin added for safety.

Residual-based adaptive strategies (enriching the basis when the reduced residual in the transfer function exceeds a target tolerance) are directly transposable from the WCAWE workflow.

---

## 6. Projection and BSP compatibility

Once the basis $\mathbf{Q}_n$ is assembled, the reduced model is obtained by a Galerkin projection on Eq. (17):

$$
\mathbf{Q}_n^H\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\mathbf{Q}_n\,\boldsymbol{\alpha}(k) = \mathbf{Q}_n^H \mathbf{F}_q.
$$

As for WCAWE and the QEP complex modal basis, the **block-structure-preserving (BSP) expansion** of Section 3.4 of the paper must be applied to protect against the magnitude imbalance between the pressure sub-block $\mathbf{K}$ and the boundary sub-blocks $\mathbf{G}_{12}, \mathbf{E}_1$. Decomposing each column of $\mathbf{Q}_n$ as

$$
\mathbf{q}_i = \begin{bmatrix}\mathbf{q}_{p,i} \\ \mathbf{q}_{q,i}\end{bmatrix}
$$

and forming

$$
\tilde{\mathbf{Q}}_n = \begin{bmatrix}\mathbf{Q}_p & \mathbf{0} \\ \mathbf{0} & \mathbf{Q}_q\end{bmatrix} \in \mathbb{C}^{(n_p+n_q) \times 2n},
$$

yields a reduced system of size $2n \times 2n$ whose sub-block structure mirrors Eq. (16). This is the same BSP treatment applied to WCAWE in Section 3.4 and to the QEP complex modes in the companion note: it ensures numerical consistency across the three reduction strategies compared in the paper.

---

## 7. Summary of the procedure

1. **Assemble** the frequency-independent matrices $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3$ from Eq. (16).
2. **Choose** the shift $\sigma = j\,2\pi f_0 / c$ and the SOAR order $n$.
3. **Form** the shifted operators $\mathbf{A}_0, \mathbf{A}_1, \mathbf{A}_2$ and factorize $\mathbf{A}_0$ once (e.g. MUMPS LU).
4. **Run** the SOAR recurrence of §3, applying modified Gram–Schmidt at each step, with deflation if needed, to obtain the orthonormal basis $\mathbf{Q}_n$.
5. **Apply the BSP expansion** to obtain $\tilde{\mathbf{Q}}_n$.
6. **Project** Eq. (17) onto $\tilde{\mathbf{Q}}_n$ to obtain a reduced system of size $2n \times 2n$.
7. **Sweep** the frequency band on the reduced system. No reassembly is required, since the reduced system inherits the polynomial frequency dependence of the full-order system.

---

## 8. Implementation notes for the FEniCSx / petsc4py / SLEPc stack

- The factorization of $\mathbf{A}_0$ is best carried out via **MUMPS through PETSc**, matching the factorization already used for the full-order sweep and for the shift-invert step of the QEP solver. The same `KSP` object can be reused across all $n$ SOAR iterations, amortizing the factorization cost.
- Sparse matrix-vector products with $\mathbf{A}_1$ and $\mathbf{A}_2$ are standard PETSc operations. The auxiliary vector $\mathbf{p}_j$ should be stored as a dense PETSc vector of size $(n_p + n_q)$, not expanded to the linearized size.
- Modified Gram–Schmidt is preferable to classical Gram–Schmidt for this non-symmetric, heavily damped problem. A single re-orthogonalization pass can be added if numerical orthogonality of $\mathbf{Q}_n$ degrades past a threshold (typically $10^{-10}$).
- SLEPc does **not** expose a public `BV`-based SOAR routine in the same way it does for Arnoldi, but the algorithm is short enough to be coded directly on top of `petsc4py`. Alternatively, SLEPc's `PEP` module with `PEPSetType(pep, PEPTOAR)` offers TOAR (Two-level Orthogonal ARnoldi, Lu et al., 2016), a numerically more robust successor of SOAR that can be used as a drop-in replacement for eigenvalue-based selection of the basis.
- The reduced dense system of size $2n \times 2n$ assembled at projection time can be handled by NumPy/SciPy or by PETSc `seqdense` matrices converted to `seqaij` before the per-frequency MUMPS solve (matching the approach already used in the WCAWE and complex-modal online stages).

---

## 9. Expected behavior and limitations

**Strengths**

- Preserves the second-order structure of the QEP: operates on vectors of size $(n_p + n_q)$, not $2(n_p + n_q)$.
- Single linear solve per iteration — the same cost profile as WCAWE, and an order of magnitude cheaper per basis vector than the QEP complex-modal approach.
- Moment-matching guarantees at the chosen shift, with a clean path to multi-point and adaptive enrichment.
- Fully compatible with the BSP machinery and with the direct-factorization infrastructure already in place.

**Limitations**

- The basis quality is sensitive to the choice of shift; a shift close to a resonance of the QEP makes $\mathbf{A}_0$ ill-conditioned and degrades the recurrence.
- For very large orders $n$, modified Gram–Schmidt eventually loses orthogonality and a re-orthogonalization pass (or the TOAR variant) becomes necessary.
- Does not provide a physically interpretable basis: the columns of $\mathbf{Q}_n$ are Krylov iterates, not modes. When physical interpretability of the reduced coordinates is desired, the QEP complex-modal projection remains preferable.
- Like WCAWE, SOAR is a single-input-single-output moment-matching method as stated above; vector-valued or parametric inputs require a block SOAR variant.

Overall, SOAR sits naturally between WCAWE and QEP complex modal projection in the reduction landscape of Eq. (17): it shares the second-order preservation and interpretability of the latter, the cost profile of the former, and is compatible with the BSP expansion common to all three.
