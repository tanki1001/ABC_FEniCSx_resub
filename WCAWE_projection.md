# WCAWE: Well-Conditioned Asymptotic Waveform Evaluation for the Frequency-Independent BGT System

A moment-matching model order reduction strategy that matches the first $n$ moments of the transfer function of Eq. (17) at a chosen shift, using a well-conditioned orthonormal basis rather than the ill-conditioned power iterates of classical AWE.

---

## 1. Context and motivation

As in the companion notes on complex modal projection and SOAR, the frequency-independent BGT formulation of Eq. (17) reads

$$
\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\mathbf{P}_q = \mathbf{F}_q,
$$

with $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3 \in \mathbb{C}^{(n_p+n_q) \times (n_p+n_q)}$ the frequency-independent block matrices of Eq. (16). Setting $\lambda = jk$ turns this into the polynomial parametric system

$$
\left[\mathbf{D}_1 + \lambda\,\mathbf{D}_2 + \lambda^2\mathbf{D}_3\right]\mathbf{P}_q(\lambda) = \mathbf{F}_q.
$$

Within the moment-matching family, three methods occupy distinct niches:

- **Classical AWE** (Asymptotic Waveform Evaluation): explicitly computes the power-series coefficients $\mathbf{r}_i$ of $\mathbf{P}_q$ at a shift $\sigma$ and uses them directly as a basis. Easy to state, but numerically unstable: the $\mathbf{r}_i$ collapse toward the dominant mode, so the basis becomes rank-deficient after only a few iterations.
- **SOAR** (Bai and Su, 2005): generates an orthonormal Krylov basis for the quadratic system through a two-term recurrence involving an auxiliary vector, preserving the second-order structure.
- **WCAWE** (Slone, Lee and Lee, 2003): generates an orthonormal basis whose span equals the same moment-matching subspace as AWE, but via a Gram–Schmidt–style recurrence augmented by **correction matrices** that track the orthogonalization history. WCAWE applies to polynomial parametric systems of arbitrary order $p$ (not only $p = 2$), and is the moment-matching engine adopted in the paper.

The motivation for WCAWE is therefore to combine:

- the **cost profile** of a Krylov iteration (a single linear solve with a frequency-independent operator per basis vector, plus cheap sparse matrix–vector products and a small dense update);
- the **moment-matching guarantee** of AWE at the shift $\sigma$ (the first $n$ moments of the reduced transfer function match those of the full-order system);
- a **well-conditioned orthonormal basis**, avoiding the rank collapse that makes naive AWE unusable past a handful of iterations;
- full compatibility with the **block-structure-preserving (BSP) expansion** of Section 3.4 of the paper.

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

Assuming $\mathbf{A}_0$ is non-singular, the local power series

$$
\mathbf{P}_q(\mu) = \sum_{i \geq 0} \mathbf{r}_i\,\mu^i
$$

has coefficients $\mathbf{r}_i$ satisfying the three-term recurrence

$$
\mathbf{r}_0 = \mathbf{A}_0^{-1}\mathbf{F}_q, \qquad
\mathbf{r}_1 = -\mathbf{A}_0^{-1}\mathbf{A}_1\mathbf{r}_0, \qquad
\mathbf{r}_i = -\mathbf{A}_0^{-1}\left(\mathbf{A}_1\mathbf{r}_{i-1} + \mathbf{A}_2\mathbf{r}_{i-2}\right),\ i \geq 2.
$$

The ideal moment-matching subspace of order $n$ is

$$
\mathcal{G}_n(\sigma) = \mathrm{span}\{\mathbf{r}_0,\ \mathbf{r}_1,\ \ldots,\ \mathbf{r}_{n-1}\}.
$$

Any orthonormal basis whose span equals $\mathcal{G}_n(\sigma)$ yields a Galerkin reduced model matching the first $n$ moments of the full-order transfer function at $\sigma$. WCAWE constructs such a basis **incrementally**, replacing the ill-conditioned $\mathbf{r}_i$ with orthonormal vectors while propagating the recursive dependence through a stack of small dense correction matrices.

---

## 3. The WCAWE recurrence for a quadratic pencil

Let $\mathbf{V}_n = [\mathbf{v}_1, \ldots, \mathbf{v}_n]$ denote the WCAWE basis. Its construction relies on upper-triangular **correction matrices** $\mathbf{U}^{(k)} \in \mathbb{C}^{n \times n}$ (one per non-leading coefficient of the polynomial pencil, so $\mathbf{U}^{(1)}$ and $\mathbf{U}^{(2)}$ for the quadratic case) that accumulate the orthogonalization coefficients and make the recursion *closed* even though the basis vectors are no longer the $\mathbf{r}_i$ themselves. At each step, a new vector is computed, orthogonalized against all previous basis vectors via modified Gram–Schmidt, normalized, and then used to update both $\mathbf{V}_n$ and the correction matrices.

**Algorithm (WCAWE, quadratic pencil).**

1. **Initialization.**
   - Solve $\mathbf{A}_0\,\tilde{\mathbf{v}}_1 = \mathbf{F}_q$.
   - Set $\eta_1 = \lVert\tilde{\mathbf{v}}_1\rVert$, $\mathbf{v}_1 = \tilde{\mathbf{v}}_1 / \eta_1$.
   - Initialize the correction matrices so that $\mathbf{U}^{(1)}_{1,1} = 1/\eta_1$ (and analogously for $\mathbf{U}^{(2)}$).

2. **Recurrence.** For $j = 2, \ldots, n$:
   1. Form the unreduced moment vector
      $$
      \tilde{\mathbf{w}}_j = -\mathbf{A}_0^{-1}\Bigl(\mathbf{A}_1\,\mathbf{V}_{j-1}\,\mathbf{u}^{(1)}_{j-1}\ +\ \mathbf{A}_2\,\mathbf{V}_{j-2}\,\mathbf{u}^{(2)}_{j-2}\Bigr),
      $$
      where $\mathbf{V}_{j-2}$ is taken empty when $j = 2$, and $\mathbf{u}^{(1)}_{j-1}$, $\mathbf{u}^{(2)}_{j-2}$ are the *last columns* of the propagated correction matrices (formally, products of appropriate sub-matrices of $\mathbf{U}^{(1)}$ and $\mathbf{U}^{(2)}$ that encode the recursive history; their exact expressions are given in Slone et al. (2003), Eqs. (18)–(22) of the original paper).
   2. **Modified Gram–Schmidt:** for $i = 1, \ldots, j-1$, set $h_{ij} = \mathbf{v}_i^H\tilde{\mathbf{w}}_j$ and $\tilde{\mathbf{w}}_j \leftarrow \tilde{\mathbf{w}}_j - h_{ij}\,\mathbf{v}_i$.
   3. $h_{jj} = \lVert\tilde{\mathbf{w}}_j\rVert$. If $h_{jj} < \text{tol}$ trigger a **deflation** step (see §4); otherwise set $\mathbf{v}_j = \tilde{\mathbf{w}}_j / h_{jj}$.
   4. Update $\mathbf{U}^{(1)}$ and $\mathbf{U}^{(2)}$ by inserting the column $(h_{1j}, \ldots, h_{jj})^T$ appropriately, so that the identity $\mathrm{span}\{\mathbf{v}_1, \ldots, \mathbf{v}_j\} = \mathcal{G}_j(\sigma)$ is preserved.

Each iteration requires exactly **one linear solve** with $\mathbf{A}_0$ and at most two sparse matrix–vector products (one with $\mathbf{A}_1$ and one with $\mathbf{A}_2$), plus the small dense bookkeeping of the correction matrices. The linear solve dominates the cost; the factorization of $\mathbf{A}_0$ is therefore computed once and reused for all $n$ iterations.

The correction matrices are the mechanism that distinguishes WCAWE from naive orthogonalization of AWE moments: without them, orthonormalizing the $\mathbf{r}_i$ step-by-step would yield vectors that *do not* span the moment subspace after rank collapse. The $\mathbf{U}^{(k)}$ keep track of "what was removed from each $\mathbf{r}_i$ during orthogonalization" and re-inject it through the right-hand side at subsequent iterations, so that the span identity $\mathrm{span}\{\mathbf{V}_j\} = \mathcal{G}_j(\sigma)$ is maintained exactly.

---

## 4. Breakdown and deflation

As in SOAR, two breakdown modes must be handled.

- **Lucky breakdown** ($h_{jj} \approx 0$ with a non-trivial right-hand side at the next step): the moment subspace has saturated locally but the pencil still has unexplored content. A fresh random vector orthogonalized against $\mathbf{V}_{j-1}$ is introduced and the recursion continues, with the correction matrices re-initialized on the new column.
- **Hard breakdown** ($h_{jj} \approx 0$ and the updated right-hand side also vanishes): the subspace is exhausted. The iteration terminates with the current $\mathbf{V}_{j-1}$.

For the BGT pencil, breakdown is rare when the shift $\sigma$ is placed inside the band and $n$ remains at realistic sizes (a few tens to a few hundreds). A single re-orthogonalization pass is recommended when $\lvert h_{jj}\rvert / \lVert \tilde{\mathbf{w}}_j\rVert_{\text{initial}}$ drops below a threshold such as $10^{-10}$.

---

## 5. Shift, order and multi-point strategies

Two parameters drive the quality of the reduced model:

- The **shift** $\sigma = j\,2\pi f_0 / c$ controls *where* in the frequency band the moments are matched. A single shift at the geometric mean of $[f_{\min}, f_{\max}]$ is usually adequate for moderate bandwidths. As in SOAR, a shift that lands close to a resonance of the QEP makes $\mathbf{A}_0$ ill-conditioned.
- The **order** $n$ sets the number of matched moments. A practical rule of thumb on damped radiating problems is to take $n$ on the order of twice the number of resonances falling inside $[f_{\min}, f_{\max}]$, with a conservative safety margin.

For broader bands, **multi-point WCAWE** (sometimes called MWCAWE) runs the recurrence at several shifts $\{\sigma_1, \ldots, \sigma_m\}$, concatenates the resulting bases, and applies a global SVD-based rank truncation to remove near-collinear directions between the sub-bases. Residual-based **adaptive enrichment** — adding moments or shifts when the reduced residual in the transfer function exceeds a target tolerance — is a direct extension and is commonly used in the paper's workflow.

---

## 6. Projection and BSP compatibility

Once the basis $\mathbf{V}_n$ is assembled, the reduced model follows from the Galerkin projection of Eq. (17):

$$
\mathbf{V}_n^H\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\mathbf{V}_n\,\boldsymbol{\alpha}(k) = \mathbf{V}_n^H \mathbf{F}_q.
$$

As for SOAR and the QEP complex modal basis, the **block-structure-preserving (BSP) expansion** of Section 3.4 of the paper must be applied to protect against the magnitude imbalance between the pressure sub-block $\mathbf{K}$ and the boundary sub-blocks $\mathbf{G}_{12}, \mathbf{E}_1$. Decomposing each column of $\mathbf{V}_n$ as

$$
\mathbf{v}_i = \begin{bmatrix}\mathbf{v}_{p,i} \\ \mathbf{v}_{q,i}\end{bmatrix}
$$

and forming

$$
\tilde{\mathbf{V}}_n = \begin{bmatrix}\mathbf{V}_p & \mathbf{0} \\ \mathbf{0} & \mathbf{V}_q\end{bmatrix} \in \mathbb{C}^{(n_p+n_q) \times 2n},
$$

yields a reduced system of size $2n \times 2n$ whose sub-block structure mirrors Eq. (16). The BSP expansion is essential here: without it, the projected pressure and boundary blocks mix and the conditioning of the reduced polynomial pencil deteriorates sharply.

---

## 7. Summary of the procedure

1. **Assemble** the frequency-independent matrices $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3$ from Eq. (16).
2. **Choose** the shift $\sigma = j\,2\pi f_0 / c$ and the WCAWE order $n$.
3. **Form** the shifted operators $\mathbf{A}_0, \mathbf{A}_1, \mathbf{A}_2$ and factorize $\mathbf{A}_0$ once (e.g. MUMPS LU).
4. **Run** the WCAWE recurrence of §3 with modified Gram–Schmidt, propagating the correction matrices $\mathbf{U}^{(1)}, \mathbf{U}^{(2)}$, and deflating if needed, to obtain the orthonormal basis $\mathbf{V}_n$.
5. **Apply the BSP expansion** to obtain $\tilde{\mathbf{V}}_n$.
6. **Project** Eq. (17) onto $\tilde{\mathbf{V}}_n$ to obtain a reduced system of size $2n \times 2n$.
7. **Sweep** the frequency band on the reduced system. No reassembly is required, since the reduced system inherits the polynomial frequency dependence of the full-order system.

---

## 8. Implementation notes for the FEniCSx / petsc4py stack

- The factorization of $\mathbf{A}_0$ is best carried out via **MUMPS through PETSc**, matching the factorization used for the full-order sweep, for the shift-invert step of the QEP solver, and for the SOAR recurrence. The same `KSP` object should be reused across all $n$ WCAWE iterations to amortize the factorization cost.
- The correction matrices $\mathbf{U}^{(1)}, \mathbf{U}^{(2)}$ are small upper-triangular matrices of size $n \times n$ and are stored densely in NumPy. Their updates at each step amount to inserting one column of Gram–Schmidt coefficients.
- Modified Gram–Schmidt is preferred over classical Gram–Schmidt for this non-symmetric, heavily damped problem, with an optional re-orthogonalization pass as a safety net.
- Neither SLEPc nor DOLFINx ships a ready-made WCAWE routine; the recurrence is short enough to be coded directly on top of `petsc4py`, using PETSc `Vec` operations for the orthogonalization and PETSc `Mat`–`Vec` products for the sparse applications of $\mathbf{A}_1$ and $\mathbf{A}_2$.
- The reduced dense system of size $2n \times 2n$ assembled at projection time can be handled by NumPy/SciPy or by PETSc `seqdense` matrices converted to `seqaij` before the per-frequency MUMPS solve — the same pattern used in the SOAR and complex-modal online stages.
- In the current codebase, the sub-blocks $(\mathbf{K}, \mathbf{M}, \mathbf{C}, \mathbf{G}_{12}, \mathbf{G}_3, \mathbf{G}_4, \mathbf{E}_1, \mathbf{E}_2)$ of Eq. (16) are already cached on the operator object, so the BSP-projected reduced matrices can be built block by block via sparse $\times$ dense products without reassembling the global $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3$.

---

## 9. Expected behavior and limitations

**Strengths**

- Well-conditioned orthonormal basis: no rank collapse, unlike naive AWE.
- Same cost profile as SOAR: one linear solve with $\mathbf{A}_0$ per iteration, plus sparse matrix–vector products and small dense bookkeeping.
- Matches the first $n$ moments of the full-order transfer function at the chosen shift, providing a quantifiable local accuracy guarantee.
- Naturally extends to multi-point and adaptive strategies.
- Applicable to polynomial pencils of arbitrary order, not only quadratic; the quadratic specialization above is what is needed for Eq. (17), but the machinery transfers unchanged to higher-order formulations.
- Fully compatible with the BSP expansion of Section 3.4.

**Limitations**

- Accuracy is inherently **local** around $\sigma$: away from the shift, the reduced model degrades and eventually loses accuracy. Multi-point or adaptive variants are the standard remedy but add algorithmic complexity.
- Like SOAR, WCAWE is sensitive to the choice of shift: a shift close to a resonance of the QEP makes $\mathbf{A}_0$ ill-conditioned and degrades the recurrence.
- Does not provide a physically interpretable basis: the columns of $\mathbf{V}_n$ are moment-matching iterates, not modes. When physical interpretability of the reduced coordinates is required, the QEP complex-modal projection remains preferable.
- As stated above, WCAWE is a single-input-single-output method; vector-valued or parametric inputs require a block WCAWE variant.

Overall, WCAWE is the moment-matching workhorse of the paper. Together with SOAR (a Krylov alternative with equivalent cost profile but specialized to quadratic pencils) and the QEP complex modal projection (a modal alternative with higher offline cost but globally valid over the band), it spans the three principal MOR strategies compared in this work for the frequency-independent BGT formulation of Eq. (17).
