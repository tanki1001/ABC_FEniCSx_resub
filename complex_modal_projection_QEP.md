# Complex Modal Projection via the Quadratic Eigenvalue Problem

A model order reduction strategy for the frequency-independent BGT formulation of Eq. (17)

---

## 1. Context and motivation

The frequency-independent BGT formulation derived in the paper leads to the assembled system

$$
\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\mathbf{P}_q = \mathbf{F}_q,
$$

where $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3 \in \mathbb{C}^{(n_p+n_q) \times (n_p+n_q)}$ are the frequency-independent block matrices defined in Eq. (16). The structural features of this system are:

- **Non-symmetry** of $\mathbf{D}_1$ and $\mathbf{D}_2$, induced by the auxiliary-variable coupling between the acoustic pressure block and the boundary unknowns;
- **Strong effective damping**, carried by the $jk\,\mathbf{D}_2$ term, which encodes the radiation losses imposed by the BGT operator on $\Gamma_3$;
- **Block coupling** between pressure degrees of freedom and auxiliary degrees of freedom, requiring the BSP expansion during projection (Eq. (29)).

Standard real-valued modal truncation built on the undamped problem $\mathbf{K}\boldsymbol{\phi} = \omega^2 \mathbf{M}\boldsymbol{\phi}$ is not well suited here: such modes are agnostic to the radiation damping and therefore fail to represent the dominant dynamics when the BGT contribution is significant.

A principled alternative in the modal family is to perform the projection on a basis of **complex modes** obtained from the Quadratic Eigenvalue Problem (QEP) associated with Eq. (17). This preserves the modal projection paradigm while honestly accounting for both the non-symmetry and the damping.

---

## 2. Formulation of the Quadratic Eigenvalue Problem

Setting the right-hand side of Eq. (17) to zero and looking for non-trivial solutions of the form $\mathbf{P}_q = \boldsymbol{\phi}\, e^{j\omega t}$ with $\lambda = jk$ yields the QEP

$$
\left[\mathbf{D}_1 + \lambda\,\mathbf{D}_2 + \lambda^2 \tilde{\mathbf{D}}_3\right]\boldsymbol{\phi} = \mathbf{0},
$$

where $\tilde{\mathbf{D}}_3 = \mathbf{D}_3 / \text{(sign convention adjusted to match }\lambda = jk\text{)}$. In practice, since $-k^2 = (jk)^2 = \lambda^2$, the coefficient of the quadratic term is simply $\mathbf{D}_3$ with a sign flip, and the QEP becomes

$$
\left[\mathbf{D}_1 + \lambda\,\mathbf{D}_2 + \lambda^2\,\mathbf{D}_3\right]\boldsymbol{\phi} = \mathbf{0}.
$$

This problem admits $2(n_p + n_q)$ complex eigenpairs $(\lambda_i, \boldsymbol{\phi}_i)$. Each eigenvalue $\lambda_i$ carries physical meaning:

- $\Im(\lambda_i)$ relates to a resonance frequency of the coupled acoustic–BGT system;
- $\Re(\lambda_i)$ encodes the effective damping of that resonance (in the exterior acoustic setting, this reflects how efficiently the BGT condition radiates energy away through $\Gamma_3$).

The complex eigenvectors $\boldsymbol{\phi}_i$ are the **complex modes** of the radiating cavity with the BGT condition in place. They are the natural modal basis for this problem.

---

## 3. Linearization of the QEP

QEPs are typically solved by first **linearizing** them into a generalized eigenvalue problem (GEP) of twice the dimension. A standard companion linearization reads

$$
\mathbf{A}\,\mathbf{z} = \lambda\,\mathbf{B}\,\mathbf{z}, \qquad \mathbf{z} = \begin{bmatrix}\boldsymbol{\phi} \\ \lambda\,\boldsymbol{\phi}\end{bmatrix},
$$

with

$$
\mathbf{A} = \begin{bmatrix}\mathbf{0} & \mathbf{I} \\ -\mathbf{D}_1 & -\mathbf{D}_2\end{bmatrix}, \qquad
\mathbf{B} = \begin{bmatrix}\mathbf{I} & \mathbf{0} \\ \mathbf{0} & \mathbf{D}_3\end{bmatrix}.
$$

The linearized pencil $(\mathbf{A}, \mathbf{B})$ is of size $2(n_p + n_q)$. This doubling of dimension is the main upfront cost of the method.

In practice, one does not need to form $\mathbf{A}$ and $\mathbf{B}$ explicitly. Both SLEPc's `PEP` module (which handles the QEP directly) and SciPy's sparse eigensolvers (via an explicit linearization) can be used.

---

## 4. Mode selection strategy

Since only a small subset of the $2(n_p + n_q)$ eigenpairs is needed, a **shift-invert** strategy targeting the frequency band of interest is essential.

Let $[f_{\min}, f_{\max}]$ be the band of interest and $f_0$ a central frequency (for instance the geometric mean). The corresponding target eigenvalue is

$$
\lambda_0 = j\,\frac{2\pi f_0}{c}.
$$

The eigensolver is then configured to compute the $N$ eigenvalues closest to $\lambda_0$ in the complex plane. A typical heuristic is to take $N$ such that the retained modes cover the band $[f_{\min}, f_{\max}]$ plus a buffer on either side, since modes outside the band still contribute to the response inside it through their tails.

For a well-conditioned basis, it is good practice to:

- Retain both modes from each complex-conjugate pair if they exist;
- Avoid near-duplicate modes by applying a small tolerance on $|\lambda_i - \lambda_j|$;
- Optionally, include a few **static correction vectors** (solutions of $\mathbf{D}_1 \mathbf{v} = \mathbf{F}_q$ at selected frequencies) to capture the quasi-static component of the response that pure modes may miss.

---

## 5. Projection and BSP compatibility

Once the $N$ complex modes are assembled into

$$
\boldsymbol{\Phi} = [\boldsymbol{\phi}_1, \ldots, \boldsymbol{\phi}_N] \in \mathbb{C}^{(n_p+n_q) \times N},
$$

the reduced model is obtained by Galerkin projection:

$$
\boldsymbol{\Phi}^H\left[\mathbf{D}_1 + jk\,\mathbf{D}_2 - k^2\mathbf{D}_3\right]\boldsymbol{\Phi}\,\boldsymbol{\alpha}(k) = \boldsymbol{\Phi}^H \mathbf{F}_q.
$$

As for WCAWE in the paper, the **block-structure-preserving (BSP) expansion** should be applied to protect against the magnitude imbalance between the pressure sub-block $\mathbf{K}$ and the boundary sub-blocks $\mathbf{G}_{12}, \mathbf{E}_1$. Decomposing each mode as

$$
\boldsymbol{\phi}_i = \begin{bmatrix}\boldsymbol{\phi}_{p,i} \\ \boldsymbol{\phi}_{q,i}\end{bmatrix}
$$

and forming

$$
\tilde{\boldsymbol{\Phi}} = \begin{bmatrix}\boldsymbol{\Phi}_p & \mathbf{0} \\ \mathbf{0} & \boldsymbol{\Phi}_q\end{bmatrix} \in \mathbb{C}^{(n_p+n_q) \times 2N},
$$

yields a reduced system of size $2N \times 2N$ whose sub-block structure mirrors Eq. (16). This is the same treatment used in Section 3.4 of the paper and ensures numerical consistency between the two approaches.

---

## 6. Summary of the procedure

1. **Assemble** the frequency-independent matrices $\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3$ from Eq. (16).
2. **Linearize** the QEP into a GEP of size $2(n_p + n_q)$ (or use a direct polynomial solver).
3. **Solve** the (linearized) eigenproblem via a shift-invert Krylov method (Arnoldi) targeting the frequency band of interest, retaining $N$ complex modes.
4. **Assemble** the modal basis $\boldsymbol{\Phi}$ and apply the BSP expansion to obtain $\tilde{\boldsymbol{\Phi}}$.
5. **Project** Eq. (17) onto $\tilde{\boldsymbol{\Phi}}$ to obtain a reduced system of size $2N \times 2N$.
6. **Sweep** the frequency band on the reduced system. No reassembly is required, since the reduced system inherits the polynomial frequency dependence of the full-order system.

---

## 7. Implementation notes for the FEniCSx / petsc4py / SLEPc stack

The current implementation already relies on PETSc and DOLFINx, which makes SLEPc the natural eigensolver:

- SLEPc's `PEP` module (Polynomial Eigenvalue Problem) accepts the coefficient matrices $(\mathbf{D}_1, \mathbf{D}_2, \mathbf{D}_3)$ directly. Internally, it performs the linearization and calls an Arnoldi-type solver on the linearized pencil.
- A shift-invert spectral transformation (`STSetType(st, STSINVERT)`) is applied around the target eigenvalue $\lambda_0$.
- The target is set via `PEPSetTarget(pep, target)` with `target = 1j * 2 * pi * f_0 / c`.
- `PEPSetWhichEigenpairs(pep, PEP_TARGET_MAGNITUDE)` selects eigenvalues by proximity to the target.

For the linear solver used inside shift-invert, a direct factorization (MUMPS through PETSc) is robust and aligns with what is already used for the full-order sweep.

---

## 8. Expected behavior and limitations

**Strengths**

- Natively handles non-symmetry and heavy damping: the BGT-induced radiation losses are built into the eigenproblem rather than treated as a perturbation.
- Provides a physically meaningful basis: each basis vector corresponds to a resonance of the coupled acoustic–radiating system.
- Reuses the frequency-independent structure of Eq. (17): no reassembly at any stage of the reduced sweep.
- Compatible with the BSP approach already developed for WCAWE, preserving consistency with the rest of the paper's framework.

**Limitations**

- Upfront cost: the QEP solution dominates the offline phase and is significantly more expensive than constructing a WCAWE basis, because it operates on a system of doubled size.
- Basis size: achieving accuracy comparable to WCAWE generally requires more modes, since complex modes are less "tailored" to a given excitation than the moment-matching vectors produced by WCAWE.
- Sensitivity to mode selection: missing a few high-residue modes near the band of interest can noticeably degrade accuracy; adaptive enrichment may be necessary.

Overall, complex modal projection via the QEP is a defensible modal-family alternative to WCAWE. It does not aim to outperform WCAWE in efficiency, but provides a rigorous modal benchmark that honestly addresses the coupling and damping of the proposed formulation.
