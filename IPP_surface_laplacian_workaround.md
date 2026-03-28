# Integration by Parts of the Surface Laplacian in a Mixed-Domain FEniCSx Setting

## 1. Context: the BGT-2 block system

The BGT-2 absorbing boundary condition leads to a coupled block system in $(p, q)$ where $p \in P$ (3D volume mesh) and $q \in Q$ (2D submesh extracted from the radiation boundary $\Gamma$). The global matrix is polynomial in $jk$:

$$\mathbf{Z}(k) = \mathbf{D}_1 + (jk)\,\mathbf{D}_2 + (jk)^2\,\mathbf{D}_3$$

Each $\mathbf{D}_i$ is a $2 \times 2$ block matrix:

$$\mathbf{D}_1 = \begin{bmatrix} K & C \\ G_{1} + G_4 & E_1 \end{bmatrix}$$

The block $(1,0)$ contains the term $G_1$ which, in the **strong form** (`B2p_tang`), reads:

$$G_1[u, p] = \int_\Gamma (-\Delta_s p)\, u \, d\Gamma$$

where $\Delta_s$ is the surface Laplacian (Laplace-Beltrami operator) and $u \in Q$ is the test function on the submesh.

---

## 2. Goal: integration by parts (IPP)

The strong form requires second surface derivatives of $p$, which are problematic for $H^1$-conforming Lagrange elements (derivatives are discontinuous across elements). The IPP transfers one derivative to the test function:

$$\int_\Gamma (-\Delta_s p)\, u \, d\Gamma = \int_\Gamma \nabla_s p \cdot \nabla_s u \, d\Gamma \;-\; \int_{\partial\Gamma} (\nabla_s p \cdot \nu)\, u \, dl$$

For a **closed surface** $\Gamma$ (e.g. the biSpherical geometry where all facets are tagged as radiation boundary), $\partial\Gamma = \emptyset$ and the boundary term vanishes. For domains with symmetry planes, $\nabla_s p \cdot \nu = 0$ on $\partial\Gamma$ by symmetry.

So the IPP form is:

$$G_1^{\text{ipp}}[u, p] = \int_\Gamma \nabla_s p \cdot \nabla_s u \, d\Gamma$$

---

## 3. The problem: cross-mesh gradients in FEniCSx

### 3.1 The two meshes

| Mesh | Dimension | Function space | Integration measure |
|------|-----------|---------------|-------------------|
| Parent mesh (3D tetrahedra) | $d = 3$ | $P$ (Lagrange, degree $k$) | `dx`, `ds(3)` |
| Submesh (2D triangles on $\Gamma$) | $d = 2$ embedded in $\mathbb{R}^3$ | $Q$ (Lagrange, degree $k$) | `dx1` |

The coupling between $P$ and $Q$ is handled by **entity maps** that associate submesh cells to parent mesh facets.

### 3.2 Attempt 1: `ds(3)` with tangential projections

```python
n = FacetNormal(mesh)
gradt_p = tangential_proj(grad(p), n)   # p in P (3D)
gradt_u = tangential_proj(grad(u), n)   # u in Q (2D submesh)
g1 = inner(gradt_p, gradt_u) * ds(3)
```

Here `ds(3)` integrates over **parent mesh facets**. At each quadrature point:
- `grad(p)` is computed using the **3D cell Jacobian** -- correct
- `grad(u)` must be computed for a submesh function at parent mesh quadrature points, using the **submesh cell Jacobian**

On **curved P2 elements**, the parent mesh facet geometry and the submesh cell geometry may not be perfectly consistent: they use different geometric mappings (3D cell restricted to a face vs. standalone 2D cell). This causes `grad(u)` to be evaluated with an inconsistent Jacobian, producing **numerically wrong surface gradients**.

**Result**: wildly oscillating pressure values (see plot), completely wrong compared to the COMSOL reference and the strong-form `B2p_tang`.

### 3.3 Attempt 2: `dx1` with both gradients

```python
g1 = inner(grad(p), grad(u)) * dx1
```

Here `dx1` integrates over **submesh cells**. Both gradients would be computed on the same 2D geometry. However, `p` is a `TrialFunction(P)` defined on the 3D parent mesh.

**Result**: FFCx compilation error. FFCx needs to tabulate the 3D basis functions of $P$ on the 2D submesh reference element, which is not supported:
```
UnboundLocalError: cannot access local variable 't' where it is not associated with a value
```
in `ffcx/ir/elementtables.py`, because the element table for a 3D Lagrange element does not exist on a 2D reference cell.

### 3.4 Summary of the fundamental limitation

| Approach | `grad(p)` | `grad(u)` | Measure | Issue |
|----------|-----------|-----------|---------|-------|
| `ds(3)` + tangential proj. | 3D Jacobian (correct) | Submesh Jacobian at parent mesh quad. points (inconsistent on curved elements) | Parent mesh facets | Wrong values |
| `dx1` | Cannot tabulate 3D basis on 2D ref. element | 2D Jacobian (correct) | Submesh cells | FFCx compilation error |

**Core issue**: there is no way in FEniCSx to compute `grad` of a function from one mesh at the quadrature points of another mesh with a different topological dimension.

---

## 4. The solution: $\mathbf{K}_{\text{surf}} \cdot \mathbf{I}_{PQ}$ decomposition

### 4.1 Mathematical idea

We decompose the bilinear form $G_1^{\text{ipp}}$ using the **restriction operator** $R: P \to Q$ that maps a function in $P$ to its trace on $\Gamma$ expressed in $Q$:

$$\int_\Gamma \nabla_s p \cdot \nabla_s u \, d\Gamma = \int_\Gamma \nabla_s (Rp) \cdot \nabla_s u \, d\Gamma$$

Since $Rp \in Q$, the right-hand side is a **purely submesh bilinear form** (both trial and test in $Q$). Writing:

- $\mathbf{K}_{\text{surf}}$: the surface stiffness matrix on $Q$, with entries $[\mathbf{K}_{\text{surf}}]_{ij} = \int_\Gamma \nabla_s \phi_j^Q \cdot \nabla_s \phi_i^Q \, d\Gamma$
- $\mathbf{I}_{PQ}$: the matrix representation of $R$, mapping $P$ DOF coefficients to $Q$ DOF coefficients

Then the $G_1$ block contribution is:

$$\boxed{\mathbf{G}_1 = \mathbf{K}_{\text{surf}} \cdot \mathbf{I}_{PQ}}$$

### 4.2 Why $\mathbf{I}_{PQ}$ is trivial for Lagrange elements

For Lagrange elements of the same degree $k$, the DOFs are point evaluations at geometric nodes. The submesh $\Gamma$ is extracted from the parent mesh boundary facets, so:

- Each $Q$ DOF at position $\mathbf{x}_i$ on the submesh corresponds to **exactly one** $P$ DOF at the same position $\mathbf{x}_i$ on the parent mesh boundary
- Therefore $\mathbf{I}_{PQ}$ is a **Boolean matrix** of size $n_Q \times n_P$ with exactly one entry of 1 per row

$$[\mathbf{I}_{PQ}]_{ij} = \begin{cases} 1 & \text{if } \mathbf{x}_i^Q = \mathbf{x}_j^P \\ 0 & \text{otherwise} \end{cases}$$

### 4.3 Why $\mathbf{K}_{\text{surf}}$ is correct

Since both trial and test functions live on $Q$ (the 2D submesh), `grad` computes the **surface gradient** using the submesh's own Jacobian. No cross-mesh evaluation, no tangential projection needed:

$$\nabla_s \phi_i^Q = J_{\text{sub}}^{-T} \hat{\nabla} \hat{\phi}_i$$

where $J_{\text{sub}}$ is the $3 \times 2$ Jacobian of the submesh cell mapping and $\hat{\nabla}$ is the gradient in the 2D reference coordinates.

---

## 5. Implementation

### Step 1: Surface stiffness matrix $\mathbf{K}_{\text{surf}}$ ($n_Q \times n_Q$)

```python
# q = TrialFunction(Q), u = TestFunction(Q), dx1 = Measure("dx", domain=submesh)
g1_surf = inner(grad(q), grad(u)) * dx1

K_surf = petsc.assemble_matrix(form(g1_surf))
K_surf.assemble()
```

This is a standard stiffness matrix assembled purely on the submesh. FFCx compiles it without issues because both functions are on the same 2D mesh.

### Step 2: Interpolation matrix $\mathbf{I}_{PQ}$ ($n_Q \times n_P$)

```python
from scipy.spatial import cKDTree

Q_coords = Q.tabulate_dof_coordinates()
P_coords = P.tabulate_dof_coordinates()
nQ_local = Q.dofmap.index_map.size_local * Q.dofmap.index_map_bs
nP_local = P.dofmap.index_map.size_local * P.dofmap.index_map_bs

# For each Q DOF, find the closest P DOF (should be at distance ~0)
tree = cKDTree(P_coords[:nP_local])
dists, p_idx = tree.query(Q_coords[:nQ_local])

# Build the sparse Boolean matrix
I_PQ = PETSc.Mat().create()
I_PQ.setSizes((nQ_local, nP_local))
I_PQ.setType("aij")
I_PQ.setPreallocationNNZ(1)  # exactly 1 nonzero per row
I_PQ.setUp()
for i in range(nQ_local):
    I_PQ.setValue(i, int(p_idx[i]), 1.0)
I_PQ.assemble()
```

The `cKDTree` efficiently matches DOF coordinates between the two spaces. Since both use degree-$k$ Lagrange on geometrically matching meshes, `dists` should be $\sim 10^{-15}$.

### Step 3: Compose $\mathbf{G}_1 = \mathbf{K}_{\text{surf}} \cdot \mathbf{I}_{PQ}$ ($n_Q \times n_P$)

```python
G1_mat = K_surf.matMult(I_PQ)
```

This PETSc matrix multiplication produces the correctly assembled IPP surface Laplacian block, mapping $P$ trial DOFs to $Q$ test DOFs.

### Step 4: Insert into the monolithic block matrix $\mathbf{D}_1$

The block-assembled $\mathbf{D}_1$ has the structure:

$$\mathbf{D}_1 = \begin{bmatrix} \underbrace{K}_{n_P \times n_P} & \underbrace{C}_{n_P \times n_Q} \\ \underbrace{G_4}_{n_Q \times n_P} & \underbrace{E_1}_{n_Q \times n_Q} \end{bmatrix}$$

where $G_4$ is assembled via `ds(3)` (no gradient of $u$, works correctly). The monolithic matrix has rows $[0, n_P)$ for $P$ DOFs and rows $[n_P, n_P + n_Q)$ for $Q$ DOFs.

To add $\mathbf{G}_1$ at the $(1,0)$ block position:

```python
offset = nP_local
D1.setOption(PETSc.Mat.Option.NEW_NONZERO_ALLOCATION_ERR, False)
for i in range(nQ_local):
    cols, vals = G1_mat.getRow(i)
    if len(cols) > 0:
        # Row i of G1_mat -> row (i + offset) in the monolithic matrix
        # Columns are P DOF indices (0-based), no shift needed
        D1.setValues([i + offset], cols.tolist(), vals.tolist(),
                     PETSc.InsertMode.ADD_VALUES)
D1.assemble()
```

The `NEW_NONZERO_ALLOCATION_ERR = False` flag is needed because $\mathbf{G}_1$ may have nonzero entries at positions not present in $G_4$'s sparsity pattern (surface stiffness couples DOFs through shared edges, which may extend beyond the element-local coupling of the mass-like $G_4$ term).

---

## 6. Diagram

```
                  nP cols              nQ cols
              ┌──────────────────┬──────────────────┐
              │                  │                  │
   nP rows    │        K         │        C         │
              │                  │                  │
              ├──────────────────┼──────────────────┤
              │                  │                  │
   nQ rows    │   G4 + G1_mat   │       E1         │
              │                  │                  │
              └──────────────────┴──────────────────┘

   G4:     assembled via ds(3) with entity_maps    (no grad(u), works fine)
   G1_mat: K_surf @ I_PQ, assembled on submesh     (grad on same mesh, correct)
```

---

## 7. Why this works

| Component | Mesh | Gradient computation | Status |
|-----------|------|---------------------|--------|
| $\mathbf{K}_{\text{surf}}$ | Submesh only | `grad(q)` and `grad(u)` both on 2D submesh, using the same Jacobian | Correct |
| $\mathbf{I}_{PQ}$ | Coordinate matching | No gradient, just DOF correspondence | Exact for same-degree Lagrange |
| $\mathbf{G}_1 = \mathbf{K}_{\text{surf}} \cdot \mathbf{I}_{PQ}$ | Composition | Inherits correctness from both factors | Correct |

The key insight: **separate the "where to compute the gradient" question from the "which DOFs to couple" question**. $\mathbf{K}_{\text{surf}}$ handles the gradient computation on the correct mesh geometry, and $\mathbf{I}_{PQ}$ handles the DOF mapping between spaces.
