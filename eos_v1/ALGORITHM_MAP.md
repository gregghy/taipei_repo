# EOS v1: Comprehensive Algorithm and Implementation Map

## Document purpose

This document maps the algorithms implemented in the current working tree of `eos_v1`. It is intended as a technical-report source: it gives the execution graph, mathematical equations, data structures, numerical update order, adaptive-mesh logic, boundary treatment, scenario variants, diagnostics, verification coverage, computational complexity, and implementation caveats.

The wording deliberately distinguishes four categories:

- **Executed**: reached by a current solver's `step` path.
- **Available**: implemented but not selected by the current path.
- **Benchmark-only**: defined in a benchmark subclass rather than the production solver.
- **Stated but not implemented**: mentioned by comments/configuration but absent from the executed algorithm.

Source citations use `file.py:Lx-Ly` links and refer to the current working tree.

## Contents

1. Executive algorithm map
2. Notation
3. Configuration and default numerical case
4. State and memory model
5. Initial conditions
6. Quadratic B-spline interpolation
7. Standard 2-D MPM algorithm
8. Standard 3-D path
9. Constitutive algorithms
10. Standard enhanced boundary conditions
11. Adaptive grid geometry
12. Adaptive boundary penalty
13. Adaptive MPM substep
14. Refinement target algorithms
15. Particle splitting and merging
16. Dynamic refinement-window motion
17. Poiseuille algorithm
18. Continuous inflow algorithm
19. Legacy base engine
20. Benchmark-only algorithms
21. Output and visualization algorithms
22. Verification map
23. Computational complexity
24. Conservation and stability properties
25. Important implementation caveats
26. Report-ready pseudocode
27. Source-by-source responsibility index
28. Bottom-line characterization

---

# 1. Executive algorithm map

## 1.1 Project identity

The project is an explicit Material Point Method (MPM) fluid simulator implemented with Taichi. Its central numerical ingredients are:

1. material points carrying mass, position, velocity, affine velocity information, deformation, stress, and pressure;
2. a temporary Cartesian background grid cleared every substep;
3. quadratic B-spline particle-grid transfers;
4. APIC momentum transfer and affine reconstruction;
5. explicit nodal momentum integration;
6. weakly compressible water or regularized Bingham stress;
7. nodal F-bar volume smoothing in the standard solver;
8. nested-patch adaptive MPM in the adaptive solver;
9. force-based enhanced boundary conditions in the standard solver and virtual penalty mass/momentum in the adaptive solver;
10. VTK output for ParaView.

The main production entry point is [`main.py:28-216`](main.py#L28-L216). The default configuration is a **2-D, standard-grid, water, full-width hydrostatic pool** selected under the name `DAM_BREAK`; adaptive MPM is off by default ([`config.py:7-14`](config.py#L7-L14)).

## 1.2 Top-level dependency graph

```text
config.py
   |
   +--> main.py
          |
          +--> boundary-field initialization
          +--> scenario dispatch
          |      +--> StandardSolver
          |      +--> AdaptiveMPMSolver2D
          |      +--> PoiseuilleSolver
          |      `--> Inflow_Solver
          |
          +--> optional damped relaxation
          +--> frame/substep loop
          `--> VTK extraction and export

StandardSolver
   +--> ParticleSystem (dense f32 particles)
   +--> Grid (dense f32 Cartesian nodes)
   +--> quadratic B-splines
   +--> APIC P2G/G2P
   +--> stress and gravity force scatter
   +--> grid/EBC boundaries
   `--> nodal F-bar + constitutive stress

AdaptiveMPMSolver2D
   +--> QuadtreeGrid2D (nested dense f64 patches)
   +--> AdaptiveParticleSystem2D (variable-mass f64 particles)
   +--> multilevel APIC P2G/force scatter
   +--> penalty boundary mass/momentum
   +--> parent-to-child interface velocity interpolation
   +--> APIC G2P + constitutive update
   `--> particle merge/split adaptation
```

## 1.3 Scenario/solver dispatch matrix

The executable dispatch is in [`main.py:40-74`](main.py#L40-L74).

| Scenario | Solver selected | Dimensions | Current status |
|---|---|---:|---|
| `DAM_BREAK` | `StandardSolver`, or `AdaptiveMPMSolver2D` if `USE_ADAPTIVE_MPM` | standard: 2-D/3-D; adaptive: 2-D only | Main supported path; default is standard 2-D |
| `IMMERSED` | `StandardSolver`, or adaptive in 2-D | standard: 2-D/3-D; adaptive: 2-D only | Implemented moving rectangular/box platform path |
| `ADAPTIVE_MPM` | `AdaptiveMPMSolver2D` | forced to 2-D | Dedicated AMR configuration path |
| `POISEUILLE` | `PoiseuilleSolver` | effectively 2-D | Solver exists, but configuration is incomplete and its `step()` signature is incompatible with `main.py` |
| `INFLOW` | `Inflow_Solver` | 2-D code path | Solver exists, but required configuration is missing |

Important naming point: the default `DAM_BREAK` configuration fills the complete domain width with fluid and only `0.15` of its height ([`config.py:181-185`](config.py#L181-L185), [`config.py:152-160`](config.py#L152-L160)). It is therefore a hydrostatic pool in the default state, not the classical narrow-column dam-break. The dedicated benchmark creates the narrow column explicitly ([`benchmarks/dam_break/run_experiment.py:72-95`](benchmarks/dam_break/run_experiment.py#L72-L95)).

## 1.4 One-frame execution

The frame loop executes `SUBSTEPS` calls to `solver.step`, then copies particle fields to NumPy and writes one VTK file ([`main.py:162-175`](main.py#L162-L175)). The configured frame interval is

$$
\Delta t_{frame}=1/60\;\text{s},
$$

and the substep is adjusted so an integer number of steps exactly fills one frame ([`config.py:247-250`](config.py#L247-L250)).

For `DAM_BREAK` and `IMMERSED`, 30 damped frames are run before the main loop ([`main.py:141-157`](main.py#L141-L157)). The main simulation time is then reset to zero at [`main.py:162`](main.py#L162). This reset is irrelevant for a static dam-break boundary, but it causes a moving immersed platform's prescribed-time trajectory to restart after relaxation.

---

# 2. Notation

| Symbol | Meaning | Code field |
|---|---|---|
| $p$ | particle index | loop variable `p` |
| $I$ | grid node multi-index | vector/index `I` |
| $\mathbf{x}_p$ | particle position | `particles.x[p]` |
| $\mathbf{v}_p$ | particle velocity | `particles.v[p]` |
| $m_p$ | particle mass | `config.P_MASS` standard; `particles.mass[p]` adaptive |
| $V_p^0$ | reference particle volume | `config.P_VOL` standard; `particles.volume0[p]` adaptive |
| $\mathbf{C}_p$ | APIC affine velocity matrix/estimated velocity gradient | `particles.C[p]` |
| $\mathbf{F}_p$ | deformation gradient | `particles.F[p]` |
| $J_p$ | `det(F_p)` | computed locally |
| $\boldsymbol\sigma_p$ | Cauchy stress | `particles.stress[p]` |
| $P_p$ | reported scalar pressure | `particles.pressure[p]` |
| $N_{Ip}$ | quadratic B-spline shape weight | product of one-dimensional `w` values |
| $\nabla N_{Ip}$ | physical-space shape gradient | product of `dw` and `w` values |
| $m_I$ | nodal mass | `grid.m[I]` |
| $\mathbf{p}_I$ | nodal momentum | stored temporarily in `grid.v[I]` before normalization |
| $\mathbf{v}_I$ | nodal velocity | `grid.v[I]` after normalization |
| $\mathbf{f}_I$ | nodal force | `grid.f[I]` |
| $h_\ell$ | adaptive level-$\ell$ cell width | `grid.dx[level]` / `level_dx[level]` |
| $K$ | bulk modulus | `RHO_0 * C_0**2` |

The 2-D code implicitly represents unit out-of-plane thickness. Consequently, `P_VOL` is numerically an area and `P_MASS = rho * area`; a strict dimensional report should describe this as a plane model with unit thickness.

---

# 3. Configuration and default numerical case

## 3.1 Standard grid

For 2-D, the physical grid is configured as `100 x 100` cells over `1.0 x 1.0`, with three padding cells on each side. The allocated node field therefore has

$$
N_x=N_y=100+2(3)+1=107
$$

nodes per axis, with $\Delta x=\Delta y=0.01$ ([`config.py:81-97`](config.py#L81-L97)). The physical domain starts at `(PADDING*DX, PADDING*DY) = (0.03, 0.03)` in the standard scenario ([`config.py:171-179`](config.py#L171-L179)).

For 3-D, the configured domain is `2.0 x 0.75 x 2.0` with `64^3` cells, so `DX != DY` ([`config.py:99-122`](config.py#L99-L122)). The current transfer kernels nevertheless use `INV_DX` and `DX` on every coordinate. This is a substantive 3-D inconsistency discussed in Section 25.

## 3.2 Default particle discretization

The standard 2-D default has:

- `MP_WIDTH = 1.0`;
- `MP_HEIGHT = 0.15`;
- two particles per cell axis;
- `NUM_MP_WIDTH = 200`;
- `NUM_MP_HEIGHT = 30`;
- `TOTAL_NUM_MP = 6000`;
- `P_VOL = 1.0*0.15/6000 = 2.5e-5`;
- `P_MASS = 1000*2.5e-5 = 0.025`.

These quantities are derived at [`config.py:205-223`](config.py#L205-L223).

## 3.3 Weak-compressibility and time step

The code estimates the free-fall speed and chooses an artificial acoustic speed ten times larger:

$$
V_{max}=\sqrt{2gH}, \qquad c_0=10\sqrt{2gH},
$$

$$
c_{max}=c_0+V_{max}, \qquad
\Delta t_{acoustic}=CFL\frac{\Delta x}{c_{max}}.
$$

For Bingham flow it also applies

$$
\nu_{max}=\frac{\mu_p+m\tau_y}{\rho_0}, \qquad
\Delta t_{viscous}=\frac{1}{2}\frac{\Delta x^2}{\nu_{max}},
$$

and takes the smaller stability limit. These equations are implemented at [`config.py:229-250`](config.py#L229-L250).

For the default 2-D pool, the resulting frame subdivision is 315 substeps and the exact adjusted substep is approximately `5.291005e-5 s`. A 300-frame main run represents 5 seconds, preceded by 0.5 seconds of damped relaxation.

## 3.4 Adaptive grid defaults

The dedicated AMR domain is `0.02 x 0.1`, base resolution `32 x 160`, base spacing `6.25e-4`, maximum level 6, and finest spacing

$$
h_6 = 6.25\times10^{-4}/64 = 9.765625\times10^{-6}.
$$

The configuration labels a `10e-6` reference spacing, but actual geometry is derived from base spacing and level count ([`config.py:15-41`](config.py#L15-L41)). `AMR_REFERENCE_FINE_DX` is descriptive only; it is not consumed by the grid builder.

The adaptive solver optionally replaces the already-computed standard time step by the finest-grid CFL limit, then recomputes an integer number of frame substeps ([`solver/adaptive_engine.py:53-63`](solver/adaptive_engine.py#L53-L63)). There is **one global fine-level time step**; no level subcycling exists.

---

# 4. State and memory model

## 4.1 Standard particles

`ParticleSystem` allocates fixed-size `f32` Taichi fields for position, velocity, affine matrix, deformation gradient, Cauchy stress, and pressure ([`core/particles.py:9-24`](core/particles.py#L9-L24)). `active_count` exists for the inflow pool but ordinary standard solver kernels iterate over all `n_particles`.

The standard particle count and mass are globally uniform. No material identifier, per-particle mass, plastic state, or adaptive level is stored.

## 4.2 Standard grid

`Grid` is a dense `f32` node array containing mass, velocity/momentum, force, old velocity, two unused cell-F-bar volume fields, and two active nodal-F-bar volume fields ([`core/grid.py:8-31`](core/grid.py#L8-L31)). Every substep clears every node and all auxiliary volume fields ([`core/grid.py:33-48`](core/grid.py#L33-L48)).

`grid.v` is overloaded:

1. during P2G it accumulates momentum;
2. after normalization it stores velocity;
3. `v_old` stores the pre-force velocity, although current standard APIC G2P does not use it.

## 4.3 Adaptive particles

`AdaptiveParticleSystem2D` uses `f64` and adds material ID, plastic volume state `Jp`, current level, variable mass, reference volume, gradient target level, and active/capacity counters ([`core/adaptive_particles.py:13-45`](core/adaptive_particles.py#L13-L45)).

Its memory capacity is

$$
C_p=\left\lceil N_{initial}\max(f_{capacity},1)\right\rceil,
$$

when splitting is enabled. Split overflow is counted rather than allowing an out-of-bounds write ([`core/adaptive_particles.py:20-25`](core/adaptive_particles.py#L20-L25), [`core/adaptive_particles.py:181-217`](core/adaptive_particles.py#L181-L217)).

Merge bin arrays and a second complete particle buffer support parallel accumulation and compaction ([`core/adaptive_particles.py:46-74`](core/adaptive_particles.py#L46-L74)).

## 4.4 Adaptive grid

For each level, `QuadtreeGrid2D` allocates a separate dense `f64` rectangular patch containing mass, velocity/momentum, force, old velocity, axis-dependent boundary mass, and boundary momentum ([`core/quadtree_grid.py:74-135`](core/quadtree_grid.py#L74-L135)). It also stores NumPy and Taichi copies of each patch's region, origin, spacing, dynamic shift, platform bounds, and motion parameters.

This is not a pointer-based recursive quadtree, a Morton-coded tree, or a sparse cell graph. It is a **hierarchy of nested dense Cartesian patches**, plus a non-overlapping leaf-cell list used for initial particle generation and visualization.

---

# 5. Initial conditions

## 5.1 Standard particle placement

Particles are placed at centers of a uniform subcell lattice:

$$
\mathbf{x}_{ij}=
\mathbf{x}_{min}+
\left((i+1/2)\frac{\Delta x}{n_{ppc}},
      (j+1/2)\frac{\Delta y}{n_{ppc}}\right).
$$

The 3-D version adds the analogous `z` coordinate. Index flattening and field initialization are at [`core/particles.py:43-109`](core/particles.py#L43-L109).

## 5.2 Hydrostatic state

At vertical coordinate $y_p$, depth below the initial free surface is

$$
d_p=\max(H-(y_p-y_{bottom}),0),
$$

and pressure is initialized to

$$
P_p^0=\rho_0(9.81)d_p.
$$

The equation of state is inverted to initialize

$$
J_p^0=\frac{1}{1+P_p^0/K}, \qquad K=\rho_0 c_0^2.
$$

The standard 2-D code sets `F = diag(1, J_init)` and `stress = -P I`; the 3-D code uses `diag(1, J_init, 1)` ([`core/particles.py:60-109`](core/particles.py#L60-L109)). The adaptive initializer uses the same construction with the configured AMR fluid top ([`core/adaptive_particles.py:147-160`](core/adaptive_particles.py#L147-L160)).

The acceleration `9.81` is hard-coded in both initializers rather than derived from `config.GRAVITY`. A zero-gravity or altered-gravity case must override particle state if it requires a consistent initial pressure.

## 5.3 Adaptive particle placement

There are two modes ([`core/adaptive_particles.py:89-145`](core/adaptive_particles.py#L89-L145)):

1. **Fixed initial level** (`AMR_INITIAL_PARTICLE_LEVEL >= 0`): fill cells in that level's region with `ppc_axis^2` particles per cell.
2. **Leaf based** (`-1`): fill every non-overlapped leaf cell at its native level.

For level $\ell$,

$$
V_p^0 = \frac{h_\ell^2}{n_{ppc}^2},
\qquad m_p=\rho_0V_p^0.
$$

Only cells whose centers lie inside the configured fluid box are filled. The center test means partially intersected cells are either entirely included or excluded.

## 5.4 Inflow pool

The inflow initializer sets `active_count=0`, moves every slot to a `-1000` sentinel position, and initializes all other state to zero/identity ([`core/particles.py:111-126`](core/particles.py#L111-L126)). Slots become active when emitted by `Inflow_Solver`.

---

# 6. Quadratic B-spline interpolation

Given a particle coordinate and inverse cell width, the code computes

$$
\mathbf{f}=\mathbf{x}_p/h,
\qquad
\mathbf{b}=\operatorname{int}(\mathbf{f}-1/2),
\qquad
\mathbf{d}=\mathbf{f}-\mathbf{b}.
$$

For each coordinate, the three weights are

$$
w_0(d)=\frac12(1.5-d)^2,
$$
$$
w_1(d)=0.75-(d-1)^2,
$$
$$
w_2(d)=\frac12(d-0.5)^2.
$$

Their physical-coordinate derivatives are

$$
w'_0=(d-1.5)/h,
\qquad w'_1=-2(d-1)/h,
\qquad w'_2=(d-0.5)/h.
$$

The implementation is in [`core/functions.py:3-29`](core/functions.py#L3-L29), and the adaptive `f64` equivalent is in [`solver/adaptive_engine.py:65-95`](solver/adaptive_engine.py#L65-L95).

In 2-D, each particle has a `3 x 3` node stencil:

$$
N_{Ip}=w_i(d_x)w_j(d_y),
$$

$$
\nabla N_{Ip}=\left(w_i'(d_x)w_j(d_y),
                         w_i(d_x)w_j'(d_y)\right).
$$

In 3-D the support is `3 x 3 x 3` and the tensor product includes the third coordinate.

The adaptive diagnostic directly measures

$$
\sum_I N_{Ip}\stackrel{ideal}{=}1,
\qquad
\left\|\sum_I\nabla N_{Ip}\right\|\stackrel{ideal}{=}0,
$$

using only in-bounds nodes ([`solver/adaptive_engine.py:332-373`](solver/adaptive_engine.py#L332-L373)).

---

# 7. Standard 2-D MPM algorithm

The executed sequence is fixed by [`solver/standard_engine.py:903-932`](solver/standard_engine.py#L903-L932).

## 7.1 Exact substep order

```text
1. clear all grid and F-bar fields
2. APIC particle-to-grid mass and momentum transfer
3. compute stress divergence and gravity forces
4. optionally add obstacle/platform EBC force
5. explicit nodal velocity update, damping, and safety clamp
6. enforce outer free-slip grid boundary
7. APIC grid-to-particle velocity/affine/position update
8. update deformation gradient
9. scatter reference/current volume to F-bar grid fields
10. gather nodal volume ratio, replace F by isotropic F-bar state,
    compute constitutive stress, and update reported pressure
```

The force at step 3 uses the stress stored at the end of the previous substep. The newly computed stress at step 10 is therefore used on the next substep.

## 7.2 APIC P2G

For every particle and support node, the standard solver accumulates

$$
m_I \mathrel{+}= N_{Ip}m_p,
$$

$$
\mathbf{p}_I \mathrel{+}=
N_{Ip}m_p\left[\mathbf{v}_p+
\mathbf{C}_p(\mathbf{x}_I-\mathbf{x}_p)\right].
$$

It then normalizes

$$
\mathbf{v}_I=\mathbf{p}_I/m_I
$$

for nonzero mass. This is implemented at [`solver/standard_engine.py:23-47`](solver/standard_engine.py#L23-L47).

The method is APIC rather than FLIP/PIC blending on the active standard path: velocity is gathered as PIC, while `C` carries the local affine velocity field. A FLIP/PIC kernel exists only in the unused base engine ([`solver/engine.py:97-133`](solver/engine.py#L97-L133)).

## 7.3 Force scatter

Current particle volume is taken as

$$
V_p=J_pV_p^0.
$$

The nodal force contribution is

$$
\mathbf{f}_I \mathrel{+}=
N_{Ip}m_p\mathbf{g}
-
V_p\boldsymbol\sigma_p\nabla N_{Ip}.
$$

Code: [`solver/standard_engine.py:225-248`](solver/standard_engine.py#L225-L248). Stress is treated as Cauchy stress, which is why the current rather than reference particle volume is used.

## 7.4 Nodal integration

At an active node,

$$
\mathbf{v}_I^{pre}=\mathbf{v}_I^n,
\qquad
\mathbf{v}_I^{n+1}=d\left(\mathbf{v}_I^n+
\Delta t\frac{\mathbf{f}_I}{m_I}\right),
$$

where `d` is the supplied relaxation damping. Each 2-D component is then clamped independently to `[-15, 15]`. Nodes with mass at or below `1e-7` are zeroed ([`solver/standard_engine.py:478-497`](solver/standard_engine.py#L478-L497)).

This is explicit momentum integration. Because particle advection uses the updated velocity, the complete transfer-update-advection sequence is commonly described as symplectic/semi-implicit Euler in particle position, although the grid force update itself is explicit.

## 7.5 Outer grid boundary

At node layers on or outside the physical box, only outward normal velocity is removed. Tangential velocity remains, giving a free-slip/no-penetration boundary ([`solver/standard_engine.py:517-527`](solver/standard_engine.py#L517-L527)).

## 7.6 APIC G2P

The particle velocity and APIC moment matrix are

$$
\mathbf{v}_p^{n+1}=\sum_I N_{Ip}\mathbf{v}_I^{n+1},
$$

$$
\mathbf{B}_p=\sum_I N_{Ip}\mathbf{v}_I^{n+1}
(\mathbf{x}_I-\mathbf{x}_p)^T.
$$

The standard uniform-grid approximation sets

$$
\mathbf{C}_p^{n+1}=\frac{4}{h^2}\mathbf{B}_p.
$$

Then

$$
\mathbf{x}_p^{n+1}=\mathbf{x}_p^n+
\Delta t\mathbf{v}_p^{n+1}.
$$

Code: [`solver/standard_engine.py:544-591`](solver/standard_engine.py#L544-L591). The factor `4/h^2` is the inverse APIC inertia for the interior quadratic B-spline stencil. The full `D` matrix calculation is present only as commented code in this kernel; the Poiseuille solver computes and inverts `D` explicitly.

After advection, each coordinate is hard-clamped `0.001` inside the physical domain. This is an additional particle containment mechanism independent of the grid boundary.

## 7.7 Immersed G2P position correction

For the active 2-D immersed path, G2P leaves the gathered velocity and affine matrix unchanged, predicts position, evaluates the moving rectangle SDF, and, if the point is inside, pushes it to the closest surface plus `1e-5` ([`solver/standard_engine.py:688-769`](solver/standard_engine.py#L688-L769)).

This “bumper” corrects position but does not remove penetrating velocity. Normal momentum control is expected to come from the grid EBC force earlier in the same step.

## 7.8 Deformation update

The standard solver advances

$$
\mathbf{F}_p^*=\left(\mathbf{I}+\Delta t\mathbf{C}_p^{n+1}\right)
\mathbf{F}_p^n
$$

at [`solver/standard_engine.py:868-875`](solver/standard_engine.py#L868-L875).

## 7.9 Nodal F-bar algorithm

The F-bar stabilization has two transfer passes.

### Volume scatter

For each particle,

$$
V_{I}^{0}\mathrel{+}=N_{Ip}V_p^0,
\qquad
J_{p,c}=\max(J_p^*,0.1),
\qquad
V_{I}^{cur}\mathrel{+}=N_{Ip}V_p^0J_{p,c}.
$$

The clamp is applied to the particle determinant before current volume is formed. Code: [`solver/standard_engine.py:49-71`](solver/standard_engine.py#L49-L71).

### Volume-ratio gather

For nodes with sufficient reference volume,

$$
J_I=V_I^{cur}/V_I^0,
$$

and the particle gathers

$$
\bar J_p=
\frac{\sum_{I\in valid}N_{Ip}J_I}
     {\sum_{I\in valid}N_{Ip}}.
$$

If no valid weight exists, it falls back to the particle's own `J` ([`solver/standard_engine.py:103-134`](solver/standard_engine.py#L103-L134)).

### F replacement

Although comments retain the conventional formula

$$
\mathbf{F}_{bar}=\left(\frac{\bar J}{J}\right)^{1/d}\mathbf{F},
$$

the executed 2-D code discards the complete deviatoric part and sets

$$
\mathbf{F}_p^{n+1}=\sqrt{\bar J_p}\,\mathbf{I}.
$$

That follows because the code computes `scale=(Jbar/J)^(1/d)`, obtains `J_new=J*scale^d=Jbar`, and constructs a fresh diagonal matrix ([`solver/standard_engine.py:135-165`](solver/standard_engine.py#L135-L165)). This is appropriate only if deformation is being retained as a volumetric fluid state; it is not a general solid F-bar update.

Finally, stress is recomputed from the selected fluid model and exported pressure is

$$
P_p=\max\left[K\left(1/\bar J_p-1\right),0\right].
$$

The stress model evaluates the determinant of the reconstructed `F` and clamps it internally, whereas exported pressure uses the separately gathered `J_bar_p` without that constitutive clamp. Reported pressure can therefore differ from the isotropic pressure actually used in stress under strong compression.

---

# 8. Standard 3-D path

The 3-D sequence is [`solver/standard_engine.py:934-947`](solver/standard_engine.py#L934-L947): 3-D P2G, 3-D force scatter, 3-D EBC, grid update, six-face free-slip boundary, G2P/platform projection, combined deformation/F-bar volume scatter, and 3-D F-bar gather/stress.

The tensor-product transfer uses 27 nodes ([`solver/standard_engine.py:73-99`](solver/standard_engine.py#L73-L99)); force gradients are expanded to three components ([`solver/standard_engine.py:250-279`](solver/standard_engine.py#L250-L279)).

The 3-D F-bar gather differs from 2-D in two ways:

1. a nodal volume threshold proportional to `P_VOL` is used;
2. the scalar correction is clamped to `[0.95, 1.05]` per step.

It then constructs

$$
\mathbf{F}=\bar J^{1/3}\mathbf{I}
$$

([`solver/standard_engine.py:167-223`](solver/standard_engine.py#L167-L223)).

The current 3-D implementation should not be reported as validated for the configured anisotropic cells. Interpolation, node positions, APIC scaling, and force gradients use `DX/INV_DX` for every axis, even though `DY` differs from `DX` ([`config.py:99-119`](config.py#L99-L119), [`solver/standard_engine.py:74-99`](solver/standard_engine.py#L74-L99), [`solver/standard_engine.py:251-279`](solver/standard_engine.py#L251-L279)). In addition, water and Bingham form the trace from only components `00` and `11`, then subtract one half of that trace from every diagonal component. That is the 2-D deviatoric projection, not the 3-D `tr(D)/3` projection ([`physics/constitutive_model.py:21-25`](physics/constitutive_model.py#L21-L25), [`physics/constitutive_model.py:103-114`](physics/constitutive_model.py#L103-L114)).

---

# 9. Constitutive algorithms

## 9.1 Weakly compressible water

The water model uses

$$
K=\rho_0c_0^2,
\qquad
J_c=\max(\det\mathbf{F},0.96),
$$

$$
P=\max\left[K(1/J_c-1),0\right].
$$

Thus tension is suppressed and compression beyond `J=0.96` does not increase the stress pressure. With

$$
\mathbf{D}=\frac12(\mathbf{C}+\mathbf{C}^T),
$$

the 2-D deviatoric viscous stress is

$$
\boldsymbol\tau=2\mu
\left(\mathbf{D}-\frac12\operatorname{tr}(\mathbf{D})\mathbf{I}\right),
\qquad \mu=10^{-3}.
$$

Total Cauchy stress is

$$
\boldsymbol\sigma=-P\mathbf{I}+\boldsymbol\tau.
$$

Code: [`physics/constitutive_model.py:7-30`](physics/constitutive_model.py#L7-L30).

The pressure relation is a linear bulk-modulus EOS expressed in terms of `1/J`; it is not the general exponentiated Tait equation.

## 9.2 Water with artificial viscosity

An available, currently uncalled standard-grid variant adds compression-only artificial pressure

$$
q=
-\alpha_L\rho_0c_0h\nabla\cdot\mathbf{v}
+
\alpha_Q\rho_0(h\nabla\cdot\mathbf{v})^2,
\qquad \nabla\cdot\mathbf{v}<0,
$$

with `alpha_L=0.5`, `alpha_Q=1.0`, and global `h=DX`. Stress uses `P+q` ([`physics/constitutive_model.py:32-70`](physics/constitutive_model.py#L32-L70)). It is imported by standard/inflow modules but not selected by their executed stress updates.

## 9.3 Adaptive water

The adaptive solver always uses `StressUsingWaterAdaptive`, which applies the same compression-only artificial viscosity with the particle's current level spacing `h_l` ([`physics/constitutive_model.py:72-90`](physics/constitutive_model.py#L72-L90)). This makes numerical shock damping scale with local resolution.

The adaptive path does not use nodal F-bar. Its `F`, stress, and pressure are updated directly during G2P.

## 9.4 Regularized Bingham fluid

The Bingham model shares the no-tension weakly compressible pressure, with `J` clamped to `0.95`. It computes

$$
\mathbf{D}'=\mathbf{D}-\frac12\operatorname{tr}(\mathbf{D})\mathbf{I},
$$

$$
\dot\gamma=\max\left(\sqrt{2\mathbf{D}':\mathbf{D}'},10^{-8}\right),
$$

$$
\mu_{eff}=\mu_p+\frac{\tau_y}{\dot\gamma}
\left(1-e^{-m\dot\gamma}\right),
$$

$$
\boldsymbol\sigma=-P\mathbf{I}+2\mu_{eff}\mathbf{D}'.
$$

Code: [`physics/constitutive_model.py:92-118`](physics/constitutive_model.py#L92-L118). This is a Papanastasiou-type regularization of a Bingham yield stress.

Bingham is selectable in the standard F-bar solver, Poiseuille solver, and inflow solver. It is not selected by the base adaptive solver, which always invokes adaptive water.

## 9.5 Benchmark-only jelly and snow

Solid-like laws appear only in `ThreeMaterialAdaptiveMPMSolver2D`, a benchmark subclass ([`benchmarks/three_materials/run_experiment.py:27-100`](benchmarks/three_materials/run_experiment.py#L27-L100)). They are not general production material dispatch.

For non-fluid particles, the benchmark computes `F=U Sigma V^T`, a rotation `R=UV^T`, and a fixed-corotated-type Cauchy/Kirchhoff expression

$$
\boldsymbol\sigma=
2\mu(\mathbf{F}-\mathbf{R})\mathbf{F}^T
+
\lambda J(J-1)\mathbf{I}.
$$

For snow, singular values are projected to `[0.975, 1.0045]`, `Jp` is updated by the removed singular stretch, and hardening is

$$
e^{10(1-J_p)}.
$$

Jelly and benchmark solid-block particles use a constant hardening multiplier `0.3`; snow uses the exponential factor. All materials share one grid velocity field, so inter-material interaction is numerical single-field sticking/mixing rather than a dedicated contact algorithm.

---

# 10. Standard enhanced boundary conditions

## 10.1 Outer box

Two generic kernels exist in [`physics/boundary.py:19-52`](physics/boundary.py#L19-L52): free-slip removes only outward normal velocity; no-slip removes the whole vector. `StandardSolver` duplicates its own free-slip implementation with slightly different boundary indices.

## 10.2 Signed distance functions

The rectangle and box functions implement standard axis-aligned signed distance:

$$
\mathbf{q}=|\mathbf{x}-\mathbf{c}|-\mathbf{h},
$$

$$
r=\|\max(\mathbf{q},0)\|+
\min(\max(q_x,q_y[,q_z]),0).
$$

Normals point outward from the solid. See [`physics/boundary.py:142-182`](physics/boundary.py#L142-L182) and [`physics/boundary.py:208-243`](physics/boundary.py#L208-L243). A circular SDF is also available but is not called by a current solver ([`physics/boundary.py:184-206`](physics/boundary.py#L184-L206)).

## 10.3 EBC spatial weight

For signed distance `r`, influence thickness `h=DECAY_ZONE`, and normal from solid to fluid,

$$
\gamma(r)=
\begin{cases}
1,&r\le0,\\
(1-r/h)^3,&0<r<h,\\
0,&r\ge h.
\end{cases}
$$

The force algorithm is [`physics/boundary.py:54-140`](physics/boundary.py#L54-L140).

## 10.4 EBC modes

### No-slip (`bc_type=1`)

$$
\mathbf{f}_{bc}=\gamma
\left(-\frac{m_I\mathbf{v}_I}{\Delta t}-\mathbf{f}_I\right).
$$

At full influence this makes the predicted nodal velocity zero in one step.

### Stationary slip (`bc_type=0`)

The unconstrained predicted momentum is

$$
\mathbf{p}^{pred}=m_I\mathbf{v}_I+\Delta t\mathbf{f}_I.
$$

If `p_pred dot n < 0`, the force cancels its normal component; tangential momentum is unchanged.

### Prescribed velocity (`bc_type=2`)

An implemented but currently unused branch attempts to impose wall velocity. Its `delta_p_normal` expression includes the complete target momentum and only projects the current momentum, so it is not a pure symmetric normal projection.

### Generalized moving slip (`bc_type=3`)

With wall momentum `p_wall=m_I v_wall`, penetration is detected from

$$
(\mathbf{p}^{pred}-\mathbf{p}_{wall})\cdot\mathbf{n}<0.
$$

The force then makes the fluid's predicted normal momentum equal the moving wall's normal momentum while retaining tangential slip.

## 10.5 Static obstacle and moving platform

A stationary dam-break obstacle uses rectangle SDF plus `bc_type=0` ([`solver/standard_engine.py:281-323`](solver/standard_engine.py#L281-L323)). This path requires `INT_SQUARE_*` variables that are not defined in the current `config.py`.

The active immersed 2-D grid EBC uses three motion phases ([`solver/standard_engine.py:368-438`](solver/standard_engine.py#L368-L438)):

$$
v(t)=
\begin{cases}
v_0,&t<T_s,\\
v_0\left(1-\frac{t-T_s}{T_d}\right),&T_s\le t<T_s+T_d,\\
0,&t\ge T_s+T_d,
\end{cases}
$$

with displacement obtained by integrating this piecewise velocity. It applies generalized slip against the translated rectangle.

## 10.6 NTU line geometry

The inflow solver uses a hand-coded polyline composed of lower and upper wall segments ([`physics/boundary.py:245-361`](physics/boundary.py#L245-L361)). The helper returns the **absolute** nearest-segment distance and a prescribed geometric normal. Therefore `r` is never negative for this geometry; the inflow G2P branch that tests `r_new < 0` cannot act as a true inside/outside position correction ([`solver/Inflow_engine.py:273-280`](solver/Inflow_engine.py#L273-L280)).

---

# 11. Adaptive grid geometry

## 11.1 Hierarchy construction

The level spacing is

$$
h_\ell=h_0/2^\ell,
\qquad \ell=0,\ldots,L.
$$

Level zero covers the complete domain. For each finer level, the desired finest box is expanded by

$$
g_\ell=(L-\ell)N_{buffer}h_{\ell-1},
$$

snapped outward to the parent lattice, clipped to the domain, and checked for nesting ([`core/quadtree_grid.py:554-592`](core/quadtree_grid.py#L554-L592)). Each patch origin is offset by its own padding:

$$
\mathbf{o}_\ell=\mathbf{r}_{min,\ell}-N_{padding}h_\ell.
$$

Patch resolution is physical cells plus two padding bands and one terminal node.

The expanded intermediate boxes usually create graded one-level transition shells, but there is no graph-based 2:1-neighbor balance algorithm or explicit balance validator.

## 11.2 Leaf tiling

At each level, cells whose centers are covered by the next level are removed from that level's leaf set. Concatenating all remaining cells creates `(leaf_level, leaf_origin, leaf_size)` arrays ([`core/quadtree_grid.py:594-622`](core/quadtree_grid.py#L594-L622)). The sum of leaf areas is checked against total domain area ([`core/quadtree_grid.py:624-631`](core/quadtree_grid.py#L624-L631)).

These leaf arrays are static after construction. Dynamic refinement translates patch bounds/origins but does not rebuild the leaf arrays. Runtime level lookup uses current region bounds, so simulation adaptation follows the moving hierarchy; the leaf list remains a record of the initial tiling and is also used in a conservative capacity estimate.

## 11.3 Finest-level lookup

A point starts at level zero and is promoted for every nested region that contains it, yielding the deepest containing level ([`core/quadtree_grid.py:633-641`](core/quadtree_grid.py#L633-L641)). Region maxima are exclusive.

## 11.4 Node mass cutoff

Each level ignores fluid mass below

$$
m_{cut,\ell}=10^{-6}\rho_0(h_\ell/n_{ppc})^2,
$$

which is one millionth of a native particle mass on that level ([`core/quadtree_grid.py:62-63`](core/quadtree_grid.py#L62-L63)).

---

# 12. Adaptive boundary penalty

## 12.1 Interpretation

The AMR solver does not call the standard force-based EBC. It augments each velocity component with an axis-specific **virtual boundary mass and momentum**. Horizontal walls contribute only to the `y` component; vertical walls contribute only to `x`.

## 12.2 Segment quadrature

Each wall segment is partitioned into `ceil(length/h_l)` pieces. Two Gauss points with local coordinates

$$
\xi=\frac12\pm\frac{1}{2\sqrt3}
$$

are evaluated per piece. A penalty coefficient is

$$
\beta_\ell=\texttt{AMR\_BOUNDARY\_PENALTY\_NORMAL}\,\rho_0h_\ell^2.
$$

The configuration value is `1e4` in the main setup. Quadrature contributions are scattered through the same `3 x 3` B-spline stencil ([`core/quadtree_grid.py:151-205`](core/quadtree_grid.py#L151-L205)).

For axis switch $s_a$, quadrature weight $q$, and wall velocity $u_{w,a}$, the conceptual update is

$$
m^b_{I,a}\mathrel{+}=\beta_\ell qN_I(\mathbf{x}_q)s_a,
$$

$$
p^b_{I,a}\mathrel{+}=m^b_{I,a}u_{w,a}.
$$

Domain wall stencils are precomputed on the CPU ([`core/quadtree_grid.py:207-243`](core/quadtree_grid.py#L207-L243)).

## 12.3 Momentum normalization

After P2G, each component is normalized as

$$
v_{I,a}=
\frac{p^{fluid}_{I,a}+p^b_{I,a}}
     {m_I+m^b_{I,a}}.
$$

Code: [`core/quadtree_grid.py:685-696`](core/quadtree_grid.py#L685-L696). The subsequent force acceleration uses the same effective component mass:

$$
v_{I,a}\mathrel{+}=\Delta t\frac{f_{I,a}}
{m_I+m^b_{I,a}}.
$$

Thus the virtual mass both blends velocity toward the wall state and reduces acceleration normal to the wall ([`solver/adaptive_engine.py:158-172`](solver/adaptive_engine.py#L158-L172)). Nodes are processed only if their **fluid** mass exceeds the level cutoff; boundary mass alone does not activate a node.

## 12.4 Moving platform

All four platform edges are quadrature-scattered at every level with axis switches appropriate to their normals ([`core/quadtree_grid.py:245-277`](core/quadtree_grid.py#L245-L277)). The mass stencil is recomputed only after platform displacement differs from the cached position by more than one quarter of the finest cell. Between rebuilds, cached mass is paired with the current wall velocity on the GPU ([`core/quadtree_grid.py:519-542`](core/quadtree_grid.py#L519-L542)).

A post-advection projection additionally pushes any particle lying inside the moving rectangle to its nearest face ([`solver/adaptive_engine.py:173-201`](solver/adaptive_engine.py#L173-L201)).

---

# 13. Adaptive MPM substep

The exact executed order is [`solver/adaptive_engine.py:375-399`](solver/adaptive_engine.py#L375-L399).

```text
0. At configured intervals, possibly translate refinement patches;
   if translated, completely adapt particles to new geometry.
1. Clear every level's fluid and boundary working fields.
2. Load cached physical-domain boundary mass.
3. If immersed, add moving-platform boundary mass/momentum.
4. APIC P2G to particle level and, normally, all ancestors.
5. Normalize each level's momentum including boundary penalty.
6. Scatter particle stress/gravity forces to the same levels.
7. Explicitly integrate every level's velocities.
8. Replace fine ghost/interface-band velocity with parent interpolation.
9. G2P only from each particle's own level; advect, update F/stress/pressure.
10. Recompute gradient target levels.
11. Merge particles that should coarsen, then perform one split level.
12. Check dynamic-refinement split capacity and increment step counter.
```

## 13.1 Multilevel APIC P2G

For particle level $l_p$, the default `AMR_SCATTER_TO_ANCESTORS=True` scatters mass/momentum to every level $0\le\ell\le l_p$ ([`solver/adaptive_engine.py:97-124`](solver/adaptive_engine.py#L97-L124)). Each level receives

$$
m_I^\ell\mathrel{+}=N_{Ip}^\ell m_p,
$$

$$
\mathbf{p}_I^\ell\mathrel{+}=
N_{Ip}^\ell m_p
\left(\mathbf{v}_p+\mathbf{C}_p(\mathbf{x}_I^\ell-\mathbf{x}_p)\right).
$$

The same particle's mass therefore appears on multiple independent level grids. It must not be summed across levels as physical mass. Physical conservation is measured from particle mass; ancestor copies provide the coarse representation used for coupling.

## 13.2 Multilevel force scatter

Stress and gravity are likewise scattered to the particle level and ancestors ([`solver/adaptive_engine.py:126-156`](solver/adaptive_engine.py#L126-L156)):

$$
\mathbf{f}_I^\ell\mathrel{+}=
N_{Ip}^\ell m_p\mathbf{g}
-
V_p^0J_p\boldsymbol\sigma_p\nabla N_{Ip}^\ell.
$$

No clamp is applied to `J` when current volume is formed. A negative determinant would reverse the volume/stress-force sign even though constitutive pressure separately clamps `J`.

## 13.3 Grid update and clamps

The grid update uses the axis-dependent effective boundary mass, multiplies by relaxation damping, and norm-clamps velocity to

$$
V_{clamp}=10\,\texttt{MAX_WAVE_SPEED}.
$$

With the main configuration, `MAX_WAVE_SPEED = C_0 + V_MAX_ESTIMATE`. Code: [`solver/adaptive_engine.py:12-15`](solver/adaptive_engine.py#L12-L15), [`config.py:235-238`](config.py#L235-L238), [`solver/adaptive_engine.py:158-172`](solver/adaptive_engine.py#L158-L172).

## 13.4 Coarse-fine interface coupling

After level solves, every fine node is marked as a ghost if:

- fluid mass is at/below cutoff; or
- it lies within `AMR_GHOST_BAND_CELLS*h_l` of an interior patch face.

Faces coincident with the physical domain are excluded from the interface-band rule. Marked velocities are overwritten by quadratic B-spline interpolation of the immediate parent level ([`core/quadtree_grid.py:698-749`](core/quadtree_grid.py#L698-L749)).

This is a one-way parent-to-child velocity prolongation after both levels have been integrated. Fine-to-coarse influence occurs earlier because fine particles scatter to ancestors. There is no composite linear solve, flux refluxing, nodal force restriction pass, or momentum correction after ghost overwrite. Consequently, local interface velocity consistency is imposed, but exact composite-grid nodal momentum conservation is not proven by the implementation.

## 13.5 G2P, advection, and state update

A particle gathers only from its own current level:

$$
\mathbf{v}_p^{n+1}=\sum_I N_{Ip}^{l_p}\mathbf{v}_I^{l_p},
$$

$$
\mathbf{C}_p^{n+1}=\frac{4}{h_{l_p}^2}
\sum_I N_{Ip}^{l_p}\mathbf{v}_I^{l_p}
(\mathbf{x}_I^{l_p}-\mathbf{x}_p)^T.
$$

Both velocity and `C` norm are clamped. Position is advanced, projected out of the platform when applicable, and kept `0.1*h_l` inside the domain. Then

$$
\mathbf{F}_p^{n+1}=(\mathbf{I}+\Delta t\mathbf{C}_p^{n+1})\mathbf{F}_p^n,
$$

adaptive-water stress is computed using `h_l`, and reported pressure is

$$
P_p=\max[K(1/\max(J_p,0.1)-1),0].
$$

Code: [`solver/adaptive_engine.py:203-251`](solver/adaptive_engine.py#L203-L251). Stress pressure clamps at `J=0.96`, while the stored pressure clamps at `J=0.1`; these fields can therefore diverge significantly under strong compression.

---

# 14. Refinement target algorithms

## 14.1 Geometric target

`finest_level_at(x)` gives the deepest patch containing the particle. This is the target for ordinary geometric and dynamic refinement.

## 14.2 Gradient target

After G2P, the code computes

$$
\eta_p=\|\mathbf{C}_p\|_F h_{l_p}.
$$

It initializes `target=0` and, for each permitted `k`, assigns `target=k` if

$$
\eta_p > \eta_0 2^k.
$$

Code: [`solver/adaptive_engine.py:296-324`](solver/adaptive_engine.py#L296-L324).

Exact semantic consequence: because passing the `k=0` threshold assigns target **zero**, a level-zero particle is not promoted until it also passes the `k=1` threshold, `2*eta_0`. A target level 2 requires `4*eta_0`. This differs from comments that describe the level-zero threshold as triggering refinement.

`AMR_GRADIENT_PRESSURE_THRESHOLD` and the configuration comment about `|J-1|` do not participate in this kernel ([`config.py:43-53`](config.py#L43-L53)). Gradient adaptation is velocity-gradient-only.

## 14.3 Target combination

The particle target is ([`core/adaptive_particles.py:162-168`](core/adaptive_particles.py#L162-L168)):

```text
geometric_target = deepest containing patch
if grid is static AND gradient refinement is enabled:
    target = min(gradient_target, geometric_target)
else:
    target = geometric_target
```

Therefore:

- static gradient mode cannot refine beyond available grid geometry;
- dynamic refinement ignores the gradient target and follows geometry only;
- disabling gradient refinement activates geometric refinement;
- with the default static AMR configuration, leaf-initialized fine particles can merge to level zero when their gradient target remains zero.

The smoke test demonstrates this last behavior: its initial `[1664, 1056, 1024, 3584]` particles by level become `[2048, 0, 0, 0]` after three resting-fluid steps.

---

# 15. Particle splitting and merging

## 15.1 Native mass

The native mass at level $\ell$ is

$$
m_{native,\ell}=\rho_0(h_\ell/n_{ppc})^2.
$$

This lets the solver distinguish a coarse particle that needs four children from a particle already small enough to change grid level without another split ([`core/adaptive_particles.py:40-45`](core/adaptive_particles.py#L40-L45)).

## 15.2 Split

If `target > current`, only one level is advanced per split pass. If

$$
m_p>1.5m_{native,\ell+1},
$$

the parent becomes four children with

$$
m_c=m_p/4,\qquad V_c^0=V_p^0/4,
$$

$$
\mathbf{x}_c=\mathbf{x}_p+
(\pm a,\pm a),
\qquad a=\frac14\sqrt{V_p^0},
$$

$$
\mathbf{v}_c=\mathbf{v}_p+
\mathbf{C}_p(\mathbf{x}_c-\mathbf{x}_p).
$$

`C`, `F`, stress, pressure, material, `Jp`, and `gradient_level` are copied ([`core/adaptive_particles.py:170-217`](core/adaptive_particles.py#L170-L217)). Symmetric offsets and quarter masses preserve total mass, center of mass, and the linear momentum represented by the affine field.

The split is **atomic at the parent level**: before reserving slots, `_children_fit_level` checks that all four child centers lie inside the next-level patch region. If any child would fall outside the refined region, the split is deferred to a later substep rather than producing a partial two-child group ([`core/adaptive_particles.py:171-177`](core/adaptive_particles.py#L171-L177)). This prevents malformed half-mass particles at refinement interfaces.

If mass is below the split threshold, only the level is incremented. New children are not recursively processed in the same kernel because the loop bound is the pre-split active count. A normal substep performs one pass; a dynamic grid shift performs up to `max_level` passes ([`solver/adaptive_engine.py:261-267`](solver/adaptive_engine.py#L261-L267)).

## 15.3 Capacity handling

A split reserves three new slots atomically because the parent slot is reused as one child. On failure, the reservation is rolled back and `split_overflow` increments. Dynamic AMR fails immediately with a recommended capacity factor; static mode only prints a warning every 500 steps ([`solver/adaptive_engine.py:275-287`](solver/adaptive_engine.py#L275-L287), [`solver/adaptive_engine.py:395-399`](solver/adaptive_engine.py#L395-L399)).

## 15.4 Merge bins

For each coarsening target level, particles are binned by:

- material;
- target-level subcell of spacing `h_target/ppc_axis`.

The slot mapping is in [`core/adaptive_particles.py:219-229`](core/adaptive_particles.py#L219-L229).

A bin is merged only when **all** conservative conditions hold ([`core/adaptive_particles.py:283-316`](core/adaptive_particles.py#L283-L316)):

1. **Minimum count**: `merge_count >= merge_min_particles`, where `merge_min_particles` is `max(4, AMR_MERGE_MIN_PARTICLES)`.
2. **Uniform coarsening**: `merge_coarsen_count == merge_count`, i.e. every particle in the bin is arriving from a finer level. Mixed groups with same-level particles are rejected.
3. **Native mass**: `|merge_mass - native_mass[target]| < 1e-6 * native_mass[target]`.

If any condition fails, the bin is left unchanged and its particles remain at their current level. This means incomplete sibling groups are retained rather than collapsed into malformed half-mass or mixed-level particles.

When a bin is accepted, all eligible same-material particles in the bin are replaced by the lowest-index particle with

$$
m=\sum_qm_q,\qquad V^0=\sum_qV_q^0,
$$

$$
\mathbf{x}=\frac{\sum_qm_q\mathbf{x}_q}{m},
\quad
\mathbf{v}=\frac{\sum_qm_q\mathbf{v}_q}{m},
\quad
\mathbf{C}=\frac{\sum_qm_q\mathbf{C}_q}{m},
$$

$$
\mathbf{F}=\frac{\sum_qV_q^0\mathbf{F}_q}{V^0},
\quad
\boldsymbol\sigma=\frac{\sum_qV_q^0\boldsymbol\sigma_q}{V^0},
$$

with pressure and `Jp` also volume-weighted ([`core/adaptive_particles.py:298-316`](core/adaptive_particles.py#L298-L316)). Removed records are marked and compacted through temporary arrays ([`core/adaptive_particles.py:335-373`](core/adaptive_particles.py#L335-L373)).

Mass and linear momentum are conserved by these averages. Deformation determinant, constitutive energy, angular momentum, and APIC affine moments are not explicitly constrained. Averaging `F` can produce a state not equivalent to the merged constitutive energy.

## 15.5 Unsupported-particle demotion

After merge finalization, `_demote_unsupported_particles` checks whether any remaining high-level particle's quadratic B-spline stencil still fits inside its grid patch ([`core/adaptive_particles.py:318-333`](core/adaptive_particles.py#L318-L333)). If the particle has moved outside the supported region and its target is coarser, its level is demoted to the target so it can participate in the next P2G on a valid grid. This is also called when merging is disabled.

---

# 16. Dynamic refinement-window motion

## 16.1 Platform trajectory

The AMR platform trajectory consistently uses constant motion, linear deceleration, and rest, implemented in NumPy and Taichi forms ([`core/quadtree_grid.py:287-299`](core/quadtree_grid.py#L287-L299), [`core/quadtree_grid.py:371-388`](core/quadtree_grid.py#L371-L388)).

## 16.2 Physics-based criterion centers

For non-platform criteria, the code copies active particle arrays to NumPy and computes a weighted centroid ([`core/quadtree_grid.py:327-369`](core/quadtree_grid.py#L327-L369)):

- velocity: select `|v| > fraction*V_MAX_ESTIMATE`, weight by `mass*|v|`;
- pressure: select `P > fraction*rho*C0^2`, weight by `mass*P`;
- deformation: select `|J-1| > threshold`, weight by `mass*|J-1|`;
- combined: add all three weights.

The combined sum mixes quantities with different physical units, so pressure will generally dominate unless values are separately normalized. If no particle passes, the algorithm silently falls back to platform displacement.

The first nonempty centroid becomes the reference. Subsequent patch displacement is `current_center - initial_center`; patch size does not change. `AMR_REFINEMENT_MARGIN` is not used by this production function to resize the patch; `sub_001` manually builds its input box from that margin.

## 16.3 Shift snapping and constraints

For every level above zero, desired translation is rounded to the nearest parent-cell spacing:

$$
snap(d,h)=\operatorname{sign}(d)
\left\lfloor |d|/h+1/2\right\rfloor h.
$$

It is then clamped so the patch remains in the physical domain and inside its already-shifted parent ([`core/quadtree_grid.py:301-320`](core/quadtree_grid.py#L301-L320)). Level zero never moves.

When shifts change, Taichi and NumPy region/origin copies are updated, domain-boundary penalty stencils are rebuilt, and the moving-platform penalty cache is invalidated ([`core/quadtree_grid.py:390-419`](core/quadtree_grid.py#L390-L419)).

The centroid computation and field transfers are host-device synchronization points every regrid interval.

---

# 17. Poiseuille algorithm

`PoiseuilleSolver.step` is defined at [`solver/Poiseuille_engine.py:282-300`](solver/Poiseuille_engine.py#L282-L300):

```text
clear
P2G mass and unnormalized momentum
periodically fold left/right ghost mass and momentum
normalize momentum
scatter forces
periodically fold forces
grid update
set top/bottom wall nodes to zero (no-slip)
copy periodic internal velocities to ghost nodes
G2P and periodic particle wrap
update F
nodal F-bar volume scatter/gather and stress
```

## 17.1 Periodic mapping

Ghost contributions are added to nodes one domain width away, then combined internal values are copied back to ghosts ([`solver/Poiseuille_engine.py:44-72`](solver/Poiseuille_engine.py#L44-L72)). Forces receive the same treatment ([`solver/Poiseuille_engine.py:99-116`](solver/Poiseuille_engine.py#L99-L116)). Updated velocity is copied, not added, to ghost nodes before G2P ([`solver/Poiseuille_engine.py:126-144`](solver/Poiseuille_engine.py#L126-L144)).

Particles wrap by

$$
x_p\leftarrow x_{min}+((x_p-x_{min})\bmod L_x).
$$

Code: [`solver/Poiseuille_engine.py:146-181`](solver/Poiseuille_engine.py#L146-L181).

## 17.2 Full APIC inertia

Unlike the standard solver's scalar interior approximation, Poiseuille forms

$$
\mathbf{D}_p=\sum_IN_{Ip}\mathbf{d}_{Ip}\mathbf{d}_{Ip}^T,
$$

and computes

$$
\mathbf{C}_p=\mathbf{B}_p\mathbf{D}_p^{-1}.
$$

This appears at [`solver/Poiseuille_engine.py:147-176`](solver/Poiseuille_engine.py#L147-L176).

## 17.3 Current integration limitations

The current `config.py` has no `POISEUILLE` scenario branch defining particle geometry and gravity/drive before globally derived particle quantities are evaluated. Selecting it directly makes `MP_WIDTH` unavailable at [`config.py:207`](config.py#L207). In addition, `main.py` always calls `step(damping=..., current_time=...)`, while `PoiseuilleSolver.step` accepts no arguments. No pressure-gradient-specific force exists in this solver; driving would need to be encoded in `config.GRAVITY` or another external modification.

---

# 18. Continuous inflow algorithm

The inflow step is [`solver/Inflow_engine.py:397-417`](solver/Inflow_engine.py#L397-L417): emit particles, clear, APIC P2G, force scatter, NTU wall EBC, grid update, outer free-slip, NTU-aware G2P, one-way inlet valve, deformation update, and nodal F-bar.

## 18.1 Emission accumulator

Each substep accumulates stream travel

$$
d_{pending}\mathrel{+}=u_{in,x}\Delta t.
$$

When it exceeds horizontal particle spacing, the integer number of complete vertical particle columns is emitted and the residual distance is retained ([`solver/Inflow_engine.py:25-42`](solver/Inflow_engine.py#L25-L42)). This produces a rate-consistent lattice without requiring a column every step.

## 18.2 Slot activation

A kernel atomically reserves `num_columns*NUM_MP_HEIGHT` slots, lays out each column, sets inlet velocity, identity `F`, zero stress, and zero pressure ([`solver/Inflow_engine.py:44-69`](solver/Inflow_engine.py#L44-L69)).

The reservation increments `active_count` even when a slot is beyond `MAX_PARTICLE_INFLOW`; later kernels iterate to `active_count`. Also, the underlying `ParticleSystem` capacity is `TOTAL_NUM_MP`, not independently `MAX_PARTICLE_INFLOW`. Robust use requires these capacities to be identical and the counter to be capped, which current code does not enforce.

## 18.3 Maze and valve

The stationary NTU polyline uses generalized slip EBC on grid nodes ([`solver/Inflow_engine.py:123-156`](solver/Inflow_engine.py#L123-L156)). G2P also removes inward normal particle velocity inside the regularization distance using

$$
\boldsymbol\Lambda=\mathbf{I}-\delta\mathbf{n}\mathbf{n}^T,
$$

with a cubic distance factor ([`solver/Inflow_engine.py:208-293`](solver/Inflow_engine.py#L208-L293)). A broad inlet region then enforces `v_x >= inlet_vx` on particles ([`solver/Inflow_engine.py:158-174`](solver/Inflow_engine.py#L158-L174)).

## 18.4 Current integration limitations

`INFLOW_DURATION`, `INFLOW_VELOCITY`, and `MAX_PARTICLE_INFLOW` are referenced but absent from current `config.py`. As with Poiseuille, no `INFLOW` branch defines `MP_WIDTH/MP_HEIGHT` before global particle derivation. The scenario is therefore an available algorithm requiring configuration repair, not a currently runnable top-level case.

`main.py` also creates a `grid_mask` for geometry output but never populates it, so `geometry_check.vtk` contains zeros ([`main.py:126-139`](main.py#L126-L139)).

---

# 19. Legacy base engine

[`solver/engine.py`](solver/engine.py) is not imported by the current scenario dispatcher. It contains historical alternatives:

- kinematic P2G;
- normalized and unnormalized APIC P2G;
- `0.98` FLIP/PIC blending;
- full-inertia APIC G2P;
- periodic mappings;
- a divergence-smoothing pressure-rate update;
- an `mp_update` that always calls Bingham stress.

Its `step` is Poiseuille-specific ([`solver/engine.py:461-475`](solver/engine.py#L461-L475)). The rate-form pressure smoother projects particle divergence to nodes, gathers a smoothed divergence, and integrates

$$
P^{n+1}=P^n-K\Delta t\overline{\nabla\cdot\mathbf{v}},
$$

then uses `stress=-P I` ([`solver/engine.py:407-459`](solver/engine.py#L407-L459)). This algorithm is not executed by `main.py` and should be described as legacy/reference code.

---

# 20. Benchmark-only algorithms

## 20.1 Gradient dam-break

The dedicated benchmark starts all particles at level zero, gives every grid level full-domain coverage, and lets the velocity-gradient target split particles ([`benchmarks/dam_break/run_experiment.py:47-106`](benchmarks/dam_break/run_experiment.py#L47-L106)). It checks partition-of-unity diagnostics every frame but does not assert them ([`benchmarks/dam_break/run_experiment.py:176-201`](benchmarks/dam_break/run_experiment.py#L176-L201)).

The file-level comments still mention a pressure/deformation split criterion, but production `compute_gradient_levels` no longer implements that criterion.

## 20.2 Moving refinement benchmark

`sub_001` manually builds an asymmetric refinement box around the immersed platform, validates that planned platform motion can move the finest patch, runs relaxation with platform time fixed at zero, and reports partition-of-unity every frame ([`benchmarks/sub_001/run_experiment.py:13-60`](benchmarks/sub_001/run_experiment.py#L13-L60), [`benchmarks/sub_001/run_experiment.py:106-171`](benchmarks/sub_001/run_experiment.py#L106-L171)).

## 20.3 Three-material impact

The benchmark creates a moving material-4 block and a column labeled fluid, jelly, or snow. A benchmark subclass overrides G2P/constitutive behavior, while inherited P2G and force scatter couple all material particles through one shared velocity grid ([`benchmarks/three_materials/run_experiment.py:18-100`](benchmarks/three_materials/run_experiment.py#L18-L100)). The block's gathered `v_x` is forced to at least `OBJECT_SPEED` until it reaches `DRIVE_STOP_X`.

It checks per-material mass and finite positions; “block cleared column” is returned as a metric rather than asserted ([`benchmarks/three_materials/run_experiment.py:250-292`](benchmarks/three_materials/run_experiment.py#L250-L292)).

## 20.4 Translating and rotating blocks

`three_blocks` prescribes every particle's rigid translational state before and after each solver step, resetting `C`, `F`, stress, pressure, and `Jp` ([`benchmarks/three_blocks/run_experiment.py:163-214`](benchmarks/three_blocks/run_experiment.py#L163-L214)). It primarily tests transfer, split/merge, coarsening, and conservation rather than free constitutive evolution.

`rotating_blocks` prescribes

$$
\mathbf{v}=\left(u-\omega(y-c_y),\;\omega(x-c_x)\right),
\qquad
\mathbf{C}=\begin{bmatrix}0&-\omega\\\omega&0\end{bmatrix},
$$

with piecewise acceleration/cruise/deceleration ([`benchmarks/rotating_blocks/run_experiment.py:136-167`](benchmarks/rotating_blocks/run_experiment.py#L136-L167), [`benchmarks/rotating_blocks/run_experiment.py:202-230`](benchmarks/rotating_blocks/run_experiment.py#L202-L230)). It checks mass, finite state, coarsening, boundary clearance, and a manually reset traction-free final state.

These tests compare two adaptation modes:

- fixed nested-patch geometry (“quadtree”);
- full-domain level availability plus gradient particle targets (“gradient”).

---

# 21. Output and visualization algorithms

## 21.1 Particle VTK

`write_vtk` converts nonfinite positions, pressure, and velocity to zero, writes all particles as one VTK `POLYDATA` polyvertex cell, and optionally writes material IDs ([`utils/exporter.py:57-126`](utils/exporter.py#L57-L126)). Sanitizing nonfinite values keeps ParaView readable but can hide a numerical failure in exported data; verification should inspect simulation arrays before export.

## 21.2 Geometry VTK

The exporter supports:

- 2-D/3-D outer-domain wireframes;
- current AMR region outlines;
- stationary square obstacles;
- time-indexed 2-D/3-D moving platforms;
- NTU polylines;
- grid-normal vectors.

See [`utils/exporter.py:128-366`](utils/exporter.py#L128-L366). `write_quadtree_grid_vtk` exports patch rectangles, not every leaf cell. `benchmarks/run_benchmark.py` has a separate true leaf-cell polygon exporter ([`benchmarks/run_benchmark.py:31-47`](benchmarks/run_benchmark.py#L31-L47)).

## 21.3 Plotting

`visualize_initial_state` reconstructs and plots the uniform particle lattice and padded/physical grid. `plot_velocity_profile` plots particle `v_x` against `y` for Poiseuille analysis ([`utils/visualization.py:12-102`](utils/visualization.py#L12-L102)).

---

# 22. Verification map

The following repository checks were executed against the documented working tree:

```text
.venv/bin/python -m benchmarks.smoke_adaptive_mpm
.venv/bin/python verify_penalty_boundary.py
.venv/bin/python verify_dynamic_refinement.py
.venv/bin/python verify_particle_merge.py
.venv/bin/python verify_adaptive.py
```

All completed with exit code zero. Taichi emitted cache-lock warnings for `~/.cache/taichi/ticache/ticache.lock`; these did not prevent compilation or execution.

## 22.1 Smoke construction

[`benchmarks/smoke_adaptive_mpm.py`](benchmarks/smoke_adaptive_mpm.py) constructs four levels, reports geometry/counts, advances three steps, and prints one position. It contains no assertions. In the validated working tree it completed successfully and demonstrated resting gradient-mode coarsening from 7328 to 2048 active particles.

## 22.2 Penalty boundary verification

[`verify_penalty_boundary.py:20-78`](verify_penalty_boundary.py#L20-L78) checks:

- analytical total vertical/horizontal wall virtual mass;
- directional switches at bottom, left, and interior probes;
- componentwise velocity normalization;
- moving-platform momentum equal to virtual mass times wall velocity.

Validated output:

```text
beta = 2.500000e+02
boundary mass sums = 4.000000e+03, 2.000000e+03
bottom probe velocity = [0.5, -0.005277044854881266]
penalty boundary verification passed
```

## 22.3 Dynamic refinement verification

[`verify_dynamic_refinement.py:57-194`](verify_dynamic_refinement.py#L57-L194) asserts:

- cached and rebuilt boundary penalty sums agree;
- moving-platform penalty mass/momentum agree;
- per-level shifts are nested, domain-bounded, and parent-grid quantized;
- finest-level lookup follows the shifted box;
- particle levels match current geometric targets after steps;
- no particle lies inside the platform;
- particle mass is conserved to `rtol=1e-12`, `atol=1e-15`;
- the dynamic split-overflow error path raises immediately when the test manually seeds a nonzero overflow counter.

Validated output reported shift `[0.00125, -0.00625]`, probe levels `[0, 2]`, active particles `5484 -> 5389`, and passed.

## 22.4 Merge verification

[`verify_particle_merge.py`](verify_particle_merge.py) tests the conservative split/merge interface:

- exact four-child splits with mass and linear-momentum conservation;
- rejection of partial/mixed merges (incomplete groups are retained);
- forward and backward level transitions;
- mass and linear-momentum conservation through round-trip split/merge.

Validated output reported `level 0/1 and level 1/2 split/merge conservation OK` and passed.

## 22.5 Hydrostatic adaptive diagnostic

[`verify_adaptive.py:49-92`](verify_adaptive.py#L49-L92) runs 6000 damped plus 1000 undamped steps, then prints total mass, speed, hydrostatic pressure relative errors, fine/coarse pressure ratios, level counts, and split overflow. Its only numerical assertion is the absence of NaNs in position and velocity at [`verify_adaptive.py:67`](verify_adaptive.py#L67). Despite its regression description, mass and pressure accuracy are currently diagnostics, not pass/fail tolerances.

The validated run completed successfully at `dt=1.603e-6 s` and `t=11.22 ms`. It reported total mass `0.8` (expected `0.8`), maximum speed `1e-4 m/s`, bottom-half hydrostatic relative error `0.0009` mean / `0.0028` maximum, pressure ratios `1.0025` both inside and outside the fine region, particle levels `[2048, 0, 0, 0]`, and zero split overflow.

## 22.6 Performance benchmark

[`benchmarks/run_benchmark.py:50-174`](benchmarks/run_benchmark.py#L50-L174) measures setup, relaxation, simulation, and VTK wall time and records Taichi kernel profiles. Stored results show:

| Solver | Initial/final particles | dt | main steps | ms/step | sim ms/frame |
|---|---:|---:|---:|---:|---:|
| Standard | 6000 | `5.291005e-5` | 94,500 | 0.631 | 198.8 |
| Adaptive L0-L2 | 26,784 / 79,737 | `1.323802e-5` | 377,700 | 1.505 | 1894.2 |

Sources: [`benchmarks/standard_2D/benchmark_summary.txt`](benchmarks/standard_2D/benchmark_summary.txt), [`benchmarks/adaptive_quadtree_2D/benchmark_summary.txt`](benchmarks/adaptive_quadtree_2D/benchmark_summary.txt).

The adaptive profile is dominated by P2G (~30.2%) and force scatter (~22.1%) ([`benchmarks/adaptive_quadtree_2D/kernel_profile.txt:4-27`](benchmarks/adaptive_quadtree_2D/kernel_profile.txt#L4-L27)). The comparison is not equal-time-step or equal-particle-count: adaptive uses four times as many substeps per frame and its population grows substantially.

## 22.7 Coverage gaps

No current automated assertion establishes:

- standard solver mass/momentum/pressure convergence;
- 3-D correctness;
- Bingham constitutive response;
- Poiseuille analytical profile;
- inflow rate/capacity behavior;
- EBC force convergence;
- energy behavior across split/merge;
- exact interface momentum conservation;
- jelly/snow constitutive unit tests;
- pressure-error acceptance in `verify_adaptive.py`.

`test.py` is an unrelated Mode-II crack plastic-zone plotting script and does not test MPM ([`test.py:1-47`](test.py#L1-L47)). `mpm_project.txt` is an outdated project-tree sketch.

---

# 23. Computational complexity

Let `Np` be active particles, `Ng` standard nodes, `L+1` AMR levels, `Ng_l` nodes in level `l`, and `l_p` a particle's level.

## 23.1 Standard 2-D per substep

- P2G: `O(9*Np)` plus `O(Ng)` normalization.
- force scatter: `O(9*Np)`.
- grid update/boundary: `O(Ng)`.
- G2P: `O(9*Np)`.
- F update: `O(Np)`.
- F-bar scatter and gather: `O(18*Np)`.
- grid clear: `O(Ng)`.

Overall:

$$
O(N_p+N_g)
$$

with a large constant of roughly five 9-node particle stencil passes.

In 3-D, every stencil pass uses 27 nodes.

## 23.2 Adaptive per substep

With ancestor scattering:

$$
P2G/forces=O\left(9\sum_p(l_p+1)\right).
$$

G2P is `O(9*Np)`. Grid clearing, normalization, update, and interface fill cost

$$
O\left(\sum_{\ell=0}^{L}N_{g,\ell}\right).
$$

Merge is potentially expensive:

$$
O\left(L(C_{merge}+N_p)\right),
$$

because all merge bins are cleared and particles are accumulated/finalized for every target level. Splitting is `O(Np)` per pass. A complete adaptation can invoke `L` split passes.

Dynamic physics-based criteria copy arrays to the CPU and cost `O(Np)` plus synchronization at each regrid interval. Penalty-stencil rebuild cost scales with boundary length divided by each level spacing times the 9-node stencil.

## 23.3 Memory

The standard solver is dense and scales as `O(Ng + Np)`. Adaptive memory is

$$
O\left(\sum_lN_{g,l}+C_p+C_{merge}\right),
$$

with multiple vector fields per level, complete temporary particle buffers, and potentially large merge-bin fields. The hierarchy is patch-dense, so its memory benefit depends on fine patches occupying a small fraction of the domain.

---

# 24. Conservation and stability properties

## 24.1 Exactly constructed particle conservation

- Split conserves particle mass exactly by four quarter masses.
- Split is atomic: all four child centers must fit inside the next-level patch or the split is deferred.
- Symmetric APIC child velocities preserve parent linear momentum for the represented affine field.
- Merge conserves particle mass and linear momentum through sums/mass-weighted velocity.
- Merge is conservative: only complete, uniformly coarsening, native-mass groups are accepted; incomplete groups are retained.
- Unsupported high-level particles are demoted to their target level after merge.
- Dynamic adaptation has a dedicated particle-mass assertion.

## 24.2 Transfer properties

For complete interior stencils, quadratic B-splines partition unity, giving conservative single-level P2G mass transfer. APIC is designed to preserve affine velocity fields and angular momentum under its assumptions.

On the adaptive hierarchy, the same mass is deliberately represented on multiple levels; only particle mass is the physical global invariant. Parent interpolation overwrites fine velocities without a post-correction momentum transfer, so exact composite-grid momentum conservation is not established.

## 24.3 Numerical stabilizers

The project uses several independent stabilizers:

- weak-compressible sound speed ten times free-fall speed;
- conservative CFL `0.1`;
- Bingham viscous time limit;
- dynamic relaxation damping `0.98`;
- `J` clamps in constitutive pressure;
- no-tension pressure cutoff;
- nodal F-bar volume smoothing in standard solver;
- artificial viscosity in adaptive water;
- grid and particle velocity/affine clamps;
- nodal mass cutoffs;
- hard particle domain clamps;
- obstacle penetration projection;
- fine-grid parent-velocity interface fill.

These mechanisms improve robustness but mean a result can be influenced by nonphysical clamps. A technical report should list the active stabilizers alongside governing equations.

---

# 25. Important implementation caveats

## 25.1 Configuration and reachability

1. **Poiseuille and inflow cannot be selected by editing only `ACTIVE_SCENARIO`.** Their scenario geometry variables are absent before global particle derivation ([`config.py:171-216`](config.py#L171-L216)).
2. **Poiseuille step signature is incompatible with `main.py`.** Main passes `damping/current_time`; Poiseuille accepts no parameters ([`main.py:163-168`](main.py#L163-L168), [`solver/Poiseuille_engine.py:282`](solver/Poiseuille_engine.py#L282)).
3. **Inflow requires undefined settings:** `INFLOW_DURATION`, `INFLOW_VELOCITY`, and `MAX_PARTICLE_INFLOW`.
4. **Dam-break obstacle requires undefined `INT_SQUARE_*` settings** when `IS_DAMBREAK_WITH_OBSTACLE=True`.
5. `AMR_DYNAMIC_REFINEMENT`, merge options, material count, and dynamic platform Y margin are obtained with defaults or injected by benchmarks rather than all being declared in `config.py`.
6. For the dedicated `ADAPTIVE_MPM` scenario, the physical AMR domain begins at `(0,0)`, but `main.py` draws the exported outer boundary from `(PADDING*DX, PADDING*DY)`. The simulation boundary and displayed boundary are therefore offset in that scenario ([`config.py:186-202`](config.py#L186-L202), [`main.py:85-100`](main.py#L85-L100)).
7. `init_boundary_fields` always allocates a 2-D normal field even when `DIM=3`; it is harmless on current 3-D paths because those normals are used only by the 2-D inflow visualization ([`physics/boundary.py:7-17`](physics/boundary.py#L7-L17)).

## 25.2 Comments/configuration versus code

1. `AMR_GRADIENT_PRESSURE_THRESHOLD` is unused.
2. Gradient refinement does not inspect `|J-1|`.
3. Gradient threshold semantics are one level lower than the comments imply because passing threshold `k` assigns target `k`.
4. `AMR_MERGE_MIN_PARTICLES` is enforced (clamped to a minimum of 4); bins below the threshold are never merged.
5. `AMR_REFINEMENT_MARGIN` does not size a production dynamic box automatically.
6. `AMR_REFERENCE_FINE_DX` is not used to derive actual spacing.
7. The so-called quadtree is a nested dense-patch hierarchy, not an explicit recursive tree.

## 25.3 Standard solver physics/numerics

1. Default `DAM_BREAK` is full-width hydrostatic water, not a classic dam column.
2. F-bar replaces `F` by a purely isotropic matrix, eliminating all deviatoric deformation history.
3. Reported pressure and stress pressure use different `J` clamps.
4. Standard artificial viscosity is defined but not called.
5. `MU_FRIC` is configured but no friction law consumes it.
6. Componentwise standard speed clamp permits vector norm above 15.
7. Outer boundary implementation differs by one node between generic and solver-local kernels.
8. Immersed relaxation advances platform time, then main time resets to zero.
9. The 3-D grid is anisotropic but kernels assume one isotropic `DX`.
10. The 3-D water/Bingham deviatoric operator remains two-dimensional.
11. 3-D platform force and particle projection use different stopping/deceleration details.

## 25.4 Adaptive solver limitations

1. There is no level subcycling.
2. There is no composite solve or refluxing at coarse-fine interfaces.
3. Dynamic refinement translates existing boxes but does not change size or rebuild leaf arrays.
4. Gradient target is ignored when dynamic refinement is active.
5. Combined dynamic criterion adds dimensionally inconsistent weights.
6. Merge preserves mass/linear momentum but not constitutive energy or exact affine/angular moments. Conservative merge requires complete sibling groups; incomplete groups are retained, which can leave fine particles temporarily above their target level.
7. Boundary virtual mass is not a force-based contact law and acts only where fluid mass activates a node.
8. Stress uses `J>=0.96`; reported pressure uses `J>=0.1`; current volume uses unclamped `J`.
9. Default leaf initialization followed by static gradient adaptation can allocate fine particles and promptly merge them to coarse.
10. Static split overflow warns late rather than failing immediately.

## 25.5 Output and validation

1. VTK export replaces NaN/Inf by zero.
2. `verify_adaptive.py` prints mass/pressure errors but asserts only finite position/velocity.
3. Smoke and several benchmarks print diagnostics without pass/fail tolerances.
4. Stored standard/adaptive performance results are not an equal-resolution or equal-time-step comparison.

---

# 26. Report-ready pseudocode

## 26.1 Standard 2-D solver

```text
Given particles {x, v, C, F, stress, pressure}:

for each substep:
    zero grid {mass, momentum/velocity, force, F-bar volumes}

    for each particle p:
        for each of 9 quadratic B-spline nodes I:
            mass[I] += N(I,p) * particle_mass
            momentum[I] += N(I,p) * particle_mass
                           * (v[p] + C[p] * (x[I] - x[p]))
    velocity[I] = momentum[I] / mass[I]

    for each particle p:
        current_volume = reference_volume * det(F[p])
        for each support node I:
            force[I] += N(I,p) * particle_mass * gravity
                        - current_volume * stress[p] * grad_N(I,p)

    add optional immersed/obstacle EBC force
    velocity[I] = damping * (velocity[I] + dt * force[I] / mass[I])
    clamp nodal velocity and remove outward domain-normal velocity

    for each particle p:
        v[p] = sum_I N(I,p) * velocity[I]
        B[p] = sum_I N(I,p) * velocity[I] outer (x[I]-x[p])
        C[p] = 4/dx^2 * B[p]
        x[p] += dt * v[p]
        project/clamp particle position
        F[p] = (I + dt*C[p]) * F[p]

    scatter reference_volume and reference_volume*det(F) to nodes
    gather nodal current/reference volume ratio J_bar
    F[p] = J_bar^(1/dimension) * I
    stress[p] = selected_fluid_stress(F[p], C[p])
    pressure[p] = max(K*(1/J_bar - 1), 0)
```

## 26.2 Adaptive 2-D solver

```text
Construct nested dense patches with h[level] = h0 / 2^level.
Construct variable-mass particles from leaf cells or a fixed initial level.

for each substep:
    if dynamic regrid interval:
        compute platform/physics criterion displacement
        snap each level shift to parent lattice and enforce nesting/domain bounds
        if hierarchy moved:
            rebuild domain penalty stencils
            merge complete coarse-going groups
            demote unsupported high-level particles
            repeatedly split until particles match geometric targets (four children must fit)

    clear all level fields
    load physical-domain virtual boundary mass
    add cached moving-platform virtual mass/momentum when immersed

    for each particle p:
        scatter APIC mass/momentum to own level and every ancestor
    normalize each level with fluid + virtual boundary mass/momentum

    for each particle p:
        scatter stress and gravity force to own level and every ancestor
    integrate every level using fluid + virtual boundary effective mass

    for each fine level:
        overwrite low-mass/interface-band velocities from parent interpolation

    for each particle p:
        gather velocity and affine matrix from its own level only
        advect and project/clamp position
        update F = (I + dt*C)*F
        compute adaptive-water stress with local h
        compute reported EOS pressure

    compute velocity-gradient target eta = norm(C)*h
    merge complete coarse-going groups (min count, uniform coarsening, native mass)
    demote unsupported high-level particles
    split particles by at most one level toward finer target (four children must fit)
```

---

# 27. Source-by-source responsibility index

| File | Algorithmic responsibility |
|---|---|
| [`config.py`](config.py) | scenarios, dimensions, grid/AMR/material/time/boundary constants |
| [`main.py`](main.py) | GPU initialization, solver dispatch, relaxation, frame loop, exports |
| [`core/functions.py`](core/functions.py) | uniform-grid quadratic B-spline weights/gradients |
| [`core/grid.py`](core/grid.py) | standard dense grid fields and clear kernel |
| [`core/particles.py`](core/particles.py) | standard particles, hydrostatic initialization, inflow pool |
| [`physics/constitutive_model.py`](physics/constitutive_model.py) | water, artificial-viscosity water, adaptive water, Bingham stress |
| [`physics/boundary.py`](physics/boundary.py) | free/no-slip kernels, EBC force, SDFs, NTU polyline normals |
| [`solver/standard_engine.py`](solver/standard_engine.py) | active standard 2-D/3-D APIC, EBC, F-bar step |
| [`solver/adaptive_engine.py`](solver/adaptive_engine.py) | adaptive APIC transfers, forces, interface fill orchestration, G2P, gradient target |
| [`core/quadtree_grid.py`](core/quadtree_grid.py) | nested patches, leaf tiling, dynamic shifts, penalty mass, parent interpolation |
| [`core/adaptive_particles.py`](core/adaptive_particles.py) | variable-mass particles, level targets, split, merge, compaction |
| [`solver/Poiseuille_engine.py`](solver/Poiseuille_engine.py) | periodic APIC channel algorithm |
| [`solver/Inflow_engine.py`](solver/Inflow_engine.py) | distance-accumulated emission, NTU wall coupling, inlet valve |
| [`solver/engine.py`](solver/engine.py) | legacy FLIP/APIC/pressure-rate experiments |
| [`utils/exporter.py`](utils/exporter.py) | particle, boundary, patch, platform, NTU, and normal VTK output |
| [`utils/visualization.py`](utils/visualization.py) | initial-grid and velocity-profile plotting |
| [`verify_penalty_boundary.py`](verify_penalty_boundary.py) | virtual boundary mass/momentum assertions |
| [`verify_dynamic_refinement.py`](verify_dynamic_refinement.py) | moving hierarchy, levels, platform exclusion, mass, capacity assertions |
| [`verify_particle_merge.py`](verify_particle_merge.py) | merge mass/count check |
| [`verify_adaptive.py`](verify_adaptive.py) | long hydrostatic diagnostic |
| [`benchmarks/run_benchmark.py`](benchmarks/run_benchmark.py) | standard/adaptive runtime and kernel profiling |
| [`benchmarks/three_materials/run_experiment.py`](benchmarks/three_materials/run_experiment.py) | benchmark-only water/jelly/snow/block material extension |

---

# 28. Bottom-line characterization

The core standard method is a weakly compressible, single-field APIC-MPM fluid integrator with explicit grid momentum evolution and a nodal-volume F-bar stabilization. Its most actively used production path is 2-D water with free-slip outer walls and optional immersed EBC.

The adaptive method is a 2-D hierarchy of nested dense patches. Variable-mass particles carry the physical state; fine particles also populate ancestor grids, and fine interface velocities are overwritten from parent interpolation. Adaptivity is realized by conservative-mass particle split/merge operations and by translating nested boxes when dynamic refinement is enabled. It is not a recursive sparse quadtree, does not subcycle in time, and does not solve a globally conservative composite-grid system.

For a formal report, the strongest validated claims are:

- exact-by-construction split/merge particle mass conservation, with dedicated tests;
- correct directional virtual penalty mass assembly and moving-wall momentum in the tested case;
- nested, snapped, bounded dynamic patch motion and geometric particle reassignment in the tested case;
- successful short adaptive execution;
- finite-state hydrostatic execution when the long diagnostic passes.

Claims about 3-D accuracy, Bingham validation, analytical Poiseuille flow, inflow capacity, full coarse-fine momentum conservation, or constitutive convergence require additional verification before being stated as demonstrated results.
