"""Shared material models from benchmarks/99_bench/mpm99.py.

Provides liquid (material 0), jelly (material 1), and snow (material 2)
constitutive updates so that benchmarks can select either a single material
or run all three simultaneously via the multi-material solver.

Liquid (material == 0):
    mu = 0
    F reset to sqrt(J) * I
    stress = lambda * J * (J - 1) * I

Jelly (material == 1):
    h = 0.3                            (constant softening)
    mu = mu_0 * h,  la = lambda_0 * h
    F evolves freely (no reset, no plasticity)
    stress = 2 * mu * (F - U @ V^T) @ F^T + I * la * J * (J - 1)

Snow (material == 2):
    h = exp(10 * (1 - Jp))           (hardening)
    mu = mu_0 * h,  la = lambda_0 * h
    singular values clamped to [1 - 2.5e-2, 1 + 4.5e-3]  (plasticity)
    F = U @ sig @ V^T
    stress = 2 * mu * (F - U @ V^T) @ F^T + I * la * J * (J - 1)
"""

import math

import taichi as ti
import config
from solver.adaptive_engine import AdaptiveMPMSolver2D

# --- Reference constants from 99_bench ---------------------------------------
YOUNGS_MODULUS = 0.1e4          # E = 1000
POISSON_RATIO = 0.2             # nu
DENSITY = 1.0                   # p_rho

# Lamé parameters (reference, before hardening)
MU_0 = YOUNGS_MODULUS / (2.0 * (1.0 + POISSON_RATIO))
LAMBDA_0 = YOUNGS_MODULUS * POISSON_RATIO / ((1.0 + POISSON_RATIO) * (1.0 - 2.0 * POISSON_RATIO))

# Snow plasticity parameters from 99_bench
SNOW_SIG_MIN = 1.0 - 2.5e-2     # 0.975
SNOW_SIG_MAX = 1.0 + 4.5e-3     # 1.0045
SNOW_HARDENING = 10.0

# Jelly softening (constant, from 99_bench material 1)
JELLY_SOFTENING = 0.3

# Liquid has zero shear
LIQUID_SHEAR_MODULUS = 0.0

# Wave speeds (used for CFL / velocity clamping)
LIQUID_WAVE_SPEED = math.sqrt(LAMBDA_0 / DENSITY)
SNOW_WAVE_SPEED = math.sqrt((LAMBDA_0 + 2.0 * MU_0) / DENSITY)
JELLY_WAVE_SPEED = math.sqrt((JELLY_SOFTENING * LAMBDA_0 + 2.0 * JELLY_SOFTENING * MU_0) / DENSITY)

MATERIALS = ("liquid", "snow", "jelly")
MULTI_MATERIAL_IDS = {"liquid": 0, "jelly": 1, "snow": 2}

# Target finest dx across all comparison cases (adaptive level-2 / 1500-case grid)
TARGET_FINEST_DX = 0.00125


def cfl_stable_dt(finest_dx, material="liquid", v_max_estimate=2.0, cfl=0.1):
    """Return the CFL-stable dt for a given dx and material."""
    if material == "liquid":
        wave_speed = LIQUID_WAVE_SPEED
    elif material == "snow":
        wave_speed = SNOW_WAVE_SPEED
    elif material == "jelly":
        wave_speed = JELLY_WAVE_SPEED
    elif material == "multi":
        wave_speed = SNOW_WAVE_SPEED  # most restrictive
    else:
        raise ValueError(f"unsupported material '{material}', expected one of {MATERIALS + ('multi',)}")
    max_wave_speed = wave_speed + v_max_estimate
    return cfl * finest_dx / max_wave_speed


def apply_mpm99_properties(material="liquid"):
    """Set config density / wave speed / Lamé constants from 99_bench."""
    config.RHO_0 = DENSITY
    if material == "liquid":
        config.C_0 = LIQUID_WAVE_SPEED
    elif material == "snow":
        config.C_0 = SNOW_WAVE_SPEED
    elif material == "jelly":
        config.C_0 = JELLY_WAVE_SPEED
    elif material == "multi":
        config.C_0 = SNOW_WAVE_SPEED  # most restrictive for CFL
    else:
        raise ValueError(f"unsupported material '{material}', expected one of {MATERIALS + ('multi',)}")
    config.MPM99_YOUNGS_MODULUS = YOUNGS_MODULUS
    config.MPM99_POISSON_RATIO = POISSON_RATIO
    config.MPM99_MU_0 = MU_0
    config.MPM99_LAMBDA_0 = LAMBDA_0


def mpm99_properties(material="liquid"):
    """Return a JSON-serialisable dict describing the chosen material model."""
    if material == "liquid":
        return {
            "source": "benchmarks/99_bench/mpm99.py",
            "material": "liquid",
            "material_id": 0,
            "density": DENSITY,
            "youngs_modulus": YOUNGS_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "shear_modulus": LIQUID_SHEAR_MODULUS,
            "lame_lambda": LAMBDA_0,
            "equivalent_wave_speed": LIQUID_WAVE_SPEED,
            "viscosity": 0.0,
            "deformation_reset": "F = sqrt(J) I",
            "cauchy_stress": "sigma = lambda J (J - 1) I",
        }
    elif material == "jelly":
        return {
            "source": "benchmarks/99_bench/mpm99.py",
            "material": "jelly",
            "material_id": 1,
            "density": DENSITY,
            "youngs_modulus": YOUNGS_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "softening": JELLY_SOFTENING,
            "shear_modulus": MU_0 * JELLY_SOFTENING,
            "lame_lambda": LAMBDA_0 * JELLY_SOFTENING,
            "equivalent_wave_speed": JELLY_WAVE_SPEED,
            "viscosity": 0.0,
            "deformation_reset": "none (F evolves freely)",
            "cauchy_stress": "sigma = 2 mu (F - U V^T) F^T + I la J (J - 1)",
        }
    elif material == "snow":
        return {
            "source": "benchmarks/99_bench/mpm99.py",
            "material": "snow",
            "material_id": 2,
            "density": DENSITY,
            "youngs_modulus": YOUNGS_MODULUS,
            "poisson_ratio": POISSON_RATIO,
            "shear_modulus": MU_0,
            "lame_lambda": LAMBDA_0,
            "hardening_coefficient": SNOW_HARDENING,
            "plasticity_clamp": [SNOW_SIG_MIN, SNOW_SIG_MAX],
            "equivalent_wave_speed": SNOW_WAVE_SPEED,
            "viscosity": 0.0,
            "deformation_reset": "F = U @ sig @ V^T (after plasticity clamp)",
            "cauchy_stress": "sigma = 2 mu (F - U V^T) F^T + I la J (J - 1)",
        }
    elif material == "multi":
        return {
            "source": "benchmarks/99_bench/mpm99.py",
            "material": "multi",
            "materials": [mpm99_properties("liquid"), mpm99_properties("jelly"), mpm99_properties("snow")],
            "equivalent_wave_speed": SNOW_WAVE_SPEED,
        }
    raise ValueError(f"unsupported material '{material}', expected one of {MATERIALS + ('multi',)}")


def create_mpm99_solver(material="liquid", **kwargs):
    """Factory that returns the appropriate solver subclass."""
    if material == "liquid":
        return MPM99LiquidSolver2D(**kwargs)
    elif material == "snow":
        return MPM99SnowSolver2D(**kwargs)
    elif material == "jelly":
        return MPM99JellySolver2D(**kwargs)
    elif material == "multi":
        return MPM99MultiMaterialSolver2D(**kwargs)
    raise ValueError(f"unsupported material '{material}', expected one of {MATERIALS + ('multi',)}")


# --- Solver classes ----------------------------------------------------------

@ti.data_oriented
class _MPM99BaseSolver2D(AdaptiveMPMSolver2D):
    """Common G2P velocity update shared by liquid and snow variants."""

    @ti.func
    def _g2p_velocity_update(self, p, level: ti.template(), t: float):
        x_p = self.particles.x[p]
        base, w0, w1, w2, _, _, _ = self._weights(level, x_p)
        v_new = ti.Vector.zero(ti.f64, 2)
        B_new = ti.Matrix.zero(ti.f64, 2, 2)
        for i, j in ti.static(ti.ndrange(3, 3)):
            I = base + ti.Vector([i, j])
            if self.grid.in_bounds(level, I):
                weight = (
                    self._weight_component(0, i, w0, w1, w2)
                    * self._weight_component(1, j, w0, w1, w2)
                )
                v_I = self.grid.v[level][I]
                dpos = self.grid.node_position(level, I) - x_p
                v_new += weight * v_I
                B_new += weight * v_I.outer_product(dpos)
        C_new = B_new * (4.0 * self.grid.level_inv_dx[level] * self.grid.level_inv_dx[level])
        velocity_limit = ti.cast(10.0 * config.MAX_WAVE_SPEED, ti.f64)
        speed = v_new.norm()
        if speed > velocity_limit:
            v_new *= velocity_limit / speed
        affine_limit = velocity_limit * self.grid.level_inv_dx[level]
        affine_norm = C_new.norm()
        if affine_norm > affine_limit:
            C_new *= affine_limit / affine_norm
        self.particles.v[p] = v_new
        self.particles.C[p] = C_new
        new_x = x_p + v_new * config.DT
        if ti.static(config.ACTIVE_SCENARIO == "IMMERSED"):
            new_x = self._project_from_platform(new_x, level, t)
        clearance = 0.1 * self.grid.level_dx[level]
        domain_min = self.grid.dynamic_domain_min[None]
        domain_max = self.grid.dynamic_domain_max[None]
        new_x[0] = ti.max(domain_min[0] + clearance, ti.min(new_x[0], domain_max[0] - clearance))
        new_x[1] = ti.max(domain_min[1] + clearance, ti.min(new_x[1], domain_max[1] - clearance))
        self.particles.x[p] = new_x
        identity = ti.Matrix.identity(ti.f64, 2)
        F_trial = (identity + C_new * config.DT) @ self.particles.F[p]
        U, singular, V = ti.svd(F_trial)
        for d in ti.static(range(2)):
            singular[d, d] = ti.max(singular[d, d], 1e-6)
        return U, singular, V, identity


@ti.data_oriented
class MPM99LiquidSolver2D(_MPM99BaseSolver2D):
    """Liquid (material 0): zero shear, isotropic volumetric stress."""

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        U, singular, V, identity = self._g2p_velocity_update(p, level, t)
        J = ti.cast(1.0, ti.f64)
        for d in ti.static(range(2)):
            J *= singular[d, d]
        F_new = identity * ti.sqrt(J)
        stress_value = ti.cast(LAMBDA_0, ti.f64) * J * (J - 1.0)
        self.particles.F[p] = F_new
        self.particles.stress[p] = identity * stress_value
        self.particles.pressure[p] = ti.max(-stress_value, 0.0)


@ti.data_oriented
class MPM99SnowSolver2D(_MPM99BaseSolver2D):
    """Snow (material 2): hardening + plasticity + full Cauchy stress."""

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        U, singular, V, identity = self._g2p_velocity_update(p, level, t)
        # Plasticity: clamp singular values and update Jp
        J = ti.cast(1.0, ti.f64)
        for d in ti.static(range(2)):
            new_sig = ti.min(ti.max(singular[d, d], SNOW_SIG_MIN), SNOW_SIG_MAX)
            self.particles.Jp[p] *= singular[d, d] / new_sig
            singular[d, d] = new_sig
            J *= new_sig
        F_new = U @ singular @ V.transpose()
        h = ti.exp(ti.cast(SNOW_HARDENING, ti.f64) * (1.0 - self.particles.Jp[p]))
        mu = ti.cast(MU_0, ti.f64) * h
        la = ti.cast(LAMBDA_0, ti.f64) * h
        stress = (
            2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose()
            + identity * la * J * (J - 1.0)
        )
        self.particles.F[p] = F_new
        self.particles.stress[p] = stress
        self.particles.pressure[p] = ti.max(-la * J * (J - 1.0), 0.0)


@ti.data_oriented
class MPM99JellySolver2D(_MPM99BaseSolver2D):
    """Jelly (material 1): constant softening, no F reset, no plasticity."""

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        U, singular, V, identity = self._g2p_velocity_update(p, level, t)
        J = ti.cast(1.0, ti.f64)
        for d in ti.static(range(2)):
            J *= singular[d, d]
        # Jelly: F evolves freely (no reset, no plasticity clamp)
        F_new = U @ singular @ V.transpose()
        h = ti.cast(JELLY_SOFTENING, ti.f64)
        mu = ti.cast(MU_0, ti.f64) * h
        la = ti.cast(LAMBDA_0, ti.f64) * h
        stress = (
            2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose()
            + identity * la * J * (J - 1.0)
        )
        self.particles.F[p] = F_new
        self.particles.stress[p] = stress
        self.particles.pressure[p] = ti.max(-la * J * (J - 1.0), 0.0)


@ti.data_oriented
class MPM99MultiMaterialSolver2D(_MPM99BaseSolver2D):
    """All three mpm99 materials in one solver, dispatched by particles.material[p].

    material == 0: liquid  (mu=0, F reset to sqrt(J)*I)
    material == 1: jelly   (h=0.3, F evolves freely)
    material == 2: snow    (hardening + plasticity)

    Includes hysteresis in gradient refinement: a particle already at level k
    only demotes when deform drops below half the split threshold, preventing
    the split/merge pulsing that would otherwise occur because dx changes
    with level.
    """

    @ti.kernel
    def compute_gradient_levels(self):
        """Gradient refinement with hysteresis.

        Without hysteresis, deform = |C| * dx is level-dependent (dx doubles
        at coarser levels), so a particle oscillates: coarse → wants split →
        fine → wants merge → coarse → ...  With hysteresis, a particle at
        level k keeps that level unless deform drops below half the threshold
        that would have promoted it.
        """
        grad_threshold = ti.cast(config.AMR_GRADIENT_REFINE_THRESHOLD, ti.f64)
        grad_max = ti.static(getattr(config, 'AMR_GRADIENT_MAX_LEVEL', self.grid.num_levels - 1))
        hysteresis = ti.cast(0.1, ti.f64)  # demote at 10% of promote threshold
        for p in range(self.particles.active_count[None]):
            lvl = self.particles.level[p]
            dx = self.grid.level_dx[lvl]
            C_norm = self.particles.C[p].norm()
            deform = C_norm * dx
            # Compute promote target (same as base)
            promote_target = 0
            for k in ti.static(range(self.grid.num_levels)):
                if k <= grad_max:
                    if deform > grad_threshold * (2.0 ** k):
                        promote_target = k
            # Hysteresis: if already above the promote target, only demote
            # when deform drops below half the threshold for current level
            target = promote_target
            if lvl > promote_target:
                demote_threshold = hysteresis * grad_threshold * (2.0 ** lvl)
                if deform >= demote_threshold:
                    target = lvl  # stay at current level
            self.particles.gradient_level[p] = target

    @ti.func
    def _g2p_level(self, p, level: ti.template(), t: float):
        U, singular, V, identity = self._g2p_velocity_update(p, level, t)
        mat = self.particles.material[p]
        if mat == 0:
            # Liquid: zero shear, F reset to sqrt(J)*I
            J = ti.cast(1.0, ti.f64)
            for d in ti.static(range(2)):
                J *= singular[d, d]
            F_new = identity * ti.sqrt(J)
            stress_value = ti.cast(LAMBDA_0, ti.f64) * J * (J - 1.0)
            self.particles.F[p] = F_new
            self.particles.stress[p] = identity * stress_value
            self.particles.pressure[p] = ti.max(-stress_value, 0.0)
        elif mat == 1:
            # Jelly: constant softening, F evolves freely
            J = ti.cast(1.0, ti.f64)
            for d in ti.static(range(2)):
                J *= singular[d, d]
            F_new = U @ singular @ V.transpose()
            h = ti.cast(JELLY_SOFTENING, ti.f64)
            mu = ti.cast(MU_0, ti.f64) * h
            la = ti.cast(LAMBDA_0, ti.f64) * h
            stress = (
                2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose()
                + identity * la * J * (J - 1.0)
            )
            self.particles.F[p] = F_new
            self.particles.stress[p] = stress
            self.particles.pressure[p] = ti.max(-la * J * (J - 1.0), 0.0)
        else:
            # Snow: hardening + plasticity
            J = ti.cast(1.0, ti.f64)
            for d in ti.static(range(2)):
                new_sig = ti.min(ti.max(singular[d, d], SNOW_SIG_MIN), SNOW_SIG_MAX)
                self.particles.Jp[p] *= singular[d, d] / new_sig
                singular[d, d] = new_sig
                J *= new_sig
            F_new = U @ singular @ V.transpose()
            h = ti.exp(ti.cast(SNOW_HARDENING, ti.f64) * (1.0 - self.particles.Jp[p]))
            mu = ti.cast(MU_0, ti.f64) * h
            la = ti.cast(LAMBDA_0, ti.f64) * h
            stress = (
                2.0 * mu * (F_new - U @ V.transpose()) @ F_new.transpose()
                + identity * la * J * (J - 1.0)
            )
            self.particles.F[p] = F_new
            self.particles.stress[p] = stress
            self.particles.pressure[p] = ti.max(-la * J * (J - 1.0), 0.0)
