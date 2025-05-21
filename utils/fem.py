"""
utils/fem.py
------------
Homogenisation of a 2-phase 2-D composite with FEniCS.

* 0 = stiff phase  (E_STIFF , NU_STIFF)
* 1 = compliant    (E_COMP  , NU_COMP)

`evaluate_composite(design)` returns the effective Young’s
modulus (plane-stress) in MPa.

Author: 2025-05-14  (rewrite)
"""

from dolfin import *
import numpy as np
from config import MATRIX_SIZE, E_STIFF, NU_STIFF , E_COMP, NU_COMP

# -------------------------------------------------------------------------
# Global parameters – edit here first
# -------------------------------------------------------------------------

SUBDIVISIONS        = 1           # mesh cells per design pixel, per edge
BLOCK_SIZE          = 5.0         # mm – physical edge length of one pixel
PLANE_STRAIN        = False       # True → plane-strain, False → plane-stress
DISPLACEMENT_RATIO  = 1e-3        # applied ε = DISPLACEMENT_RATIO
ELEMENT_ORDER       = 1           # 1 = P1, 2 = P2

# Silence FEniCS screen chatter
set_log_level(LogLevel.WARNING)

# -------------------------------------------------------------------------
# FEM core
# -------------------------------------------------------------------------

def evaluate_composite(design: np.ndarray) -> float:
    """
    Compute the effective Young’s modulus of a 0/1 design.

    Parameters
    ----------
    design : ndarray, shape (MATRIX_SIZE, MATRIX_SIZE), values {0,1}
        0 = stiff pixel, 1 = compliant pixel

    Returns
    -------
    float
        Effective modulus in MPa (abs value); np.nan if the solve fails.
    """
    if design.shape != (MATRIX_SIZE, MATRIX_SIZE):
        raise ValueError(f"design must be {MATRIX_SIZE}×{MATRIX_SIZE}")

    # --- build mesh -------------------------------------------------------
    nx = ny = MATRIX_SIZE * SUBDIVISIONS
    mesh = UnitSquareMesh(nx, ny)
    domain_side = MATRIX_SIZE * BLOCK_SIZE
    mesh.coordinates()[:] *= domain_side

    # --- tag cells --------------------------------------------------------
    material_tags = MeshFunction("size_t", mesh, mesh.topology().dim())
    material_tags.set_all(0)

    # fast numpy-based mapping from cell midpoints to pixel indices
    coords = mesh.coordinates()
    for cell in cells(mesh):
        x, y = cell.midpoint().x(), cell.midpoint().y()
        i = min(int(x // BLOCK_SIZE), MATRIX_SIZE - 1)               # col
        j = MATRIX_SIZE - 1 - min(int(y // BLOCK_SIZE), MATRIX_SIZE - 1)  # row
        material_tags[cell] = int(design[j, i])                      # 0/1

    # --- material fields (DG-0) ------------------------------------------
    Q  = FunctionSpace(mesh, "DG", 0)
    E  = Function(Q)
    nu = Function(Q)

    # vectorised assignment (avoids Python loop)
    tags = material_tags.array()
    E_values  = np.where(tags == 0, E_STIFF,  E_COMP)
    nu_values = np.where(tags == 0, NU_STIFF, NU_COMP)
    E.vector().set_local(E_values)
    nu.vector().set_local(nu_values)

    # Lamé parameters for plane-stress / plane-strain
    mu     = E / (2.0 * (1.0 + nu))
    if PLANE_STRAIN:
        lmbda = (E * nu) / ((1 + nu) * (1 - 2 * nu))
    else:
        lmbda = (E * nu) / (1.0 - nu**2)

    # --- variational problem --------------------------------------------
    V = VectorFunctionSpace(mesh, "P", ELEMENT_ORDER)

    def eps(u):
        return sym(grad(u))

    def sigma(u):
        return 2 * mu * eps(u) + lmbda * tr(eps(u)) * Identity(2)

    u, v = TrialFunction(V), TestFunction(V)
    a = inner(sigma(u), eps(v)) * dx
    L = dot(Constant((0, 0)), v) * dx

    # --- boundaries & BCs -----------------------------------------------
    class Bottom(SubDomain):
        def inside(_, x, on_b): return on_b and near(x[1], 0.0, DOLFIN_EPS)

    class Top(SubDomain):
        def inside(_, x, on_b): return on_b and near(x[1], domain_side, DOLFIN_EPS)

    class LeftCorner(SubDomain):
        def inside(_, x, on_b):
            return near(x[0], 0.0, DOLFIN_EPS) and near(x[1], 0.0, DOLFIN_EPS)

    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    Bottom().mark(boundaries, 1)
    Top().mark(boundaries, 2)

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    δ = DISPLACEMENT_RATIO * domain_side   # prescribed vertical displacement
    bcs = [
        DirichletBC(V.sub(1), Constant(0.0), boundaries, 1),          # Uy = 0 bottom
        DirichletBC(V.sub(1), Constant(-δ), boundaries, 2),           # Uy = −δ top
        DirichletBC(V.sub(0), Constant(0.0), LeftCorner(), "pointwise")  # Ux pin
    ]

    # --- solve -----------------------------------------------------------
    u_sol = Function(V)
    try:
        solve(a == L, u_sol, bcs, solver_parameters=dict(linear_solver="mumps"))
    except RuntimeError:
        return np.nan

    # --- reaction force on top ------------------------------------------
    n = FacetNormal(mesh)
    traction = dot(sigma(u_sol), n)
    F_y = -assemble(traction[1] * ds(2))        # sign: reaction
    A   = domain_side * 1.0                     # unit thickness (mm)
    stress = F_y / A
    strain = abs(δ) / domain_side

    return abs(stress / strain)


# -------------------------------------------------------------------------
# Simple analytical bounds
# -------------------------------------------------------------------------

def voigt_model(phi: float) -> float:
    """Voigt upper bound."""
    return phi * E_STIFF + (1 - phi) * E_COMP

def reuss_model(phi: float) -> float:
    """Reuss lower bound."""
    denom = phi * E_COMP + (1 - phi) * E_STIFF
    return np.nan if abs(denom) < 1e-12 else (E_STIFF * E_COMP) / denom


# -------------------------------------------------------------------------
# Quick demo
# -------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Running FEM demo (MATRIX_SIZE = {MATRIX_SIZE})\n")

    # --- sanity: all-stiff, all-compliant --------------------------------
    all_stiff = np.zeros((MATRIX_SIZE, MATRIX_SIZE))
    all_compl = np.ones((MATRIX_SIZE, MATRIX_SIZE))

    for label, design in [("all-stiff", all_stiff), ("all-compliant", all_compl)]:
        E_eff = evaluate_composite(design)
        print(f"{label:<14s}  →  {E_eff:8.2f} MPa")

    # --- random design ---------------------------------------------------
    rnd = np.random.randint(0, 2, (MATRIX_SIZE, MATRIX_SIZE))
    phi = (rnd == 0).mean()                     # stiff volume fraction
    E_eff = evaluate_composite(rnd)
    E_voigt = voigt_model(phi)
    E_reuss = reuss_model(phi)

    print("\nRandom design:")
    print(rnd)
    print(f"\nφ_stiff = {phi:.2f}")
    print(f"FEM      = {E_eff:8.2f} MPa")
    print(f"Reuss ⩽  E ⩽ Voigt : {E_reuss:8.2f} … {E_voigt:8.2f} MPa")
