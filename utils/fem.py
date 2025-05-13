# utils/fem.py
# Requires FEniCS!
# from dolfin import *
# set_log_active(False)
import numpy as np
from config import MATRIX_SIZE, E_STIFF, NU_STIFF, E_COMP, NU_COMP

def evaluate_composite(design):
    """
    Evaluate the effective Young's modulus of a composite design.
    Parameters:
    design : MATRIX_SIZE×MATRIX_SIZE numpy array with 0 (stiff) or 1 (compliant)
    Returns:
    Effective Young's modulus (MPa)
    """
    subdivisions_per_block = 8 # Number of subdivisions per design block
    total_blocks_x = MATRIX_SIZE
    total_blocks_y = MATRIX_SIZE
    # Create refined mesh: total blocks * subdivisions per block per side.
    mesh = UnitSquareMesh(total_blocks_x * subdivisions_per_block,
                          total_blocks_y * subdivisions_per_block)
    # Assume each block is 5 mm, so the total domain side is:
    domain_side = MATRIX_SIZE * 5.0
    mesh.coordinates()[:] *= domain_side

    # Assign materials to cells based on the design matrix.
    materials = MeshFunction("size_t", mesh, mesh.topology().dim())
    materials.set_all(0)
    for cell in cells(mesh):
        x, y = cell.midpoint().x(), cell.midpoint().y()
        # Map the (x, y) coordinate to the corresponding block in the design.
        i_block = int(x // 5) # column index (each block is 5 mm)
        j_block = MATRIX_SIZE - 1 - int(y // 5) # row index, flipped vertically
        # Ensure indices are within bounds (can happen with floating point precision)
        i_block = min(max(i_block, 0), MATRIX_SIZE - 1)
        j_block = min(max(j_block, 0), MATRIX_SIZE - 1)
        materials[cell] = int(design[j_block, i_block])

    # Define function spaces.
    V = VectorFunctionSpace(mesh, "P", 1)
    Q = FunctionSpace(mesh, "DG", 0)

    # Create functions for material properties.
    E_func = Function(Q)
    nu_func = Function(Q)

    # Optimized material property assignment (avoid iterating cells twice)
    cell_indices = np.array(list(range(mesh.num_cells())))
    material_values = materials.array() # Get material assignments for all cells

    E_values = np.where(material_values == 0, E_STIFF, E_COMP)
    nu_values = np.where(material_values == 0, NU_STIFF, NU_COMP)

    E_func.vector()[:] = E_values[cell_indices] # Assign based on cell index
    nu_func.vector()[:] = nu_values[cell_indices]

    # Define plane stress parameters.
    mu_expr = E_func / (2.0 * (1.0 + nu_func))
    lmbda_expr = (E_func * nu_func) / (1.0 - nu_func**2) # Plane stress lambda

    # Define strain and stress functions.
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)

    def sigma(u):
        # Plain stress constitutive relation
        return 2.0 * mu_expr * epsilon(u) + lmbda_expr * tr(epsilon(u)) * Identity(2)

    # Variational formulation.
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(Constant((0, 0)), v) * dx # No body force

    # Define boundaries for loading.
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            # Use DOLFIN_EPS for robust floating point comparison
            return on_boundary and near(x[1], 0.0, DOLFIN_EPS)

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return on_boundary and near(x[1], domain_side, DOLFIN_EPS)

    class LeftCorner(SubDomain):
        def inside(self, x, on_boundary):
            # Pin a single point to remove rigid body motion in x
            return near(x[0], 0.0, DOLFIN_EPS) and near(x[1], 0.0, DOLFIN_EPS)

    # Create mesh functions for boundaries
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0) # Default marker
    BottomBoundary().mark(boundaries, 1)
    TopBoundary().mark(boundaries, 2)
    # No marker needed for LeftCorner as it's applied pointwise

    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)

    # Apply boundary conditions.
    # Fix bottom edge in y-direction
    bc_bottom = DirichletBC(V.sub(1), Constant(0.0), boundaries, 1)
    # Apply vertical displacement on top edge
    applied_displacement = -0.001 * domain_side # Example: 0.1% strain
    bc_top = DirichletBC(V.sub(1), Constant(applied_displacement), boundaries, 2)
    # Fix left bottom corner in x-direction to prevent rigid body motion
    bc_left_corner = DirichletBC(V.sub(0), Constant(0.0), LeftCorner(), method="pointwise")

    bcs = [bc_bottom, bc_top, bc_left_corner]

    # Solve the FEM problem.
    u_sol = Function(V)
    try:
        solve(a == L, u_sol, bcs)
    except RuntimeError as e:
        print(f"FEM solver failed: {e}. Returning NaN.")
        # Check for issues like singular matrix (e.g., fully compliant material)
        return np.nan # Or some other indicator of failure

    # Compute effective Young's modulus from the reaction force on the top boundary.
    n = FacetNormal(mesh)
    traction = dot(sigma(u_sol), n) # Traction vector t = sigma * n
    # Reaction force component in y-direction on the top boundary (ds(2))
    # Note: Reaction force is typically opposite to traction under applied displacement
    F_y_reaction = -assemble(traction[1] * ds(2)) # Integral of ty over top surface

    thickness = 5.0 # Assume 5mm thickness (consistent with block size interpretation)
    area = domain_side * thickness # Cross-sectional area
    # Average stress = Force / Area
    stress_avg = F_y_reaction / area
    # Applied strain = Displacement / Original Length
    strain_applied = abs(applied_displacement / domain_side)

    if abs(strain_applied) < DOLFIN_EPS: # Avoid division by zero
        print("Warning: Applied strain is near zero. Effective modulus cannot be calculated.")
        return np.nan

    # Effective Young's Modulus E = Stress / Strain
    E_eff = stress_avg / strain_applied

    # Return positive value for modulus
    return abs(E_eff)


# --- Analytical Bounds (Unchanged) ---
def voigt_model(phi):
    """
    Voigt Model (Upper Bound):
    E_Voigt = phi * E_STIFF + (1 - phi) * E_COMP
    Parameters:
    phi : float
        Volume fraction of the stiff material.
    Returns:
    Upper bound for the effective modulus (MPa)
    """
    return phi * E_STIFF + (1 - phi) * E_COMP

def reuss_model(phi):
    """
    Reuss Model (Lower Bound):
    E_Reuss = (E_STIFF * E_COMP) / (phi * E_COMP + (1 - phi) * E_STIFF)
    Parameters:
    phi : float
        Volume fraction of the stiff material.
    Returns:
    Lower bound for the effective modulus (MPa)
    """
    # Avoid division by zero if denominator is somehow zero
    denominator = (phi * E_COMP + (1 - phi) * E_STIFF)
    if abs(denominator) < 1e-9: # Use a small epsilon
        print("Warning: Reuss model denominator is near zero.")
        # Decide on return value: Maybe Voigt, maybe NaN, maybe very large number
        # Returning Voigt might be misleading. NaN is safer.
        return np.nan
    return (E_STIFF * E_COMP) / denominator


if __name__ == "__main__":
    print(f"Running FEM script with MATRIX_SIZE={MATRIX_SIZE}")
    # sample_design = np.random.randint(0, 2, size=(MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)
    sample_design = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)

    if sample_design.shape != (MATRIX_SIZE, MATRIX_SIZE):
        print(f"Warning: Sample design shape {sample_design.shape} does not match MATRIX_SIZE ({MATRIX_SIZE},{MATRIX_SIZE}). Using random design.")
        sample_design = np.random.randint(0, 2, size=(MATRIX_SIZE, MATRIX_SIZE), dtype=np.float32)

    print("Evaluating sample design:")
    print(sample_design)

    # --- Using Voigt/Reuss average as placeholder ---
    vol_frac_stiff_sample = (MATRIX_SIZE * MATRIX_SIZE - np.sum(sample_design)) / (MATRIX_SIZE * MATRIX_SIZE)
    E_approx = (voigt_model(vol_frac_stiff_sample) + reuss_model(vol_frac_stiff_sample)) / 2
    print(f"\n--- Using Voigt/Reuss average (Placeholder) ---")
    print(f"Approx Effective Modulus: {E_approx:.2f} MPa")
    print(f"Stiff Volume Fraction: {vol_frac_stiff_sample:.2f}")

    # --- Uncomment to run actual FEM (requires FEniCS) ---
    # print("\n--- Running FEniCS Evaluation ---")
    # try:
    #     E_eff_fem = evaluate_composite(sample_design)
    #     if not np.isnan(E_eff_fem):
    #         print(f"FEM Effective Young's Modulus: {E_eff_fem:.2f} MPa")
    #     else:
    #         print("FEM evaluation failed.")
    # except Exception as e:
    #     print(f"An error occurred during FEniCS evaluation: {e}")
    #     import traceback
    #     traceback.print_exc()
    # --- End FEniCS section ---