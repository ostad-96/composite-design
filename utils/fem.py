# fem.py
# Requires FEniCS!
from dolfin import *
set_log_active(False)
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
    subdivisions_per_block = 8  # Number of subdivisions per design block
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
        i_block = int(x // 5)              # column index (each block is 5 mm)
        j_block = MATRIX_SIZE - 1 - int(y // 5)  # row index, flipped vertically
        materials[cell] = int(design[j_block, i_block])
    
    # Define function spaces.
    V = VectorFunctionSpace(mesh, "P", 1)
    Q = FunctionSpace(mesh, "DG", 0)
    
    # Create functions for material properties.
    E_func = Function(Q)
    nu_func = Function(Q)
    
    for cell in cells(mesh):
        if materials[cell] == 0:
            E_func.vector()[cell.index()] = E_STIFF
            nu_func.vector()[cell.index()] = NU_STIFF
        else:
            E_func.vector()[cell.index()] = E_COMP
            nu_func.vector()[cell.index()] = NU_COMP
    
    # Define plane stress parameters.
    mu_expr = E_func / (2.0 * (1.0 + nu_func))
    lmbda_expr = (E_func * nu_func) / (1.0 - nu_func**2)
    
    # Define strain and stress functions.
    def epsilon(u):
        return 0.5 * (grad(u) + grad(u).T)
    
    def sigma(u):
        return 2 * mu_expr * epsilon(u) + lmbda_expr * tr(epsilon(u)) * Identity(2)
    
    # Variational formulation.
    u = TrialFunction(V)
    v = TestFunction(V)
    a = inner(sigma(u), epsilon(v)) * dx
    L = dot(Constant((0, 0)), v) * dx
    
    # Define boundaries for loading.
    class BottomBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], 0.0) and on_boundary

    class TopBoundary(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[1], domain_side) and on_boundary

    class LeftCorner(SubDomain):
        def inside(self, x, on_boundary):
            return near(x[0], 0.0) and near(x[1], 0.0)
    
    boundaries = MeshFunction("size_t", mesh, mesh.topology().dim() - 1)
    boundaries.set_all(0)
    BottomBoundary().mark(boundaries, 1)
    TopBoundary().mark(boundaries, 2)
    
    ds = Measure("ds", domain=mesh, subdomain_data=boundaries)
    
    # Apply boundary conditions.
    bc_bottom = DirichletBC(V.sub(1), Constant(0.0), boundaries, 1)
    bc_top = DirichletBC(V.sub(1), Constant(-0.0025), boundaries, 2)
    bc_left = DirichletBC(V.sub(0), Constant(0.0), LeftCorner(), method="pointwise")
    
    # Solve the FEM problem.
    u_sol = Function(V)
    solve(a == L, u_sol, [bc_bottom, bc_top, bc_left])
    
    # Compute effective Young's modulus from the reaction on the top boundary.
    n = FacetNormal(mesh)
    traction = dot(sigma(u_sol), n)
    F_y = assemble(traction[1] * ds(2))
    
    thickness = 5.0  # mm
    area = domain_side * thickness
    stress_avg = (F_y * thickness) / area
    strain_applied = 1e-4
    E_eff = stress_avg / strain_applied
    
    return -E_eff

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
    return (E_STIFF * E_COMP) / (phi * E_COMP + (1 - phi) * E_STIFF)

if __name__ == "__main__":
    sample_design = np.array([
        [0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0],
        [0, 1, 1, 1, 0],
        [0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0]
    ], dtype=np.float32)
    E_eff = evaluate_composite(sample_design)
    # Compute stiff material fraction
    vol_frac_stiff = (MATRIX_SIZE * MATRIX_SIZE - np.sum(sample_design)) / (MATRIX_SIZE * MATRIX_SIZE)
    print(f"Effective Young's Modulus: {E_eff:.2f} MPa, Stiff Volume Fraction: {vol_frac_stiff:.2f}")
