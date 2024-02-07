import dolfin as df
import numpy as np

df.parameters['ghost_mode'] = 'shared_vertex'

# We have function in H(div) trace space and want to get/set its values
# on a subsect of mesh entities 

def is_dlt_scalar(V):
    '''Scalar HDiv Trace or go home'''
    assert V.ufl_element().family() == 'HDiv Trace'
    # For simplicity
    assert V.ufl_element().value_shape() == ()

    return True


def get_indices(V, facet_f, tags):
    '''Pick dof indices of u on tagged facets'''
    assert tags
    assert is_dlt_scalar(V)

    mesh = V.mesh()
    tdim = mesh.topology().dim()
    assert facet_f.dim() == tdim - 1
    
    dm = V.dofmap()
    first, last = dm.ownership_range()

    indices = np.array(dm.entity_dofs(mesh, tdim-1))
    # At this point we have dofs of all the facets
    ndofs_per_facet = len(dm.tabulate_entity_dofs(tdim-1, 0))
    indices = indices.reshape((-1, ndofs_per_facet))
    # Look at marked facets
    marked_facets = np.sort(np.unique(np.concatenate([np.where(facet_f.array() == tag)[0] for tag in tags])))
    assert len(marked_facets) <= len(indices), (len(marked_facets), len(indices))
    # Candidate dofs are those who are on marked
    indices = indices[marked_facets]
    # Select those that we own, keep the mask because we will use for facets
    # too
    owned = lambda d: first <= dm.local_to_global_index(d) < last
    owned_indices_mask = np.fromiter(map(owned, indices.ravel()), dtype=bool)
    owned_indices_mask = owned_indices_mask.reshape((-1, ndofs_per_facet))
    owned_indices = (indices.ravel()[owned_indices_mask.ravel()]).reshape((-1, ndofs_per_facet))
    # We keep the facets that owned the dofs
    owned_facets = marked_facets[np.where(owned_indices_mask[:, 0])]
    
    return owned_facets, owned_indices


def get_values(u, indices):
    '''Get array of dof values of u in indices'''
    # NOTE: this is not a view
    assert is_dlt_scalar(u.function_space())    
    return u.vector().get_local()[indices]


def set_values(u, indices, values):
    '''Set dof values of u at indices to values'''
    assert is_dlt_scalar(u.function_space())    
    assert len(indices) == len(values)
    all_values = 0*u.vector().get_local()
    all_values[indices] = values

    u.vector().set_local(all_values)
    df.as_backend_type(u.vector()).update_ghost_values()

    return u

# --------------------------------------------------------------------

if __name__ == '__main__':

    dim = 3
    
    if dim == 2:
        mesh = df.UnitSquareMesh(32, 32)

        chis = {
            1: 'near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))',
            2: 'near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))',
            3: 'near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))',                          
            4: 'near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS))'
        }

        tags = (1, 2, 3, 4)

    elif dim == 3:

        mesh = df.UnitCubeMesh(8, 8, 8)

        chis = {
            1: 'near(x[0], 0.25) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[2]) && (x[2] < 0.75+DOLFIN_EPS))',
            2: 'near(x[0], 0.75) && ((0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[2]) && (x[2] < 0.75+DOLFIN_EPS))',
            3: 'near(x[1], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[2]) && (x[2] < 0.75+DOLFIN_EPS))',
            4: 'near(x[1], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[2]) && (x[2] < 0.75+DOLFIN_EPS))',
            5: 'near(x[2], 0.25) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))',
            6: 'near(x[2], 0.75) && ((0.25-DOLFIN_EPS < x[0]) && (x[0] < 0.75+DOLFIN_EPS) && (0.25-DOLFIN_EPS < x[1]) && (x[1] < 0.75+DOLFIN_EPS))'}

        tags = (1, 2, 3, 4, 5, 6)
    else:
        raise ValueError

    # The good stuff
    facet_f = df.MeshFunction('size_t', mesh, mesh.topology().dim()-1, 0)
    for tag, chi in chis.items():
        df.CompiledSubDomain(chi).mark(facet_f, tag)

    V = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 1)
    # Make up a function that we want to modify
    r = df.SpatialCoordinate(mesh)
    x, y = r[0], r[1]

    u, v = df.TrialFunction(V), df.TestFunction(V)
    a = df.inner(u('+'), v('+'))*df.dS + df.inner(u, v)*df.ds
    Lx = df.inner(x, v('+'))*df.dS + df.inner(x, v)*df.ds
    Ly = df.inner(y, v('+'))*df.dS + df.inner(y, v)*df.ds    

    A, bx, by = map(df.assemble, (a, Lx, Ly))
    
    solver = df.PETScKrylovSolver('cg', 'hypre_amg')
    solver.parameters['relative_tolerance'] = 1E-40
    solver.parameters['absolute_tolerance'] = 1E-14
    solver.set_operators(A, A)

    x_h, y_h = df.Function(V), df.Function(V)
    solver.solve(x_h.vector(), bx)
    solver.solve(y_h.vector(), by)    

    # We want to make a function x + 2*y but only on marked facets    
    owned_facets, owned_indices = get_indices(V, facet_f, tags=tags)
    owned_indices = owned_indices.ravel()
    x_values = get_values(x_h, owned_indices)
    y_values = get_values(y_h, owned_indices)
    # Do it
    values = x_values + 2*y_values
    uh = df.Function(V)
    uh = set_values(uh, owned_indices, values)

    # For camparison we define the function everywhere
    L = df.inner(x+2*y, v('+'))*df.dS + df.inner(x+2*y, v)*df.ds
    b = df.assemble(L)
    
    true = df.Function(V)
    solver.solve(true.vector(), b)
    # ... but compute the error only on the regions where we cooked uh ...
    dGamma = df.Measure('dS', subdomain_data=facet_f)
    error = df.sqrt(df.assemble(sum((uh('+') - true('+'))**2*dGamma(tag) for tag in tags)))
    print(error)
    # ... everywhere else it should be 0
    error = df.sqrt(df.assemble(uh('+')**2*dGamma(0) + df.assemble(uh**2*df.ds)))
    print(error)
