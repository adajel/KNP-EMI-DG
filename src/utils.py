import dolfin as df
import numpy as np
import sys

code = """
#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
namespace py = pybind11;

#include <dolfin/function/Expression.h>
#include <dolfin/mesh/MeshFunction.h>
#include <dolfin/mesh/Cell.h>

class PCWS : public dolfin::Expression
{
public:

  PCWS() : dolfin::Expression() {}

  void eval(Eigen::Ref<Eigen::VectorXd> values, Eigen::Ref<const Eigen::VectorXd> x, const ufc::cell& c) const override
  {
    const uint cell_index = c.index;

    const std::size_t value = (*subdomains)[c.index];
    values[0] = value;
  }

  std::shared_ptr<dolfin::MeshFunction<std::size_t>> subdomains;

};

PYBIND11_MODULE(SIGNATURE, m)
{
  py::class_<PCWS, std::shared_ptr<PCWS>, dolfin::Expression>
    (m, "PCWS")
    .def(py::init<>())
    .def_readwrite("subdomains", &PCWS::subdomains);
}
"""

df.parameters['ghost_mode'] = 'shared_facet'
aux = df.compile_cpp_code(code)

def subdomain_marking_foo(subdomains, V=None, aux=aux):
    '''Function in P0 space with cell values given by tags of the cell'''
    mesh = subdomains.mesh()

    assert mesh.topology().dim() == subdomains.dim()
    # As Expression
    f = df.CompiledExpression(aux.PCWS(), subdomains=subdomains, degree=0)

    if V is not None:
        assert V.ufl_element().value_size() == 1
        assert V.ufl_element().family() == 'Discontinuous Lagrange'

        return df.interpolate(f, V)

    V = df.FunctionSpace(mesh, 'DG', 0)
    return df.interpolate(f, V)

def interface_normal(subdomains, mesh):

    '''
    Computes a [DLT]^d function representing the facet normal vector
    such that on the interface between the subdomains it points from the
    higher (tag) value to a lower value.
    '''

    # Represent cell tags as P0 function so that we can query
    chi = subdomain_marking_foo(subdomains)

    # Normal computation
    V = df.VectorFunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)
    v = df.TestFunction(V)

    n, hA = df.FacetNormal(mesh), df.FacetArea(mesh)
    n_ = df.Function(V)  # Our oriented normal
    # FEniCS will orient the normal form '+' to '-'. This is what we want
    # if the plus side is greater than minus side
    switch = df.conditional(df.ge(chi('+'), chi('-')), n('-'), n('+'))
    # Here we do L^2 projection to get the normal
    df.assemble((1/df.avg(hA))*df.inner(switch, v('+'))*df.dS + (1/hA)*df.inner(n, v)*df.ds,
                n_.vector())

    return n_

def plus(phi, normal):
    '''Restriction of phi to the cell from which the normal originates'''
    n = df.FacetNormal(normal.function_space().mesh())    
    switch = df.conditional(df.ge(df.dot(normal('+'), n('+')), df.Constant(0)), phi('+'), phi('-'))
    return switch


def minus(phi, normal):
    '''Restriction of phi to the cell at which the normal ends'''
    n = df.FacetNormal(normal.function_space().mesh())
    switch = df.conditional(df.ge(df.dot(normal('+'), n('+')), df.Constant(0)), phi('-'), phi('+'))    
    return switch

def pcws_constant_project(f, V, fV=None):
    '''Project f onto V where V is some piecewise-constant space'''
    assert V.ufl_element().degree() == 0

    v = df.TestFunction(V)
    assert v.ufl_shape == f.ufl_shape

    mesh = V.mesh()
    hV, hA = df.CellVolume(mesh), df.FacetArea(mesh)
    # Might be useful to reuse the function for projecting
    if fV is None:
        fV = df.Function(V)
    x = fV.vector()
    # Normally we would assemble linear system (A, b) and solve for x being
    # the coefficient vector of fV. Here we want to directly assemble the action
    # of inv(A) onto b. This is possible since A is diagonal
    projection_forms = {
        'HDiv Trace': lambda f: (1/df.avg(hA))*df.inner(f, v('+'))*df.dS
    }

    form = projection_forms[V.ufl_element().family()](f)
    # Assemble action into x
    df.assemble(form, x)

    return fV

def CellCenterDistance(mesh):
    '''Discontinuous Lagrange Trace function that holds the cell-to-cell distance'''
    # Cell-cell distance for the interior facet is defined as a distance 
    # of midpoints of the cells that share the facet. For exterior facet
    # we take the distance of cell midpoint and the facet midpoint
    Q = df.FunctionSpace(mesh, 'DG', 0)
    V = df.FunctionSpace(mesh, 'CG', 1)
    L = df.FunctionSpace(mesh, 'Discontinuous Lagrange Trace', 0)

    gdim = mesh.geometry().dim()

    cK, fK = df.CellVolume(mesh), df.FacetArea(mesh)
    q, l = df.TestFunction(Q), df.TestFunction(L)
    # The idea here to first assemble component by component the cell 
    # and (exterior) facet midpoint
    cell_centers, facet_centers = [], []
    for xi in df.SpatialCoordinate(mesh):
        qi = df.Function(Q)
        # Pretty much use the definition that a midpoint is int_{cell} x_i/vol(cell)
        # It's thanks to eval in q that we take values :)
        df.assemble((1/cK)*df.inner(xi, q)*df.dx, tensor=qi.vector())
        cell_centers.append(qi)
        # Same here but now our mean is over an edge
        li = df.Function(L)
        df.assemble((1/fK)*df.inner(xi, l)*df.ds, tensor=li.vector())
        facet_centers.append(li)
    # We build components to vectors
    cell_centers, facet_centers = map(df.as_vector, (cell_centers, facet_centers))

    distances = df.Function(L)
    # FIXME: This might not be necessary but it's better to be certain
    dS_, ds_ = df.dS(metadata={'quadrature_degree': 0}), df.ds(metadata={'quadrature_degree': 0})
    # Finally we assemble magniture of the vector that is determined by the
    # two centers
    df.assemble(((1/fK('+'))*df.inner(df.sqrt(df.dot(df.jump(cell_centers), df.jump(cell_centers))), l('+'))*dS_+
              (1/fK)*df.inner(df.sqrt(df.dot(cell_centers-facet_centers, cell_centers-facet_centers)), l)*ds_),
             distances.vector())

    return distances
