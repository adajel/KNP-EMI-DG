# KNP-EMI-DG

### About

Code for solving the KNP-EMI problem using a DG fem method. The numerical
method is described in
[Ellingsrud et al. 2025](https://doi.org/10.1137/24M1653367 "Ellingsrud, Benedusi, and Kuchta. A splitting, discontinuous Galerkin solver for the cell-by-cell electroneutral Nernst–Planck framework.SIAM Journal on Scientific Computing 47.2 (2025): B477-B504.")

### Dependencies

All dependencies are listed in [environment.yml](./environment.yml).
To create an environment with these dependencies use either `conda` or `mamba` and run

```bash
conda env create -f environment.yml
```

or

```bash
mamba env create -f environment.yml
```

Then call

```bash
conda activate KNP-EMI
```

### Installation

To install the package run
```bash
python3 -m pip install -e .
```

### Idealized geometries
The directory [examples/idealized-geometries](https://github.com/adajel/KNP-EMI-DG/tree/main/examples/idealized-geometries)
contains code for running 2D and 3D simulations on idealized geometries 
representing neurons surrounded by ECS with Hodgkin-Huxley membrane dynamics.

### EMIx simulation
The directory
[examples/emix-simulations](https://github.com/adajel/KNP-EMI-DG/tree/main/examples/emix-simulations)
contains an example where the KNP-EMI DG code is used to run an
electrodiffusive simulation on a realistic 3D geometry representing
brain tissue generated via the
[emimesh](https://github.com/scientificcomputing/emimesh/tree/main) pipeline.

The initial conditions for the coupled KNP-EMI system are calibrated by solving
an extended system of ODEs to ensure an initial steady state - see
mm_calibrated.py and run_calibration.py for 
[mm_calibration.py](https://github.com/adajel/KNP-EMI-DG/blob/main/examples/emix-simulations/mm_calibration.py), 
[run_calibration.py](https://github.com/adajel/KNP-EMI-DG/blob/main/examples/emix-simulations/run_calibration.py), 
for further details.

### Selected files

- [solver.py](https://github.com/adajel/KNP-EMI-DG/tree/main/src/knpemidg/solver.py): class for PDE solver.
- [membrane.py](https://github.com/adajel/KNP-EMI-DG/tree/main/src/knpemidg/membrane.py): class for membrane model (ODE stepping, functions for communication
  between PDE and ODE solver etc.).
- [mm_hh.py](https://github.com/adajel/KNP-EMI-DG/tree/main/examples/idealized-geometries/mm_hh.py): Hodkin Huxley model (with ODEs)
- run\_\*.py: scripts for running various simulations. Contains PDE parameters
  (mesh, physical and numerical parameters)
 `make*mesh\_\*\*.py`: scripts for generating idealized 2D and 3D meshes

### Geometry

The code assumes ECS cells are tagged with 0 and ICS cells are tagged with
1,2,3, ... and that all interior facets are tagged with 0. The membrane
potential is defined as phi_i - phi_e (i.e. phi_1 - phi_2). Since we have
marked cell in ECS with 0 and cells in ICS with 1 we have an interface
normal pointing inwards. In general, normals will always point from lower to
higher (e.g. from 0 -> 1)

### License

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community

Contact ada@simula.no for questions or to report issues with the software.
