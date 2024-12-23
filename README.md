# KNP-EMI-DG

### About

Code for solving the KNP--EMI problem using a DG fem method.

### Dependencies code

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

### Reproduce results paper

```bash

# run MMS test in space
python run_MMS_space.py

# run MMS test in space
python run_MMS_time.py

# run simulation on idealized 2D geometry
python run_2D.py

# run simulation on idealized 3D geometry
python run_3D.py

# run simulation on realistic 3D geometry of rat neuron
python run_rat_neuron.py

```

### Files

- [solver.py](./src/solver.py): class for PDE solver.

- [membrane.py](./src/membrane.py): class for membrane model (ODE stepping, functions for communication
  between PDE and ODE solver etc.).

- `mm\_\*.py`: spesification of membrane model (including all membrane parameters)

  - [mm_hh.py](./src/mm_hh.py): Hodkin Huxley model (with ODEs)
  - [mm_leak.py](./src/mm_leak.py): passive leak model (no ODEs)

- run\_\*.py: scripts for running various simulations. Contains PDE parameters
  (mesh, physical and numerical parameters)

  - [run_MMS_space.py](./src/run_MMS_space.py): MMS test in space
  - [run_MMS_time.py](./src/run_MMS_time.py): MMS test in time
  - [run_2D.py](./src/run_2D.py): simulation with Hodkin Huxley dynamics in idealized 3D axons
  - [run_3D.py](./src/run_3D.py): simulation with Hodkin Huxley dynamics in idealized 3D axons
  - [run_rat_neuron.py](./src/run_rat_neuron.py): simulation with spatially varying membrane mechanisms in realistic 3D geometry

- `make*mesh*\*.py`: scripts for generating idealized 2D and 3D meshes

- `make*mesh*\*.py`: scripts for generating figures

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
