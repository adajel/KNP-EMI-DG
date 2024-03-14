# KNP-EMI-DG

### About ###
Code for solving the KNP-EMI problem using a DG fem method.

- solver.py: contains class for PDE solver. Contains the following files:
- membrane.py: containt class for solving ODEs at membrane and communication beteween PDE and ODE solver (e.g. membrane potential, src terms, etc.)
- run_*.py files: scripts for running various simulations
    - run_mms_space.py: run MMS test in space
    - run_2D.py: run simulation with HH in idealized 3D axons
    - run_3D.py: run simulation with HH in idealized 3D axons
- mm_*.py files: spesification of membrane model (including all membrane parameters)
    - mm_HH.py: Hodkin Huxley model
    - mm_leak.py: leak model

### Geometry ###

The code assumes all ECS cells are tagged with 2 and ICS cells are tagged with
1, and that all interior facets are tagged with 0. The membrane
potential is defined as phi_i - phi_e (i.e. phi_1 - phi_2). Since we have
marked cell in ECS with 2 and cells in ICS with 1 we have an interface
normal pointing inwards. In general, normals will always point from higher to
lower (e.g. from 2 -> 1)

### Dependencies code ###

To setup environment, run:

    conda create --name <env> --file environment.txt

### Usage ###

```python

# run MMS test in space
python run_MMS_space.py

# run MMS test in space
python run_MMS_tim.py

# run simulation on idealized 2D geometry
python run_2D.py

# run simulation on idealized 3D geometry
python run_3D.py
```

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
