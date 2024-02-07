# KNP-EMI-DG

Code for solving the KNP-EMI problem using a DG fem method.

### Geometry ###

The code assumes all ECS cells are tagget with 2 and ICS cells are tagged with 1, and further that all interior
facets are tagged with 0. The membrane potential is defined as phi_i - phi_e (i.e. phi_1 - phi_2). Since we have marked cell in
ECS with 2 and cells in ICS with 1 we have an interface normal pointing inwards. In general, normals will always point from higher to lower (e.g. from 2 -> 1)

### Dependencies code ###

*To setup environment, run:

    conda create --name <env> --file environment.txt

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
