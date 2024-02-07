# KNP-EMI-DG

Code for solving the KNP-EMI problem using a DG fem method.

### Geometry ###

The code assumes all membrane facets are tagged with 1, that all interior
facets are tagged with 0, and that the ECS is tagged with 2 and ICS cells are tagged with 1. 

The membrane potential is defined as phi_i - phi_e. since we have marked cell in
ECS with 2 and cells in ICS with 1 we have an interface normal pointing inwards.

Normal will always point from higher to lower (e.g. from 2 -> 1)

### Dependencies code ###


### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
