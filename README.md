# KNP-EMI-DG

Code for solving the KNP-EMI problem using a DG fem method.
We here approximate the following system:

- d(c_k)/dt + div(J_k) = 0, (emi)
- F sum_k z^k div(J_k) = 0, (knp)
where

-J_k(c_k, phi) = - D grad(c_k) - z_k D_k psi c_k grad(phi)

We solve the system iteratively, by decoupling the first and second
equation, yielding the following system: Given c_a_ and c_b_,
iterate over the two following steps:

    step I:  (emi) find phi by solving (2), with J^a(c_a_, phi) and J^b(c_b_, phi)
    step II: (knp) find c_a and c_b by solving (1) with J^k(c_a, c_b, phi_), where phi_
                is the solution from step I
   (step III: solve ODEs at interface, and update membrane potential)

 Membrane potential is defined as phi_i - phi_e, since we have
 marked cell in ECS with 2 and cells in ICS with 1 we have an
 interface normal pointing inwards
    ____________________
   |                    |
   |      ________      |
   |     |        |     |
   | ECS |   ICS  |     |
   |  2  |->  1   |     |
   | (+) |   (-)  |     |
   |     |________|     |
   |                    |
   |____________________|

Normal will always point from higher to lower (e.g. from 2 -> 1)

NB! The code assumes all membrane facets are tagged with 1, and that all interior
facets are tagged with 0.


### Dependencies P1 code ###

Get the environment needed (all dependencies etc.), build and
and run the Docker container *ceciledc/fenics_mixed_dimensional:13-03-20* by:

* Installing docker: https://docs.docker.com/engine/installation/
* Build and start docker container with:

        docker run -t -v $(pwd):/home/fenics -i ceciledc/fenics_mixed_dimensional:13-03-20
        pip3 install vtk
        cd ulfy-master
        python3 setup.py install

### License ###

The software is free: you can redistribute it and/or modify it under the terms
of the GNU Lesser General Public License as published by the Free Software
Foundation, either version 3 of the License, or (at your option) any later
version.

### Community ###

Contact ada@simula.no for questions or to report issues with the software.
