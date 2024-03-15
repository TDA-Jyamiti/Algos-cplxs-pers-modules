# Efficient Algorithms for Complexes of Persistence Modules with Applications

This repository contains an implementation of the algorithms proposed in the paper (link).

## Organization of the repository

The repository is organized as follows:

`compute_pres.py` contains an implentation of the algorithm PresPersMod as described in section 4 of the paper.

`compute_hom_pres.py` contains an implementation of the algorithm PresHom as described in section 3 of the paper.

`main.py` contains an example for a persistent sheaf over a simplicial complex (Example 24 in the paper).

`main_graph.py` contains an example which has a persistent sheaf over a graph and computes the persistent sheaf cohomology. It also contains a way to generate filtration matrices.

## Dependencies

1. numpy
   
2. scipy

The code has been tested on `numpy 1.24.4` and `scipy 1.8.0`.

## License

THIS SOFTWARE IS PROVIDED "AS-IS". THERE IS NO WARRANTY OF ANY KIND. NEITHER THE AUTHORS NOR PURDUE UNIVERSITY WILL BE LIABLE FOR ANY DAMAGES OF ANY KIND, EVEN IF ADVISED OF SUCH POSSIBILITY.

This software was developed (and is copyrighted by) the CGTDA research group at Purdue University. Please do not redistribute this software. This program is for academic research use only.

## Citation

The paper is accepted at SoCG 2024.



