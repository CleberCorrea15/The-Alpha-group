# Alpha Group Simulations

This directory contains Python scripts implementing simulations related to the Alpha Group framework, as described in the series of articles on the Alpha Group, including *The Alpha Group Dynamic Mapping*.

## Overview

The Alpha Group is a novel algebraic and topological framework proposed to describe the emergence of the \( S^4 \) topology and the coherent structure of space. The simulations here explore the dynamical behavior of matrices \( M(\theta) \) from the Alpha Group, analyzing how topological features emerge through iterative dynamics.

## Contents

- `Alpha_group_ODE_solution_0_radians.py`
- `phase_lyapunov_solution_0_radians.py`  
  Simulation script for the dynamic mapping of the Alpha Group matrix \( M(0 radians) \).

- `Alpha_group_ODE_solution_1.57_radians.py`
- `phase_lyapunov_solution_1.57_radians.py`
  
Simulation script for the dynamic mapping of the Alpha Group matrix \( M(1.57 radians) \).


## How to Run

1. Ensure you have the required Python packages installed:  
   ```bash
   pip install numpy matplotlib ripser persim scipy
   ```

2. Run the main simulation script:  
   ```bash
   python Alpha_group_ODE_solution_1.57_radians.py

3. Alpha Group – ψ Orbit Near π/2

This script simulates and visualizes the **vibrational trajectory** of the quantum-like state vector \( \psi \) governed by the Alpha Group dynamic mapping matrix \( M(\theta) \).

The simulation focuses on the **topological transition region** near \( \theta \approx \pi/2 \), where coherence drops and the Lyapunov exponent reaches its peak.

## Features
- Generates the **complex state evolution** using a 4×4 Alpha Group matrix.
- Normalizes the state at each step to preserve unit norm.
- Plots the **imaginary part** of \(\psi_1, \psi_2, \psi_3\) as a **3D trajectory**.
- Interactive slider to explore values of \( \theta \) close to \( \pi/2 \).

## Usage
```bash
!pip install numpy matplotlib ipywidgets

   ```

3. Use the visualization and analysis scripts to explore the generated data.

## References

Corrêa, C. S., & De Melo, T. B. (2025). The Alpha Group Dynamic Mapping. ArXiv. https://arxiv.org/abs/2507.18303.

Corrêa, C. S., De Melo, T. B., & Custódio, D. M. (2025). The Alpha Group Tensorial Metric. ArXiv: arXiv:2507.16954 [math.DG] https://doi.org/10.47976/RBHM2024v24n4851-57

CORRÊA, C. S.; DE MELO, T. B.; CUSTÓDIO, D. M. Proposing the Alpha Group. International Journal for Research in Engineering Application & Management (IJREAM), v. 8, n. 5, p. 66-71, 2022. DOI: https://doi.org/10.35291/22454-9150.2022.0421.

CORRÊA, Cleber Souza; DE MELO, Thiago Braido Nogueira. Multiscale topology and dynamic internal vectorial geometry-alpha group. Studies in Engineering and Exact Sciences, v. 6, n. 1, p. e17403-e17403, 2025. DOI: https://doi.org/10.54021/seesv6n1-042.

CORRÊA, Cleber Souza; DE MELO, Thiago Braido Nogueira. Division as a radial vector relationship–Alpha group. Studies in Engineering and Exact Sciences, v. 6, n. 1, p. e16083-e16083, 2025. DOI: https://doi.org/10.54021/seesv6n1-037. 

Souza Correa, Cleber & Nogueira de Melo, T. B. (2025). The Alpha Group 4D Geometry: Symmetric Structures and Topological Transitions (https://github.com/CleberCorrea15/The-Alpha-group). Zenodo. https://doi.org/10.5281/zenodo.16815767

Cleber Souza Corrêa, Thiago Braido Nogueira de Melo. Topological Differences between Riemann Geometry and the Alpha Group via Graph Methods. 2025. ⟨hal-05281857⟩
 Zenodo. https://doi.org/10.5281/zenodo.17196204

## Contact

For questions or collaboration, please contact [Cleber Correa](https://github.com/CleberCorrea15).

## ResearchGate Profile

You can find my publications and collaborations at [ResearchGate](https://www.researchgate.net/profile/Cleber-Souza-Correa?ev=hdr_xprf).


Cleber Souza Correa - https://sciprofiles.com/profile/4669718
