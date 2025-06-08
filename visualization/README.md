Still Editing......
# Poisson Solver Visualization
This Python program visualizes the convergence of different numerical methods (Jacobi, Gauss-Seidel, and Multigrid) when solving the Poisson equation. It specifically focuses on the pressure projection step commonly found in Stable Fluid simulations.

---

# Features
Simulates the pressure projection process on a 128x128 2D grid.
Generates an initial divergence field comprising both high and low-frequency components.
Implements three distinct solver methods:
- Jacobi Iteration
- Gauss-Seidel Iteration
- Multigrid V-cycle
Provides visualizations including:
3D Residual Surface Plot: Shows residual values on the x-y grid plane.
Convergence Curve: Plots iteration count against residual values.
# Installation
```Bash
pip install -r requirements.txt
```
# Usage
Simply run the main program:
```Bash
python poisson_solver_visualization.py
```
The program will automatically:
- Generate the initial divergence field.
- Solve the Poisson equation using all three methods.
- Display the 3D residual plots.
- Show the convergence curves.
# Output Details
3D Residual Plots: These plots illustrate the distribution of residuals across the grid, offering a clear visual of how each method handles high and low-frequency errors.
Convergence Curves: These plots demonstrate the convergence behavior of each method over iterations, with residuals displayed on a logarithmic scale.
Notes
The program defaults to a 128x128 grid.
The initial divergence field is composed of 70% low-frequency and 30% high-frequency components.
The Multigrid method uses 3 grid levels by default.
