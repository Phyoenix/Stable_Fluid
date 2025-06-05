import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from matplotlib.widgets import Slider
import matplotlib.animation as animation

class PoissonSolver:
    def __init__(self, nx=128, ny=128):
        self.nx = nx
        self.ny = ny
        self.dx = 1.0 / nx
        self.dy = 1.0 / ny
        
    def generate_divergence_field(self):
        """Generate a divergence field with white noise"""
        # Generate white noise
        divergence = np.random.normal(0, 1, (self.nx, self.ny))
        
        # Normalize
        divergence = divergence / np.max(np.abs(divergence))
        return divergence
    
    def compute_fft(self, field):
        """Compute 2D FFT and return the magnitude spectrum"""
        fft = np.fft.fft2(field)
        fft_shift = np.fft.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        return magnitude_spectrum
    
    def plot_spectrum(self, spectrum, title):
        """Plot the frequency spectrum"""
        plt.figure(figsize=(10, 8))
        plt.imshow(np.log1p(spectrum), cmap='viridis')
        plt.colorbar(label='log(magnitude)')
        plt.title(title)
        plt.xlabel('Frequency X')
        plt.ylabel('Frequency Y')
        plt.show()
    
    def compute_divergence(self, p):
        """Compute divergence of the pressure field with proper boundary conditions"""
        divergence = np.zeros_like(p)
        
        # Interior points
        divergence[1:-1, 1:-1] = (p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1] - 
                                 4 * p[1:-1, 1:-1]) / (self.dx * self.dx)
        
        # Boundary points (using one-sided differences)
        # Left boundary
        divergence[1:-1, 0] = (p[1:-1, 1] - p[1:-1, 0]) / self.dx
        # Right boundary
        divergence[1:-1, -1] = (p[1:-1, -1] - p[1:-1, -2]) / self.dx
        # Top boundary
        divergence[0, 1:-1] = (p[1, 1:-1] - p[0, 1:-1]) / self.dx
        # Bottom boundary
        divergence[-1, 1:-1] = (p[-1, 1:-1] - p[-2, 1:-1]) / self.dx
        
        # Corners
        divergence[0, 0] = (p[1, 0] - p[0, 0]) / self.dx + (p[0, 1] - p[0, 0]) / self.dx
        divergence[0, -1] = (p[1, -1] - p[0, -1]) / self.dx + (p[0, -1] - p[0, -2]) / self.dx
        divergence[-1, 0] = (p[-1, 0] - p[-2, 0]) / self.dx + (p[-1, 1] - p[-1, 0]) / self.dx
        divergence[-1, -1] = (p[-1, -1] - p[-2, -1]) / self.dx + (p[-1, -1] - p[-1, -2]) / self.dx
        
        return divergence
    
    def jacobi_iteration(self, p, b, omega=1.0, max_iter=20, tol=1e-6):
        """Jacobi iteration method with proper boundary conditions"""
        residuals = []
        divergence_fields = []
        p_new = p.copy()
        
        for iter in range(max_iter):
            p_old = p_new.copy()
            
            # Update interior points
            p_new[1:-1, 1:-1] = (1-omega) * p_old[1:-1, 1:-1] + \
                               omega * 0.25 * (p_old[1:-1, 2:] + p_old[1:-1, :-2] + 
                                             p_old[2:, 1:-1] + p_old[:-2, 1:-1] - 
                                             b[1:-1, 1:-1] * self.dx * self.dx)
            
            # Apply boundary conditions (Dirichlet)
            p_new[0, :] = 0  # Top
            p_new[-1, :] = 0  # Bottom
            p_new[:, 0] = 0  # Left
            p_new[:, -1] = 0  # Right
            
            # Compute divergence field
            divergence_field = self.compute_divergence(p_new)
            divergence_fields.append(divergence_field)
            
            # Compute residual norm (L2 norm of the divergence field)
            residual = np.sqrt(np.sum(divergence_field**2))
            residuals.append(residual)
            
            if residual < tol:
                break
                
        return p_new, residuals, divergence_fields
    
    def gauss_seidel_iteration(self, p, b, omega=1.0, max_iter=20, tol=1e-6):
        """Gauss-Seidel iteration method with proper boundary conditions"""
        residuals = []
        divergence_fields = []
        p_new = p.copy()
        
        for iter in range(max_iter):
            p_old = p_new.copy()
            
            # Update interior points
            for i in range(1, self.nx-1):
                for j in range(1, self.ny-1):
                    p_new[i,j] = (1-omega) * p_old[i,j] + \
                                omega * 0.25 * (p_new[i,j+1] + p_new[i,j-1] + 
                                              p_new[i+1,j] + p_new[i-1,j] - 
                                              b[i,j] * self.dx * self.dx)
            
            # Apply boundary conditions (Dirichlet)
            p_new[0, :] = 0  # Top
            p_new[-1, :] = 0  # Bottom
            p_new[:, 0] = 0  # Left
            p_new[:, -1] = 0  # Right
            
            # Compute divergence field
            divergence_field = self.compute_divergence(p_new)
            divergence_fields.append(divergence_field)
            
            # Compute residual norm (L2 norm of the divergence field)
            residual = np.sqrt(np.sum(divergence_field**2))
            residuals.append(residual)
            
            if residual < tol:
                break
                
        return p_new, residuals, divergence_fields
    
    def multigrid_vcycle(self, p, b, level=0, max_level=3, max_iter=20, tol=1e-6):
        """Multigrid V-cycle method with proper boundary conditions"""
        residuals = []
        divergence_fields = []
        p_new = p.copy()
        
        for iter in range(max_iter):
            p_old = p_new.copy()
            
            # One V-cycle
            p_new = self._vcycle(p_new, b, level, max_level)
            
            # Apply boundary conditions (Dirichlet)
            p_new[0, :] = 0  # Top
            p_new[-1, :] = 0  # Bottom
            p_new[:, 0] = 0  # Left
            p_new[:, -1] = 0  # Right
            
            # Compute divergence field
            divergence_field = self.compute_divergence(p_new)
            divergence_fields.append(divergence_field)
            
            # Compute residual norm (L2 norm of the divergence field)
            residual = np.sqrt(np.sum(divergence_field**2))
            residuals.append(residual)
            
            if residual < tol:
                break
                
        return p_new, residuals, divergence_fields
    
    def _vcycle(self, p, b, level, max_level):
        """Helper function for one V-cycle with proper boundary conditions"""
        if level == max_level:
            # Solve directly on coarsest grid
            return self.jacobi_iteration(p, b, max_iter=10, tol=1e-6)[0]
        
        # Pre-smoothing
        p, _, _ = self.jacobi_iteration(p, b, max_iter=3)
        
        # Apply boundary conditions
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0
        
        # Compute residual
        residual = self.compute_divergence(p)
        
        # Restrict residual to coarser grid
        coarse_residual = self.restrict(residual)
        coarse_p = np.zeros_like(coarse_residual)
        
        # Recursive call
        coarse_p = self._vcycle(coarse_p, coarse_residual, level+1, max_level)
        
        # Prolongate correction
        correction = self.prolongate(coarse_p)
        p = p + correction
        
        # Apply boundary conditions
        p[0, :] = 0
        p[-1, :] = 0
        p[:, 0] = 0
        p[:, -1] = 0
        
        # Post-smoothing
        p, _, _ = self.jacobi_iteration(p, b, max_iter=3)
        
        return p
    
    def restrict(self, fine):
        """Restrict fine grid to coarse grid"""
        coarse = np.zeros((fine.shape[0]//2, fine.shape[1]//2))
        for i in range(coarse.shape[0]):
            for j in range(coarse.shape[1]):
                coarse[i,j] = 0.25 * (fine[2*i,2*j] + fine[2*i+1,2*j] + 
                                    fine[2*i,2*j+1] + fine[2*i+1,2*j+1])
        return coarse
    
    def prolongate(self, coarse):
        """Prolongate coarse grid to fine grid"""
        fine = np.zeros((coarse.shape[0]*2, coarse.shape[1]*2))
        for i in range(coarse.shape[0]):
            for j in range(coarse.shape[1]):
                fine[2*i,2*j] = coarse[i,j]
                fine[2*i+1,2*j] = coarse[i,j]
                fine[2*i,2*j+1] = coarse[i,j]
                fine[2*i+1,2*j+1] = coarse[i,j]
        return fine

def create_interactive_3d_plot(divergence_fields, title):
    """Create an interactive 3D plot with a slider to show divergence at different iterations"""
    fig = plt.figure(figsize=(12, 8))
    
    # Create 3D subplot
    ax = fig.add_subplot(111, projection='3d')
    
    # Create meshgrid
    x = np.arange(divergence_fields[0].shape[0])
    y = np.arange(divergence_fields[0].shape[1])
    X, Y = np.meshgrid(x, y)
    
    # Initial plot
    surf = ax.plot_surface(X, Y, divergence_fields[0].T, cmap='viridis')
    
    # Add colorbar
    fig.colorbar(surf, ax=ax, shrink=0.5, aspect=5)
    
    # Add slider
    ax_slider = plt.axes([0.2, 0.02, 0.6, 0.03])
    slider = Slider(ax_slider, 'Iteration', 0, len(divergence_fields)-1,
                   valinit=0, valstep=1)
    
    def update(val):
        iteration = int(slider.val)
        ax.clear()
        surf = ax.plot_surface(X, Y, divergence_fields[iteration].T, cmap='viridis')
        ax.set_title(f'{title} - Iteration {iteration}')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Divergence')
        fig.canvas.draw_idle()
    
    slider.on_changed(update)
    ax.set_title(f'{title} - Iteration 0')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Divergence')
    
    plt.show()

def plot_convergence(residuals, labels):
    """Plot convergence curves"""
    plt.figure(figsize=(10, 6))
    for residual, label in zip(residuals, labels):
        plt.semilogy(residual, label=label)
    
    # Set x-axis ticks to show every 5 iterations
    plt.xticks(np.arange(0, len(residuals[0]), 5))
    
    plt.xlabel('Iteration')
    plt.ylabel('L2 Norm of Divergence\n(Measure of how far the velocity field is from being divergence-free)')
    plt.title('Convergence Comparison\nSmaller values indicate better divergence-free property')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_convergence_rate(residuals, labels):
    """Plot convergence rate (ratio of consecutive residuals)"""
    plt.figure(figsize=(10, 6))
    for residual, label in zip(residuals, labels):
        # Compute convergence rate
        rates = np.array(residual[1:]) / np.array(residual[:-1])
        plt.semilogy(rates, label=label)
    
    # Set x-axis ticks to show every 5 iterations
    plt.xticks(np.arange(0, len(rates), 5))
    
    plt.xlabel('Iteration')
    plt.ylabel('Convergence Rate\n(Ratio of current to previous divergence)')
    plt.title('Convergence Rate Comparison\nValues closer to 1 indicate slower convergence')
    plt.legend()
    plt.grid(True)
    plt.show()

def main():
    # Initialize solver
    solver = PoissonSolver(nx=128, ny=128)
    
    # Generate initial divergence field with white noise
    b = solver.generate_divergence_field()
    
    # Plot initial spectrum
    initial_spectrum = solver.compute_fft(b)
    solver.plot_spectrum(initial_spectrum, 'Initial Frequency Spectrum')
    
    # Initial pressure field
    p0 = np.zeros((solver.nx, solver.ny))
    
    # Solve using different methods
    p_jacobi, residuals_jacobi, divergence_fields_jacobi = solver.jacobi_iteration(p0.copy(), b)
    p_gs, residuals_gs, divergence_fields_gs = solver.gauss_seidel_iteration(p0.copy(), b)
    p_mg, residuals_mg, divergence_fields_mg = solver.multigrid_vcycle(p0.copy(), b)
    
    # Plot final spectra for each method
    final_spectrum_jacobi = solver.compute_fft(divergence_fields_jacobi[-1])
    final_spectrum_gs = solver.compute_fft(divergence_fields_gs[-1])
    final_spectrum_mg = solver.compute_fft(divergence_fields_mg[-1])
    
    solver.plot_spectrum(final_spectrum_jacobi, 'Final Frequency Spectrum - Jacobi')
    solver.plot_spectrum(final_spectrum_gs, 'Final Frequency Spectrum - Gauss-Seidel')
    solver.plot_spectrum(final_spectrum_mg, 'Final Frequency Spectrum - Multigrid')
    
    # Create interactive 3D plots
    create_interactive_3d_plot(divergence_fields_jacobi, 'Jacobi Method Divergence')
    create_interactive_3d_plot(divergence_fields_gs, 'Gauss-Seidel Method Divergence')
    create_interactive_3d_plot(divergence_fields_mg, 'Multigrid Method Divergence')
    
    # Plot convergence curves
    plot_convergence([residuals_jacobi, residuals_gs, residuals_mg], 
                    ['Jacobi', 'Gauss-Seidel', 'Multigrid'])
    
    # Plot convergence rates
    plot_convergence_rate([residuals_jacobi, residuals_gs, residuals_mg],
                         ['Jacobi', 'Gauss-Seidel', 'Multigrid'])

if __name__ == "__main__":
    main()