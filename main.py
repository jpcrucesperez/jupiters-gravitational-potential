import numpy as np
import matplotlib.pyplot as plt

def gravitational_potential(r, theta, phi, mu, R, J_coefficients):
    """
    Calculate the gravitational potential including zonal harmonics.

    Parameters:
    r : float
        Distance from the planet center (meters).
    theta : float
        Colatitude in radians (0 at north pole).
    phi : float
        Longitude in radians.
    mu : float
        Standard gravitational parameter (m^3/s^2).
    R : float
        Planet's mean radius (meters).
    J_coefficients : list
        List of zonal harmonics coefficients [J2, J3, ..., J8].

    Returns:
    V : float
        Gravitational potential at the given position.
    """
    # Initial potential from point mass gravity
    V = -mu / r
    
    # Add contributions from zonal harmonics
    P = [1.0, np.cos(theta)]  # Initialize Legendre polynomials
    for n, Jn in enumerate(J_coefficients, start=2):
        # Compute the associated Legendre polynomial Pn(cos(theta))
        if len(P) <= n:  # Generate next Legendre polynomial if needed
            Pn = ((2*n - 1) * np.cos(theta) * P[-1] - (n - 1) * P[-2]) / n
            P.append(Pn)
        else:
            Pn = P[n]
        # Add zonal harmonic contribution
        V += (R / r) ** n * Jn * Pn * mu / r
    
    return V

def generate_heatmap_grid(mu, radius, J_coefficients):
    """
    Generate a 2D grid of gravitational potential for visualization.

    Parameters:
    mu : float
        Standard gravitational parameter (m^3/s^2).
    radius : float
        Planet's mean radius (meters).
    J_coefficients : list
        List of zonal harmonics coefficients.

    Returns:
    r_values : ndarray
        Array of radial distances.
    theta_values : ndarray
        Array of colatitudes.
    V_grid : ndarray
        2D array of gravitational potential values.
    """
    theta_values = np.linspace(0, np.pi, 100)  # Colatitude from 0 to π (radians)
    r_values = np.linspace(radius, 2 * radius, 100)  # Radius from planet's surface to 2R

    # Create a grid for potential values
    V_grid = np.zeros((len(r_values), len(theta_values)))

    for i, r in enumerate(r_values):
        for j, theta in enumerate(theta_values):
            V_grid[i, j] = gravitational_potential(r, theta, 0, mu, radius, J_coefficients)

    return r_values, theta_values, V_grid

# Constants for Earth and Jupiter
earth_mu = 3.986004418e14  # m^3/s^2
earth_radius = 6378137.0  # meters
earth_J = [-0.1082635854e-2, 0.2532435346e-5, 0.1619331205e-5, 0.2277161016e-6, -0.5396484906e-6, 0.3513684422e-6, 0.2025187152e-6]  # J2 to J8 for Earth

jupiter_mu = 1.26686534e17  # m^3/s^2
jupiter_radius = 71492000.0  # meters
jupiter_J = [14696.5735e-6, -0.045e-6, -586.6085e-6, 0.0723e-6, 34.2007e-6, 0.120e-6, -2.422e-6]  # J2 to J8 for Jupiter

# Generate grids for Earth and Jupiter
r_earth, theta_earth, V_earth = generate_heatmap_grid(earth_mu, earth_radius, earth_J)
r_jupiter, theta_jupiter, V_jupiter = generate_heatmap_grid(jupiter_mu, jupiter_radius, jupiter_J)

# Plot side-by-side heatmaps
fig, axes = plt.subplots(1, 2, figsize=(14, 6), constrained_layout=True)

# Earth Heatmap
im1 = axes[0].contourf(np.degrees(theta_earth), r_earth / 1000, V_earth, levels=100, cmap="viridis")
axes[0].set_title("Gravitational Potential Heatmap (Earth)")
axes[0].set_xlabel("Colatitude (degrees)")
axes[0].set_ylabel("Radius (km)")
fig.colorbar(im1, ax=axes[0], label="Gravitational Potential (m^2/s^2)")

# Jupiter Heatmap
im2 = axes[1].contourf(np.degrees(theta_jupiter), r_jupiter / 1000, V_jupiter, levels=100, cmap="viridis")
axes[1].set_title("Gravitational Potential Heatmap (Jupiter)")
axes[1].set_xlabel("Colatitude (degrees)")
axes[1].set_ylabel("Radius (km)")
fig.colorbar(im2, ax=axes[1], label="Gravitational Potential (m^2/s^2)")

plt.show()
