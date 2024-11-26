import numpy as np

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

# Example: Earth and Jupiter Constants
earth_mu = 3.986004418e14  # m^3/s^2
earth_radius = 6378137.0  # meters
jupiter_mu = 1.26686534e17  # m^3/s^2
jupiter_radius = 71492000.0  # meters

# Placeholder for zonal coefficients, replace these with your values
earth_J = [-0.1082635854e-2, 0.2532435346e-5, 0.1619331205e-5, 0.2277161016e-6, -0.5396484906e-6, 0.3513684422e-6, 0.2025187152e-6]  # J2 to J8 for Earth
jupiter_J = [14696.5735e-6, -0.045e-6, -586.6085e-6, 0.0723e-6, 34.2007e-6, 0.120e-6, -2.422e-6]  # J2 to J8 for Jupiter

# Example calculation
r_earth = 7000000.0  # Position (m) from Earth's center
theta = np.radians(45)  # Colatitude in radians
phi = np.radians(30)    # Longitude in radians

V_earth = gravitational_potential(r_earth, theta, phi, earth_mu, earth_radius, earth_J)
print(f"Earth Gravitational Potential: {V_earth:.4e} m^2/s^2")

r_jupiter = 80000000.0  # Position (m) from Jupiter's center
V_jupiter = gravitational_potential(r_jupiter, theta, phi, jupiter_mu, jupiter_radius, jupiter_J)
print(f"Jupiter Gravitational Potential: {V_jupiter:.4e} m^2/s^2")
