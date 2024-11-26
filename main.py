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

# Constants for Earth and Jupiter
earth_mu = 3.986004418e14  # m^3/s^2
earth_radius = 6378137.0  # meters
jupiter_mu = 1.26686534e17  # m^3/s^2
jupiter_radius = 71492000.0  # meters

# Zonal coefficients (J2 to J8) for Earth and Jupiter
earth_J = [-0.1082635854e-2, 0.2532435346e-5, 0.1619331205e-5, 0.2277161016e-6, -0.5396484906e-6, 0.3513684422e-6, 0.2025187152e-6]
jupiter_J = [14696.5735e-6, -0.045e-6, -586.6085e-6, 0.0723e-6, 34.2007e-6, 0.120e-6, -2.422e-6]

# Range of radial distances (from planet's surface outward)
r_earth = np.linspace(earth_radius, 10 * earth_radius, 500)
r_jupiter = np.linspace(jupiter_radius, 10 * jupiter_radius, 500)

# Different latitudes to evaluate (0° to 90°)
latitudes_deg = [0, 15, 30, 45, 60, 75, 90]  # List of latitudes
latitudes_rad = np.radians(latitudes_deg)  # Convert to radians

# Create the plot for Earth and Jupiter
plt.figure(figsize=(12, 12))

# Loop through each latitude for Earth
for lat in latitudes_rad:
    V_earth = [gravitational_potential(r, lat, 0, earth_mu, earth_radius, earth_J) for r in r_earth]
    plt.subplot(2, 1, 1)  # First subplot for Earth
    plt.plot(r_earth / 1e6, V_earth, label=f"Latitude {latitudes_deg[np.where(latitudes_rad == lat)[0][0]]}°")

# Loop through each latitude for Jupiter
for lat in latitudes_rad:
    V_jupiter = [gravitational_potential(r, lat, 0, jupiter_mu, jupiter_radius, jupiter_J) for r in r_jupiter]
    plt.subplot(2, 1, 2)  # Second subplot for Jupiter
    plt.plot(r_jupiter / 1e6, V_jupiter, label=f"Latitude {latitudes_deg[np.where(latitudes_rad == lat)[0][0]]}°")

# Add titles, labels, and legends for Earth
plt.subplot(2, 1, 1)
plt.title("Gravitational Potential of Earth at Different Latitudes")
plt.xlabel("Distance from Earth's Center (10^6 meters)")
plt.ylabel("Gravitational Potential (m^2/s^2)")
plt.legend()
plt.grid()

# Add titles, labels, and legends for Jupiter
plt.subplot(2, 1, 2)
plt.title("Gravitational Potential of Jupiter at Different Latitudes")
plt.xlabel("Distance from Jupiter's Center (10^6 meters)")
plt.ylabel("Gravitational Potential (m^2/s^2)")
plt.legend()
plt.grid()

# Show the plot
plt.tight_layout()
plt.show()
