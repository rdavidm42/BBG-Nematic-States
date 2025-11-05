"""
Energy Dispersion Calculator for Bernal Bilayer Graphene

This module computes the electronic band structure and eigenvectors for a tight-binding
model of Bernal bilayer graphene. The implementation uses Numba JIT compilation for
high-performance parallel computation on HPC clusters.

Key Features:
    - Parallel eigenvalue decomposition across momentum space
    - Four distinct Hamiltonian configurations (different valley/layer combinations)
    - Optimized for large-scale momentum grids
"""

import numpy as np
import numba


@numba.jit(nopython=True, parallel=True)
def dispersion(z, hamiltonian1, hamiltonian2, hamiltonian3, hamiltonian4):
    """
    Compute energy bands and eigenvectors for four Hamiltonian configurations.
    
    This function performs parallel diagonalization of tight-binding Hamiltonians
    across momentum space. Each Hamiltonian represents a different combination of
    valley (K/K') and layer configuration in bilayer graphene.
    
    Parameters
    ----------
    z : float
        Applied perpendicular electric field (displacement field) in meV
    hamiltonian1 : ndarray, shape (4, 4, lengthm)
        First Hamiltonian configuration (K valley, spin up)
    hamiltonian2 : ndarray, shape (4, 4, lengthm)
        Second Hamiltonian configuration (K' valley, spin up)
    hamiltonian3 : ndarray, shape (4, 4, lengthm)
        Third Hamiltonian configuration (K valley, spin down)
    hamiltonian4 : ndarray, shape (4, 4, lengthm)
        Fourth Hamiltonian configuration (K' valley, spin down)
    
    Returns
    -------
    energy1, energy2, energy3, energy4 : ndarray, shape (lengthm,)
        Energy dispersions for the second band (index 1) of each configuration,
        adjusted by the applied electric field (-energy - z/2)
    vectors1, vectors2, vectors3, vectors4 : ndarray, shape (4, lengthm)
        Corresponding eigenvectors for each configuration
    
    Notes
    -----
    - Uses parallel execution across momentum points for optimal HPC performance
    - Extracts second eigenvalue (index 1) which corresponds to the
      valence band edge in gapped bilayer graphene
    """
    lengthm = hamiltonian1.shape[2]
    
    # Initialize energy arrays for each configuration
    energy1 = np.zeros(lengthm)
    energy2 = np.zeros(lengthm)
    energy3 = np.zeros(lengthm)
    energy4 = np.zeros(lengthm)

    # Initialize eigenvector arrays (4 bands per momentum point)
    vectors1 = np.zeros((4, lengthm), dtype=np.complex128)
    vectors2 = np.zeros((4, lengthm), dtype=np.complex128)
    vectors3 = np.zeros((4, lengthm), dtype=np.complex128)
    vectors4 = np.zeros((4, lengthm), dtype=np.complex128)
    
    # Parallel loop over all momentum points
    for i in numba.prange(lengthm):
        # Diagonalize each Hamiltonian at momentum point i
        # sys[0] contains eigenvalues, sys[1] contains eigenvectors
        sys1 = np.linalg.eigh(hamiltonian1[:, :, i])
        sys2 = np.linalg.eigh(hamiltonian2[:, :, i])
        sys3 = np.linalg.eigh(hamiltonian3[:, :, i])
        sys4 = np.linalg.eigh(hamiltonian4[:, :, i])
        
        # Extract second eigenvalue (band of interest)
        energy1[i] = sys1[0][1]
        energy2[i] = sys2[0][1]
        energy3[i] = sys3[0][1]
        energy4[i] = sys4[0][1]

        # Extract corresponding eigenvectors
        vectors1[:, i] = sys1[1][:, 1]
        vectors2[:, i] = sys2[1][:, 1]
        vectors3[:, i] = sys3[1][:, 1]
        vectors4[:, i] = sys4[1][:, 1]
    
    # Return energies with electric field correction and all eigenvectors
    return (-energy1 - z/2, -energy2 - z/2, -energy3 - z/2, -energy4 - z/2,
            vectors1, vectors2, vectors3, vectors4)


def getting_energies(v, momenta):
    """
    Construct tight-binding Hamiltonians and compute band structures.
    
    This function builds the full tight-binding Hamiltonian for Bernal bilayer
    graphene including intralayer and interlayer hopping terms. It sweeps over
    a range of perpendicular electric fields and momentum points.
    
    Parameters
    ----------
    v : ndarray, shape (lengthv,)
        Array of perpendicular electric field values (meV) to sweep
    momenta : ndarray, shape (lengthm, 2)
        2D momentum grid points (kx, ky) in the Brillouin zone
    
    Returns
    -------
    energy_bands : ndarray, shape (4, lengthv, lengthm)
        Energy dispersions for all four configurations across field and momentum
    vectors_bands : ndarray, shape (4, lengthv, 4, lengthm)
        Corresponding eigenvectors for all configurations
    
    Notes
    -----
    The four Hamiltonians correspond to:
        1. K valley, spin up
        2. K' valley, spin up
        3. K valley, spin down
        4. K' valley, spin down
    """
    # Extract momentum components
    kx = momenta[:, 0]
    ky = momenta[:, 1]
    
    # Compute structure factors for K and K' valleys
    # These encode the hexagonal lattice geometry through phase factors
    # f1, f1c correspond to K valley; f2, f2c correspond to K' valley
    f1 = (np.exp(1j * ky / np.sqrt(3)) + 
          2 * np.exp(-1j * ky / (2 * np.sqrt(3))) * 
          np.cos((kx - 4 * np.pi / 3 * np.ones(momenta.shape[0])) / 2))
    
    f1c = (np.exp(-1j * ky / np.sqrt(3)) + 
           2 * np.exp(1j * ky / (2 * np.sqrt(3))) * 
           np.cos((kx - 4 * np.pi / 3 * np.ones(momenta.shape[0])) / 2))
    
    f2 = (np.exp(1j * ky / np.sqrt(3)) + 
          2 * np.exp(-1j * ky / (2 * np.sqrt(3))) * 
          np.cos((kx + 4 * np.pi / 3 * np.ones(momenta.shape[0])) / 2))
    
    f2c = (np.exp(-1j * ky / np.sqrt(3)) + 
           2 * np.exp(1j * ky / (2 * np.sqrt(3))) * 
           np.cos((kx + 4 * np.pi / 3 * np.ones(momenta.shape[0])) / 2))
    
    lengthv = len(v)
    lengthm = len(momenta)
    
    # Tight-binding parameters (meV units)
    t0 = 3100  # Intralayer nearest-neighbor hopping
    t1 = 380   # Interlayer vertical hopping (dimer sites)
    t3 = 290   # Interlayer skew hopping
    t4 = 138   # Interlayer skew hopping
    w = 10.5   # Sublattice potential difference
    l = 0      # Spin orbit coupling
    
    # Initialize output arrays
    energy_1 = np.ones((lengthv, lengthm))
    energy_2 = np.ones((lengthv, lengthm))
    energy_3 = np.ones((lengthv, lengthm))
    energy_4 = np.ones((lengthv, lengthm))
    
    vectors1 = np.zeros((lengthv, 4, lengthm), dtype=np.complex128)
    vectors2 = np.zeros((lengthv, 4, lengthm), dtype=np.complex128)
    vectors3 = np.zeros((lengthv, 4, lengthm), dtype=np.complex128)
    vectors4 = np.zeros((lengthv, 4, lengthm), dtype=np.complex128)
    
    # Loop over electric field values
    for z in range(lengthv):
        # Construct four 4x4 tight-binding Hamiltonians
        # Basis: [A1, B1, A2, B2] where 1,2 are layer indices and A,B are sublattices
        
        # Hamiltonian 1: K valley spin up
        hamiltonian1 = np.array([
            [v[z]/2 * np.ones(lengthm), -t0*f1, t4*f1, t3*f1c],
            [-t0*f1c, w + v[z]/2 * np.ones(lengthm), t1*np.ones(lengthm), t4*f1],
            [t4*f1c, t1*np.ones(lengthm), w - v[z]/2 * np.ones(lengthm) + l/2, -t0*f1],
            [t3*f1, t4*f1c, -t0*f1c, -v[z]/2 * np.ones(lengthm) + l/2]
        ])
        
        # Hamiltonian 2: K' valley spin up
        hamiltonian2 = np.array([
            [v[z]/2 * np.ones(lengthm), -t0*f2, t4*f2, t3*f2c],
            [-t0*f2c, w + v[z]/2 * np.ones(lengthm), t1*np.ones(lengthm), t4*f2],
            [t4*f2c, t1*np.ones(lengthm), w - v[z]/2 * np.ones(lengthm) - l/2, -t0*f2],
            [t3*f2, t4*f2c, -t0*f2c, -v[z]/2 * np.ones(lengthm) - l/2]
        ])
        
        # Hamiltonian 3: K valley spin down
        hamiltonian3 = np.array([
            [v[z]/2 * np.ones(lengthm), -t0*f1, t4*f1, t3*f1c],
            [-t0*f1c, w + v[z]/2 * np.ones(lengthm), t1*np.ones(lengthm), t4*f1],
            [t4*f1c, t1*np.ones(lengthm), w - v[z]/2 * np.ones(lengthm) - l/2, -t0*f1],
            [t3*f1, t4*f1c, -t0*f1c, -v[z]/2 * np.ones(lengthm) - l/2]
        ])
        
        # Hamiltonian 4: K' valley spin down
        hamiltonian4 = np.array([
            [v[z]/2 * np.ones(lengthm), -t0*f2, t4*f2, t3*f2c],
            [-t0*f2c, w + v[z]/2 * np.ones(lengthm), t1*np.ones(lengthm), t4*f2],
            [t4*f2c, t1*np.ones(lengthm), w - v[z]/2 * np.ones(lengthm) + l/2, -t0*f2],
            [t3*f2, t4*f2c, -t0*f2c, -v[z]/2 * np.ones(lengthm) + l/2]
        ])
        
        # Compute dispersions for this electric field value
        (energy_1[z], energy_2[z], energy_3[z], energy_4[z],
         vectors1[z], vectors2[z], vectors3[z], vectors4[z]) = dispersion(
            v[z], hamiltonian1, hamiltonian2, hamiltonian3, hamiltonian4)
    
    return (np.array([energy_1, energy_2, energy_3, energy_4]),
            np.array([vectors1, vectors2, vectors3, vectors4]))