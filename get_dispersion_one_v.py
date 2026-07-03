"""
Energy Dispersion Calculator for Bernal Bilayer Graphene

This module computes the electronic band structure and eigenvectors for a tight-binding
model of Bernal bilayer graphene. The implementation uses Numba JIT compilation for 
efficient parallel computation across momentum space.

Key Features:
    - Parallel eigenvalue decomposition across momentum space
    - Four distinct Hamiltonian configurations (different valley/layer combinations)
    - Optimized for large-scale momentum grids
"""

import numpy as np
import numba
import os

@numba.jit(nopython=True,parallel=True)
def dispersion(v,hamiltonian1,hamiltonian2,hamiltonian3,hamiltonian4):
    """
    Compute energy bands and eigenvectors for four Hamiltonian configurations.
    
    This function performs parallel diagonalization of tight-binding Hamiltonians
    across momentum space. Each Hamiltonian represents a different combination of
    valley (K/K') and layer configuration in bilayer graphene.
    
    Parameters
    ----------
    v : float
        Applied perpendicular electric field (displacement field) in meV
    hamiltonian1 : ndarray, shape (4, 4, lengthm)
        First isospin Hamiltonian (K valley, spin up)
    hamiltonian2 : ndarray, shape (4, 4, lengthm)
        Second isospin Hamiltonian (K' valley, spin up)
    hamiltonian3 : ndarray, shape (4, 4, lengthm)
        Third isospin Hamiltonian (K valley, spin down)
    hamiltonian4 : ndarray, shape (4, 4, lengthm)
        Fourth isospin Hamiltonian (K' valley, spin down)
    
    Returns
    -------
    energy1, energy2, energy3, energy4 : ndarray, shape (lengthm,)
        Dispersions for the higher energy hole (index 1) of each configuration,
        tuned by electric field so that at the K/K' points, the valence band is 
        at zero energy.
    vectors1, vectors2, vectors3, vectors4 : ndarray, shape (4, lengthm)
        Corresponding eigenvectors for each isospin.
    
    """
    lengthm = hamiltonian1.shape[2]

    # Initialize arrays to hold energies and eigenvectors
    energy1 = np.zeros(lengthm)
    energy2 = np.zeros(lengthm)
    energy3 = np.zeros(lengthm)
    energy4 = np.zeros(lengthm)

    vectors1 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors2 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors3 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors4 = np.zeros((4,lengthm),dtype=np.complex128)

    for i in numba.prange(lengthm):
        # Get eigenvalues and eigenvectors for each Hamiltonian
        sys1 = np.linalg.eigh(hamiltonian1[:,:,i])
        sys2 = np.linalg.eigh(hamiltonian2[:,:,i])
        sys3 = np.linalg.eigh(hamiltonian3[:,:,i])
        sys4 = np.linalg.eigh(hamiltonian4[:,:,i])
        
        energy1[i] = sys1[0][1]
        energy2[i] = sys2[0][1]
        energy3[i] = sys3[0][1]
        energy4[i] = sys4[0][1]

        vectors1[:,i] = sys1[1][:,1]
        vectors2[:,i] = sys2[1][:,1]
        vectors3[:,i] = sys3[1][:,1]
        vectors4[:,i] = sys4[1][:,1]
    return -energy1-v/2, -energy2-v/2, -energy3-v/2, -energy4-v/2,vectors1,vectors2,vectors3,vectors4

def getting_energies(v,momenta):   
    kx = momenta[:,0]
    ky = momenta[:,1]

    # Define the form factors for the tight-binding model
    f1 = np.exp(1j*ky/np.sqrt(3)) + 2*np.exp(-1j*ky/(2*np.sqrt(3)))*np.cos((kx-4*np.pi/3*np.ones(momenta.shape[0]))/2);
    f1c = np.exp(-1j*ky/np.sqrt(3)) + 2*np.exp(1j*ky/(2*np.sqrt(3)))*np.cos((kx-4*np.pi/3*np.ones(momenta.shape[0]))/2);
    f2 = np.exp(1j*ky/np.sqrt(3)) + 2*np.exp(-1j*ky/(2*np.sqrt(3)))*np.cos((kx+4*np.pi/3*np.ones(momenta.shape[0]))/2);
    f2c = np.exp(-1j*ky/np.sqrt(3)) + 2*np.exp(1j*ky/(2*np.sqrt(3)))*np.cos((kx+4*np.pi/3*np.ones(momenta.shape[0]))/2);
    
    lengthm = len(momenta)
    
    # Tight binding parameters for Bernal bilayer graphene
    t0 = 2610
    t1 = 361
    t3 = 283
    t4 = 138
    w = 15
    l = 0
    
    energy_1 = np.ones(lengthm)
    energy_2 = np.ones(lengthm)
    energy_3 = np.ones(lengthm)
    energy_4 = np.ones(lengthm)
    
    vectors1 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors2 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors3 = np.zeros((4,lengthm),dtype=np.complex128)
    vectors4 = np.zeros((4,lengthm),dtype=np.complex128)

    #Hamiltonians for each valley/spin configuration in sublattice basis (A1,B1,A2,B2)
    hamiltonian1 = np.array([[v/2*np.ones(lengthm),-t0*f1,t4*f1,t3*f1c]
                              ,[-t0*f1c,w+v/2*np.ones(lengthm),t1*np.ones(lengthm),t4*f1]
                              ,[t4*f1c,t1*np.ones(lengthm),w-v/2*np.ones(lengthm)+l/2,-t0*f1]
                              ,[t3*f1,t4*f1c,-t0*f1c,-v/2*np.ones(lengthm)+l/2]])
    hamiltonian2 = np.array([[v/2*np.ones(lengthm),-t0*f2,t4*f2,t3*f2c]
                              ,[-t0*f2c,w+v/2*np.ones(lengthm),t1*np.ones(lengthm),t4*f2]
                              ,[t4*f2c,t1*np.ones(lengthm),w-v/2*np.ones(lengthm)-l/2,-t0*f2]
                              ,[t3*f2,t4*f2c,-t0*f2c,-v/2*np.ones(lengthm)-l/2]])
    hamiltonian3 = np.array([[v/2*np.ones(lengthm),-t0*f1,t4*f1,t3*f1c]
                              ,[-t0*f1c,w+v/2*np.ones(lengthm),t1*np.ones(lengthm),t4*f1]
                              ,[t4*f1c,t1*np.ones(lengthm),w-v/2*np.ones(lengthm)-l/2,-t0*f1]
                              ,[t3*f1,t4*f1c,-t0*f1c,-v/2*np.ones(lengthm)-l/2]])
    hamiltonian4 = np.array([[v/2*np.ones(lengthm),-t0*f2,t4*f2,t3*f2c]
                              ,[-t0*f2c,w+v/2*np.ones(lengthm),t1*np.ones(lengthm),t4*f2]
                              ,[t4*f2c,t1*np.ones(lengthm),w-v/2*np.ones(lengthm)+l/2,-t0*f2]
                              ,[t3*f2,t4*f2c,-t0*f2c,-v/2*np.ones(lengthm)+l/2]])
    energy_1, energy_2, energy_3, energy_4,vectors1,vectors2,vectors3,vectors4 = dispersion(v,hamiltonian1,hamiltonian2,hamiltonian3,hamiltonian4)
    
    # Save the tight-binding parameters to a JSON file for reproducibility
    path = os.getcwd() + '/../../'
    path = os.path.abspath(path)
    
    output_path = path + "/tight_binding.json"
    if not os.path.exists(output_path):
        import json
        
        params = {"t0":t0,"t1":t1,"t3":t3,"t4":t4,"delta":w,"spin_orbit":l}
        
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
    
    return np.array([energy_1,energy_2,energy_3,energy_4]),np.array([vectors1,vectors2,vectors3,vectors4])