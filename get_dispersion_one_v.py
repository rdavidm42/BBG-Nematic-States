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
import jax.numpy as jnp
import os

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
    
    h1_batched = jnp.transpose(hamiltonian1, (2, 0, 1))
    h2_batched = jnp.transpose(hamiltonian2, (2, 0, 1))
    h3_batched = jnp.transpose(hamiltonian3, (2, 0, 1))
    h4_batched = jnp.transpose(hamiltonian4, (2, 0, 1))

    vals1, vecs1 = jnp.linalg.eigh(h1_batched)
    vals2, vecs2 = jnp.linalg.eigh(h2_batched)
    vals3, vecs3 = jnp.linalg.eigh(h3_batched)
    vals4, vecs4 = jnp.linalg.eigh(h4_batched)

    # 4. Extract specific eigenvalues (lengthm,) and eigenvectors (lengthm, 4)
    energy_1, vectors1 = vals1[:, 1], vecs1[:, :, 1]
    energy_2, vectors2 = vals2[:, 1], vecs2[:, :, 1]
    energy_3, vectors3 = vals3[:, 1], vecs3[:, :, 1]
    energy_4, vectors4 = vals4[:, 1], vecs4[:, :, 1]
    
    # Save the tight-binding parameters to a JSON file for reproducibility
    path = os.getcwd() + '/../../'
    path = os.path.abspath(path)
    
    output_path = path + "/tight_binding.json"
    if not os.path.exists(output_path):
        import json
        
        params = {"t0":t0,"t1":t1,"t3":t3,"t4":t4,"delta":w,"spin_orbit":l}
        
        with open(output_path, "w") as f:
            json.dump(params, f, indent=2)
    energies = jnp.array([-energy_1-v/2,-energy_2-v/2,-energy_3-v/2,-energy_4-v/2])
    vectors = jnp.array([jnp.transpose(vectors1),jnp.transpose(vectors2),jnp.transpose(vectors3),jnp.transpose(vectors4)])
    return energies, vectors