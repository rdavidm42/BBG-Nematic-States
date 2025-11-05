"""
Phase Diagram Solver for Bilayer Graphene Hartree-Fock Theory

This script systematically explores the ground state phase diagram of Bernal bilayer
graphene by testing multiple competing order parameter configurations. It utilizes 
transfer learning by loading pre-computed trial states and determines which 
configuration minimizes the total energy across a 2D parameter space (electric field 
vs. particle density).

The algorithm tests 18 different competing phases including:
    - Valley-polarized states (occupation in K or K' valley only)
    - Spin-polarized states (ferromagnetism)
    - Combinations of the above with different Fermi surface topologies

Output:
    All results saved to results/ directory:
        - normal_state_bands.npy: Energy of system in the absence of interactions
        - delta.npy: Converged order parameters for all configurations
        - ef.npy: Chemical potentials
        - ef_norm.npy: Non-interacting Fermi levels
        - error.npy: Convergence errors
        - occupation.npy: Band occupation fractions
        - best_energies.npy: Total energies (use to determine ground state)

Usage:
    python solve_gap_equation.py
    
    Prerequisites: Must run get_trial_states.py first to generate initial states
    Typical runtime: 6-12 hours on HPC cluster
"""

import numpy as np
import os
from get_dispersion import getting_energies
from hartree_fock_solver import main

# ============================================================================
# Setup: Create output directory
# ============================================================================
try:
    os.mkdir('results')
except FileExistsError:
    pass

directory = 'results/'

# ============================================================================
# Define Brillouin Zone Momentum Grid
# ============================================================================
# Reciprocal lattice vectors for hexagonal lattice
b1 = np.array([2*np.pi, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi, -2*np.pi/np.sqrt(3)])

# System size (must match get_trial_states.py)
L = 250 * 6  # System size in terms of multiples of the unit cell

# Generate momentum grid: circular cutoff in Brillouin zone
momenta = np.array([
    (n1*b1 + n2*b2) / L 
    for n1 in range(-200, 200) 
    for n2 in range(-200, 200) 
    if np.linalg.norm((n1*b1 + n2*b2) / L) < 4*np.pi/3 * 0.06
])

lengthm = len(momenta)


# ============================================================================
# Physical Parameters
# ============================================================================
# Material and interaction parameters
epsilon = 55.26349406 / (1000 * 1000)  # Effective dielectric constant
a = 0.246  # Lattice constant (nm)
A = L**2 * 0.052  # System area (nm^2)

# Coulomb interaction strength
u_calculated = a / (2 * epsilon * A)
u = u_calculated / 10  # Account for screening

# Simulation parameters
t = 0.005  # Temperature (meV) - effectively zero temperature
d_gate = 200  # Gate screening distance (nm)


# ============================================================================
# Parameter Space: Electric Field and Density
# ============================================================================
# Density sweep: 1×10^10 to 10×10^11 cm^-2
lengthn = 35
total_number = np.linspace(1e10, 10e11, lengthn)

# Electric field sweep: 25 to 200 meV
lengthv = 35
v = np.linspace(25, 200, lengthv)

# ============================================================================
# Load Pre-computed Trial States
# ============================================================================

# Single and double pocket configurations (K and K' valleys)
one_k = np.load('trial_states/one_pocket_k.npy')
two_k = np.load('trial_states/two_pocket_k.npy')
one_k_prime = np.load('trial_states/one_pocket_k_prime.npy')
two_k_prime = np.load('trial_states/two_pocket_k_prime.npy')

# Three-pocket and large-pocket configurations
three_k = np.load('trial_states/three_pockets_k_hex.npy')
three_k_prime = np.load('trial_states/three_pockets_k_prime_hex.npy')
large_k = np.load('trial_states/large_pocket_k_hex.npy')
large_k_prime = np.load('trial_states/large_pocket_k_prime_hex.npy')

# Empty configuration (no particles in that band)
empty = np.zeros((lengthv, lengthm))

# Compute Band Structure
energy, vectors = getting_energies(v, momenta)

np.save(directory + 'normal_state_bands.npy', energy)

# Setup Interaction Matrix
# Momentum transfer: |k - k'| with regularization to avoid division by zero
vec_diff = np.linalg.norm((momenta[None, :] - momenta[:, None]), axis=-1) + 0.00001

# Screened Coulomb interaction: V(q) = u*tanh(d_gate*|q|)/|q|
interaction0 = u * np.tanh(d_gate / a * vec_diff) / vec_diff


# ============================================================================
# Define Competing Phase Configurations
# ============================================================================
# Each row represents a different trial configuration across the 4 bands:
# [band_0 (K, spin up), band_1 (K', spin up), band_2 (K, spin down), band_3 (K', spin down)]
#
# Physical interpretation of configurations:
#   - three_k/three_k_prime: Three-pocket Fermi surface
#   - large_k/large_k_prime: Single large Fermi pocket
#   - one_k/one_k_prime: Single small pocket
#   - two_k/two_k_prime: Two pockets
#   - empty: No Fermi surface (band is unoccupied)

initial_guess = np.array([
    # Fully valley and spin-polarized states
    [three_k, empty, empty, empty],
    [large_k, empty, empty, empty],
    
    # Partially valley and spin-polarized states
    [large_k, three_k_prime, three_k, three_k_prime],
    [large_k, one_k_prime, one_k, one_k_prime],
    [large_k, two_k_prime, two_k, two_k_prime],
    
    # Fully Valley polarized (equivalent with spin polarized in mean field)
    [large_k, empty, three_k, empty],
    [large_k, empty, one_k, empty],
    [large_k, empty, two_k, empty],
    [large_k, empty, large_k, empty],
    
    # Partially Valley polarized
    [large_k, three_k_prime, large_k, three_k_prime],
    [large_k, one_k_prime, large_k, one_k_prime],
    [large_k, two_k_prime, large_k, two_k_prime],
    
    # Paritally valley polarized and fully spin polaried in one valley
    [large_k, large_k_prime, large_k, empty],
    [large_k, large_k_prime, large_k, three_k_prime],
    [large_k, large_k_prime, large_k, large_k_prime],
    
    # Symmetric valley occupations
    [three_k, three_k_prime, three_k, three_k_prime],
    [one_k, one_k_prime, one_k, one_k_prime],
    [two_k, two_k_prime, two_k, two_k_prime],
])

num_configs = initial_guess.shape[0]


# ============================================================================
# Initialize Storage Arrays
# ============================================================================
# Primary results arrays
d = np.moveaxis(np.zeros((lengthn, *initial_guess.shape)), 3, 0)
ef = np.zeros((lengthv, lengthn, num_configs))
ef_norm = np.zeros((lengthv, lengthn, num_configs))
occupation = np.zeros((lengthv, lengthn, num_configs, energy.shape[0]))

# Convergence tracking (nested lists for variable-length convergence histories)
best_max_error = [[[[] for i in range(lengthn)] 
                    for z in range(lengthv)] 
                   for j in range(num_configs)]
total_energy = [[[[] for i in range(lengthn)] 
                 for z in range(lengthv)] 
                for j in range(num_configs)]


# ============================================================================
# Main Computation Loop: Solve for All Configurations
# ============================================================================

total_iterations = lengthv * num_configs * lengthn

for z in range(lengthv):
    
    # Compute wavefunction-weighted interaction matrix
    # Weight by eigenvectors to account for change from sublattice basis to band basis
    interaction = interaction0 * np.array([
        np.abs(vector[z].T.conj() @ vector[z])**2 
        for vector in vectors
    ])
    
    for j in range(num_configs):
        
        # Extract initial order parameter for this configuration
        dinitial = initial_guess[j, :, z]
        
        # Sweep over particle densities
        for i in range(lengthn):
            # Solve self-consistent Hartree-Fock equations
            (d[z, i, j], ef[z, i, j], ef_norm[z, i, j], 
             best_max_error[j][z][i], occupation[z, i, j], 
             total_energy[j][z][i]) = main(
                lengthm, L, energy[:, z], total_number[i], 
                interaction, t, dinitial
            )
# Get final energy and error from convergence history (last element of each list)
best_energies = np.array([[[x[-1] for x in y] for y in z] for z in total_energy])
errors = np.array([[[x[-1] for x in y] for y in z] for z in best_max_error])

# ============================================================================
# Save Results
# ============================================================================
np.save(directory + 'delta.npy', d)
np.save(directory + 'ef.npy', ef)
np.save(directory + 'ef_norm.npy', ef_norm)
np.save(directory + 'error.npy', errors)
np.save(directory + 'occupation.npy', occupation)
np.save(directory + 'best_energies.npy', best_energies)
