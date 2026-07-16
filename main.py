"""
Phase Diagram Solver for Bernal Bilayer Graphene with Self-Consistent Hartree-Fock Theory

This script systematically explores the ground state phase diagram of Bernal bilayer
graphene by testing multiple competing order parameter configurations. It utilizes 
transfer learning by creating trial states solved for a single isospin, and using those 
trials states as a starting point for the full four-isospin self-consistent Hartree-Fock solver. 
This is done across a range of applied perpendicular electric fields and total particle densities 
to create a phase diagram for the system.

The algorithm tests 18 different competing phases including:
    - Valley-polarized states (occupation in K or K' valley only)
    - Spin-polarized states (ferromagnetism)
    - Combinations of the above with nematic Fermi surfaces

Output:
    All results saved to results/ directory:
        - delta.npy: Converged order parameters for all configurations
        - ef.npy: Chemical potentials
        - ef_norm.npy: Non-interacting Fermi levels
        - best_max_error.npy: Convergence errors
        - occupation.npy: Band occupation fractions
        - total_energy.npy: Total free energies (use to determine ground state)
        - momenta.npy: Momentum grid points used in calculations

Usage:
    This file is intended to be run through a job submission script on an HPC cluster. It takes three command line arguments:
        1. Applied perpendicular electric field (v) in meV
        2. Total particle density (total_number) in cm^-2
        3. Interaction strength scaling factor (er) to adjust the effective interaction strength
    These parameters are all entered via the looprun.sh script, which submits multiple jobs to the cluster for different parameter combinations.
"""


import numpy as np
from get_dispersion_one_v import getting_energies
from solve_sc import main, particle_num
import os
import sys
import matplotlib.pyplot as plt
import scipy.optimize as opt
from scipy.interpolate import griddata

# ============================================================================
# Define Brillouin Zone Momentum Grid
# ============================================================================
# Reciprocal lattice vectors for hexagonal lattice
b1 = np.array([2*np.pi, 2*np.pi/np.sqrt(3)])
b2 = np.array([2*np.pi,-2*np.pi/np.sqrt(3)])

# Initial system size and momentum cutoff for the Brillouin zone sampling
L = 1000*6
cutoff = .12
# Generate momentum grid with circular cutoff
momenta = np.array([(n1*b1+n2*b2)/L for n1 in range(-200,200) for n2 in range(-200,200) if np.linalg.norm((n1*b1+n2*b2)/L)<cutoff])

lengthm = len(momenta)


# ============================================================================
# Physical Parameters
# ============================================================================
t = .005 #Temperature in meV
d_gate = 20 # Gate distance to sample in nm
v = float(sys.argv[1]) # Applied perpendicular electric field in meV
total_number = float(sys.argv[2]) # Total particle density in cm^-2
er = float(sys.argv[3]) # Dielectric constant used to scale the effective interaction strength
epsilon = 55.26349406/(1000*1000) # Permittivity of free space in meV/nm
a = .246 # Lattice constant in nm

energy,_ = getting_energies(v,momenta) # Get non-interacting energy dispersions and eigenvectors for the four isospin configurations for dense momentum grid

ef = opt.fsolve(particle_num,[np.min(energy)],args=(np.zeros_like(energy[0]),energy[0],L**2*5.24e-16*total_number,t))[0] # Chemical potential for the non-interacting system at the given density and temperature

kx = momenta[:,0]
ky = momenta[:,1]

# ============================================================================
# Getting Adaptive Momentum Cutoff and System Size
# ============================================================================

grid_x = np.linspace(kx.min(), kx.max(), 400)
grid_y = np.linspace(ky.min(), ky.max(), 400)
X, Y = np.meshgrid(grid_x, grid_y)
Z = griddata((kx, ky), energy[0], (X, Y), method='cubic')
contour = plt.contour(X,Y,Z,[ef])
plt.close()
segs = contour.allsegs[0]

try:
    cutoff = max([np.max(x) for x in segs])*1.1
except:
    cutoff = .12
L = np.around(np.sqrt(20000*8*np.pi/np.sqrt(3))/cutoff*1/6)*6
momenta = np.array([(n1*b1+n2*b2)/L for n1 in range(-200,200) for n2 in range(-200,200) if np.linalg.norm((n1*b1+n2*b2)/L)<cutoff])

del grid_x, grid_y, X, Y, Z, contour, kx, ky

A = L**2*.0524 # Sample area in nm^2

u = a/(2*epsilon*er*A) # Interaction strength prefactor in meV

lengthm = len(momenta)
# Get non-interacting energy dispersions and eigenvectors for the four isospin configurations for adaptive momentum grid
energy,vectors = getting_energies(v,momenta)

# ============================================================================
# Get Trial States for Each Isospin Configuration
# ============================================================================
polarized = np.array([u*total_number/4*(2.46e-8)**2*np.ones(lengthm)])
full_metal_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),total_number/4*.9,np.array([vectors[0]]),t,polarized,momenta,u,d_gate,a)
full_metal_k_prime,_,_,_,_,_ = main(lengthm,L,np.array([energy[1]]),total_number/4*.9,np.array([vectors[1]]),t,polarized,momenta,u,d_gate,a)
half_metal_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),total_number/2*.9,np.array([vectors[0]]),t,polarized,momenta,u,d_gate,a)
quarter_metal_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),total_number*.9,np.array([vectors[0]]),t,polarized,momenta,u,d_gate,a)
three_quarter_metal_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),total_number/3*.9,np.array([vectors[0]]),t,polarized,momenta,u,d_gate,a)
three_quarter_metal_k_prime,_,_,_,_,_ = main(lengthm,L,np.array([energy[1]]),total_number/3*.9,np.array([vectors[1]]),t,polarized,momenta,u,d_gate,a)

nematic_guess = np.array([momenta[:,0]])
one_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),.25e10,np.array([vectors[0]]),t,nematic_guess,momenta,u,d_gate,a)
one_k_prime,_,_,_,_,_ = main(lengthm,L,np.array([energy[1]]),.25e10,np.array([vectors[1]]),t,-nematic_guess,momenta,u,d_gate,a)
two_k,_,_,_,_,_ = main(lengthm,L,np.array([energy[0]]),.25e10,np.array([vectors[0]]),t,-nematic_guess,momenta,u,d_gate,a)
two_k_prime,_,_,_,_,_ = main(lengthm,L,np.array([energy[1]]),.25e10,np.array([vectors[1]]),t,nematic_guess,momenta,u,d_gate,a)

empty = np.zeros_like(full_metal_k)

# Building the initial guesses for the 18 competing phases to be tested in the self-consistent Hartree-Fock solver
initial_guess = np.array([
    [quarter_metal_k,empty,empty,empty],
    [one_k,empty,empty,empty],
    [two_k,empty,empty,empty],
    [quarter_metal_k,full_metal_k_prime,full_metal_k,full_metal_k_prime],
    [quarter_metal_k,one_k_prime,one_k,one_k_prime],
    [quarter_metal_k,two_k_prime,two_k,two_k_prime],
    [quarter_metal_k,empty,full_metal_k,empty],
    [quarter_metal_k,empty,one_k,empty],
    [quarter_metal_k,empty,two_k,empty],
    [half_metal_k,empty,half_metal_k,empty],
    [half_metal_k,full_metal_k_prime,half_metal_k,full_metal_k_prime],
    [half_metal_k,one_k_prime,half_metal_k,one_k_prime],
    [half_metal_k,two_k_prime,half_metal_k,two_k_prime],
    [three_quarter_metal_k,three_quarter_metal_k_prime,three_quarter_metal_k,empty],
    [three_quarter_metal_k,three_quarter_metal_k_prime,three_quarter_metal_k,full_metal_k_prime],
    [full_metal_k,full_metal_k_prime,full_metal_k,full_metal_k_prime],
    [one_k,one_k_prime,one_k,one_k_prime],
    [two_k,two_k_prime,two_k,two_k_prime],
])[:,:,0]

# ============================================================================
# Solving for Ground State
# ============================================================================
# Initialize arrays to hold the results
d = np.zeros_like(initial_guess)
ef = np.zeros(initial_guess.shape[0])
ef_norm = np.zeros(initial_guess.shape[0])
best_max_error = np.zeros(initial_guess.shape[0])
occupation = np.zeros((initial_guess.shape[0],energy.shape[0]))
total_energy = np.zeros(initial_guess.shape[0])

# Run the self-consistent Hartree-Fock solver for each competing phase
for j in range(initial_guess.shape[0]):
    dinitial = initial_guess[j]
    d[j],ef[j],ef_norm[j],best_max_error[j],occupation[j],total_energy[j] = main(lengthm,L,energy,total_number,vectors,t,dinitial,momenta,u,d_gate,a)

# Getting variables that correspond to the lowest energy solution across all competing phases
d = d[np.argmin(total_energy)]
ef = ef[np.argmin(total_energy)]
occupation = occupation[np.argmin(total_energy)]
ef_norm = ef_norm[0]
best_max_error = best_max_error[np.argmin(total_energy)]

total_energy = np.min(total_energy)

# ============================================================================
# Saving Results
# ============================================================================
# Move back two folders
path = os.getcwd() + '/../../'
path = os.path.abspath(path)

# Creating save directory
directory = os.path.join(path, 'data_files'+str(int(er))+str(d_gate)+'/')

# Save all relevant data to the specified directory
data_to_save = {
    'd': d,
    'ef': ef,
    'ef_norm': ef_norm,
    'best_max_error': best_max_error,
    'occupation': occupation,
    'total_energy': total_energy,
    'momenta': momenta
}
os.makedirs(directory,exists_ok=True)
# Save each data object in its own subdirectory for organization
for name, data_object in data_to_save.items():
    target_subdir = os.path.join(directory, name)
    os.makedirs(target_subdir, exist_ok=True)
    filename = f"{v}${total_number}.npy"
    full_save_path = os.path.join(target_subdir, filename)
    np.save(full_save_path, data_object)

output_path = os.path.join(path, "params.json")

if not os.path.exists(output_path):
    import json
    # Save the simulation parameters to a JSON file
    params = {"L":L,
    "cutoff":cutoff,
    "epsilon":er,
    "t":t,
    "d_gate":d_gate,
    "version":"v2"}
    
    with open(output_path, "w") as f:
        json.dump(params, f, indent=2)