"""
Self-Consistent Hartree-Fock Solver for Bilayer Graphene

This module implements a self-consistent field (SCF) solver for the Hartree-Fock
equations in bilayer graphene. The algorithm iteratively solves for the order
parameter (interaction-induced gap) and chemical potential until convergence.

Key Components:
    - Fermi-Dirac distribution for finite temperature
    - Root finding for chemical potential (particle number conservation)
    - Self-consistent iteration loop with convergence monitoring
    - Total energy calculation
"""

import numpy as np
import numba
import scipy.optimize as opt


@numba.njit
def fermi(e, mu, T):
    """
    Compute the Fermi-Dirac distribution function.
    
    Parameters
    ----------
    e : float or ndarray
        Energy values
    mu : float
        Chemical potential (Fermi level)
    T : float
        Temperature in energy units (meV)
    
    Returns
    -------
    float or ndarray
        Occupation probability at given energy
    
    Notes
    -----
    This is the standard Fermi-Dirac distribution: f(E) = 1 / (1 + exp((E-μ)/T))
    """
    return 1 / (1 + np.exp((e - mu) / T))


@numba.njit
def particle_num(mu, d, e, num, t):
    """
    Constraint function for particle number conservation.
    
    This function computes the difference between the target particle number
    and the actual occupied states. Used with root finder to determine chemical
    potential that satisfies particle number constraint.
    
    Parameters
    ----------
    mu : float
        Chemical potential to test
    d : ndarray, shape (n_bands, lengthm)
        Current order parameter (interaction-induced gap)
    e : ndarray, shape (n_bands, lengthm)
        Energy dispersions
    num : float
        Target total particle number
    t : float
        Temperature
    
    Returns
    -------
    float
        Difference between target and actual particle number
    """
    result = num - np.sum(fermi(e - d, mu, t))
    return result


@numba.njit
def fixed_log(mu, e, t):
    """
    Numerically stable logarithm for free energy calculation.
    
    Computes log terms in the grand potential while avoiding numerical
    overflow for large arguments. Uses conditional evaluation to maintain
    stability.
    
    Parameters
    ----------
    mu : float
        Chemical potential
    e : ndarray
        Energy values
    t : float
        Temperature
    
    Returns
    -------
    ndarray
        Stable logarithm values for entropy term
    """
    return np.where(mu - e > 0,
                    (mu - e) / t + np.log(1 + np.exp((e - mu) / t)),
                    np.log(1 + np.exp((mu - e) / t)))


@numba.jit(nopython=True, parallel=True)
def loops(dlast, mu, e, lengthm, interaction, t):
    """
    Compute updated order parameter from self-consistency equation.
    
    This is the core self-consistent field calculation. For each momentum point,
    the new order parameter is computed as the sum over all momentum points of
    the interaction matrix weighted by the occupation.
    
    Parameters
    ----------
    dlast : ndarray, shape (lengthm,)
        Previous iteration's order parameter
    mu : float
        Chemical potential
    e : ndarray, shape (lengthm,)
        Energy dispersion
    lengthm : int
        Number of momentum points
    interaction : ndarray, shape (lengthm, lengthm)
        Interaction matrix V(k, k')
    t : float
        Temperature
    
    Returns
    -------
    ndarray, shape (lengthm,)
        Updated order parameter: d_new(k) = Σ_k' V(k,k') * f(E(k') - d_old(k'))
    
    Notes
    -----
    Parallelized over momentum points for HPC efficiency.
    """
    d = np.zeros(lengthm)
    for i in numba.prange(lengthm):
        d[i] = np.sum(fermi(e - dlast, mu, t) * interaction[i])
    return d


def get_order_parameters(num, lengthm, l, energy, interaction, initial_guess, t):
    """
    Solve self-consistent Hartree-Fock equations iteratively.
    
    This function implements the main SCF loop:
    1. Initialize order parameter (interaction gap)
    2. Find chemical potential satisfying particle number
    3. Update order parameter using self-consistency equation
    4. Repeat until convergence or maximum iterations
    
    Parameters
    ----------
    num : float
        Total particle number
    lengthm : int
        Number of momentum grid points
    l : float
        System size (for density calculations)
    energy : ndarray, shape (n_bands, lengthm)
        Non-interacting energy dispersions
    interaction : ndarray, shape (n_bands, lengthm, lengthm)
        Interaction matrices for each band
    initial_guess : ndarray, shape (n_bands, lengthm)
        Starting guess for order parameter
    t : float
        Temperature (meV)
    
    Returns
    -------
    true_d : ndarray
        Converged order parameter
    true_ef : float
        Converged chemical potential
    ef_norm : float
        Non-interacting Fermi level (for reference)
    maxerror : list
        Maximum error at each iteration (convergence history)
    total_energy : list
        Total energy at each iteration
    
    Notes
    -----
    Convergence criteria:
        - Maximum error < 1e-5 (self-consistency tolerance)
        - Maximum 250 iterations
        - Early stopping if error doesn't improve for 20 consecutive iterations
    
    The algorithm uses scipy.optimize.fsolve to find chemical potential at each
    iteration, ensuring particle number conservation.
    """
    dinitial = np.copy(initial_guess)
    
    # Calculate non-interacting Fermi level (for reference)
    ef_norm = opt.fsolve(particle_num, [np.min(energy[0])],
                        args=(np.zeros(dinitial.shape), energy, num, t))[0]
    
    # Initial chemical potential with interactions
    ef = opt.fsolve(particle_num, [np.min(energy - dinitial)],
                   args=(dinitial, energy, num, t))[0]

    # Initialize convergence tracking
    maxerror = [1]
    best_max_error = 10000
    i = 0  # Iteration counter
    j = 0  # Counter for iterations without improvement
    
    # Calculate initial total energy (kinetic + interaction contributions)
    total_energy = [np.sum((energy - dinitial/2) * fermi(energy - dinitial, ef, t))]
    
    iteration = 0
    
    # Self-consistent iteration loop
    while best_max_error > 1e-5 and i < 250:
        # Update order parameter using self-consistency equation
        # d_new(k) = Σ_k' V(k,k') * f(E(k') - d_old(k'))
        d = np.array([loops(x, ef, e, lengthm, inter, t) 
                     for x, e, inter in zip(dinitial, energy, interaction)])
        
        # Compute convergence metric (maximum pointwise change)
        maxerror.append(np.max(np.abs(d - dinitial)))
        dinitial = d
        
        # Update chemical potential to maintain particle number
        ef = opt.fsolve(particle_num, [np.min(energy - dinitial)],
                       args=(dinitial, energy, num, t))[0]
        
        # Calculate total energy: E_total = Σ_k (E_k - Δ_k/2) * f_k
        total_energy.append(np.sum((energy - dinitial/2) * 
                                  fermi(energy - dinitial, ef, t)))
        
        # Track best solution (in case of oscillations)
        if maxerror[i + 1] < best_max_error:
            true_d = d
            true_ef = ef
            j = 0
            best_max_error = maxerror[i + 1]
            iteration = i
        else:
            j += 1
        
        # Early stopping if no improvement for 20 iterations
        if j >= 20:
            break
        
        i += 1
    
    return true_d, true_ef, ef_norm, maxerror, total_energy


def main(lengthm, l, energy, total_number, interaction, t, initial_guess):
    """
    Main entry point for Hartree-Fock calculation.
    
    This wrapper function converts physical units and calls the SCF solver,
    then computes additional derived quantities like band occupation.
    
    Parameters
    ----------
    lengthm : int
        Number of momentum points
    l : float
        System size (nm)
    energy : ndarray, shape (n_bands, lengthm)
        Energy dispersions (meV)
    total_number : float
        Particle density (cm^-2)
    interaction : ndarray, shape (n_bands, lengthm, lengthm)
        Interaction matrices
    t : float
        Temperature (meV)
    initial_guess : ndarray
        Initial order parameter guess
    
    Returns
    -------
    d : ndarray
        Converged order parameter (meV)
    ef : float
        Chemical potential (meV)
    ef_norm : float
        Non-interacting Fermi level (meV)
    maxerror : list
        Convergence history
    occupation : ndarray
        Fractional occupation of each band
    total_energy : list
        Energy evolution during convergence
    
    Notes
    -----
    Conversion factor: 5.24e-16 converts system size (nm^2) to physical area
    and total_number (cm^-2) to absolute particle count.
    """
    # Convert density (cm^-2) to absolute particle number
    number = l**2 * 5.24e-16 * total_number
    
    # Solve self-consistent equations
    d, ef, ef_norm, maxerror, total_energy = get_order_parameters(
        number, lengthm, l, energy, interaction, initial_guess, t)
    
    # Calculate occupation fraction for each band
    occupation = np.array([np.sum(fermi(e - x, ef, t)) / number 
                          for e, x in zip(energy, d)])
    
    return d, ef, ef_norm, maxerror, occupation, total_energy