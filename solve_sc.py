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
import jax
import jax.numpy as jnp
from jaxopt import ScipyRootFinding


def fermi_no_jit(e,mu,T):
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
    """
    return 1/(1+jnp.exp((e-mu)/T))

fermi = jax.jit(fermi_no_jit)

def particle_num_no_jit(mu,d,e,num,t):
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
    result = num - jnp.sum(fermi(e-d,mu,t))
    return result

particle_num = jax.jit(particle_num_no_jit)

def fixed_log_no_jit(mu,e,t):
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
        Stable logarithm values for free energy
    """
    return jnp.where(mu-e>0,(mu-e)/t+jnp.log(1+jnp.exp((e-mu)/t)), jnp.log(1+jnp.exp((mu-e)/t)))

fixed_log = jax.jit(fixed_log_no_jit)

def loops(dlast,mu,e,interaction,t):
    """
    Compute updated order parameter from self-consistency equation.
    
    This is the core self-consistent field calculation. For each momentum point,
    the new order parameter is computed as the sum over all momentum points of
    the interaction matrix weighted by the occupation. The interaction is computed using 
    a screened Coulomb potential with a hyperbolic tangent cutoff.
    
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
    vector : ndarray, shape (lengthm, lengthm)
        Interaction matrix V(k, k')
    t : float
        Temperature
    momenta : ndarray, shape (lengthm, 2)
        Momentum grid points
    u : float
        Prefactor determining interaction strength
    d_gate : float
        Gate distance for screening
    a : float
        Lattice constant
    
    Returns
    -------
    ndarray, shape (lengthm,)
        Updated order parameter: d_new(k) = Σ_k' V(k,k') * f(E(k') - d_old(k'))
    
    Notes
    -----
    First loop is parallelized over momentum points.
    """
    fermi_vals = fermi(e-dlast,mu,t)
    # Ensure the interaction matrix is contiguous for efficient access by MSI cluster
    return interaction @ fermi_vals

def get_order_parameters(num,lengthm,energy,interaction,initial_guess,t):
    """
    Solve self-consistent Hartree-Fock equations iteratively.
    
    This function implements the main SCHF loop:
    1. Initialize order parameter with initial guess
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
    interaction : ndarray, shape (n_bands, 4, lengthm)
        Interaction matrix with coherence factors
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
    """

    dinitial = jnp.copy(initial_guess)
    particle_solver = ScipyRootFinding(optimality_fun=particle_num, method="hybr")
    # Calculate non-interacting Fermi level (for reference)
    ef_norm = particle_solver.run(init_params=jnp.min(energy[0]), 
                                  d = jnp.zeros_like(dinitial), 
                                  e = energy, 
                                  num = num, 
                                  t = t).params

    # Initial chemical potential with interactions
    ef = particle_solver.run(init_params=jnp.min(energy-dinitial), 
                                d = dinitial, 
                                e = energy, 
                                num = num, 
                                t = t).params

    # Initialize convergence tracking
    maxerror = [1]
    best_max_error = 10000
    i = 0
    j = 0
    # Calculate initial free energy
    total_energy = ef*num + jnp.sum(1/2*dinitial*fermi(energy-dinitial,ef,t)-t*fixed_log(ef,energy-dinitial,t))
    # Self-consistent iteration loop
    while best_max_error>1e-5 and i<250:
        # Update order parameter using self-consistency equation
        d = jnp.array([loops(x,ef,e,inter,t) for x,e,inter in zip(dinitial,energy,interaction)])
        # Compute convergence metric (maximum pointwise change)
        maxerror.append(jnp.max(jnp.abs(d-dinitial)))
        dinitial = d
        # Update chemical potential to maintain particle number
        ef = particle_solver.run(init_params=jnp.min(energy-dinitial), 
                                    d = dinitial, 
                                    e = energy, 
                                    num = num, 
                                    t = t).params
        if maxerror[i+1]<best_max_error:
            # Save the best converged values
            true_d = d
            true_ef = ef
            j = 0
            best_max_error = maxerror[i+1]
            total_energy = ef*num + jnp.sum(1/2*dinitial*fermi(energy-dinitial,ef,t)-t*fixed_log(ef,energy-dinitial,t))
        else:
            j+=1
        # Early stopping if no improvement for 20 iterations
        if j >= 20:
            break
        i += 1
    return true_d,true_ef,ef_norm,best_max_error,float(total_energy)

def main(lengthm,l,energy,total_number,interaction,t,initial_guess):
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
    interaction : ndarray, shape (n_bands, 4, lengthm)
        Interaction, with coherence factors (meV)
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
    """

    # Convert density (cm^-2) to absolute particle number
    number = l**2*5.24e-16*total_number
    # Solve self-consistent equations
    d,ef,ef_norm,maxerror,total_energy = get_order_parameters(number,
                                                              lengthm,
                                                              energy,
                                                              interaction,
                                                              initial_guess,
                                                              t)
    # Calculate occupation fraction for each isospin
    occupation = np.array([float(np.sum(fermi(e-x,ef,t))/number) for e,x in zip(energy,d)])
    return np.array(d),ef,ef_norm,maxerror,occupation,total_energy

