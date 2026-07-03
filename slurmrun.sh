#!/bin/bash
#SBATCH --job-name=one_job           # Job name
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=3           # Adjust this to the number of CPU cores you want to use
#SBATCH --mem=512mb                   # Job memory request
#SBATCH -p msismall                  # Partition name
#SBATCH --time=24:00:00              # Time limit hrs:min:sec
#SBATCH -o slurm_%j.out
#SBATCH -e slurm_%j.err
pwd; hostname; date
echo ${SLURM_JOBID}

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMBA_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

module load python3/3.10.9_anaconda2023.03_libmamba
python $1 $2 $3 $4