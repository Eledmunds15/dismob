#!/bin/bash
#SBATCH --job-name=simulation
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=02:30:00
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=256
#SBATCH --exclude=node132

module load OpenMPI/4.1.4-GCC-12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export APPTAINER_TMPDIR=$HOME/.apptainer_tmp
export APPTAINER_CACHEDIR=$HOME/.apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

HOST_MPI_PATH="/opt/apps/testapps/el7/software/staging/OpenMPI/4.1.4-GCC-12.2.0"

CONTAINER_PATH="/mnt/parscratch/users/mtp24ele/dismob/00_envs/lmp_CPU_22Jul2025.sif"

INPUT="/mnt/parscratch/users/mtp24ele/dismob/03_shear/run.py"

srun --export=ALL bash -c '
    cd /mnt/parscratch/users/mtp24ele/dismob || { echo "Directory not found on $(hostname)"; exit 1; }
    sleep $((RANDOM % 10))
    apptainer exec \
        --bind '"$HOST_MPI_PATH:$HOST_MPI_PATH"' \
        '"$CONTAINER_PATH"' \
        python3 '"$INPUT"'
'
