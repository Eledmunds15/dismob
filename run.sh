#!/bin/bash
#SBATCH --job-name=4-simulation
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=05:00:00
#SBATCH --mem-per-cpu=750MB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=128
#SBATCH --exclude=node052

CONTAINER_PATH="/mnt/parscratch/users/mtp24ele/dismob/00_envs/lmp_CPU_22Jul2025.sif"
INPUT="/mnt/parscratch/users/mtp24ele/dismob/05_shear/run.py"

STUDY_NAME="112_disl_200K"
CONFIG_FILE="4_stN_dtE_T200_FS500_S7811.yaml"

# Setting up environment
module load OpenMPI/4.1.4-GCC-12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export APPTAINER_TMPDIR=$HOME/.apptainer_tmp
export APPTAINER_CACHEDIR=$HOME/.apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

HOST_MPI_PATH="/opt/apps/testapps/el7/software/staging/OpenMPI/4.1.4-GCC-12.2.0"

CONFIG_PATH="/mnt/parscratch/users/mtp24ele/dismob/000_data/05_shear/${STUDY_NAME}/configs/${CONFIG_FILE}"

# Run Command
srun --export=ALL \
    apptainer exec \
        --bind ${HOST_MPI_PATH}:${HOST_MPI_PATH} \
        ${CONTAINER_PATH} \
        python3 ${INPUT} \
        --config ${CONFIG_PATH}
