#!/bin/bash
#SBATCH --job-name=analysis
#SBATCH --mail-user=eledmunds1@sheffield.ac.uk
#SBATCH --time=00:10:00
#SBATCH --mem-per-cpu=5GB
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=40
#SBATCH --exclude=node052

CONTAINER_PATH="/mnt/parscratch/users/mtp24ele/dismob/00_envs/lmp_CPU_22Jul2025.sif"
INPUT="/mnt/parscratch/users/mtp24ele/dismob/05_post_processing/analysis.py"

CASE_DIR="stP_dtE_T900_FS1_R15_D5_S7252"
STUDY_NAME="precipitate_climb"

# Setting up environment
module load OpenMPI/4.1.4-GCC-12.2.0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

export APPTAINER_TMPDIR=$HOME/.apptainer_tmp
export APPTAINER_CACHEDIR=$HOME/.apptainer_cache
mkdir -p "$APPTAINER_TMPDIR" "$APPTAINER_CACHEDIR"

HOST_MPI_PATH=/opt/apps/testapps/el7/software/staging/OpenMPI/4.1.4-GCC-12.2.0

CASE_PATH="/mnt/parscratch/users/mtp24ele/dismob/000_data/04_shear/${STUDY_NAME}/${CASE_DIR}"

# Run Command
srun --export=ALL \
     apptainer exec \
     --bind $HOST_MPI_PATH:$HOST_MPI_PATH \
     $CONTAINER_PATH \
     python3 $INPUT \
     --config $CASE_PATH \
     --dxa