# =============================================================
# LAMMPS Monopole Input Generation
# Author: Ethan L. Edmunds
# Version: v1.0
# Description: Python script to produce input for precipitate calculations.
# Note: Dislocation is aligned along Z, glide plane along X axis, climb plane is Y axis.
# Command: apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 02_input/run.py
# =============================================================

# =============================================================
# IMPORT LIBRARIES
# =============================================================
import os
import numpy as np
import subprocess

from matscipy.calculators.eam import EAM
from matscipy.dislocation import get_elastic_constants

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data')) # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '02_input')) # Stage data directory

OUTPUT_DIR = os.path.join(STAGE_DATA_DIR, 'output') # Output folder
DUMP_DIR = os.path.join(STAGE_DATA_DIR, 'dump') # Dump folder
LOG_DIR = os.path.join(STAGE_DATA_DIR, 'logs') # Log folder

for directory in [OUTPUT_DIR, DUMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

POTENTIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_potentials')) # Potentials Directory
POTENTIAL_FILE = os.path.join(POTENTIALS_DIR, 'malerba.fs') # Potential file

# =============================================================
# SIMULATION PARAMETERS
# =============================================================
# Lattice dimensions (angstrom)
X_LEN = 100  # LENGTH ALONG X
Y_LEN = 30  # LENGTH ALONG Y
Z_LEN = 30  # LENGTH ALONG Z

X_DIR = '[111]'
Z_DIR = '[1-10]'
Y_DIR = '[11-2]'

# =============================================================
# FILENAMES
# =============================================================

OUTPUT_FILENAME = f'edge_dislo_{X_LEN}_{Y_LEN}_{Z_LEN}.lmp'

UNITCELL = 'unitcell.cfg'
TOP = 'top.cfg'
BOTTOM = 'bottom.cfg'
TMP_FILE = OUTPUT_FILENAME

OUTPUT_PATH = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

# =============================================================
# MAIN FUNCTION
# =============================================================
def main():

    # ---------------------------
    # Check if files are removed
    # ---------------------------
    for file in [UNITCELL, TOP, BOTTOM, TMP_FILE]:
        if os.path.exists(file):
            os.remove(file)

    # ---------------------------
    # Load EAM potential
    # ---------------------------
    eam_calc = EAM(POTENTIAL_FILE)

    # Get lattice constant and elastic constants for Fe
    alat, C11, C12, C44 = get_elastic_constants(calculator=eam_calc, symbol="Fe", verbose=True)

    # ---------------------------
    # Define dislocation
    # ---------------------------

    subprocess.run(['atomsk', '--create', 'bcc', str(alat), 'Fe', 'orient', X_DIR, Y_DIR, Z_DIR, UNITCELL])

    subprocess.run(['atomsk', UNITCELL, '-duplicate', str(X_LEN), str(Y_LEN), str(Z_LEN), '-deform', 'X', str(0.5/X_LEN), '0.0', BOTTOM])

    subprocess.run(['atomsk', UNITCELL, '-duplicate', str(X_LEN+1), str(Y_LEN), str(Z_LEN), '-deform', 'X', str(-0.5/(X_LEN+1)), '0.0', TOP])

    subprocess.run(['atomsk', '--merge', 'Y', '2', BOTTOM, TOP, TMP_FILE])

    subprocess.run(['cp', TMP_FILE, OUTPUT_PATH])

    # ---------------------------
    # Remove files
    # ---------------------------
    for file in [UNITCELL, TOP, BOTTOM, TMP_FILE]:
        if os.path.exists(file):
            os.remove(file)

# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":
    main()
