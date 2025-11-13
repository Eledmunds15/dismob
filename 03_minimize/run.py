# =============================================================
# LAMMPS Monopole Input Generation
# Author: Ethan L. Edmunds
# Version: v1.5
# Description: Python script to produce input for precipitate calculations.
# Note: Dislocation is aligned along X, glide plane along Y axis.
# Command (single): apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun.openmpi -np 4 /opt/venv/bin/python3 03_minimize/run.py --config 000_data/02_input/output/edge_dislo_100_30_30.lmp
# Command (suite):  apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun.openmpi -np 4 /opt/venv/bin/python3 03_minimize/run.py --config 000_data/02_input/output --suite
# =============================================================

# =============================================================
# IMPORT LIBRARIES
# =============================================================
import sys, os, subprocess, argparse
import numpy as np
from mpi4py import MPI
from lammps import lammps

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
params = {}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# PATH SETTINGS
# =============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data'))  # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '03_minimize'))  # Stage data directory

OUTPUT_DIR = os.path.join(STAGE_DATA_DIR, 'output')  # Output folder
DUMP_DIR = os.path.join(STAGE_DATA_DIR, 'dump')      # Dump folder
LOG_DIR = os.path.join(STAGE_DATA_DIR, 'logs')       # Log folder

for directory in [OUTPUT_DIR, DUMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

POTENTIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_potentials'))  # Potentials Directory
POTENTIAL_FILE = os.path.join(POTENTIALS_DIR, 'malerba.fs')  # Potential file

# =============================================================
# SIMULATION PARAMETERS
# =============================================================
ENERGY_TOL = 1e-6  # Energy tolerance for minimization
FORCE_TOL = 1e-8   # Force tolerance for minimization
BUFF = 2

# =============================================================
# RUN SINGLE FILE
# ============================================================
def lammpsSim(input_file: str):
    """Run LAMMPS minimization for a single input file."""

    if not input_file:
        raise FileNotFoundError(f"Input file not found: {input_file}")

    base_name = os.path.basename(input_file)
    name_no_ext = os.path.splitext(base_name)[0]

    LOG_PATH = os.path.join(LOG_DIR, f"log_{name_no_ext}.lammps")
    DUMP_PATH = os.path.join(DUMP_DIR, base_name + "_m")
    OUTPUT_PATH = os.path.join(OUTPUT_DIR, base_name)

    if rank == 0:
        print(f"\n[INFO] 🔹 Running minimization for: {base_name}")

    # ---------- Initialize LAMMPS ----------------------------
    lmp = lammps(comm=comm)
    lmp.cmd.clear()
    lmp.cmd.log(LOG_PATH)

    lmp.cmd.units('metal')
    lmp.cmd.dimension(3)
    lmp.cmd.boundary('p', 'f', 'p')

    lmp.cmd.read_data(input_file)
    lmp.cmd.pair_style('eam/fs')
    lmp.cmd.pair_coeff('*', '*', POTENTIAL_FILE, 'Fe')

    lmp.cmd.neighbor(2.0, "bin")
    lmp.cmd.neigh_modify("delay", 10, "check", "yes")

    lmp.cmd.group('all', 'type', '1')
    lmp.cmd.compute('peratom', 'all', 'pe/atom')

    lmp.cmd.minimize(ENERGY_TOL, FORCE_TOL, 1000, 10000)

    lmp.cmd.write_dump('all', 'custom', DUMP_PATH, 'id', 'x', 'y', 'z', 'c_peratom')
    lmp.cmd.write_data(OUTPUT_PATH)

    if rank == 0:
        print(f"\n[DONE] ✅ Output written to {OUTPUT_PATH}")
        print(f"[LOG]  🧾 Log file: {LOG_PATH}\n")

    lmp

    return None


import os
import argparse

# =============================================================
# INPUT COMMAND STUFF
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Dislocation Minimisation with LAMMPS")

    group = parser.add_mutually_exclusive_group(required=True)

    group.add_argument(
        "--config",
        type=str,
        help="Path to input .lmp file"
    )

    group.add_argument(
        "--suite",
        type=str,
        help="Path to directory containing multiple .lmp files"
    )

    return parser.parse_args()


# =============================================================
# ENTRY POINT
# =============================================================
if __name__ == "__main__":

    args = parse_args()

    # --- Suite mode ---
    if args.suite:
        full_path = os.path.abspath(os.path.expanduser(args.suite))

        if not os.path.isdir(full_path):
            raise NotADirectoryError(f"[ERROR] Suite directory not found: {full_path}")

        lmp_files = [
            os.path.join(full_path, f)
            for f in os.listdir(full_path)
            if f.endswith(".lmp")
        ]

        if not lmp_files:
            raise FileNotFoundError(f"[ERROR] No .lmp files found in directory: {full_path}")

        print(f"[INFO] 🔹 Running minimization suite with {len(lmp_files)} files in {full_path}")

        for file_path in lmp_files:
            print(f"[INFO] 🔸 Running: {os.path.basename(file_path)}")
            lammpsSim(file_path)

    # --- Single-file mode ---
    elif args.config:
        full_path = os.path.abspath(os.path.expanduser(args.config))

        if not os.path.isfile(full_path):
            raise FileNotFoundError(f"[ERROR] File not found: {full_path}")

        if not full_path.endswith(".lmp"):
            raise ValueError(f"[ERROR] Expected a .lmp file but got: {full_path}")

        print(f"[INFO] 🔹 Running single minimization for: {os.path.basename(full_path)}")
        lammpsSim(full_path)
