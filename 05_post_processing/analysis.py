# =============================================================
# LAMMPS Dislocation-Void Interaction Simulation
# Author: Ethan L. Edmunds
# Version: v1.2
# Description: Python script to perform DXA and/or Wigner-Seitz
#              analyses for dislocation-void interaction studies.
#
# Run examples:
#   # Run DXA only
#   apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 03_shear/analysis.py \
#       --config /abs/path/to/000_data/03_shear/void_case --dxa
#
#   # Run Wigner-Seitz only
#   apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 03_shear/analysis.py \
#       --config /abs/path/to/000_data/03_shear/void_case \
#       --wignerseitz /abs/path/to/000_data/02_minimize/dump/edge_dislo_100_30_40_dump
#
#   # Run both
#   apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 03_shear/analysis.py \
#       --config /abs/path/to/000_data/03_shear/void_case \
#       --dxa \
#       --wignerseitz /abs/path/to/000_data/02_minimize/dump/edge_dislo_100_30_40_dump
# =============================================================

# =============================================================
# IMPORT LIBRARIES
# =============================================================

import os
import re
import numpy as np
import argparse
from mpi4py import MPI
import traceback

from ovito.io import import_file, export_file
from ovito.modifiers import (
    DislocationAnalysisModifier,
    WignerSeitzAnalysisModifier,
    DeleteSelectedModifier,
    InvertSelectionModifier,
    ExpressionSelectionModifier,
)
from ovito.pipeline import FileSource

# =============================================================
# INITIALISE MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# ARGUMENT PARSING
# =============================================================
parser = argparse.ArgumentParser(
    description="LAMMPS Dislocation-Void Interaction Simulation"
)
parser.add_argument(
    "--config",
    type=str,
    required=True,
    help="Absolute path to the case directory (CASE_DIR)",
)
parser.add_argument(
    "--dxa",
    action="store_true",
    help="Enable Dislocation Extraction Analysis (DXA)",
)
parser.add_argument(
    "--wignerseitz",
    type=str,
    help="Enable Wigner-Seitz analysis and specify absolute path to reference file",
)
args = parser.parse_args()

# =============================================================
# VALIDATION
# =============================================================
CASE_DIR = os.path.abspath(args.config)

if rank == 0:
    if not os.path.isabs(CASE_DIR):
        raise ValueError(f"--config must be an absolute path. Received: {CASE_DIR}")
    if not os.path.exists(CASE_DIR):
        raise FileNotFoundError(f"Specified CASE_DIR does not exist: {CASE_DIR}")

    if not args.dxa and not args.wignerseitz:
        raise ValueError("You must specify at least one analysis: --dxa or --wignerseitz")

    if args.wignerseitz:
        REFERENCE_FILE = os.path.abspath(args.wignerseitz)
        if not os.path.isabs(REFERENCE_FILE):
            raise ValueError(f"--wignerseitz path must be absolute. Received: {REFERENCE_FILE}")
        if not os.path.exists(REFERENCE_FILE):
            raise FileNotFoundError(f"Reference file does not exist: {REFERENCE_FILE}")
    else:
        REFERENCE_FILE = None

comm.Barrier()

# Broadcast reference file path to all ranks (needed for WS)
REFERENCE_FILE = comm.bcast(args.wignerseitz if args.wignerseitz else None, root=0)

# =============================================================
# PATH SETTINGS
# =============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "000_data"))
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "03_shear"))

for directory in [STAGE_DATA_DIR, CASE_DIR]:
    if rank == 0:
        try:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory does not exist: {directory}")
        except Exception:
            print(f"[Rank {rank}] Error with directory: {directory}")
            traceback.print_exc()
            raise
    comm.Barrier()

DXA_DIR = os.path.join(CASE_DIR, "dxa")
DXA_SUMMARY_DIR = os.path.join(CASE_DIR, "dxa_summary")
DXA_ATOMS_DIR = os.path.join(CASE_DIR, "dxa_atoms")
WS_VAC_DIR = os.path.join(CASE_DIR, "wigner_seitz_vacs")
WS_SIA_DIR = os.path.join(CASE_DIR, "wigner_seitz_sias")

for directory in [DXA_DIR, DXA_SUMMARY_DIR, DXA_ATOMS_DIR, WS_VAC_DIR, WS_SIA_DIR]:
    os.makedirs(directory, exist_ok=True)

DATA_DIR = os.path.abspath(os.path.join(CASE_DIR, "dump"))

if rank == 0:
    print(f"\n[INFO] Using CASE_DIR: {CASE_DIR}")
    if args.dxa:
        print("[INFO] DXA analysis enabled.")
    if args.wignerseitz:
        print(f"[INFO] Wigner-Seitz analysis enabled with reference: {REFERENCE_FILE}")
    print("")

# =============================================================
# MAIN FUNCTION
# =============================================================

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    dump_files = None
    if rank == 0:
        dump_files = get_filenames(DATA_DIR)

    dump_files = comm.bcast(dump_files, root=0)
    start, end = split_indexes(len(dump_files), rank, size)

    print(f"Rank {rank} of size {size} processing files from {start} to {end}", flush=True)

    process_file(dump_files[start:end])

    comm.Barrier()

    if rank == 0:
        print("Successfully processed all files...")

    return None

# =============================================================
# ANALYSIS FUNCTIONS
# =============================================================

def process_file(dump_chunk):
    input_paths = [os.path.join(DATA_DIR, dump_file) for dump_file in dump_chunk]
    for frame in input_paths:
        pipeline = import_file(frame)
        data = pipeline.compute()

        if args.dxa:
            performDXA(data.clone())

        if args.wignerseitz:
            performWS(data.clone(), REFERENCE_FILE)

        print(f"Successfully processed frame {frame}...", flush=True)

def performDXA(data):
    dxaModifier = DislocationAnalysisModifier(
        input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC
    )
    data.apply(dxaModifier)

    expModifier = ExpressionSelectionModifier(expression="Cluster == 1")
    data.apply(expModifier)

    delModifier = DeleteSelectedModifier()
    data.apply(delModifier)

    timestep = data.attributes["Timestep"]

    export_file(data, os.path.join(DXA_DIR, f"dxa_{int(timestep)}"), "ca")

    export_file(
        data,
        os.path.join(DXA_ATOMS_DIR, f"dxa_atoms_{int(timestep)}"),
        "lammps/dump",
        columns=[
            "Particle Identifier",
            "Position.X",
            "Position.Y",
            "Position.Z",
            "c_peratom",
            "Cluster",
        ],
    )

    print(f"DXA for timestep {timestep} complete...", flush=True)
    return None

def performWS(data, reference_file):
    timestep = data.attributes["Timestep"]

    wsModifier = WignerSeitzAnalysisModifier()
    wsModifier.reference = FileSource()
    wsModifier.reference.load(reference_file)
    data.apply(wsModifier)

    occupancies = data.particles["Occupancy"]

    vac_data = data.clone()
    sia_data = data.clone()

    vac_selection = vac_data.particles_.create_property("Selection")
    sia_selection = sia_data.particles_.create_property("Selection")

    vac_selection[...] = occupancies == 0
    sia_selection[...] = occupancies == 2

    vac_data.apply(InvertSelectionModifier())
    sia_data.apply(InvertSelectionModifier())
    vac_data.apply(DeleteSelectedModifier())
    sia_data.apply(DeleteSelectedModifier())

    export_file(
        vac_data,
        os.path.join(WS_VAC_DIR, f"ws_vac_{timestep}"),
        "lammps/dump",
        columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
    )

    export_file(
        sia_data,
        os.path.join(WS_SIA_DIR, f"ws_sia_{timestep}"),
        "lammps/dump",
        columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
    )

    print(f"WS for timestep {timestep} complete...", flush=True)
    return None

# =============================================================
# UTILITIES
# =============================================================

def get_filenames(dir_path):
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(files, key=natural_sort_key)

def natural_sort_key(s):
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r"(\d+)", s)]

def split_indexes(n_files, rank, size):
    chunk_size = n_files // size
    remainder = n_files % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    return [start, end]

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main()
