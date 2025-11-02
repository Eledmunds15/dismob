# =============================================================
# LAMMPS Dislocation-Void Interaction Simulation
# Author: Ethan L. Edmunds
# Version: v1.0
# Description: Python script to produce input for void calculations.
# Note: Dislocation is aligned along X, glide plane along Y axis.
# Run: apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 03_shear/analysis.py
# =============================================================

# =============================================================
# IMPORT LIBRARIES
# =============================================================

import os
import re
import numpy as np
from mpi4py import MPI
import traceback

from ovito.io import import_file, export_file
from ovito.modifiers import DislocationAnalysisModifier, WignerSeitzAnalysisModifier, DeleteSelectedModifier, InvertSelectionModifier, ExpressionSelectionModifier
from ovito.pipeline import FileSource

# =============================================================
# INITIALISE MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data')) # Master data directory
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '03_shear')) # Stage data directory
CASE_DIR = os.path.abspath(os.path.join(STAGE_DATA_DIR, 'prec_R15_T900_V0.01_9116'))

for directory in [STAGE_DATA_DIR, CASE_DIR]:
    if rank == 0:
        try:
            if not os.path.exists(directory):
                raise FileNotFoundError(f"Directory does not exist: {directory}")
        except Exception as e:
            print(f"[Rank {rank}] Error with directory: {directory}")
            traceback.print_exc()
            raise
    comm.Barrier()  # ensure all ranks wait until rank 0 finishes the check

DXA_DIR = os.path.join(CASE_DIR, 'dxa') # File with DXA analysis
DXA_SUMMARY_DIR = os.path.join(CASE_DIR, 'dxa_summary') # File with DXA summary files
DXA_ATOMS_DIR = os.path.join(CASE_DIR, 'dxa_atoms') # File with the atoms extracted by DXA
WS_VAC_DIR = os.path.join(CASE_DIR, 'wigner_seitz_vacs') # File with wigner seitz analysis files
WS_SIA_DIR = os.path.join(CASE_DIR, 'wigner_seitz_sias') # File with wigner seitz analysis files

for directory in [DXA_DIR, DXA_SUMMARY_DIR, DXA_ATOMS_DIR, WS_VAC_DIR, WS_SIA_DIR]:
    os.makedirs(directory, exist_ok=True)

# REFERENCE FILE FOR WIGNER SEITZ ANALYSIS
REFERENCE_DIR = os.path.abspath(os.path.join(BASE_DIR, '02_minimize', 'dump')) # Input directory
REFERENCE_FILE = os.path.join(REFERENCE_DIR, 'edge_dislo_100_30_40_dump') # Input file

DATA_DIR = os.path.abspath(os.path.join(CASE_DIR, 'dump'))

# =============================================================
# MAIN FUNCTION
# =============================================================

def main():
    #--- INITIALISE MPI ---#
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    #--- INITIALISE VARIABLE ON ALL RANKS ---#
    dump_files = None

    #--- Get files ---#
    if rank == 0: dump_files = get_filenames(DATA_DIR)

    #--- BROADCAST AND DISTRIBUTE WORK ---#
    dump_files = comm.bcast(dump_files, root=0)

    # Each rank gets only its share of files to process
    start, end = split_indexes(len(dump_files), rank, size)

    print(f"Rank {rank} of size {size} processing files from {start} to {end}", flush=True)

    #--- PROCESS FILES ---#
    process_file(dump_files[start:end])

    comm.Barrier()

    if rank == 0: print("Successfully processed all files...")
    
    return None

# --------------------------- ANALYSIS ---------------------------#

def process_file(dump_chunk):

    input_paths = [os.path.join(DATA_DIR, dump_file) for dump_file in dump_chunk]

    for frame in input_paths:

        pipeline = import_file(frame)
        data = pipeline.compute()

        performDXA(data.clone())
        # performWS(data.clone())
        
        print(f"Successfully processed frame {frame}...", flush=True)

def performDXA(data):

    dxaModifier = DislocationAnalysisModifier(input_crystal_structure=DislocationAnalysisModifier.Lattice.BCC)
    
    data.apply(dxaModifier) # Run DXA
    
    # Select normal sites
    expModifier = ExpressionSelectionModifier(expression = 'Cluster == 1')
    data.apply(expModifier)

    # Delete Selected
    delModifier = DeleteSelectedModifier()
    data.apply(delModifier)

    timestep = data.attributes['Timestep']

    export_file(data, os.path.join(DXA_DIR, f'dxa_{int(timestep)}'), "ca")

    export_file(data, os.path.join(DXA_ATOMS_DIR, f'dxa_atoms_{int(timestep)}'), "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z", "c_peratom", "Cluster"])

    print(f"DXA for timestep {timestep} complete...", flush=True)

    return None

def performWS(data):

    timestep = data.attributes['Timestep']

    """
    print(f"Timestep: {timestep}")
    print(f"Number of particles: {np.sum(data.particles.count)}")
    """

    # Create Wignerseitz modifier
    wsModifier = WignerSeitzAnalysisModifier()
    wsModifier.reference = FileSource()
    wsModifier.reference.load(REFERENCE_FILE)

    data.apply(wsModifier)

    occupancies = data.particles['Occupancy']

    vac_data = data.clone()
    sia_data = data.clone()

    vac_selection = vac_data.particles_.create_property('Selection')
    sia_selection = sia_data.particles_.create_property('Selection')

    vac_selection[...] = (occupancies == 0)
    sia_selection[...] = (occupancies == 2)

    """
    print(f"Number of particles selected (vac): {np.sum(vac_selection)}")
    print(f"Number of particles selected (sia): {np.sum(sia_selection)}")
    """

    vac_data.apply(InvertSelectionModifier())
    sia_data.apply(InvertSelectionModifier())

    vac_data.apply(DeleteSelectedModifier())
    sia_data.apply(DeleteSelectedModifier())

    """
    print(f"Number of particles in vac: {vac_data.particles.count}")
    print(f"Number of particles in sia: {sia_data.particles.count}")
    """

    # Export the file
    export_file(
            vac_data,
            os.path.join(WS_VAC_DIR, f'ws_vac_{timestep}'),
            "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
        )

    export_file(
            sia_data,
            os.path.join(WS_SIA_DIR, f'ws_sia_{timestep}'),
            "lammps/dump",
            columns=["Particle Identifier", "Position.X", "Position.Y", "Position.Z"],
    )

    print(f"WS for timestep {timestep} complete...", flush=True)

    return None

# --------------------------- UTILITIES ---------------------------#

def view_information(data):
    
    print('')
    print("Available particle properties:")
    for prop in data.particles.keys():
        print(f"  - {prop}")

    print("\nAvailable global attributes:")
    for attr in data.attributes.keys():
        print(f"  - {attr}")
    print('')

def get_filenames(dir_path):
    """Returns a naturally sorted list of filenames (not paths) in the given directory."""
    files = [f for f in os.listdir(dir_path) if os.path.isfile(os.path.join(dir_path, f))]
    return sorted(files, key=natural_sort_key)

def natural_sort_key(s):
    # Split the string into digit and non-digit parts, convert digits to int
    return [int(text) if text.isdigit() else text.lower() for text in re.split(r'(\d+)', s)]

def split_indexes(n_files, rank, size):
    """Split n_files into contiguous chunks of indexes for each rank."""
    chunk_size = n_files // size
    remainder = n_files % size

    if rank < remainder:
        start = rank * (chunk_size + 1)
        end = start + chunk_size + 1
    else:
        start = rank * chunk_size + remainder
        end = start + chunk_size

    return [start, end]

# --------------------------- ENTRY POINT ---------------------------#

if __name__ == "__main__":

        main()