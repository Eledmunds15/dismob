# =============================================================
# LAMMPS Dislocation-Void Interaction Simulation
# Author: Ethan L. Edmunds
# Version: v1.0
# Description: Python script to produce input for void calculations.
# Note: Dislocation is aligned along X, glide plane along Y axis.
# Run: apptainer exec 00_envs/lmp_CPU_22Jul2025.sif mpirun.openmpi -np 16 /opt/venv/bin/python3 03_shear/run.py
# =============================================================

# =============================================================
# IMPORT LIBRARIES
# =============================================================
import os, json, datetime
import numpy as np
from mpi4py import MPI
from lammps import lammps

# =============================================================
# INITIALISE MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# PATH SETTINGS
# =============================================================

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data'))
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, '03_shear'))

INPUT_DIR = os.path.abspath(os.path.join(BASE_DIR, '02_minimize'))
INPUT_FILE = os.path.join(INPUT_DIR, 'output', 'edge_dislo_100_30_40_output.lmp')

POTENTIALS_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '00_potentials'))
POTENTIAL_FILE = os.path.join(POTENTIALS_DIR, 'malerba.fs')

# =============================================================
# SIMULATION PARAMETERS
# =============================================================

OBSTACLE_TYPES = ['void', 'prec']
OBSTACLE_TYPE = OBSTACLE_TYPES[0]

OBSTACLE_RADIUS = 20
DISLOCATION_INITIAL_DISPLACEMENT = 40
FIXED_SURFACE_DEPTH = 5

DT = 0.001
TEMPERATURE = 600
SHEAR_VELOCITY = 0.01

RUN_TIME = 150000
THERMO_FREQ = 1000
DUMP_FREQ = 1000
RESTART_FREQ = DUMP_FREQ

RANDOM_SEED = np.random.randint(1000,9999)

# =============================================================
# DIRECTORY INITIALIZATION AND CASE NAMING
# =============================================================

def make_case_name(obstacle_type, radius, temperature, control_mode, control_value):
    """Generate a consistent case directory name."""
    if control_mode == "shear_velocity":
        ctrl = f"V{control_value}"
    elif control_mode == "applied_stress":
        ctrl = f"S{control_value}MPa"
    else:
        ctrl = "unknown"
    return f"{obstacle_type}_R{radius}_T{temperature}_{ctrl}_{RANDOM_SEED}"


def initialise_output_dirs():
    """Initialize directory structure for output, dump, logs, and restarts."""
    
    global CASE_NAME, CASE_DATA_DIR, OUTPUT_DIR, DUMP_DIR, LOG_DIR, RESTART_DIR

    CASE_NAME = make_case_name(OBSTACLE_TYPE, OBSTACLE_RADIUS, TEMPERATURE, "shear_velocity", SHEAR_VELOCITY)
    CASE_DATA_DIR = os.path.abspath(os.path.join(STAGE_DATA_DIR, CASE_NAME))

    OUTPUT_DIR = os.path.join(CASE_DATA_DIR, 'output')
    DUMP_DIR = os.path.join(CASE_DATA_DIR, 'dump')
    LOG_DIR = os.path.join(CASE_DATA_DIR, 'logs')
    RESTART_DIR = os.path.join(CASE_DATA_DIR, 'restarts')

    if rank == 0:

        for directory in [CASE_DATA_DIR, OUTPUT_DIR, DUMP_DIR, LOG_DIR, RESTART_DIR]:
            os.makedirs(directory, exist_ok=True)
            
        print(f"Successfully made {CASE_DATA_DIR}", flush=True)

    comm.Barrier()

    return None

def write_metadata():
    
    if rank == 0:
        """Write metadata about the simulation setup to JSON."""
        metadata = {
            "timestamp": str(datetime.datetime.now()),
            "mpi_cores": size,
            "input_file": INPUT_FILE,
            "potential_file": POTENTIAL_FILE,
            "obstacle_type": OBSTACLE_TYPE,
            "obstacle_radius": OBSTACLE_RADIUS,
            "dislocation_displacement": DISLOCATION_INITIAL_DISPLACEMENT,
            "dt": DT,
            "temperature": TEMPERATURE,
            "shear_velocity": SHEAR_VELOCITY,
            "run_time": RUN_TIME,
            "thermo_freq": THERMO_FREQ,
            "dump_freq": DUMP_FREQ,
            "restart_freq": RESTART_FREQ
        }

        with open(os.path.join(LOG_DIR, "metadata.json"), "w") as f:
            json.dump(metadata, f, indent=2)

    comm.Barrier()

    return None

# =============================================================
# MAIN FUNCTION
# =============================================================

def main(sim_type):
    initialise_output_dirs()
    write_metadata()

    if sim_type == 'void':
        sim_void()
    elif sim_type == 'prec':
        sim_prec()
    else:
        raise ValueError(f"Unknown simulation type: {sim_type}")

    return None

# =============================================================
# LAMMPS WORKFLOWS
# =============================================================

def sim_void():
    """Run the dislocation–void interaction simulation."""

    lmp = lammps()
    lmp.cmd.clear()
    lmp.cmd.log(os.path.join(LOG_DIR, 'log.lammps'))

    lmp.cmd.units('metal')
    lmp.cmd.dimension(3)
    lmp.cmd.boundary('p', 'f', 'p')
    lmp.cmd.read_data(INPUT_FILE)

    lmp.cmd.pair_style('eam/fs')
    lmp.cmd.pair_coeff('*', '*', POTENTIAL_FILE, 'Fe')

    boxBounds = lmp.extract_box()

    box_min = boxBounds[0]
    box_max = boxBounds[1]

    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    simBoxCenter = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    # Displace and define regions
    lmp.cmd.group('all', 'type', '1')
    lmp.cmd.displace_atoms('all', 'move', OBSTACLE_RADIUS + DISLOCATION_INITIAL_DISPLACEMENT, 0, 0, 'units', 'box')
    lmp.cmd.write_dump('all', 'custom', os.path.join(OUTPUT_DIR, 'displaced_config.txt'), 'id', 'x', 'y', 'z')
    
    lmp.cmd.region('void_reg', 'sphere', simBoxCenter[0], simBoxCenter[1], simBoxCenter[2], OBSTACLE_RADIUS)
    lmp.cmd.region('top_surface_reg', 'block', 'INF', 'INF', (ymax - FIXED_SURFACE_DEPTH), 'INF', 'INF', 'INF')
    lmp.cmd.region('bottom_surface_reg', 'block', 'INF', 'INF', 'INF', (ymin + FIXED_SURFACE_DEPTH), 'INF', 'INF')

    lmp.cmd.group('top_surface', 'region', 'top_surface_reg')
    lmp.cmd.group('bottom_surface', 'region', 'bottom_surface_reg')
    lmp.cmd.group('void', 'region', 'void_reg')
    lmp.cmd.group('mobile_atoms', 'subtract', 'all', 'void', 'top_surface', 'bottom_surface')
    
    lmp.cmd.delete_atoms('group', 'void')
    lmp.cmd.write_dump('all', 'custom', os.path.join(OUTPUT_DIR, 'displaced_voided_config.txt'), 'id', 'x', 'y', 'z')

    # Compute and fix definitions
    lmp.cmd.compute('peratom', 'all', 'pe/atom')
    lmp.cmd.compute('stress', 'all', 'stress/atom', 'NULL')
    lmp.cmd.compute('temp_compute', 'all', 'temp')
    lmp.cmd.compute('press_comp', 'all', 'pressure', 'temp_compute')

    lmp.cmd.fix('1', 'all', 'nvt', 'temp', TEMPERATURE, TEMPERATURE, 100.0 * DT)
    lmp.cmd.velocity('mobile_atoms', 'create', TEMPERATURE, RANDOM_SEED, 'mom', 'yes', 'rot', 'yes')
    lmp.cmd.fix('top_surface_freeze', 'top_surface', 'setforce', 0.0, 0.0, 0.0)
    lmp.cmd.fix('bottom_surface_freeze', 'bottom_surface', 'setforce', 0.0, 0.0, 0.0)
    lmp.cmd.velocity('top_surface', 'set', -SHEAR_VELOCITY, 0.0, 0.0)
    lmp.cmd.velocity('bottom_surface', 'set', 0.0, 0.0, 0.0)

    lmp.cmd.write_dump('top_surface', 'custom', os.path.join(OUTPUT_DIR, 'top_surface_ID.txt'), 'id', 'x', 'y', 'z')
    lmp.cmd.write_dump('bottom_surface', 'custom', os.path.join(OUTPUT_DIR, 'bottom_surface_ID.txt'), 'id', 'x', 'y', 'z')

    # Outputs
    lmp.cmd.thermo_style('custom', 'step', 'temp', 'pe', 'etotal',
                         'c_press_comp[1]', 'c_press_comp[2]', 'c_press_comp[3]',
                         'c_press_comp[4]', 'c_press_comp[5]', 'c_press_comp[6]')
    lmp.cmd.thermo(THERMO_FREQ)

    dump_path = os.path.join(DUMP_DIR, 'dump_*')
    lmp.cmd.dump('1', 'all', 'custom', DUMP_FREQ, dump_path, 'id', 'x', 'y', 'z', 'c_peratom', 'c_stress[4]')

    restart_path = os.path.join(RESTART_DIR, 'restart_*')
    lmp.cmd.restart(RESTART_FREQ, restart_path)

    lmp.cmd.run(RUN_TIME)
    return None

def sim_prec():
    """Run the dislocation–precipitate interaction simulation."""

    lmp = lammps()
    lmp.cmd.clear()
    lmp.cmd.log(os.path.join(LOG_DIR, 'log.lammps'))

    lmp.cmd.units('metal')
    lmp.cmd.dimension(3)
    lmp.cmd.boundary('p', 'f', 'p')
    lmp.cmd.read_data(INPUT_FILE)

    lmp.cmd.pair_style('eam/fs')
    lmp.cmd.pair_coeff('*', '*', POTENTIAL_FILE, 'Fe')

    boxBounds = lmp.extract_box()

    box_min = boxBounds[0]
    box_max = boxBounds[1]

    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    simBoxCenter = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    lmp.cmd.group('all', 'type', '1')
    lmp.cmd.displace_atoms('all', 'move', OBSTACLE_RADIUS + DISLOCATION_INITIAL_DISPLACEMENT, 0, 0, 'units', 'box')

    lmp.cmd.region('precipitate_reg', 'sphere', simBoxCenter[0], simBoxCenter[1], simBoxCenter[2], OBSTACLE_RADIUS)
    lmp.cmd.region('top_surface_reg', 'block', 'INF', 'INF', (ymax - FIXED_SURFACE_DEPTH), 'INF', 'INF', 'INF')
    lmp.cmd.region('bottom_surface_reg', 'block', 'INF', 'INF', 'INF', (ymin + FIXED_SURFACE_DEPTH), 'INF', 'INF')

    lmp.cmd.group('top_surface', 'region', 'top_surface_reg')
    lmp.cmd.group('bottom_surface', 'region', 'bottom_surface_reg')
    lmp.cmd.group('precipitate', 'region', 'precipitate_reg')
    lmp.cmd.group('mobile_atoms', 'subtract', 'all', 'precipitate', 'top_surface', 'bottom_surface')

    #--- Define Computes ---#
    lmp.cmd.compute('peratom', 'all', 'pe/atom')
    lmp.cmd.compute('stress', 'all', 'stress/atom', 'NULL')
    lmp.cmd.compute('temp_compute', 'all', 'temp')
    lmp.cmd.compute('press_comp', 'all', 'pressure', 'temp_compute')

    lmp.cmd.fix('1', 'all', 'nvt', 'temp', TEMPERATURE, TEMPERATURE, 100.0 * DT)
    lmp.cmd.velocity('mobile_atoms', 'create', TEMPERATURE, RANDOM_SEED, 'mom', 'yes', 'rot', 'yes')

    lmp.cmd.fix('top_surface_freeze', 'top_surface', 'setforce', 0.0, 0.0, 0.0)
    lmp.cmd.fix('bottom_surface_freeze', 'bottom_surface', 'setforce', 0.0, 0.0, 0.0)
    lmp.cmd.velocity('top_surface', 'set', -SHEAR_VELOCITY, 0.0, 0.0)
    lmp.cmd.velocity('bottom_surface', 'set', 0.0, 0.0, 0.0)

    lmp.cmd.fix('precipitate_freeze', 'precipitate', 'setforce', 0.0, 0.0, 0.0)
    lmp.cmd.velocity('precipitate', 'set', 0.0, 0.0, 0.0)

    #--- Dump ID's for post-processing or future simulations ---#
    lmp.cmd.write_dump('precipitate', 'custom', os.path.join(OUTPUT_DIR, 'precipitate_ID.txt'), 'id', 'x', 'y', 'z')
    lmp.cmd.write_dump('top_surface', 'custom', os.path.join(OUTPUT_DIR, 'top_surface_ID.txt'), 'id', 'x', 'y', 'z')
    lmp.cmd.write_dump('bottom_surface', 'custom', os.path.join(OUTPUT_DIR, 'bottom_surface_ID.txt'), 'id', 'x', 'y', 'z')

    lmp.cmd.thermo_style('custom', 'step', 'temp', 'pe', 'etotal',
                         'c_press_comp[1]', 'c_press_comp[2]', 'c_press_comp[3]',
                         'c_press_comp[4]', 'c_press_comp[5]', 'c_press_comp[6]')
    lmp.cmd.thermo(THERMO_FREQ)

    dump_path = os.path.join(DUMP_DIR, 'dump_*')
    lmp.cmd.dump('1', 'all', 'custom', DUMP_FREQ, dump_path, 'id', 'x', 'y', 'z', 'c_peratom', 'c_stress[4]')
    
    restart_path = os.path.join(RESTART_DIR, 'restart_*')
    lmp.cmd.restart(RESTART_FREQ, restart_path)

    lmp.cmd.run(RUN_TIME)
    return None

# =============================================================
# ENTRY POINT
# =============================================================

if __name__ == "__main__":
    main(OBSTACLE_TYPE)
