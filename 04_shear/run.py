#!/usr/bin/env python3
"""Run the dislocation–void interaction simulation."""

# Example run:
# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 04_shear/run.py --config 000_data/03_shear/test_run/configs/1_stN_dtE_T100_FS10M_S9233.yaml --bench

import os, yaml
import numpy as np
from lammps import lammps
import datetime

# -----------------------
# Project root (absolute)
# -----------------------
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# -----------------------
# Global parameters
# -----------------------
params = {}

# -----------------------
# Extract parameters from YAML
# -----------------------
def extractParams(args_config_path, args_bench):
    """Load YAML config and resolve all relative paths."""
    global params

    # Make YAML path absolute
    if not os.path.isabs(args_config_path):
        config_path = os.path.join(PROJECT_ROOT, args_config_path)
    else:
        config_path = args_config_path

    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load YAML
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    # Store absolute YAML path
    params["config_path"] = config_path

    # Resolve paths relative to PROJECT_ROOT
    params["input_dir"] = os.path.abspath(os.path.join(PROJECT_ROOT, params.get("input_dir", ".")))
    params["potential_file"] = os.path.abspath(os.path.join(PROJECT_ROOT, params.get("potential_file")))

    # Apply benchmark mode
    if args_bench:
        params["run_time"] = 1000  # quick runtime
        params["bench"] = True
    else:
        params["bench"] = False

    print(f"[✓] Loaded configuration from {config_path}")
    print(f"Simulation ID: {params.get('sim_id', 'N/A')}")
    print(f"Benchmark mode: {params['bench']}")
    return params

# -----------------------
# Prepare output directories
# -----------------------
def prepareOutput():
    """Create simulation output directory structure based on SIM_ID and write metadata."""
    global params

    if not params:
        raise RuntimeError("Parameters not loaded. Call extractParams() first.")

    # Directory containing YAML
    yaml_dir = os.path.dirname(params["config_path"])

    # Move one level up if YAML is inside 'configs'
    if os.path.basename(yaml_dir) == "configs":
        yaml_dir = os.path.dirname(yaml_dir)

    sim_id = params.get("sim_id")
    if params.get("bench"):
        sim_id = f"{sim_id}_bench"

    # Create simulation root directory
    sim_out_dir = os.path.join(yaml_dir, sim_id)
    os.makedirs(sim_out_dir, exist_ok=True)

    # Standard subdirectories
    subdirs = ["dump", "logs", "output", "restart"]
    for sub in subdirs:
        os.makedirs(os.path.join(sim_out_dir, sub), exist_ok=True)

    # Save directories back to params
    params["output_dir"] = sim_out_dir
    params["dump_dir"] = os.path.join(sim_out_dir, "dump")
    params["logs_dir"] = os.path.join(sim_out_dir, "logs")
    params["output_subdir"] = os.path.join(sim_out_dir, "output")
    params["restart_dir"] = os.path.join(sim_out_dir, "restart")

    print(f"[✓] Prepared output directory at: {sim_out_dir}")
    print(f"Subdirectories: {', '.join(subdirs)}")

    # Write metadata
    metadata_file = os.path.join(params["logs_dir"], "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("Simulation metadata\n")
        f.write(f"Generated on: {datetime.datetime.now().isoformat()}\n")
        for key in ["sim_id", "study_name", "input_file", "temperature", "force_magnitude", "run_time", "bench"]:
            f.write(f"{key}: {params.get(key)}\n")
    print(f"[✓] Metadata written to: {metadata_file}")

# -----------------------
# LAMMPS simulation
# -----------------------
def lammpsSim():
    """Run LAMMPS with parameters loaded from YAML."""
    global params

    if not params:
        raise RuntimeError("Parameters not loaded. Call extractParams() first.")

    # Resolve absolute input file
    full_input_path = os.path.join(params["input_dir"], params["input_file"])
    if not os.path.exists(full_input_path):
        raise FileNotFoundError(f"Input file not found: {full_input_path}")

    lmp = lammps()
    lmp.cmd.clear()
    lmp.cmd.log(os.path.join(params['logs_dir'], 'log.lammps'))

    # LAMMPS setup
    lmp.cmd.units('metal')
    lmp.cmd.dimension(3)
    lmp.cmd.boundary('p', 'f', 'p')
    lmp.cmd.atom_style('atomic')
    lmp.cmd.read_data(full_input_path)

    # Pair style and potential
    lmp.cmd.pair_style('eam/fs')
    lmp.cmd.pair_coeff('* *', params['potential_file'], 'Fe')

    boxBounds = lmp.extract_box()

    box_min = boxBounds[0]
    box_max = boxBounds[1]

    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]

    simBoxCenter = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    print(
        f"Simulation box extents:\n"
        f"  X: {xmin:.3f} -> {xmax:.3f}\n"
        f"  Y: {ymin:.3f} -> {ymax:.3f}\n"
        f"  Z: {zmin:.3f} -> {zmax:.3f}\n"
        f"Box center: ({simBoxCenter[0]:.3f}, {simBoxCenter[1]:.3f}, {simBoxCenter[2]:.3f})"
    )

    # Displace and define regions
    lmp.cmd.group('all', 'type', '1')

    if params['dislocation_initial_displacement'] != None:
        lmp.cmd.displace_atoms('all', 'move', params['obstacle_radius'] + params['dislocation_initial_displacement'], 0, 0, 'units', 'box')
        lmp.cmd.write_dump('all', 'custom', os.path.join(params['output_subdir'], 'displaced_config.txt'), 'id', 'x', 'y', 'z')

    splitGeom(lmp)

    defComputes(lmp)

    lmp.cmd.fix('1', 'all', 'nvt', 'temp', params['temperature'], params['temperature'], 100.0 * params['timestep'])
    lmp.cmd.velocity('mobile_atoms', 'create', params['temperature'], params['random_seed'], 'mom', 'yes', 'rot', 'yes')
    
    if params['force_type'] == 'strain':
        lmp.cmd.fix('top_surface_freeze', 'top_surface', 'setforce', 0.0, 0.0, 0.0)
        lmp.cmd.fix('bottom_surface_freeze', 'bottom_surface', 'setforce', 0.0, 0.0, 0.0)

        height = (ymax - ymin)

        shearVelocity = params['force_magnitude']*height

        lmp.cmd.velocity('top_surface', 'set', -shearVelocity, 0.0, 0.0)
        lmp.cmd.velocity('bottom_surface', 'set', 0.0, 0.0, 0.0)

    print(f"[✓] LAMMPS run complete. Log: {os.path.join(params['logs_dir'], 'log.lammps')}")

    return None

def splitGeom(lmp, ymin, ymax):

    global params

    if (params['study_type'] == 'native'):

        lmp.cmd.region('top_surface_reg', 'block', 'INF', 'INF', (ymax - params['fixed_surface_depth']), 'INF', 'INF', 'INF')
        lmp.cmd.region('bottom_surface_reg', 'block', 'INF', 'INF', 'INF', (ymin + params['fixed_surface_depth']), 'INF', 'INF')

        lmp.cmd.group('top_surface', 'region', 'top_surface_reg')
        lmp.cmd.group('bottom_surface', 'region', 'bottom_surface_reg')
        lmp.cmd.group('mobile_atoms', 'subtract', 'all', 'top_surface', 'bottom_surface')
        
        return None

    elif (params['study_type'] == 'obstacle'):

        return None

    else:
        raise RuntimeError("Study Type not defined.")

def defComputes(lmp):

    lmp.cmd.compute('peratom', 'all', 'pe/atom')
    lmp.cmd.compute('stress', 'all', 'stress/atom', 'NULL')
    lmp.cmd.compute('temp_compute', 'all', 'temp')
    lmp.cmd.compute('press_comp', 'all', 'pressure', 'temp_compute')

# -----------------------
# Main
# -----------------------
def main():
    import argparse
    parser = argparse.ArgumentParser(description="Run a dislocation–void interaction simulation.")
    parser.add_argument("--config", required=True, help="Path to YAML config file.")
    parser.add_argument("--bench", action="store_true", help="Enable benchmark mode (default: False)")
    args = parser.parse_args()

    extractParams(args.config, args.bench)
    prepareOutput()
    lammpsSim()

if __name__ == "__main__":
    main()
