#!/usr/bin/env python3
"""
Run the dislocation–void interaction simulation.

Example run:
apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 04_shear/run.py --config 000_data/04_shear/test_run/configs/1_stN_dtE_T100_FS10M_S1781.yaml --bench
"""

import os, yaml, datetime, argparse
import numpy as np
from mpi4py import MPI
from lammps import lammps

# =============================================================
# GLOBALS & PROJECT ROOT
# =============================================================
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
params = {}

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# CONFIGURATION HANDLING
# =============================================================
def extractParams(args_config_path, args_bench):
    """Load YAML config and resolve all relative paths."""
    global params

    # --- Make YAML path absolute ---
    if not os.path.isabs(args_config_path):
        config_path = os.path.join(PROJECT_ROOT, args_config_path)
    else:
        config_path = args_config_path

    if not os.path.exists(config_path):
        if rank == 0: raise FileNotFoundError(f"Config file not found: {config_path}")
        return None

    # --- Load YAML ---
    with open(config_path, "r") as f:
        params = yaml.safe_load(f)

    # --- Store absolute YAML path ---
    params["config_path"] = config_path

    # --- Resolve relative paths ---
    params["input_dir"] = os.path.abspath(os.path.join(PROJECT_ROOT, params.get("input_dir", ".")))
    params["potential_file"] = os.path.abspath(os.path.join(PROJECT_ROOT, params.get("potential_file")))

    # --- Apply benchmark mode ---
    if args_bench:
        params["run_time"] = 100  # quick runtime
        params["bench"] = True
    else:
        params["bench"] = False

    if rank == 0:
        print(f"[✓] Loaded configuration from {config_path}")
        print(f"Simulation ID: {params.get("sim_id", "N/A")}")
        print(f"Benchmark mode: {params["bench"]}\n")
    return params

# =============================================================
# OUTPUT DIRECTORY PREPARATION
# =============================================================
def prepareOutput():
    """Create simulation output directory structure and metadata."""
    global params

    if not params:
        raise RuntimeError("Parameters not loaded. Call extractParams() first.")

    # --- Determine YAML directory ---
    yaml_dir = os.path.dirname(params["config_path"])

    # Move one level up if YAML is inside "configs"
    if os.path.basename(yaml_dir) == "configs":
        yaml_dir = os.path.dirname(yaml_dir)

    sim_id = params.get("sim_id")
    if params.get("bench"):
        sim_id = f"{sim_id}_bench"

    # --- Create simulation root directory ---
    sim_out_dir = os.path.join(yaml_dir, sim_id)
    os.makedirs(sim_out_dir, exist_ok=True)

    # --- Standard subdirectories ---
    subdirs = ["dump", "logs", "output", "restart"]
    for sub in subdirs:
        os.makedirs(os.path.join(sim_out_dir, sub), exist_ok=True)

    # --- Save directories back to params ---
    params["output_dir"] = sim_out_dir
    params["dump_dir"] = os.path.join(sim_out_dir, "dump")
    params["logs_dir"] = os.path.join(sim_out_dir, "logs")
    params["output_subdir"] = os.path.join(sim_out_dir, "output")
    params["restart_dir"] = os.path.join(sim_out_dir, "restart")

    if rank == 0:
        print(f"[✓] Prepared output directory at: {sim_out_dir}")
        print(f"Subdirectories: {", ".join(subdirs)}\n")

    # --- Write metadata ---
    metadata_file = os.path.join(params["logs_dir"], "metadata.txt")
    with open(metadata_file, "w") as f:
        f.write("Simulation metadata\n")
        f.write(f"Generated on: {datetime.datetime.now().isoformat()}\n")
        for key in ["sim_id", "study_name", "input_file", "temperature",
                    "force_magnitude", "run_time", "bench"]:
            f.write(f"{key}: {params.get(key)}\n")
    
    if rank == 0: print(f"[✓] Metadata written to: {metadata_file}")

# =============================================================
# LAMMPS SIMULATION
# =============================================================
def lammpsSim():
    """Run LAMMPS with parameters loaded from YAML."""
    global params

    if not params:
        raise RuntimeError("Parameters not loaded. Call extractParams() first.")


    # ---------------------------------------------------------
    # 1. Load input file and initialize LAMMPS
    # ---------------------------------------------------------
    full_input_path = os.path.join(PROJECT_ROOT, params["input_dir"], params["input_file"])
    if not os.path.exists(full_input_path):
        raise FileNotFoundError(f"Input file not found: {full_input_path}")

    lmp = lammps()
    lmp.cmd.clear()
    lmp.cmd.log(os.path.join(params["logs_dir"], "log.lammps"))


    # ---------------------------------------------------------
    # 2. Basic setup (units, dimensions, boundary, atom style)
    # ---------------------------------------------------------
    lmp.cmd.units("metal")
    lmp.cmd.dimension(3)
    lmp.cmd.boundary("p", "f", "p")
    lmp.cmd.atom_style("atomic")


    # ---------------------------------------------------------
    # 3. Read structure and define potential
    # ---------------------------------------------------------
    lmp.cmd.read_data(full_input_path)
    lmp.cmd.pair_style("eam/fs")
    lmp.cmd.pair_coeff("* *", params["potential_file"], "Fe")

    lmp.cmd.neighbor(2.0, "bin")
    lmp.cmd.neigh_modify("delay", 10, "check", "yes")

    # ---------------------------------------------------------
    # 4. Extract simulation box information
    # ---------------------------------------------------------
    boxBounds = lmp.extract_box()
    box_min = boxBounds[0]
    box_max = boxBounds[1]
    xmin, xmax = box_min[0], box_max[0]
    ymin, ymax = box_min[1], box_max[1]
    zmin, zmax = box_min[2], box_max[2]
    simBoxCenter = [np.mean([xmin, xmax]), np.mean([ymin, ymax]), np.mean([zmin, zmax])]

    if rank == 0: print(
        f"Simulation box extents:\n"
        f"  X: {xmin:.3f} -> {xmax:.3f}\n"
        f"  Y: {ymin:.3f} -> {ymax:.3f}\n"
        f"  Z: {zmin:.3f} -> {zmax:.3f}\n"
        f"Box center: ({simBoxCenter[0]:.3f}, {simBoxCenter[1]:.3f}, {simBoxCenter[2]:.3f})\n"
    )


    # ---------------------------------------------------------
    # 5. Define groups and initial displacement
    # ---------------------------------------------------------
    lmp.cmd.group("all", "type", "1")

    if params["dislocation_initial_displacement"] is not None:
        lmp.cmd.displace_atoms(
            "all", "move",
            params["obstacle_radius"] + params["dislocation_initial_displacement"],
            0, 0, "units", "box"
        )
        lmp.cmd.write_dump(
            "all", "custom",
            os.path.join(params["output_subdir"], "displaced_config.txt"),
            "id", "x", "y", "z"
        )


    # ---------------------------------------------------------
    # 6. Define surface regions and groups
    # ---------------------------------------------------------
    lmp.cmd.region("top_surface_reg", "block", "INF", "INF",
                   (ymax - params["fixed_surface_depth"]), "INF", "INF", "INF")
    lmp.cmd.region("bottom_surface_reg", "block", "INF", "INF", "INF",
                   (ymin + params["fixed_surface_depth"]), "INF", "INF")
    lmp.cmd.group("top_surface", "region", "top_surface_reg")
    lmp.cmd.group("bottom_surface", "region", "bottom_surface_reg")


    # ---------------------------------------------------------
    # 7. Define obstacle / void / precipitate regions
    # ---------------------------------------------------------
    if (params["study_type"] == "prec") or (params["study_type"] == "void"):
        lmp.cmd.region("obstacle_reg", "sphere",
                       simBoxCenter[0], simBoxCenter[1], simBoxCenter[2],
                       params["obstacle_radius"])
        lmp.cmd.group("obstacle", "region", "obstacle_reg")
        lmp.cmd.group("mobile_atoms", "subtract", "all", "obstacle",
                      "top_surface", "bottom_surface")

        if (params["study_type"] == "void"):
            lmp.cmd.delete_atoms("group", "obstacle")
            lmp.cmd.write_dump(
                "all", "custom",
                os.path.join(params["output_dir"], "displaced_voided_config.txt"),
                "id", "x", "y", "z"
            )
        elif (params["study_type"] == "prec"):
            lmp.cmd.write_dump(
                "obstacle", "custom",
                os.path.join(params["output_subdir"], "precipitate_ID.txt"),
                "id", "x", "y", "z"
            )
    else:
        lmp.cmd.group("mobile_atoms", "subtract", "all", "top_surface", "bottom_surface")


    # ---------------------------------------------------------
    # 8. Define computes and fixes (temperature, stress, etc.)
    # ---------------------------------------------------------
    lmp.cmd.compute("peratom", "all", "pe/atom")
    lmp.cmd.compute("stress", "all", "stress/atom", "NULL")
    lmp.cmd.compute("temp_compute", "all", "temp")
    lmp.cmd.compute("press_comp", "all", "pressure", "temp_compute")

    DT = params.get("time_step", 0.001)
    lmp.cmd.fix("1", "all", "nvt", "temp",
                params["temperature"], params["temperature"], 100.0 * DT)
    lmp.cmd.velocity("mobile_atoms", "create",
                     params["temperature"], params["random_seed"], "mom", "yes", "rot", "yes")


    # ---------------------------------------------------------
    # 9. Apply loading conditions
    # ---------------------------------------------------------
    if (params["force_type"] == "strain"):

        lmp.cmd.fix("top_surface_freeze", "top_surface", "setforce", 0.0, 0.0, 0.0)
        lmp.cmd.fix("bottom_surface_freeze", "bottom_surface", "setforce", 0.0, 0.0, 0.0)

        # Calculate strain velocity for the surface (strain rate * height of simulation cell)
        strainRate = params["force_magnitude"] # In Strain
        strainVelocity = strainRate * abs(ymax - ymin)

        lmp.cmd.velocity("top_surface", "set", -strainVelocity, 0.0, 0.0)
        lmp.cmd.velocity("bottom_surface", "set", 0.0, 0.0, 0.0)

    elif (params["force_type"] == "stress"):

        # Freeze top & bottom surfaces (no z motion)
        lmp.cmd.fix("top_surface_freeze", "top_surface", "setforce", 0.0, 0.0, 0.0)
        lmp.cmd.fix("bottom_surface_freeze", "bottom_surface", "setforce", 0.0, 0.0, 0.0)

        # Convert input pressure (MPa → Pa)
        pressure = params["force_magnitude"] * 1.0e6  # Pa = N/m²

        # Compute area of top plane (convert Å² → m²)
        planeArea_m2 = abs(xmax - xmin) *1e-10 * abs(zmax - zmin) * 1.0e-10  # m²

        # Total force applied across the top surface [N]
        forceTotal_N = pressure * planeArea_m2  # N = J/m

        # Get number of atoms in top surface group
        lmp.cmd.variable("n_top", "equal", "count(top_surface)")
        nAtomsTop = int(lmp.extract_variable("n_top", None, 0))
        if rank == 0: print(f"Number of atoms in top_surface: {nAtomsTop}", flush=True)

        # Convert to per-atom force in eV/Å
        force_per_atom_N = forceTotal_N / nAtomsTop
        force_per_atom_eV_per_A = force_per_atom_N * (6.241509E8)  # N → eV/Å
        if rank == 0: print(f"Apply force of {force_per_atom_eV_per_A}eV/Å per atom in top_surface",flush=True)

        # Apply shear load in -x direction
        lmp.cmd.fix("top_surface_shear_force", "top_surface", "aveforce",
                    -force_per_atom_eV_per_A, 0.0, 0.0)
        
        
    # ---------------------------------------------------------
    # 10. Handle precipitate motion
    # ---------------------------------------------------------
    if (params["study_type"] == "prec"):
        lmp.cmd.fix("precipitate_freeze", "obstacle", "setforce", 0.0, 0.0, 0.0)
        lmp.cmd.velocity("obstacle", "set", 0.0, 0.0, 0.0)


    # ---------------------------------------------------------
    # 11. Thermo, dump, and restart settings
    # ---------------------------------------------------------
    lmp.cmd.thermo_style("custom", "step", "temp", "pe", "etotal",
                         "c_press_comp[1]", "c_press_comp[2]", "c_press_comp[3]",
                         "c_press_comp[4]", "c_press_comp[5]", "c_press_comp[6]")
    lmp.cmd.thermo(params["thermo_freq"])

    dumpPath = os.path.join(params["dump_dir"], "dump_*")
    lmp.cmd.dump("1", "all", "custom", params["dump_freq"], dumpPath, "id", "x", "y", "z", "c_peratom")

    restartPath = os.path.join(params["restart_dir"], "restart_*")
    if params.get("restart_freq", 0):
        lmp.cmd.restart(params["restart_freq"], restartPath)


    # ---------------------------------------------------------
    # 12. Run the simulation
    # ---------------------------------------------------------
    lmp.cmd.run(params["run_time"])

    comm.Barrier()

    if rank == 0: print(f"\n[✓] LAMMPS run complete. Log: {os.path.join(params["logs_dir"], "log.lammps")}\n", flush=True)

    return None


# =============================================================
# ENTRY POINT
# =============================================================
def parse_args():

    parser = argparse.ArgumentParser(description="Run a dislocation simulation.")
    
    parser.add_argument(
        "--config", 
        required=True, 
        help="Path to YAML config file."
    )
    
    parser.add_argument(
        "--bench", 
        action="store_true", 
        help="Enable benchmark mode (default: False)")

    return parser.parse_args()


# =============================================================
# ENTRY POINT
# =============================================================
def main():
    
    args = parse_args()

    extractParams(args.config, args.bench)
    prepareOutput()
    lammpsSim()

if __name__ == "__main__":
    main()
