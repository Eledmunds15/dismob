# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif python3 01_lat_param/run.py --config 00_potentials/malerba.fs --Tmin 0 --Tmax 10 --linspace 2

# =============================================================
# LAMMPS Lattice Parameter Exploration (Modified)
# =============================================================
import os, argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from lammps import lammps
from mpi4py import MPI
from matscipy.calculators.eam import EAM
from matscipy.dislocation import get_elastic_constants

# =============================================================
# PATH SETTINGS
# =============================================================
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "000_data"))
STAGE_DATA_DIR = os.path.abspath(os.path.join(BASE_DIR, "01_lat_param"))

OUTPUT_DIR = os.path.join(STAGE_DATA_DIR, "output")
DUMP_DIR = os.path.join(STAGE_DATA_DIR, "dump")
LOG_DIR = os.path.join(STAGE_DATA_DIR, "logs")

for directory in [OUTPUT_DIR, DUMP_DIR, LOG_DIR]:
    os.makedirs(directory, exist_ok=True)

# =============================================================
# MPI
# =============================================================
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# =============================================================
# LAMMPS Lattice Parameter Exploration (Headless)
# =============================================================
def lammpsSim(temperature, potential_file, max_steps=10000, check_interval=10, rolling_window=10, tol=1e-4):
    """Run LAMMPS simulation until equilibrium based on rolling window of Lx or Pressure."""

    if temperature == 0:
        return None

    dump_dir = os.path.join(DUMP_DIR, f"dump_T{temperature}")
    if rank == 0:
        os.makedirs(dump_dir, exist_ok=True)

    lmp = lammps()
    # lmp.cmd.log(os.path.join(LOG_DIR, f"log_{temperature}.lammps"))
    lmp.cmd.log(os.path.join(LOG_DIR, f"test.lammps"))

    # ---------------------------
    # Main Settings
    # ---------------------------
    lmp.cmd.units("metal")
    lmp.cmd.dimension(3)
    lmp.cmd.boundary("p", "p", "p")
    lmp.cmd.atom_style("atomic")
    dt = 0.001
    lmp.cmd.timestep(dt)

    # ---------------------------
    # Load EAM potential
    # ---------------------------
    eam_calc = EAM(potential_file)
    alat, _, _, _ = get_elastic_constants(calculator=eam_calc, symbol="Fe", verbose=False)

    # ---------------------------
    # BCC Lattice
    # ---------------------------
    lmp.cmd.lattice("bcc", alat)
    lmp.cmd.region("box", "block", 0, 10, 0, 10, 0, 10)
    lmp.cmd.create_box(1, "box")
    lmp.cmd.create_atoms(1, "box")
    
    lmp.cmd.pair_style("eam/fs")
    lmp.cmd.pair_coeff("* *", potential_file, "Fe")
    lmp.cmd.neighbor(2.0, "bin")
    lmp.cmd.neigh_modify("delay", 10, "check", "yes")

    lmp.cmd.group("all", "type", "1")
    
    # ---------------------------
    # Thermalisation
    # ---------------------------
    lmp.cmd.velocity("all", "create", temperature, np.random.randint(1000, 9999), "mom", "yes", "rot", "yes")
    lmp.cmd.fix(2, "all", "npt", "temp", temperature, temperature, 100.0*dt, "iso", 0.0, 0.0, 100.0*dt)
    lmp.cmd.dump(1, "all", "custom", 200, os.path.join(dump_dir, "dump.*"), "id", "x", "y", "z")
    lmp.cmd.thermo_style("custom", "step", "temp", "lx", "ly", "lz", "press", "pxx", "pyy", "pzz", "ke", "pe", "etotal")
    lmp.cmd.thermo(check_interval)

    # ---------------------------
    # Run simulation with equilibrium check
    # ---------------------------
    thermo_records = []
    steps_done = 0
    equilibrated = False
    L_plot, P_plot = [], []

    while steps_done < max_steps and not equilibrated:
        lmp.cmd.run(check_interval, "pre", "no", "post", "no")
        t = lmp.last_thermo()

        thermo_records.append(t)
        steps_done += check_interval

        # collect rolling data
        L_plot.append(t["Lx"])
        P_plot.append(t["Press"])

        if len(L_plot) >= rolling_window:
            recent_L = np.array(L_plot[-rolling_window:])
            recent_P = np.array(P_plot[-rolling_window:])
            if (np.max(recent_L) - np.min(recent_L) < tol) and (np.max(recent_P) - np.min(recent_P) < tol):
                if rank == 0:
                    print(f"\n[INFO] Equilibrium reached at step {t['step']}\n")
                equilibrated = True

    # ---------------------------
    # Save thermo data to CSV
    # ---------------------------
    if rank == 0:
        df = pd.DataFrame(thermo_records)
        
        # csv_file = os.path.join(OUTPUT_DIR, f"thermo_T{temperature}.csv")
        csv_file = os.path.join(OUTPUT_DIR, f"test2.csv")
        
        df.to_csv(csv_file, index=False)
        print(f"[INFO] Thermo data saved to {csv_file}")

    lmp.close()


# =============================================================
# ARGUMENT PARSER
# =============================================================
def parse_args():
    parser = argparse.ArgumentParser(description="Generate edge dislocation configuration.")
    parser.add_argument("--config", required=True, help="Path to the potential file (e.g., malerba.fs)")
    parser.add_argument("--Tmin", type=float, required=True, help="Minimum temperature (in K)")
    parser.add_argument("--Tmax", type=float, required=True, help="Maximum temperature (in K)")
    parser.add_argument("--linspace", type=int, required=True, help="Number of temperature points between Tmin and Tmax")
    return parser.parse_args()


# =============================================================
# ENTRY POINT
# =============================================================
def main():
    args = parse_args()
    temperatures = np.linspace(args.Tmin, args.Tmax, args.linspace)
    potential_file = os.path.abspath(args.config)
    if not os.path.exists(potential_file):
        raise FileNotFoundError(f"Potential file not found: {potential_file}")

    if rank == 0:
        print("\n===== Lattice Parameter Exploration =====")
        print(f"Potential file : {potential_file}")
        print(f"Tmin           : {args.Tmin} K")
        print(f"Tmax           : {args.Tmax} K")
        print(f"Points         : {args.linspace}")
        print("=========================================\n")
        print(f"Temperatures: {temperatures}\n")

    # Divide temperature tasks across MPI ranks
    for i, temp in enumerate(temperatures):
        if i % size == rank:
            lammpsSim(temp, potential_file)


if __name__ == "__main__":
    main()
