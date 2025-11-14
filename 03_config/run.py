# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 04_config/run.py

import numpy as np
import pandas as pd
from config import generate_configs_from_df

data = {
    "input_file": 4*["edge_dislo_100_30_30_[112].lmp"],
    "potential_file": 4*["00_potentials/malerba.fs"],
    "dislo_type": 4*["edge"],
    "study_type": 4*["native"],
    "fixed_surface_depth": 4*[5.0],
    "force_type": 4*["stress"],
    "force_magnitude": [50, 100, 200, 500],
    "temperature": 4*[200],
    "run_time": 4*[500000],
    "dump_freq": 4*[1000],
    "thermo_freq": 4*[None],
    "restart_freq": 4*[None],
    "random_seed": 4*[None]
}

df = pd.DataFrame(data)
generate_configs_from_df(df, study_name="112_disl_200K")
