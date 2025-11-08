# apptainer exec 00_envs/lmp_CPU_22Jul2025.sif /opt/venv/bin/python3 03_config/run.py

import numpy as np
import pandas as pd
from config import generate_configs_from_df

data = {
    "input_file": 6*["edge_dislo_100_30_40_output.lmp"],
    "potential_file": 6*["00_potentials/malerba.fs"],
    "dislo_type": 6*["edge"],
    "study_type": 6*["native"],
    "fixed_surface_depth": 6*[5.0],
    "force_type": 6*["strain"],
    "force_magnitude": 6*[1e7],
    "temperature": [100, 200, 300, 400, 500, 600],
    "run_time": 6*[100],
    "dump_freq": 6*[1000],
    "thermo_freq": 6*[None],
    "restart_freq": 6*[None],
    "random_seed": 6*[None],
}

df = pd.DataFrame(data)
generate_configs_from_df(df, study_name="test_run")
