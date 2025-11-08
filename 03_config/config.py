#!/usr/bin/env python3
import os
import yaml
import pandas as pd
import numpy as np

# -----------------------
# Base template
# -----------------------
BASE_CONFIG = {
    "input_dir": "/000_data/02_minimize/output",
    "input_file": None,
    "potential_file": "../00_potentials/malerba.fs",
    "dislo_type": None,
    "study_type": None,
    "fixed_surface_depth": 5.0,
    "force_type": None,
    "force_magnitude": None,
    "obstacle_radius": None,
    "dislocation_initial_displacement": None,
    "timestep": 0.001,
    "temperature": None,
    "run_time": None,
    "dump_freq": 1000,
    "thermo_freq": None,
    "restart_freq": None,
    "random_seed": None,
}

# -----------------------
# Validation
# -----------------------
def validate_config(cfg):
    if cfg.get("temperature") is None:
        raise ValueError("temperature must be defined")
    if cfg.get("run_time") is None:
        raise ValueError("run_time must be defined")
    if cfg.get("timestep") is None:
        cfg["timestep"] = 0.001
    if cfg.get("dump_freq") is None:
        cfg["dump_freq"] = 1000
    if cfg.get("thermo_freq") is None:
        cfg["thermo_freq"] = cfg["dump_freq"]
    if cfg.get("restart_freq") is None:
        cfg["restart_freq"] = 0
    if cfg.get("random_seed") is None:
        cfg["random_seed"] = np.random.randint(1000, 9999)
    
    if cfg["study_type"] != "native" and cfg["obstacle_radius"] is None:
        raise ValueError(f"obstacle_radius must be defined for study_type={cfg['study_type']}")
    if cfg["study_type"] != "native" and cfg["dislocation_initial_displacement"] is None:
        raise ValueError("dislocation_initial_displacement must be defined for native study_type")
    return cfg

# -----------------------
# Filename & sim_id (UPDATE)
# -----------------------
def make_sim_id(cfg):
    """Generate a compact, readable, and unique simulation ID."""
    
    # Abbreviate categorical parameters
    study_code = cfg['study_type'][0].upper()   # first letter
    dislo_code = cfg['dislo_type'][0].upper()   # first letter
    force_code = cfg['force_type'][0].upper()   # first letter

    # Compact large numbers
    def compact_number(num):
        if num is None:
            return ""
        if num >= 1e6:
            return f"{num/1e6:.0f}M"
        elif num >= 1e3:
            return f"{num/1e3:.0f}k"
        elif isinstance(num, float):
            return f"{num:.2g}"
        else:
            return str(num)

    force_mag = compact_number(cfg['force_magnitude'])
    temp = compact_number(cfg['temperature'])
    radius = compact_number(cfg.get('obstacle_radius'))
    disp = compact_number(cfg.get('dislocation_initial_displacement'))

    # Build ID parts, omitting empty values
    parts = [
        f"st{study_code}",
        f"dt{dislo_code}",
        f"T{temp}" if temp else "",
        f"F{force_code}{force_mag}" if force_mag else "",
        f"R{radius}" if radius else "",
        f"D{disp}" if disp else "",
        f"S{cfg['random_seed']}"  # always include seed for uniqueness
    ]

    sim_id = "_".join([p for p in parts if p])  # skip empty parts

    # Optional: append short hash if ID is still very long
    if len(sim_id) > 40:
        hash_code = hashlib.md5(sim_id.encode()).hexdigest()[:6]
        sim_id = sim_id[:30] + "_" + hash_code

    return sim_id

def make_filename(sim_id):
    """Create YAML filename from sim_id."""
    return f"{sim_id}.yaml"

# -----------------------
# Generate configs
# -----------------------
def generate_configs_from_df(df: pd.DataFrame, study_name: str):
    """
    Generate YAML config files from a DataFrame into a subdirectory for a study.

    Args:
        df: pandas DataFrame with parameters for each study.
        study_name: Name of the study (used to create a subdirectory in 000_data/03_shear)
    """
    # Base study directory
    BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '000_data'))
    study_dir = os.path.join(BASE_DIR, "03_shear", study_name)
    os.makedirs(study_dir, exist_ok=True)

    # Dedicated directory to store YAMLs
    configs_dir = os.path.join(study_dir, "configs")
    os.makedirs(configs_dir, exist_ok=True)

    summary_csv = os.path.join(study_dir, "config_summary.csv")
    summary_records = []

    # Determine zero-padding length based on total number of rows
    total_configs = len(df)
    pad_len = len(str(total_configs))  # e.g., 15 configs -> pad_len=2

    for idx, row in df.iterrows():
        cfg = BASE_CONFIG.copy()
        cfg.update(row.to_dict())
        cfg = validate_config(cfg)

        # Add study_name
        cfg["study_name"] = study_name

        # Generate unique sim_id
        sim_id = make_sim_id(cfg)
        cfg["sim_id"] = sim_id

        # Zero-padded index
        file_index = str(idx + 1).zfill(pad_len)  # start from 1
        fname = f"{file_index}_{sim_id}.yaml"
        fpath = os.path.join(configs_dir, fname)

        # Write YAML
        with open(fpath, "w") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)

        # Record for summary CSV
        record = cfg.copy()
        record["config_file"] = fpath
        summary_records.append(record)

        print(f"[✓] Created config: {fpath}")

    # Save summary CSV
    summary_df = pd.DataFrame(summary_records)
    summary_df.to_csv(summary_csv, index=False)
    print(f"\n[✓] Summary CSV saved to: {summary_csv}")
    print(f"Total configs generated: {len(summary_records)}")


