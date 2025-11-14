# Series of functions for analysis
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation

import atomman as am

def extractLammpsDump(path):
    """
    Reads a LAMMPS dump file and extracts:
      - Simulation timestep
      - Number of atoms
      - Box bounds (3D)
      - Atomic data (as a Pandas DataFrame)
    """

    timestep = None
    noOfAtoms = None
    boxBounds = []
    atom_data_rows = []
    cols = []

    # --- Read and clean file lines ---
    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    i = 0
    while i < len(lines):
        line = lines[i]

        # --- Timestep ---
        if line.startswith('ITEM: TIMESTEP'):
            timestep = int(lines[i + 1])
            i += 2
            continue

        # --- Number of Atoms ---
        elif line.startswith('ITEM: NUMBER OF ATOMS'):
            noOfAtoms = int(lines[i + 1])
            i += 2
            continue

        # --- Box Bounds ---
        elif line.startswith('ITEM: BOX BOUNDS'):
            boxBounds = []
            for j in range(1, 4):
                bounds = [float(x) for x in lines[i + j].split()]
                boxBounds.append(bounds)
            i += 4
            continue

        # --- Atom Data ---
        elif line.startswith('ITEM: ATOMS'):
            cols = line.split()[2:]  # get column names
            for j in range(1, noOfAtoms + 1):
                values = [float(x) for x in lines[i + j].split()]
                atom_data_rows.append(values)
            i += noOfAtoms + 1
            continue

        else:
            i += 1

    # --- Build DataFrame ---
    atomData = pd.DataFrame(atom_data_rows, columns=cols)

    # --- Return results in structured dictionary ---
    return {
        "timestep": timestep,
        "noOfAtoms": noOfAtoms,
        "boxBounds": boxBounds,
        "columns": cols,
        "atomData": atomData
    }

def extractLogData(path):
    """
    Extracts the thermodynamic table from a LAMMPS log file and returns it as a Pandas DataFrame.
    """

    with open(path, 'r') as f:
        lines = [line.strip() for line in f.readlines()]

    thermo_df = None

    for i, line in enumerate(lines):
        if line.startswith("Step"):
            # header line
            header = line.split()
            data_rows = []

            # read subsequent lines until empty line or next section
            for data_line in lines[i+1:]:
                if not data_line or data_line.startswith("Loop time") or data_line.startswith("Per MPI"):
                    break
                values = [float(x) for x in data_line.split()]
                data_rows.append(values)

            thermo_df = pd.DataFrame(data_rows, columns=header)
            break

    return thermo_df

def extractDXADump(path, wrap=True):
    """
    Extract DXA LAMMPS dump files with simulation cell info, dislocation data,
    and vertex coordinates. Ensures connectivity-preserving wrapping of dislocations.

    Returns a dictionary with:
    - simCell_origin: [ox, oy, oz]
    - simCell_matrix: 3x3 numpy array
    - noOfDislocations: integer
    - dislocation_summary: DataFrame with one row per dislocation
    - dislocations: DataFrame with all vertices
    """
    
    simCell_origin = None
    simCell_matrix = None
    dislocations = []

    # --- Read lines ---
    with open(path, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]
    
    i = 0
    while i < len(lines):
        line = lines[i]

        if line.startswith("SIMULATION_CELL_ORIGIN"):
            simCell_origin = np.array([float(x) for x in line.split()[1:4]])
            i += 1

        elif line.startswith("SIMULATION_CELL_MATRIX"):
            mat = []
            for j in range(1,4):
                mat.append([float(x) for x in lines[i+j].split()])
            simCell_matrix = np.array(mat)
            i += 4

        elif line.startswith("DISLOCATIONS"):
            nDisl = int(line.split()[1]) if len(line.split()) > 1 else 0
            i += 1

            for _ in range(nDisl):
                disl_id = int(lines[i]); i += 1
                burgers = np.array([float(x) for x in lines[i].split()]); i += 1
                cluster_id = int(lines[i]); i += 1
                n_vertices = int(lines[i]); i += 1

                vertices = []
                for _ in range(n_vertices):
                    vertices.append([float(x) for x in lines[i].split()])
                    i += 1
                vertices = np.array(vertices)

                # --- Connectivity-preserving unwrapping ---
                L = np.diag(simCell_matrix)
                for j in range(1, len(vertices)):
                    delta = vertices[j] - vertices[j-1]
                    delta = (delta + 0.5*L) % L - 0.5*L
                    vertices[j] = vertices[j-1] + delta

                # Store dislocation
                dislocations.append({
                    "dislocation_ID": disl_id,
                    "cluster_id": cluster_id,
                    "burgers_vector": burgers,
                    "n_vertices": n_vertices,
                    "vertices": vertices
                })
        else:
            i += 1

    # --- Build summary DataFrame ---
    dislocation_summary = pd.DataFrame([
        {
            "dislocation_ID": d["dislocation_ID"],
            "cluster_id": d["cluster_id"],
            "bx": d["burgers_vector"][0],
            "by": d["burgers_vector"][1],
            "bz": d["burgers_vector"][2],
            "n_vertices": d["n_vertices"]
        }
        for d in dislocations
    ])

    # --- Build vertices DataFrame ---
    vertex_records = []
    for d in dislocations:
        for v in d["vertices"]:
            vertex_records.append({
                "dislocation_ID": d["dislocation_ID"],
                "x": v[0],
                "y": v[1],
                "z": v[2]
            })
    dislocation_vertices = pd.DataFrame(vertex_records)

    # --- Optional wrapping (connectivity-preserving) ---
    if wrap:
        ox, oy, oz = simCell_origin
        Lx, Ly, Lz = np.diag(simCell_matrix)

        wrapped_records = []
        for disl_id, sub_df in dislocation_vertices.groupby("dislocation_ID"):
            verts = sub_df[["x","y","z"]].values.copy()
            
            # --- Connectivity-preserving unwrap is already done ---
            # Shift line so it fully sits in box:
            min_coords = verts.min(axis=0)
            shift = np.array([ox, oy, oz]) - min_coords
            verts += shift

            # Optional: apply modulo to ensure all points are inside box
            verts = (verts - np.array([ox, oy, oz])) % np.array([Lx, Ly, Lz]) + np.array([ox, oy, oz])

            wrapped_records.extend([
                {"dislocation_ID": disl_id, "x": v[0], "y": v[1], "z": v[2]} 
                for v in verts
            ])
        dislocation_vertices = pd.DataFrame(wrapped_records)


    return {
        "simCell_origin": simCell_origin,
        "simCell_matrix": simCell_matrix,
        "noOfDislocations": len(dislocations),
        "dislocation_summary": dislocation_summary,
        "dislocations": dislocation_vertices
    }
