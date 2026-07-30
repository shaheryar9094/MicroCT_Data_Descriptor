## -----------------------------------------------------------------------------
# Supplementary Script for the manuscript:
#
#   "High-resolution micro-CT dataset of Mount Gambier limestone with
#    three phase reusable segmentation classifier, dual pore networks
#    model and finite volume multiscale meshes"
#
#
# Authors: Shaheryar T. Hussain, Klaus Regenauer-Lieb and Sheik S. Rahman.
#
# Corresponding author: shaheryar.hussain@curtin.edu.au
#
#
# Reproducibility statement:
#   This file is provided as supplementary research material to support
#   methodological transparency, reproducibility, and independent
#   verification of the micro-CT dataset and its segmentation validation 
#   results reported in the associated manuscript.
#
#
# Data availability:
#   All data described in the associated manuscript are publicly available
#   on the Digital Porous Media Portal (https://digitalporousmedia.org).
#   The dataset DOI is: << INSERT RESOLVED DOI ONCE DEPOSIT IS LIVE >>
#
#
# Conditions of use:
#   This script may be used freely for academic and research purposes.
#   Appropriate citation of the associated manuscript is expected whenever
#   this script, the underlying workflow, or derivative adaptations
#   contribute to published work.
#
#
# Licence:
#   Unless otherwise stated, this material is provided under the Creative
#   Commons Attribution 4.0 International Licence (CC BY 4.0).
#
## -----------------------------------------------------------------------------

"""
Dual Pore Network Model (DPNM) — STAGE 2: ANALYSIS & EXPORT
============================================================
Loads a dual pore network produced by `dpnm_generate.py` and computes
flow / geometric properties, then writes all output files. This stage is
fast: it never touches the raw micro-CT image or re-runs SNOW2.

Computes:
  * single-phase absolute permeability (OpenPNM StokesFlow)
  * statistics report

Exports:
  * CSV (pores + throats)
  * Statoil / ICL format (.dat node1/node2/link1/link2)
  * VTK PolyData (ParaView)
  * JSON metadata summary (+ permeability + full statistics)

"""

import numpy as np
import openpnm as op
import multiprocessing as mp
import sys
import os
import io
import json
import time
import pickle

# ============================================================
# USER PARAMETERS
# ============================================================
OUTPUT_DIR = "DPNM_output"
NETWORK_FILE = os.path.join(OUTPUT_DIR, "dpnm_network.pkl")

SHAPE_CORRECTION = 2

NX, NY, NZ = 1280, 1280, 1280
VOXEL_SIZE = 2.675e-6
N_CORES = max(1, mp.cpu_count() - 4)

# ============================================================
# LOAD THE GENERATED NETWORK
# ============================================================

def load_network(filepath):
    """
    Load a dual pore network saved by `dpnm_generate.py`.

    Returns the network dict plus the acquisition metadata (voxel size and
    image dimensions), reading them from the file when available and
    otherwise falling back to the module constants.
    """
    print(f"Loading dual network: {filepath}")
    if not os.path.exists(filepath):
        raise FileNotFoundError(
            f"Network file not found: {filepath}\n"
            f"Run the generation stage first:  python dpnm_generate.py"
        )

    with open(filepath, "rb") as f:
        network = pickle.load(f)

    voxel_size = float(network.get("_voxel_size", VOXEL_SIZE))
    nx = int(network.get("_nx", NX))
    ny = int(network.get("_ny", NY))
    nz = int(network.get("_nz", NZ))

    n_pores = network["pore.coords"].shape[0]
    n_throats = network["throat.conns"].shape[0]
    print(f"  Pores:      {n_pores:,}")
    print(f"  Throats:    {n_throats:,}")
    print(f"  Voxel size: {voxel_size*1e6:.3f} µm")
    print(f"  Domain:     {nx}x{ny}x{nz} voxels")

    if "pore.boundary_id" not in network or "pore.boundary_name" not in network:
        raise KeyError(
            "Loaded network is missing boundary labels "
            "(pore.boundary_id / pore.boundary_name). Re-run dpnm_generate.py."
        )

    return network, voxel_size, (nx, ny, nz)

# ============================================================
# VECTORIZED EXPORT FUNCTIONS
# ============================================================

def export_csv(network, output_dir, prefix="dpnm"):
    print(f"\nExporting to CSV in {output_dir}/...")

    pore_file = os.path.join(output_dir, f"{prefix}_pores.csv")
    n_pores = network["pore.coords"].shape[0]

    buf = io.StringIO()
    buf.write("pore_id,x_m,y_m,z_m,radius_m,volume_m3,label,boundary_id,boundary_face\n")
    for i in range(n_pores):
        c = network["pore.coords"][i]
        buf.write(f"{i},{c[0]:.8e},{c[1]:.8e},{c[2]:.8e},"
                  f"{network['pore.radius'][i]:.8e},"
                  f"{network['pore.volume'][i]:.8e},"
                  f"{network['pore.label'][i]},"
                  f"{network['pore.boundary_id'][i]},"
                  f"{network['pore.boundary_name'][i]}\n")
    with open(pore_file, "w") as f:
        f.write(buf.getvalue())
    print(f"  Saved: {pore_file}")

    throat_file = os.path.join(output_dir, f"{prefix}_throats.csv")
    n_throats = network["throat.conns"].shape[0]
    header = "throat_id,pore1,pore2,radius_m,length_m,label"
    throat_data = np.column_stack([
        np.arange(n_throats),
        network["throat.conns"],
        network["throat.radius"],
        network["throat.length"],
        network["throat.label"]
    ])
    np.savetxt(throat_file, throat_data, delimiter=",", header=header,
               comments="", fmt=["%d", "%d", "%d", "%.8e", "%.8e", "%d"])
    print(f"  Saved: {throat_file}")

    return pore_file, throat_file


def export_statoil(network, output_dir, prefix="dpnm"):
    """
    Export in Statoil/ICL format.
    """
    print(f"\nExporting Statoil format in {output_dir}/...")

    coords = network["pore.coords"]
    radii_p = network["pore.radius"]
    volumes = network["pore.volume"]
    labels_p = network["pore.label"]
    boundary_ids = network["pore.boundary_id"]
    conns = network["throat.conns"]
    radii_t = network["throat.radius"]
    lengths = network["throat.length"]
    labels_t = network["throat.label"]

    n_pores = coords.shape[0]
    n_throats = conns.shape[0]

    t0 = time.time()
    adjacency = [[] for _ in range(n_pores)]
    coord_num = np.zeros(n_pores, dtype=int)
    for i in range(n_throats):
        p1, p2 = int(conns[i, 0]), int(conns[i, 1])
        adjacency[p1].append(i)
        adjacency[p2].append(i)
        coord_num[p1] += 1
        coord_num[p2] += 1
    print(f"  Adjacency list built in {time.time()-t0:.2f}s")

    G = SHAPE_CORRECTION / (16.0 * np.pi)

    node1_path = os.path.join(output_dir, f"{prefix}_node1.dat")
    buf = io.StringIO()
    buf.write(f"{n_pores}\n")
    for i in range(n_pores):
        conn_str = " ".join(str(c) for c in adjacency[i])
        buf.write(f"{i} {coords[i,0]:.8e} {coords[i,1]:.8e} {coords[i,2]:.8e} "
                  f"{coord_num[i]} {conn_str}\n")
    with open(node1_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Saved: {node1_path}")

    node2_path = os.path.join(output_dir, f"{prefix}_node2.dat")
    buf = io.StringIO()
    buf.write(f"{n_pores}\n")
    for i in range(n_pores):
        buf.write(f"{i} {volumes[i]:.8e} {radii_p[i]:.8e} {G:.6f} "
                  f"{labels_p[i]} {boundary_ids[i]}\n")
    with open(node2_path, "w") as f:
        f.write(buf.getvalue())
    print(f"  Saved: {node2_path}")

    link1_path = os.path.join(output_dir, f"{prefix}_link1.dat")
    link1_data = np.column_stack([
        np.arange(n_throats), conns[:, 0], conns[:, 1],
        radii_t, np.full(n_throats, G), lengths, labels_t
    ])
    with open(link1_path, "w") as f:
        f.write(f"{n_throats}\n")
        np.savetxt(f, link1_data,
                   fmt=["%d", "%d", "%d", "%.8e", "%.6f", "%.8e", "%d"])
    print(f"  Saved: {link1_path}")

    link2_path = os.path.join(output_dir, f"{prefix}_link2.dat")
    half_len = lengths / 2.0
    link2_data = np.column_stack([
        np.arange(n_throats), conns[:, 0], conns[:, 1],
        half_len, half_len, labels_t
    ])
    with open(link2_path, "w") as f:
        f.write(f"{n_throats}\n")
        np.savetxt(f, link2_data,
                   fmt=["%d", "%d", "%d", "%.8e", "%.8e", "%d"])
    print(f"  Saved: {link2_path}")

    return node1_path, node2_path, link1_path, link2_path


def export_metadata(network, output_dir, voxel_size, dims, prefix="dpnm"):
    """Export a JSON metadata file summarizing the network."""
    meta = {
        "description": "Dual Pore Network Model — Mount Gambier Limestone",
        "voxel_size_m": voxel_size,
        "image_dimensions_voxels": list(dims),
        "image_physical_size_mm": [d * voxel_size * 1e3 for d in dims],
        "n_cores_used": int(network.get("_n_cores", N_CORES)),
        "n_macro_pores": int(network["_n_macro_pores"]),
        "n_micro_pores": int(network["_n_micro_pores"]),
        "n_total_pores": int(network["pore.coords"].shape[0]),
        "n_macro_throats": int(network["_n_macro_throats"]),
        "n_micro_throats": int(network["_n_micro_throats"]),
        "n_coupling_throats": int(network["_n_coupling_throats"]),
        "n_total_throats": int(network["throat.conns"].shape[0]),
        "pore_labels": {"0": "macro", "1": "micro"},
        "throat_labels": {"0": "macro", "1": "micro", "2": "coupling"},
        "boundary_labels": {
            "0": "internal",
            "1": "x_min",
            "2": "x_max",
            "3": "y_min",
            "4": "y_max",
            "5": "z_min",
            "6": "z_max"
        },
        "coordinate_units": "meters",
        "notes": (
            "Geometry-only network — no flow properties assigned. "
            "Users should assign contact angles, fluid properties, "
            "and conductance models as needed."
        )
    }

    macro_mask = network["pore.label"] == 0
    micro_mask = network["pore.label"] == 1
    meta["macro_pore_radius_stats_um"] = {
        "min": float(np.min(network["pore.radius"][macro_mask]) * 1e6) if macro_mask.any() else 0,
        "max": float(np.max(network["pore.radius"][macro_mask]) * 1e6) if macro_mask.any() else 0,
        "mean": float(np.mean(network["pore.radius"][macro_mask]) * 1e6) if macro_mask.any() else 0,
    }
    meta["micro_pore_radius_stats_um"] = {
        "min": float(np.min(network["pore.radius"][micro_mask]) * 1e6) if micro_mask.any() else 0,
        "max": float(np.max(network["pore.radius"][micro_mask]) * 1e6) if micro_mask.any() else 0,
        "mean": float(np.mean(network["pore.radius"][micro_mask]) * 1e6) if micro_mask.any() else 0,
    }

    bids = network["pore.boundary_id"]
    face_names = {0: "internal", 1: "x_min", 2: "x_max",
                  3: "y_min", 4: "y_max", 5: "z_min", 6: "z_max"}
    meta["boundary_pore_counts"] = {
        name: int(np.sum(bids == fid)) for fid, name in face_names.items()
    }

    meta_path = os.path.join(output_dir, f"{prefix}_metadata.json")
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"\n  Saved metadata: {meta_path}")
    return meta_path


def export_vtk(network, output_dir, prefix="dpnm"):
    """Export as VTK PolyData."""
    print(f"\nExporting VTK for ParaView visualization...")

    coords = network["pore.coords"]
    conns = network["throat.conns"]
    n_pores = coords.shape[0]
    n_throats = conns.shape[0]

    vtk_path = os.path.join(output_dir, f"{prefix}_network.vtk")

    buf = io.StringIO()
    buf.write("# vtk DataFile Version 3.0\n")
    buf.write("Dual Pore Network Model - Mount Gambier\n")
    buf.write("ASCII\n")
    buf.write("DATASET POLYDATA\n")

    buf.write(f"POINTS {n_pores} float\n")
    for i in range(n_pores):
        buf.write(f"{coords[i,0]:.8e} {coords[i,1]:.8e} {coords[i,2]:.8e}\n")

    buf.write(f"\nLINES {n_throats} {n_throats * 3}\n")
    for i in range(n_throats):
        buf.write(f"2 {conns[i,0]} {conns[i,1]}\n")

    buf.write(f"\nPOINT_DATA {n_pores}\n")

    buf.write("SCALARS pore_radius float 1\nLOOKUP_TABLE default\n")
    for r in network["pore.radius"]:
        buf.write(f"{r:.8e}\n")

    buf.write("SCALARS pore_volume float 1\nLOOKUP_TABLE default\n")
    for v in network["pore.volume"]:
        buf.write(f"{v:.8e}\n")

    buf.write("SCALARS pore_label int 1\nLOOKUP_TABLE default\n")
    for l in network["pore.label"]:
        buf.write(f"{l}\n")

    buf.write("SCALARS boundary_id int 1\nLOOKUP_TABLE default\n")
    for b in network["pore.boundary_id"]:
        buf.write(f"{b}\n")

    buf.write(f"\nCELL_DATA {n_throats}\n")

    buf.write("SCALARS throat_radius float 1\nLOOKUP_TABLE default\n")
    for r in network["throat.radius"]:
        buf.write(f"{r:.8e}\n")

    buf.write("SCALARS throat_length float 1\nLOOKUP_TABLE default\n")
    for l in network["throat.length"]:
        buf.write(f"{l:.8e}\n")

    buf.write("SCALARS throat_label int 1\nLOOKUP_TABLE default\n")
    for l in network["throat.label"]:
        buf.write(f"{l}\n")

    with open(vtk_path, "w") as f:
        f.write(buf.getvalue())

    print(f"  Saved: {vtk_path}")
    return vtk_path

def compute_permeability(network, voxel_size, nx, ny, nz, direction='x'):
    """
    Compute single-phase absolute permeability using OpenPNM StokesFlow.
    Uses scipy for cluster detection.
    """
    print(f"\nComputing single-phase permeability (direction={direction})...")
    
    from scipy.sparse import csr_matrix
    from scipy.sparse.csgraph import connected_components
    
    coords = network['pore.coords']
    conns = network['throat.conns']
    n_pores = coords.shape[0]
    n_throats = conns.shape[0]
    
    print("  Checking network connectivity...")
    
    row = np.concatenate([conns[:, 0], conns[:, 1]])
    col = np.concatenate([conns[:, 1], conns[:, 0]])
    data = np.ones(len(row), dtype=int)
    adj = csr_matrix((data, (row, col)), shape=(n_pores, n_pores))
    
    n_clusters, cluster_labels = connected_components(adj, directed=False)
    cluster_ids, cluster_sizes = np.unique(cluster_labels, return_counts=True)
    
    largest_id = cluster_ids[np.argmax(cluster_sizes)]
    largest_size = int(np.max(cluster_sizes))
    
    print(f"  Found {n_clusters} clusters")
    print(f"  Largest cluster: {largest_size:,} pores ({largest_size/n_pores*100:.1f}%)")
    
    keep_pores = cluster_labels == largest_id
    
    old_to_new = np.full(n_pores, -1, dtype=int)
    new_indices = np.where(keep_pores)[0]
    old_to_new[new_indices] = np.arange(len(new_indices))
    
    keep_throats = keep_pores[conns[:, 0]] & keep_pores[conns[:, 1]]
    
    new_conns = old_to_new[conns[keep_throats]]
    
    new_coords = coords[keep_pores]
    new_pore_radii = network['pore.radius'][keep_pores]
    new_pore_vols = network['pore.volume'][keep_pores]
    new_throat_radii = network['throat.radius'][keep_throats]
    new_throat_lengths = network['throat.length'][keep_throats]
    
    n_new_pores = new_coords.shape[0]
    n_new_throats = new_conns.shape[0]
    
    print(f"  After filtering: {n_new_pores:,} pores, {n_new_throats:,} throats")
    
    net = op.network.Network(conns=new_conns, coords=new_coords)
    
    mu = 1.0e-3  # water viscosity Pa.s
    r_t = new_throat_radii
    L_t = np.maximum(new_throat_lengths, voxel_size * 0.1)
    net['throat.hydraulic_conductance'] = SHAPE_CORRECTION * np.pi * r_t**4 / (8.0 * mu * L_t) 

    domain = np.array([nx, ny, nz]) * voxel_size
    dir_map = {'x': 0, 'y': 1, 'z': 2}
    ax = dir_map[direction]
    
    margin = 5 * voxel_size
    inlet_pores = new_coords[:, ax] <= margin
    outlet_pores = new_coords[:, ax] >= (domain[ax] - margin)
    
    n_inlet = int(np.sum(inlet_pores))
    n_outlet = int(np.sum(outlet_pores))
    print(f"  Inlet pores:  {n_inlet}")
    print(f"  Outlet pores: {n_outlet}")
    
    if n_inlet == 0 or n_outlet == 0:
        print("  Expanding margin to 10 voxels...")
        margin = 10 * voxel_size
        inlet_pores = new_coords[:, ax] <= margin
        outlet_pores = new_coords[:, ax] >= (domain[ax] - margin)
        n_inlet = int(np.sum(inlet_pores))
        n_outlet = int(np.sum(outlet_pores))
        print(f"  Inlet pores:  {n_inlet}")
        print(f"  Outlet pores: {n_outlet}")
        
        if n_inlet == 0 or n_outlet == 0:
            print("  ERROR: Cannot find inlet/outlet pores. Returning 0.")
            return 0.0
    
    phase = op.phase.Phase(network=net)
    phase['pore.viscosity'] = mu
    phase['throat.hydraulic_conductance'] = net['throat.hydraulic_conductance']
    
    sf = op.algorithms.StokesFlow(network=net, phase=phase)
    sf.set_value_BC(pores=inlet_pores, values=1.0)
    sf.set_value_BC(pores=outlet_pores, values=0.0)
    
    sf.run()
    
    Q = sf.rate(pores=inlet_pores, mode='group')[0]
    
    cross_axes = [i for i in range(3) if i != ax]
    A = domain[cross_axes[0]] * domain[cross_axes[1]]
    L = domain[ax]
    
    K_m2 = abs(Q) * mu * L / (A * 1.0)
    K_darcy = K_m2 / 9.869233e-13
    
    print(f"\n  Flow rate Q = {Q:.6e} m³/s")
    print(f"  Permeability = {K_m2:.6e} m²")
    print(f"  Permeability = {K_darcy:.4f} Darcy")
    print(f"  (Computed on {n_new_pores:,} of {n_pores:,} total pores)")
    
    return K_darcy


def print_network_report(network, voxel_size, nx, ny, nz, perm_darcy):
    """
    Print a comprehensive network statistics report for both sub-networks.
    """
    coords = network['pore.coords']
    conns = network['throat.conns']
    pore_radii = network['pore.radius']
    pore_vols = network['pore.volume']
    pore_labels = network['pore.label']
    throat_radii = network['throat.radius']
    throat_labels = network['throat.label']
    
    n_total_pores = coords.shape[0]
    n_total_throats = conns.shape[0]
    
    macro_p = pore_labels == 0
    micro_p = pore_labels == 1
    macro_t = throat_labels == 0
    micro_t = throat_labels == 1
    coupling_t = throat_labels == 2
    
    n_macro_p = np.sum(macro_p)
    n_micro_p = np.sum(micro_p)
    n_macro_t = np.sum(macro_t)
    n_micro_t = np.sum(micro_t)
    n_coupling_t = np.sum(coupling_t)
    
    coord_num = np.zeros(n_total_pores, dtype=int)
    for i in range(n_total_throats):
        p1, p2 = int(conns[i, 0]), int(conns[i, 1])
        coord_num[p1] += 1
        coord_num[p2] += 1
    
    macro_coord = coord_num[macro_p]
    micro_coord = coord_num[micro_p]
    
    domain_vol = (nx * voxel_size) * (ny * voxel_size) * (nz * voxel_size)
    macro_porosity = np.sum(pore_vols[macro_p]) / domain_vol * 100
    micro_porosity = np.sum(pore_vols[micro_p]) / domain_vol * 100
    total_porosity = (np.sum(pore_vols[macro_p]) + np.sum(pore_vols[micro_p])) / domain_vol * 100
    
    macro_r_p_um = pore_radii[macro_p] * 1e6
    micro_r_p_um = pore_radii[micro_p] * 1e6
    macro_r_t_um = throat_radii[macro_t] * 1e6
    micro_r_t_um = throat_radii[micro_t] * 1e6
    coupling_r_t_um = throat_radii[coupling_t] * 1e6 if n_coupling_t > 0 else np.array([0])
    
    sep = "=" * 70
    print(f"\n{sep}")
    print("DUAL PORE NETWORK MODEL — STATISTICS REPORT")
    print(f"Mount Gambier Limestone | Voxel size: {voxel_size*1e6:.3f} µm")
    print(f"Domain: {nx}×{ny}×{nz} voxels ({nx*voxel_size*1e3:.2f}×{ny*voxel_size*1e3:.2f}×{nz*voxel_size*1e3:.2f} mm³)")
    print(sep)
    
    print(f"\n{'Property':<40} {'Macro':>15} {'Micro':>15}")
    print("-" * 70)
    
    print(f"{'Number of pore bodies':<40} {n_macro_p:>15,} {n_micro_p:>15,}")
    print(f"{'Number of throats':<40} {n_macro_t:>15,} {n_micro_t:>15,}")
    print(f"{'Number of coupling throats':<40} {n_coupling_t:>15,} {'—':>15}")
    
    print(f"{'Mean coordination number':<40} {np.mean(macro_coord):>15.2f} {np.mean(micro_coord):>15.2f}")
    print(f"{'Min coordination number':<40} {np.min(macro_coord):>15d} {np.min(micro_coord):>15d}")
    print(f"{'Max coordination number':<40} {np.max(macro_coord):>15d} {np.max(micro_coord):>15d}")
    
    print(f"{'Mean pore inscribed radius (µm)':<40} {np.mean(macro_r_p_um):>15.2f} {np.mean(micro_r_p_um):>15.2f}")
    print(f"{'Min pore inscribed radius (µm)':<40} {np.min(macro_r_p_um):>15.2f} {np.min(micro_r_p_um):>15.2f}")
    print(f"{'Max pore inscribed radius (µm)':<40} {np.max(macro_r_p_um):>15.2f} {np.max(micro_r_p_um):>15.2f}")
    
    print(f"{'Mean throat inscribed radius (µm)':<40} {np.mean(macro_r_t_um):>15.2f} {np.mean(micro_r_t_um):>15.2f}")
    print(f"{'Min throat inscribed radius (µm)':<40} {np.min(macro_r_t_um):>15.2f} {np.min(micro_r_t_um):>15.2f}")
    print(f"{'Max throat inscribed radius (µm)':<40} {np.max(macro_r_t_um):>15.2f} {np.max(micro_r_t_um):>15.2f}")
    
    if n_coupling_t > 0:
        print(f"\n{'Coupling throat radius — mean (µm)':<40} {np.mean(coupling_r_t_um):>15.2f}")
        print(f"{'Coupling throat radius — min (µm)':<40} {np.min(coupling_r_t_um):>15.2f}")
        print(f"{'Coupling throat radius — max (µm)':<40} {np.max(coupling_r_t_um):>15.2f}")
    
    print(f"\n{'Sub-network porosity (%)':<40} {macro_porosity:>15.2f} {micro_porosity:>15.2f}")
    print(f"{'Total network porosity (%)':<40} {total_porosity:>15.2f}")
    print(f"{'Single-phase permeability (Darcy)':<40} {perm_darcy:>15.4f}")
    
    print(sep)
    
    report_path = os.path.join(OUTPUT_DIR, "dpnm_statistics_report.txt")
    
    report_lines = [
        "DUAL PORE NETWORK MODEL — STATISTICS REPORT",
        f"Mount Gambier Limestone | Voxel size: {voxel_size*1e6:.3f} µm",
        f"Domain: {nx}x{ny}x{nz} voxels",
        "",
        f"Macro pore bodies:          {n_macro_p:,}",
        f"Micro pore bodies:          {n_micro_p:,}",
        f"Macro throats:              {n_macro_t:,}",
        f"Micro throats:              {n_micro_t:,}",
        f"Coupling throats:           {n_coupling_t:,}",
        f"Total pores:                {n_total_pores:,}",
        f"Total throats:              {n_total_throats:,}",
        "",
        f"Macro mean coord. number:   {np.mean(macro_coord):.2f}",
        f"Micro mean coord. number:   {np.mean(micro_coord):.2f}",
        "",
        f"Macro mean pore radius:     {np.mean(macro_r_p_um):.2f} µm",
        f"Micro mean pore radius:     {np.mean(micro_r_p_um):.2f} µm",
        f"Macro mean throat radius:   {np.mean(macro_r_t_um):.2f} µm",
        f"Micro mean throat radius:   {np.mean(micro_r_t_um):.2f} µm",
        "",
        f"Macro sub-network porosity: {macro_porosity:.2f}%",
        f"Micro sub-network porosity: {micro_porosity:.2f}%",
        f"Total network porosity:     {total_porosity:.2f}%",
        f"Single-phase permeability:  {perm_darcy:.4f} Darcy",
    ]
    
    with open(report_path, "w") as f:
        f.write("\n".join(report_lines))
    print(f"\n  Report saved: {report_path}")
    
    return {
        "n_macro_pores": n_macro_p,
        "n_micro_pores": n_micro_p,
        "n_macro_throats": n_macro_t,
        "n_micro_throats": n_micro_t,
        "n_coupling_throats": n_coupling_t,
        "macro_mean_coord": float(np.mean(macro_coord)),
        "micro_mean_coord": float(np.mean(micro_coord)),
        "macro_mean_pore_radius_um": float(np.mean(macro_r_p_um)),
        "micro_mean_pore_radius_um": float(np.mean(micro_r_p_um)),
        "macro_mean_throat_radius_um": float(np.mean(macro_r_t_um)),
        "micro_mean_throat_radius_um": float(np.mean(micro_r_t_um)),
        "macro_porosity_pct": float(macro_porosity),
        "micro_porosity_pct": float(micro_porosity),
        "total_porosity_pct": float(total_porosity),
        "permeability_darcy": float(perm_darcy),
    }


# ============================================================
# MAIN PIPELINE (ANALYSIS & EXPORT ONLY)
# ============================================================
def main():
    in_notebook = any("ipykernel" in m or "IPython" in m for m in sys.modules)
    argv = [] if (in_notebook or len(sys.argv) < 2 or sys.argv[1].startswith("-")) else sys.argv[1:]
    net_file = argv[0] if len(argv) > 0 else NETWORK_FILE

    print("=" * 60)
    print("DUAL PORE NETWORK MODEL (STAGE 2: ANALYSIS & EXPORT)")
    print("Mount Gambier Limestone")
    print("=" * 60)

    t_total = time.time()

    dpnm, voxel_size, (nx, ny, nz) = load_network(net_file)

    perm_darcy = compute_permeability(dpnm, voxel_size, nx, ny, nz, direction='x')

    stats = print_network_report(dpnm, voxel_size, nx, ny, nz, perm_darcy)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    export_csv(dpnm, OUTPUT_DIR)
    export_statoil(dpnm, OUTPUT_DIR)
    export_vtk(dpnm, OUTPUT_DIR)
    export_metadata(dpnm, OUTPUT_DIR, voxel_size, (nx, ny, nz))

    def convert_numpy(obj):
        """Convert numpy types to native Python for JSON serialization."""
        if isinstance(obj, dict):
            return {k: convert_numpy(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [convert_numpy(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    meta_path = os.path.join(OUTPUT_DIR, "dpnm_metadata.json")
    with open(meta_path, "r") as f:
        meta = json.load(f)
    meta["single_phase_permeability_darcy"] = float(perm_darcy)
    meta["single_phase_permeability_m2"] = float(perm_darcy * 9.869233e-13)
    meta["network_statistics"] = convert_numpy(stats)
    meta = convert_numpy(meta)
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    elapsed = time.time() - t_total

    print("\n" + "=" * 60)
    print(f"DONE — Analysis time: {elapsed:.1f}s")
    print("=" * 60)
    print(f"\nOutput files in {OUTPUT_DIR}/:")
    print("  dpnm_pores.csv             — pore data with boundary labels")
    print("  dpnm_throats.csv           — throat data")
    print("  dpnm_node1.dat             — Statoil pore connectivity")
    print("  dpnm_node2.dat             — Statoil pore geometry")
    print("  dpnm_link1.dat             — Statoil throat geometry")
    print("  dpnm_link2.dat             — Statoil throat connectivity")
    print("  dpnm_network.vtk           — VTK for ParaView")
    print("  dpnm_metadata.json         — Network summary + permeability")
    print("  dpnm_statistics_report.txt — Publication-ready statistics")


if __name__ == "__main__":
    main()
