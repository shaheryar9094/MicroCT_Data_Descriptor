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
#   The dataset DOI is: << DOI AVAILABLE ONCE DEPOSIT IS LIVE >>
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
Dual Pore Network Model (DPNM) Extraction — STAGE 1: GENERATION
================================================================
Extracts a geometry-only dual pore network from a 3-phase segmented
micro-CT image and SAVES it to disk.

This is the computationally expensive stage (image loading + SNOW2 watershed 
extraction) and takes abour 2 hrs for Mount Gambier Sample of 1280 cubic 
voxels at 2.675 microns. Run it once; the saved network is then reused by the 
fast analysis stage(`dpnm_analyze.py`) for permeability, statistics and all 
file exports.

Phases:  0 = macropore,  1 = microporous,  2 = solid
Output:  A single pickled dual pore network (macro + micro + coupling),
         including boundary labels and acquisition metadata (voxel size
         and image dimensions), so the analysis stage is self-contained.

Resource reporting
------------------
The run is wrapped in a lightweight background monitor that records, for the
WHOLE process (and any child processes):
  * total wall-clock time,
  * peak RAM (resident set size), and
  * peak disk footprint (output dir + any temporary dpnm_* dirs).
These are printed in a summary at the end and written to a small
`*_resource_usage.json` companion file. 

"""

import numpy as np
import porespy as ps
from scipy import ndimage
from scipy.spatial import cKDTree
import multiprocessing as mp
import subprocess
import sys
import tempfile
import os
import json
import time
import math
import gc
import pickle
import inspect
import threading

try:
    import psutil
    _HAVE_PSUTIL = True
except Exception:
    _HAVE_PSUTIL = False

try:
    import resource
    _HAVE_RESOURCE = True
except Exception:
    _HAVE_RESOURCE = False

# ============================================================
# USER PARAMETERS 
# ============================================================
RAW_FILE = "MTG_1280CubicVoxels_2.675Microns_3Phase.raw"          
NX, NY, NZ = 1280, 1280, 1280                 
VOXEL_SIZE = 2.675e-6                       
DTYPE = np.uint8                       

LABEL_PORE = 0    
LABEL_MICRO = 1   
LABEL_SOLID = 2   

MACRO_SIGMA = 0.4     
MICRO_SIGMA = 0.3     
MIN_MICRO_REGION = 50 

COUPLING_DILATION = 2  
MAX_COUPLING_DIST = 30
SHAPE_CORRECTION = 2

BOUNDARY_MARGIN = 5    

N_CORES = max(1, mp.cpu_count() - 4)  

EXTRACTION_MODE = "inprocess_parallel"
SNOW2_MIN_CHUNK = 128   

OUTPUT_DIR = "DPNM_output"
NETWORK_FILE = os.path.join(OUTPUT_DIR, "dpnm_network.pkl")  

MONITOR_INTERVAL = 0.5  
# ============================================================
# RESOURCE MONITORING (total time / peak RAM / peak storage)
# ============================================================

def _fmt_bytes(n):
    n = float(n)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if n < 1024.0 or unit == "TB":
            return f"{n:.2f} {unit}"
        n /= 1024.0


def _fmt_duration(seconds):
    seconds = int(round(seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m {s:02d}s"
    if m:
        return f"{m}m {s:02d}s"
    return f"{s}s"


def _dir_size(path):
    total = 0
    if not path or not os.path.isdir(path):
        return 0
    for root, _dirs, files in os.walk(path):
        for name in files:
            fp = os.path.join(root, name)
            try:
                total += os.path.getsize(fp)
            except OSError:
                pass
    return total


def _rusage_peak_bytes():
    if not _HAVE_RESOURCE:
        return 0
    try:
        self_rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        child_rss = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss
    except Exception:
        return 0
    scale = 1024 if sys.platform.startswith("linux") else 1
    return (self_rss + child_rss) * scale


class ResourceMonitor:
    def __init__(self, watch_dirs, interval=MONITOR_INTERVAL):
        self.watch_dirs = list(watch_dirs)
        self.interval = interval
        self.peak_rss_bytes = 0
        self.peak_disk_bytes = 0
        self._stop = threading.Event()
        self._thread = None
        self._proc = psutil.Process(os.getpid()) if _HAVE_PSUTIL else None
        self._tmp_root = tempfile.gettempdir()

    def _current_rss(self):
        if self._proc is not None:
            try:
                total = self._proc.memory_info().rss
                for child in self._proc.children(recursive=True):
                    try:
                        total += child.memory_info().rss
                    except Exception:
                        pass
                return total
            except Exception:
                pass
        return _rusage_peak_bytes()

    def _current_disk(self):
        total = 0
        for d in self.watch_dirs:
            total += _dir_size(d)
        try:
            for name in os.listdir(self._tmp_root):
                if name.startswith("dpnm_"):
                    total += _dir_size(os.path.join(self._tmp_root, name))
        except OSError:
            pass
        return total

    def _sample(self):
        self.peak_rss_bytes = max(self.peak_rss_bytes, self._current_rss())
        self.peak_disk_bytes = max(self.peak_disk_bytes, self._current_disk())

    def _run(self):
        while not self._stop.is_set():
            try:
                self._sample()
            except Exception:
                pass
            self._stop.wait(self.interval)

    def start(self):
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=self.interval * 2 + 1)
        try:
            self._sample()
        except Exception:
            pass
        self.peak_rss_bytes = max(self.peak_rss_bytes, _rusage_peak_bytes())

    @property
    def backend(self):
        if _HAVE_PSUTIL:
            return "psutil"
        if _HAVE_RESOURCE:
            return "getrusage"
        return "time only"


def _detect_parallel_kwarg():
    try:
        params = inspect.signature(ps.networks.snow2).parameters
    except (ValueError, TypeError):
        return None
    if "parallel_kw" in params:
        return "parallel_kw"
    if "parallelization" in params:
        return "parallelization"
    return None

PARALLEL_KWARG = _detect_parallel_kwarg()


def load_raw(filepath, nx, ny, nz, dtype):
    print(f"Loading file: {filepath}")
    
    if filepath.endswith('.npy'):
        im = np.load(filepath)
    else:
        im = np.fromfile(filepath, dtype=dtype)
        expected = nx * ny * nz
        assert im.size == expected, (
            f"File size mismatch: got {im.size} voxels, expected {expected}"
        )
        im = im.reshape((nz, ny, nx))
    
    print(f"  Image shape: {im.shape}")
    present = np.nonzero(np.bincount(im.reshape(-1), minlength=256))[0]
    print(f"  Unique labels: {present}")
    return im

def phase_stats(im):
    total = im.size
    counts = np.bincount(im.reshape(-1), minlength=3)
    for label, name in [(LABEL_PORE, "Macropore"),
                        (LABEL_MICRO, "Microporous"),
                        (LABEL_SOLID, "Solid")]:
        count = int(counts[label])
        frac = count / total * 100
        print(f"  {name} (label={label}): {frac:.2f}% ({count:,} voxels)")


# ============================================================
# SNOW2 — IN-PROCESS MULTI-CORE (chunked watershed)
# ============================================================

def _snow2_parallel_settings(shape, n_cores, min_chunk=SNOW2_MIN_CHUNK):

    guess = max(2, math.ceil(n_cores ** (1.0 / 3.0)))
    divs = []
    for dim in shape:
        max_by_size = max(1, dim // min_chunk)
        divs.append(int(min(guess, max_by_size)))
    return {"divs": divs, "cores": int(n_cores)}


def run_snow2(binary_image, voxel_size, sigma, label):

    phases = binary_image if binary_image.dtype == bool else binary_image.astype(bool)
    base = dict(phases=phases, voxel_size=voxel_size, sigma=sigma,
                accuracy="standard")

    use_parallel = (N_CORES >= 2) and (PARALLEL_KWARG is not None)
    if use_parallel:
        settings = _snow2_parallel_settings(phases.shape, N_CORES)
        print(f"  {label} SNOW2 (parallel watershed via '{PARALLEL_KWARG}', "
              f"divs={settings['divs']}, cores={settings['cores']})...")
        try:
            t0 = time.time()
            res = ps.networks.snow2(**{**base, PARALLEL_KWARG: settings})
            print(f"    done in {time.time()-t0:.1f}s")
            return res.network, res.regions
        except Exception as e:
            print(f"    parallel watershed failed ({e}); falling back to serial")

    print(f"  {label} SNOW2 (serial)...")
    t0 = time.time()
    res = ps.networks.snow2(**base)
    print(f"    done in {time.time()-t0:.1f}s")
    return res.network, res.regions


def extract_networks(macro_binary, micro_binary, voxel_size,
                     macro_sigma, micro_sigma):

    print(f"\nExtracting macro + micro networks SEQUENTIALLY "
          f"(each SNOW2 uses up to {N_CORES} cores)...")
    t0 = time.time()

    macro_net, macro_regions = run_snow2(macro_binary, voxel_size,
                                         macro_sigma, "Macro")
    micro_net, micro_regions = run_snow2(micro_binary, voxel_size,
                                         micro_sigma, "Micro")

    elapsed = time.time() - t0
    n_mp = macro_net["pore.coords"].shape[0]
    n_mt = macro_net["throat.conns"].shape[0]
    n_up = micro_net["pore.coords"].shape[0]
    n_ut = micro_net["throat.conns"].shape[0]

    print(f"  Macro network: {n_mp} pores, {n_mt} throats")
    print(f"  Micro network: {n_up} pores, {n_ut} throats")
    print(f"  Total extraction time: {elapsed:.1f}s")

    return macro_net, macro_regions, micro_net, micro_regions


# ============================================================
# SNOW2 PARALLEL EXTRACTION  (LEGACY subprocess-concurrent mode)
# ============================================================

_SNOW2_WORKER_SCRIPT = '''
import sys
import numpy as np
import porespy as ps
import pickle

input_path = sys.argv[1]
output_path = sys.argv[2]
voxel_size = float(sys.argv[3])
sigma = float(sys.argv[4])

binary_image = np.load(input_path)
snow_result = ps.networks.snow2(
    phases=binary_image.astype(bool),
    voxel_size=voxel_size,
    sigma=sigma,
    accuracy="standard"
)

result = {
    "network": dict(snow_result.network),
}
with open(output_path, "wb") as f:
    pickle.dump(result, f, protocol=pickle.HIGHEST_PROTOCOL)
'''


def _run_snow2_sequential(binary_image, voxel_size, sigma, label):
    print(f"  Running {label} SNOW2 sequentially...")
    snow_result = ps.networks.snow2(
        phases=binary_image.astype(bool),
        voxel_size=voxel_size,
        sigma=sigma,
        accuracy='standard'
    )
    return snow_result.network, snow_result.regions


def extract_networks_parallel(macro_binary, micro_binary, voxel_size,
                              macro_sigma, micro_sigma):
    print(f"\nExtracting macro + micro networks in PARALLEL ({N_CORES} cores available)...")
    t0 = time.time()

    tmp_dir = tempfile.mkdtemp(prefix="dpnm_")
    macro_in = os.path.join(tmp_dir, "macro_in.npy")
    micro_in = os.path.join(tmp_dir, "micro_in.npy")
    macro_out = os.path.join(tmp_dir, "macro_out.pkl")
    micro_out = os.path.join(tmp_dir, "micro_out.pkl")
    worker_script = os.path.join(tmp_dir, "snow2_worker.py")

    try:
        np.save(macro_in, macro_binary)
        np.save(micro_in, micro_binary)
        with open(worker_script, "w") as f:
            f.write(_SNOW2_WORKER_SCRIPT)
        print(f"  Temp dir: {tmp_dir}")

        python_exe = sys.executable
        print(f"  Python: {python_exe}")

        proc_macro = subprocess.Popen(
            [python_exe, worker_script, macro_in, macro_out,
             str(voxel_size), str(macro_sigma)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        proc_micro = subprocess.Popen(
            [python_exe, worker_script, micro_in, micro_out,
             str(voxel_size), str(micro_sigma)],
            stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )

        print("  Both SNOW2 subprocesses launched, waiting...")

        _, stderr_macro = proc_macro.communicate()
        _, stderr_micro = proc_micro.communicate()

        if proc_macro.returncode != 0:
            raise RuntimeError(f"Macro SNOW2 failed:\n{stderr_macro.decode()}")
        if proc_micro.returncode != 0:
            raise RuntimeError(f"Micro SNOW2 failed:\n{stderr_micro.decode()}")

        with open(macro_out, "rb") as f:
            macro_result = pickle.load(f)
        with open(micro_out, "rb") as f:
            micro_result = pickle.load(f)

        macro_net = macro_result["network"]
        macro_regions = macro_result.get("regions", None)
        micro_net = micro_result["network"]
        micro_regions = micro_result.get("regions", None)

        print("  Parallel extraction SUCCEEDED")

    except Exception as e:
        print(f"  Parallel extraction failed: {e}")
        print("  Falling back to sequential...")
        macro_net, macro_regions = _run_snow2_sequential(
            macro_binary, voxel_size, macro_sigma, "Macro"
        )
        micro_net, micro_regions = _run_snow2_sequential(
            micro_binary, voxel_size, micro_sigma, "Micro"
        )

    finally:
        for fp in [macro_in, micro_in, macro_out, micro_out, worker_script]:
            try:
                os.remove(fp)
            except OSError:
                pass
        try:
            os.rmdir(tmp_dir)
        except OSError:
            pass

    elapsed = time.time() - t0
    n_mp = macro_net["pore.coords"].shape[0]
    n_mt = macro_net["throat.conns"].shape[0]
    n_up = micro_net["pore.coords"].shape[0]
    n_ut = micro_net["throat.conns"].shape[0]

    print(f"  Macro network: {n_mp} pores, {n_mt} throats")
    print(f"  Micro network: {n_up} pores, {n_ut} throats")
    print(f"  Total extraction time: {elapsed:.1f}s")

    return macro_net, macro_regions, micro_net, micro_regions


def clean_micro_phase(micro_binary, min_size):
    print(f"\nCleaning micro phase (removing clusters < {min_size} voxels)...")
    labeled, n_clusters = ndimage.label(micro_binary)
    print(f"  Found {n_clusters} microporous clusters")

    cluster_ids = np.arange(1, n_clusters + 1)
    sizes = ndimage.sum(micro_binary, labeled, cluster_ids)

    keep_lookup = np.zeros(n_clusters + 1, dtype=bool)
    keep_lookup[1:][sizes >= min_size] = True
    keep_mask = keep_lookup[labeled]

    kept = np.sum(keep_lookup)
    print(f"  Kept {kept} clusters (removed {n_clusters - kept})")
    print(f"  Micro voxels: {np.sum(micro_binary):,} → {np.sum(keep_mask):,}")
    del labeled
    gc.collect()
    return keep_mask


def find_coupling_throats(macro_net, micro_net, macro_binary, micro_binary,
                          dilation, max_dist, voxel_size):
    print("\nFinding macro-micro coupling throats...")

    struct = ndimage.generate_binary_structure(3, 1)
    macro_dilated = ndimage.binary_dilation(
        macro_binary, structure=struct, iterations=dilation
    )
    interface = macro_dilated & micro_binary
    del macro_dilated
    n_interface = np.sum(interface)
    print(f"  Interface voxels: {n_interface:,}")

    if n_interface == 0:
        print("  WARNING: No interface found between macro and micro phases!")
        return np.empty((0, 2), dtype=int), np.empty((0,)), np.empty((0, 3))

    interface_labeled, n_regions = ndimage.label(interface)
    print(f"  Interface regions: {n_regions}")

    region_ids = np.arange(1, n_regions + 1)
    centroids_vox = np.array(
        ndimage.center_of_mass(interface, interface_labeled, region_ids)
    )
    centroids_m = centroids_vox * voxel_size

    region_sizes = ndimage.sum(np.ones_like(interface, dtype=int),
                               interface_labeled, region_ids)
    del interface, interface_labeled
    gc.collect()

    macro_coords = macro_net["pore.coords"]
    micro_coords = micro_net["pore.coords"]
    macro_tree = cKDTree(macro_coords)
    micro_tree = cKDTree(micro_coords)

    dist_macro, idx_macro = macro_tree.query(centroids_m, workers=N_CORES)
    dist_micro, idx_micro = micro_tree.query(centroids_m, workers=N_CORES)

    total_dist = dist_macro + dist_micro
    max_dist_m = max_dist * voxel_size

    valid = total_dist < max_dist_m
    n_valid = np.sum(valid)

    if n_valid == 0:
        print("  No coupling throats within distance threshold!")
        return np.empty((0, 2), dtype=int), np.empty((0,)), np.empty((0, 3))

    coupling_conns = np.column_stack([idx_macro[valid], idx_micro[valid]])
    coupling_centers = centroids_m[valid]


    region_vols_m3 = region_sizes[valid] * (voxel_size ** 3)
    coupling_radii = (3 * region_vols_m3 / (4 * np.pi)) ** (1/3)

    print(f"  Coupling throats created: {n_valid}")
    return coupling_conns, coupling_radii, coupling_centers


def label_boundary_pores(coords, voxel_size, nx, ny, nz, margin):
    print(f"\nLabeling boundary pores (margin = {margin} voxels)...")

    n_pores = coords.shape[0]
    margin_m = margin * voxel_size
    domain_max = np.array([nx, ny, nz]) * voxel_size

    boundary_id = np.zeros(n_pores, dtype=int)
    name_map = {0: "internal", 1: "x_min", 2: "x_max",
                3: "y_min", 4: "y_max", 5: "z_min", 6: "z_max"}

    boundary_id[coords[:, 2] >= (domain_max[2] - margin_m)] = 6
    boundary_id[coords[:, 2] <= margin_m]                    = 5
    boundary_id[coords[:, 1] >= (domain_max[1] - margin_m)] = 4
    boundary_id[coords[:, 1] <= margin_m]                    = 3
    boundary_id[coords[:, 0] >= (domain_max[0] - margin_m)] = 2
    boundary_id[coords[:, 0] <= margin_m]                    = 1

    boundary_name = np.array([name_map[b] for b in boundary_id], dtype="U10")

    for fid, fname in sorted(name_map.items()):
        c = np.sum(boundary_id == fid)
        print(f"  {fname}: {c} pores")

    return boundary_id, boundary_name


def build_dual_network(macro_net, micro_net, coupling_conns, coupling_radii,
                       coupling_centers, voxel_size):

    print("\nBuilding combined dual pore network...")

    n_macro_pores = macro_net["pore.coords"].shape[0]
    n_micro_pores = micro_net["pore.coords"].shape[0]
    n_macro_throats = macro_net["throat.conns"].shape[0]
    n_micro_throats = micro_net["throat.conns"].shape[0]
    n_coupling = len(coupling_conns)

    total_pores = n_macro_pores + n_micro_pores
    total_throats = n_macro_throats + n_micro_throats + n_coupling

    print(f"  Total pores:   {total_pores} ({n_macro_pores} macro + {n_micro_pores} micro)")
    print(f"  Total throats: {total_throats} ({n_macro_throats} macro + {n_micro_throats} micro + {n_coupling} coupling)")

    coords = np.vstack([macro_net["pore.coords"], micro_net["pore.coords"]])

    macro_radii_p = macro_net.get("pore.inscribed_diameter",
                     macro_net.get("pore.equivalent_diameter",
                     np.zeros(n_macro_pores))) / 2.0
    micro_radii_p = micro_net.get("pore.inscribed_diameter",
                     micro_net.get("pore.equivalent_diameter",
                     np.zeros(n_micro_pores))) / 2.0
    if np.all(macro_radii_p == 0):
        macro_radii_p = macro_net.get("pore.diameter", np.ones(n_macro_pores) * voxel_size) / 2.0
    if np.all(micro_radii_p == 0):
        micro_radii_p = micro_net.get("pore.diameter", np.ones(n_micro_pores) * voxel_size) / 2.0

    pore_radii = np.concatenate([macro_radii_p, micro_radii_p])

    macro_vols = macro_net.get("pore.volume",
                  macro_net.get("pore.region_volume",
                  (4/3) * np.pi * macro_radii_p**3))
    micro_vols = micro_net.get("pore.volume",
                  micro_net.get("pore.region_volume",
                  (4/3) * np.pi * micro_radii_p**3))
    pore_volumes = np.concatenate([macro_vols, micro_vols])

    pore_label = np.concatenate([
        np.zeros(n_macro_pores, dtype=int),
        np.ones(n_micro_pores, dtype=int)
    ])

    macro_conns = macro_net["throat.conns"]
    micro_conns = micro_net["throat.conns"] + n_macro_pores

    if n_coupling > 0:
        coupling_conns_offset = coupling_conns.copy()
        coupling_conns_offset[:, 1] += n_macro_pores
        all_conns = np.vstack([macro_conns, micro_conns, coupling_conns_offset])
    else:
        all_conns = np.vstack([macro_conns, micro_conns])

    macro_radii_t = macro_net.get("throat.inscribed_diameter",
                     macro_net.get("throat.equivalent_diameter",
                     macro_net.get("throat.diameter",
                     np.ones(n_macro_throats) * voxel_size))) / 2.0
    micro_radii_t = micro_net.get("throat.inscribed_diameter",
                     micro_net.get("throat.equivalent_diameter",
                     micro_net.get("throat.diameter",
                     np.ones(n_micro_throats) * voxel_size))) / 2.0

    throat_radii = np.concatenate([macro_radii_t, micro_radii_t, coupling_radii]) \
                   if n_coupling > 0 else np.concatenate([macro_radii_t, micro_radii_t])

    throat_lengths = np.linalg.norm(
        coords[all_conns[:, 0]] - coords[all_conns[:, 1]], axis=1
    )

    throat_label = np.concatenate([
        np.zeros(n_macro_throats, dtype=int),
        np.ones(n_micro_throats, dtype=int),
        np.full(n_coupling, 2, dtype=int)
    ])

    network = {
        "pore.coords": coords,
        "pore.radius": pore_radii,
        "pore.volume": pore_volumes,
        "pore.label": pore_label,
        "throat.conns": all_conns,
        "throat.radius": throat_radii,
        "throat.length": throat_lengths,
        "throat.label": throat_label,
        "_n_macro_pores": n_macro_pores,
        "_n_micro_pores": n_micro_pores,
        "_n_macro_throats": n_macro_throats,
        "_n_micro_throats": n_micro_throats,
        "_n_coupling_throats": n_coupling,
    }

    return network


# ============================================================
# SAVE THE GENERATED NETWORK
# ============================================================

def save_network(network, filepath, voxel_size, dims, n_cores):
    """
    Persist the dual pore network so the analysis stage can reload it
    without repeating the expensive SNOW2 extraction.

    The saved pickle carries:
      * every pore.* / throat.* array
      * the sub-network counts 
      * the boundary labels (pore.boundary_id / pore.boundary_name)
      * acquisition metadata (voxel_size, nx, ny, nz, n_cores)
    so `dpnm_analyze.py` is fully self-contained and does not depend on
    its hard-coded constants matching the ones used here.
    """
    print(f"\nSaving dual network...")
    nx, ny, nz = dims

    payload = dict(network)
    payload["_voxel_size"] = float(voxel_size)
    payload["_nx"] = int(nx)
    payload["_ny"] = int(ny)
    payload["_nz"] = int(nz)
    payload["_n_cores"] = int(n_cores)
    payload["_format_version"] = 1

    out_dir = os.path.dirname(os.path.abspath(filepath))
    os.makedirs(out_dir, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)

    size_mb = os.path.getsize(filepath) / (1024 ** 2)
    print(f"  Saved: {filepath} ({size_mb:.1f} MB)")

    info = {
        "network_file": os.path.abspath(filepath),
        "voxel_size_m": float(voxel_size),
        "image_dimensions_voxels": [int(nx), int(ny), int(nz)],
        "n_cores_used": int(n_cores),
        "n_macro_pores": int(network["_n_macro_pores"]),
        "n_micro_pores": int(network["_n_micro_pores"]),
        "n_total_pores": int(network["pore.coords"].shape[0]),
        "n_macro_throats": int(network["_n_macro_throats"]),
        "n_micro_throats": int(network["_n_micro_throats"]),
        "n_coupling_throats": int(network["_n_coupling_throats"]),
        "n_total_throats": int(network["throat.conns"].shape[0]),
    }
    info_path = os.path.splitext(filepath)[0] + "_generation_info.json"
    with open(info_path, "w") as f:
        json.dump(info, f, indent=2)
    print(f"  Saved: {info_path}")

    return filepath


# ============================================================
# MAIN PIPELINE (GENERATION ONLY)
# ============================================================
def main():
    in_notebook = any("ipykernel" in m or "IPython" in m for m in sys.modules)
    argv = [] if (in_notebook or len(sys.argv) < 2 or sys.argv[1].startswith("-")) else sys.argv[1:]
    raw_file = argv[0] if len(argv) > 0 else RAW_FILE
    net_file = argv[1] if len(argv) > 1 else NETWORK_FILE

    print("=" * 60)
    print("DUAL PORE NETWORK MODEL EXTRACTION (STAGE 1: GENERATION)")
    print(f"CPU cores available: {mp.cpu_count()}, using: {N_CORES}")
    print(f"Extraction mode: {EXTRACTION_MODE}")
    if EXTRACTION_MODE == "inprocess_parallel":
        print(f"SNOW2 parallel keyword: {PARALLEL_KWARG or 'unavailable (serial)'}")
    print("Mount Gambier Limestone")
    print("=" * 60)

    out_dir = os.path.dirname(os.path.abspath(net_file))
    monitor = ResourceMonitor(watch_dirs=[out_dir])
    print(f"Resource monitor backend: {monitor.backend}"
          f"{'' if _HAVE_PSUTIL else '  (install psutil for best peak-RAM accuracy)'}")
    monitor.start()

    t_total = time.time()
    elapsed = None
    try:
        im = load_raw(raw_file, NX, NY, NZ, DTYPE)
        phase_stats(im)

        macro_binary = (im == LABEL_PORE)
        micro_binary = (im == LABEL_MICRO)

        print(f"\nMacro porosity: {np.mean(macro_binary)*100:.2f}%")
        print(f"Micro porosity: {np.mean(micro_binary)*100:.2f}%")
        print(f"Total porosity: {(np.mean(macro_binary) + np.mean(micro_binary))*100:.2f}%")

        del im
        gc.collect()

        micro_clean = clean_micro_phase(micro_binary, MIN_MICRO_REGION)
        del micro_binary
        gc.collect()

        if EXTRACTION_MODE == "subprocess_concurrent":
            macro_net, macro_regions, micro_net, micro_regions = \
                extract_networks_parallel(
                    macro_binary, micro_clean, VOXEL_SIZE, MACRO_SIGMA, MICRO_SIGMA
                )
        else:
            macro_net, macro_regions, micro_net, micro_regions = \
                extract_networks(
                    macro_binary, micro_clean, VOXEL_SIZE, MACRO_SIGMA, MICRO_SIGMA
                )

        del macro_regions, micro_regions
        gc.collect()

        coupling_conns, coupling_radii, coupling_centers = find_coupling_throats(
            macro_net, micro_net, macro_binary, micro_clean,
            COUPLING_DILATION, MAX_COUPLING_DIST, VOXEL_SIZE
        )

        del macro_binary, micro_clean
        gc.collect()

        dpnm = build_dual_network(
            macro_net, micro_net, coupling_conns, coupling_radii,
            coupling_centers, VOXEL_SIZE
        )

        boundary_id, boundary_name = label_boundary_pores(
            dpnm["pore.coords"], VOXEL_SIZE, NX, NY, NZ, BOUNDARY_MARGIN
        )
        dpnm["pore.boundary_id"] = boundary_id
        dpnm["pore.boundary_name"] = boundary_name

        save_network(dpnm, net_file, VOXEL_SIZE, (NX, NY, NZ), N_CORES)

        elapsed = time.time() - t_total

        print("\n" + "=" * 60)
        print(f"DONE — Generation time: {elapsed:.1f}s")
        print("=" * 60)
        print(f"\nSaved network: {net_file}")
        print("\nNext step — run the analysis stage:")
        print(f"  python dpnm_analyze.py {net_file}")
        print("\nThe analysis stage computes permeability, prints the")
        print("statistics report, and writes CSV / Statoil / VTK / metadata.")

    finally:
        if elapsed is None:
            elapsed = time.time() - t_total
        monitor.stop()

        final_output_bytes = _dir_size(out_dir)

        print("\n" + "=" * 60)
        print("RESOURCE USAGE SUMMARY")
        print("=" * 60)
        print(f"  Total wall-clock time : {_fmt_duration(elapsed)} ({elapsed:.1f}s)")
        print(f"  Peak RAM (RSS)        : {_fmt_bytes(monitor.peak_rss_bytes)}"
              f"{'' if _HAVE_PSUTIL else '  (getrusage estimate)'}")
        print(f"  Peak disk footprint   : {_fmt_bytes(monitor.peak_disk_bytes)}")
        print(f"  Final output on disk  : {_fmt_bytes(final_output_bytes)}  ({out_dir})")
        print("  (peak disk = output dir + any temporary dpnm_* dirs, sampled during the run)")
        print("=" * 60)

        try:
            resource_info = {
                "total_wall_clock_seconds": round(float(elapsed), 3),
                "total_wall_clock_human": _fmt_duration(elapsed),
                "peak_ram_bytes": int(monitor.peak_rss_bytes),
                "peak_ram_human": _fmt_bytes(monitor.peak_rss_bytes),
                "peak_disk_bytes": int(monitor.peak_disk_bytes),
                "peak_disk_human": _fmt_bytes(monitor.peak_disk_bytes),
                "final_output_bytes": int(final_output_bytes),
                "final_output_human": _fmt_bytes(final_output_bytes),
                "ram_backend": monitor.backend,
                "extraction_mode": EXTRACTION_MODE,
                "n_cores_used": int(N_CORES),
            }
            res_path = os.path.splitext(os.path.abspath(net_file))[0] + "_resource_usage.json"
            os.makedirs(os.path.dirname(res_path), exist_ok=True)
            with open(res_path, "w") as f:
                json.dump(resource_info, f, indent=2)
            print(f"  Saved resource usage: {res_path}")
        except Exception as e:
            print(f"  (could not write resource-usage JSON: {e})")


if __name__ == "__main__":
    main()
