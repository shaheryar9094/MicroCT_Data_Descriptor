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
NetCDF (.nc) Stack to TIFF Converter
=====================================
Reads all .nc files from a folder, extracts the image data,
and saves as a single multi-slices TIFF stack for processing.

Usage:
    1. Set NC_FOLDER to your folder containing .nc files
    2. Run the script
    3. Open the output .tif in image processing software
"""

import os
import sys
import glob
import numpy as np
import netCDF4 as nc
from tifffile import imwrite
from natsort import natsorted  

# ============================================================
# USER PARAMETERS
# ============================================================
NC_FOLDER = r"tomo_R_nc"     
OUTPUT_FILE = "MTG_Original.tif"               
VARIABLE_NAME = None                     
# ============================================================


def inspect_nc(filepath):
    ds = nc.Dataset(filepath)
    print(f"\n  File: {os.path.basename(filepath)}")
    print(f"  Dimensions: {dict(ds.dimensions)}")
    print(f"  Variables:")
    for name, var in ds.variables.items():
        print(f"    '{name}': shape={var.shape}, dtype={var.dtype}")
    ds.close()


def find_image_variable(ds):
    best_name = None
    best_size = 0

    dim_names = set(ds.dimensions.keys())

    for name, var in ds.variables.items():
        if name in dim_names and var.ndim == 1:
            continue
        size = np.prod(var.shape)
        if size > best_size:
            best_size = size
            best_name = name

    return best_name


def load_nc_slice(filepath, var_name=None):
    ds = nc.Dataset(filepath)

    if var_name is None:
        var_name = find_image_variable(ds)
        if var_name is None:
            ds.close()
            raise ValueError(f"Could not auto-detect image variable in {filepath}")

    data = np.array(ds.variables[var_name][:])
    ds.close()
    return data, var_name


def main():
    print("=" * 60)
    print("NetCDF (.nc) Stack to TIFF Converter")
    print("=" * 60)

    nc_files = glob.glob(os.path.join(NC_FOLDER, "*.nc"))
    if not nc_files:
        print(f"\nERROR: No .nc files found in: {NC_FOLDER}")
        print("  Check your NC_FOLDER path.")
        sys.exit(1)

    try:
        nc_files = natsorted(nc_files)
    except:
        nc_files = sorted(nc_files)

    print(f"\nFound {len(nc_files)} .nc files in: {NC_FOLDER}")

    inspect_nc(nc_files[0])

    first_slice, var_name = load_nc_slice(nc_files[0], VARIABLE_NAME)
    print(f"\n  Using variable: '{var_name}'")
    print(f"  Slice shape: {first_slice.shape}, dtype: {first_slice.dtype}")

    if first_slice.ndim == 2:
        mode = "2D_slices"
        ny, nx = first_slice.shape
        nz = len(nc_files)
        print(f"\n  Mode: stacking {nz} 2D slices ({ny} × {nx} each)")

    elif first_slice.ndim == 3:
        mode = "3D_chunks"
        print(f"\n  Mode: concatenating {len(nc_files)} 2D slices")

    else:
        first_slice = first_slice.squeeze()
        if first_slice.ndim == 2:
            mode = "2D_slices"
            ny, nx = first_slice.shape
            nz = len(nc_files)
            print(f"\n  Mode: stacking {nz} squeezed 2D slices ({ny} × {nx})")
        else:
            print(f"\n  ERROR: Unexpected data shape: {first_slice.shape}")
            sys.exit(1)

    print(f"\n  Loading slices...")
    slices = [first_slice.squeeze() if mode == "2D_slices" else first_slice]

    for i, fpath in enumerate(nc_files[1:], start=2):
        data, _ = load_nc_slice(fpath, var_name)
        if mode == "2D_slices":
            data = data.squeeze()
        slices.append(data)

        if i % 100 == 0 or i == len(nc_files):
            print(f"    Loaded {i}/{len(nc_files)}")

    print(f"    Loaded {len(nc_files)}/{len(nc_files)}")

    print(f"\n  Stacking into volume...")
    if mode == "2D_slices":
        volume = np.stack(slices, axis=0)
    else:
        volume = np.concatenate(slices, axis=0)

    print(f"  Volume shape: {volume.shape}")
    print(f"  Volume dtype: {volume.dtype}")
    print(f"  Value range: [{volume.min()}, {volume.max()}]")
    print(f"  Memory: {volume.nbytes / 1e9:.2f} GB")

    output_path = os.path.join(NC_FOLDER, OUTPUT_FILE)
    print(f"\n  Saving to: {output_path}")
    imwrite(output_path, volume, photometric='minisblack')

    size_mb = os.path.getsize(output_path) / 1e6
    print(f"  Done! File size: {size_mb:.1f} MB")

if __name__ == "__main__":
    main()
