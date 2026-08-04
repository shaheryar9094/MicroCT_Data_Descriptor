# Micro-CT — Analysis & Modelling Code

[![License: CC BY 4.0](https://img.shields.io/badge/License-CC%20BY%204.0-blue.svg)](https://creativecommons.org/licenses/by/4.0/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)

Supplementary code for the data paper:

> **High-resolution micro-CT dataset of Mount Gambier limestone with three-phase reusable segmentation classifier, dual pore network model and finite-volume multiscale meshes**
> Shaheryar T. Hussain, Klaus Regenauer-Lieb, Sheik S. Rahman. *(under review)*

## Overview

The scripts cover the workflow for converting micro-CT slices to a workable image stack, estimating a representative elementary volume (REV) on the segmented volume, and extracting/analysing a dual (macro + micro) pore network. Segmentation into the three phases (macropore / microporous / solid) is performed separately using the reusable classifier described in the paper; its output — a 3-phase `.raw` volume (`0 = macropore`, `1 = microporous`, `2 = solid`) — is the input to the REV and DPNM scripts.

## Repository contents

| File | Description |
|------|-------------|
| `NetCDF_to_Tif.py` | Converts a folder of reconstructed micro-CT `.nc` slices into a single multi-slice TIFF stack for downstream segmentation and processing. |
| `Statistical_REV.py` | Estimates the statistical REV of the segmented 3-phase volume via centred-subvolume convergence and moving-window heterogeneity analysis. Outputs per-window CSVs, a summary report, and convergence / heterogeneity / boxplot figures. |
| `DPNM_Stage1.py` | **Stage 1 — generation.** Extracts a geometry-only dual pore network (macro + micro + coupling throats) from the 3-phase image using PoreSpy's SNOW2 watershed. Computationally heavy (~2 h for the 1280³ sample which required 168GB of RAM); run once. |
| `DPNM_Stage2.py` | **Stage 2 — analysis & export.** Loads the saved network, computes single-phase absolute permeability (OpenPNM `StokesFlow`) and network statistics, and exports CSV, Statoil/ICL `.dat`, VTK (ParaView), and JSON metadata. Fast and re-runnable. |

The dual pore network is split into two stages so the expensive extraction (Stage 1) runs once, while analysis and exports (Stage 2) can be re-run freely.

## Requirements

Python ≥ 3.9 with:

```bash
pip install numpy scipy porespy openpnm netCDF4 tifffile natsort matplotlib psutil
```

## Usage

Each script exposes its input paths and parameters in a `USER PARAMETERS` block near the top, edit these to point at your data before running.

```bash
# 1. Reconstructed .nc slices  ->  single TIFF stack
python NetCDF_to_Tif.py

# 2. REV estimation on the segmented 3-phase volume
python Statistical_REV.py

# 3. Dual pore network — generation
python DPNM_Stage1.py 

# 4. Dual pore network — analysis & export 
python DPNM_Stage2.py 
```

## Data availability

All data described in the manuscript are publicly available on the [Digital Porous Media Portal](https://digitalporousmedia.org). *DOI available once the deposit is live*.

## Citation

If this code or dataset contributes to your work, please cite the associated manuscript/dataset:

```
Hussain, S. T., Regenauer-Lieb, K., & Rahman, S. S. High-resolution micro-CT dataset
of Mount Gambier limestone with three-phase reusable segmentation classifier, dual pore
network model and finite-volume multiscale meshes. (Manuscript under review.)

Hussain, S., Regenauer-Lieb, K., & Rahman, S. Micro-CT dataset of Mount Gambier Limestone
with Segmentation Classifier, Dual Pore Network Model and Finite Volume Meshes. Digital
Porous Media. DOI: (in process)
```

*(Full citation and DOI to be updated upon publication.)*

## License

Released under the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
