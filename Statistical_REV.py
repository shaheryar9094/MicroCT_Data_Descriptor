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
Statistical REV Estimation — Moving Window Analysis
====================================================
Reads a segmented micro-CT volume (.raw or .tiff) and estimates the
statistical representative elementary volume (REV) by computing
macro-porosity, micro-porosity, and solid fraction within cubic
subvolumes sampled across the domain.

Outputs:
  - Per-window phase fractions (CSV)
  - Summary statistics (console + text file)
  - Convergence plot: phase fraction vs. window size
  - Spatial heterogeneity map

"""

import numpy as np
import os
import csv
import time
from pathlib import Path

# ============================================================
# PARAMETERS
# ============================================================
INPUT_FILE = "MTG_1280CubicVoxels_2.675Microns_3Phase.raw"
NX, NY, NZ = 1280, 1280, 1280
DTYPE = np.uint8
VOXEL_SIZE = 2.675e-6  

LABEL_PORE = 0    
LABEL_MICRO = 1   
LABEL_SOLID = 2   

WINDOW_SIZES = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1280]
STRIDE_FRACTION = 0.5  # stride = window_size * 0.5 (50% overlap)
MIN_STRIDE = 50        

OUTPUT_DIR = "REV_estimation"
# ============================================================


def load_image(filepath, nx, ny, nz, dtype):
    print(f"Loading: {filepath}")
    
    if filepath.endswith('.npy'):
        im = np.load(filepath)
    elif filepath.endswith('.tif') or filepath.endswith('.tiff'):
        from tifffile import imread
        im = imread(filepath)
    else:
        im = np.fromfile(filepath, dtype=dtype)
        expected = nx * ny * nz
        assert im.size == expected, (
            f"Size mismatch: got {im.size}, expected {expected}"
        )
        im = im.reshape((nz, ny, nx))
    
    print(f"  Shape: {im.shape}")
    print(f"  Dtype: {im.dtype}")
    print(f"  Labels: {np.unique(im)}")
    return im


def compute_phase_fractions(subvolume):
    total = subvolume.size
    macro = np.sum(subvolume == LABEL_PORE) / total * 100
    micro = np.sum(subvolume == LABEL_MICRO) / total * 100
    solid = np.sum(subvolume == LABEL_SOLID) / total * 100
    return macro, micro, solid


def moving_window_analysis(im, window_size, stride):
    nz, ny, nx = im.shape
    results = []
    
    x_starts = list(range(0, nx - window_size + 1, stride))
    y_starts = list(range(0, ny - window_size + 1, stride))
    z_starts = list(range(0, nz - window_size + 1, stride))
    
    if x_starts[-1] + window_size < nx:
        x_starts.append(nx - window_size)
    if y_starts[-1] + window_size < ny:
        y_starts.append(ny - window_size)
    if z_starts[-1] + window_size < nz:
        z_starts.append(nz - window_size)
    
    total_windows = len(x_starts) * len(y_starts) * len(z_starts)
    print(f"  Windows: {len(x_starts)}×{len(y_starts)}×{len(z_starts)} = {total_windows}")
    
    count = 0
    for z0 in z_starts:
        for y0 in y_starts:
            for x0 in x_starts:
                sub = im[z0:z0+window_size, y0:y0+window_size, x0:x0+window_size]
                macro, micro, solid = compute_phase_fractions(sub)
                
                results.append({
                    'x0': x0, 'y0': y0, 'z0': z0,
                    'cx': x0 + window_size // 2,
                    'cy': y0 + window_size // 2,
                    'cz': z0 + window_size // 2,
                    'window_size': window_size,
                    'macro_porosity': macro,
                    'micro_porosity': micro,
                    'total_porosity': macro + micro,
                    'solid_fraction': solid,
                })
                count += 1
    
    return results


def convergence_analysis(im, window_sizes):
    nz, ny, nx = im.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    
    results = []
    for ws in window_sizes:
        if ws > min(nx, ny, nz):
            continue
        
        half = ws // 2
        x0 = max(0, cx - half)
        y0 = max(0, cy - half)
        z0 = max(0, cz - half)
        
        x0 = min(x0, nx - ws)
        y0 = min(y0, ny - ws)
        z0 = min(z0, nz - ws)
        
        sub = im[z0:z0+ws, y0:y0+ws, x0:x0+ws]
        macro, micro, solid = compute_phase_fractions(sub)
        
        physical_size_mm = ws * VOXEL_SIZE * 1e3
        
        results.append({
            'window_size_voxels': ws,
            'physical_size_mm': physical_size_mm,
            'macro_porosity': macro,
            'micro_porosity': micro,
            'total_porosity': macro + micro,
            'solid_fraction': solid,
        })
        
        print(f"  {ws:>5d} voxels ({physical_size_mm:.2f} mm): "
              f"ϕ_macro={macro:.2f}% | ϕ_micro={micro:.2f}% | "
              f"ϕ_total={macro+micro:.2f}% | solid={solid:.2f}%")
    
    return results

def export_results(all_window_results, convergence_results, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    window_csv = os.path.join(output_dir, "moving_window_results.csv")
    with open(window_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'window_size', 'x0', 'y0', 'z0', 'cx', 'cy', 'cz',
            'macro_porosity', 'micro_porosity', 'total_porosity', 'solid_fraction'
        ])
        writer.writeheader()
        for results in all_window_results.values():
            writer.writerows(results)
    print(f"  Saved: {window_csv}")
    
    conv_csv = os.path.join(output_dir, "convergence_results.csv")
    with open(conv_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=[
            'window_size_voxels', 'physical_size_mm',
            'macro_porosity', 'micro_porosity', 'total_porosity', 'solid_fraction'
        ])
        writer.writeheader()
        writer.writerows(convergence_results)
    print(f"  Saved: {conv_csv}")
    
    summary_csv = os.path.join(output_dir, "summary_statistics.csv")
    with open(summary_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow([
            'window_size', 'n_windows',
            'macro_mean', 'macro_std', 'macro_min', 'macro_max',
            'micro_mean', 'micro_std', 'micro_min', 'micro_max',
            'total_mean', 'total_std', 'total_min', 'total_max',
            'solid_mean', 'solid_std', 'solid_min', 'solid_max',
        ])
        for ws, results in sorted(all_window_results.items()):
            macro = np.array([r['macro_porosity'] for r in results])
            micro = np.array([r['micro_porosity'] for r in results])
            total = np.array([r['total_porosity'] for r in results])
            solid = np.array([r['solid_fraction'] for r in results])
            
            writer.writerow([
                ws, len(results),
                f"{np.mean(macro):.2f}", f"{np.std(macro):.2f}",
                f"{np.min(macro):.2f}", f"{np.max(macro):.2f}",
                f"{np.mean(micro):.2f}", f"{np.std(micro):.2f}",
                f"{np.min(micro):.2f}", f"{np.max(micro):.2f}",
                f"{np.mean(total):.2f}", f"{np.std(total):.2f}",
                f"{np.min(total):.2f}", f"{np.max(total):.2f}",
                f"{np.mean(solid):.2f}", f"{np.std(solid):.2f}",
                f"{np.min(solid):.2f}", f"{np.max(solid):.2f}",
            ])
    print(f"  Saved: {summary_csv}")
    
    return window_csv, conv_csv, summary_csv


def print_summary(all_window_results, convergence_results):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    sep = "=" * 80
    lines = []
    
    lines.append(sep)
    lines.append("STATISTICAL REV ESTIMATION — MOVING WINDOW ANALYSIS")
    lines.append(f"Mount Gambier Limestone | Voxel size: {VOXEL_SIZE*1e6:.3f} µm")
    lines.append(sep)
    
    lines.append("\nCONVERGENCE ANALYSIS (centred subvolumes):")
    lines.append("-" * 80)
    lines.append(f"{'Size (vox)':<12} {'Size (mm)':<10} {'ϕ_macro%':<10} "
                 f"{'ϕ_micro%':<10} {'ϕ_total%':<10} {'Solid%':<10}")
    lines.append("-" * 80)
    for r in convergence_results:
        lines.append(f"{r['window_size_voxels']:<12d} "
                     f"{r['physical_size_mm']:<10.2f} "
                     f"{r['macro_porosity']:<10.2f} "
                     f"{r['micro_porosity']:<10.2f} "
                     f"{r['total_porosity']:<10.2f} "
                     f"{r['solid_fraction']:<10.2f}")
    
    lines.append(f"\n{'SPATIAL HETEROGENEITY (moving window statistics)':}")
    lines.append("-" * 80)
    lines.append(f"{'Size':<8} {'N':<6} {'ϕ_total mean±std':<20} "
                 f"{'ϕ_total range':<20} {'CoV%':<8}")
    lines.append("-" * 80)
    
    for ws in sorted(all_window_results.keys()):
        results = all_window_results[ws]
        total = np.array([r['total_porosity'] for r in results])
        mean = np.mean(total)
        std = np.std(total)
        cov = (std / mean * 100) if mean > 0 else 0
        
        lines.append(f"{ws:<8d} {len(results):<6d} "
                     f"{mean:.2f} ± {std:.2f}{'':>8} "
                     f"[{np.min(total):.2f} – {np.max(total):.2f}]{'':>5} "
                     f"{cov:.1f}")
    
    lines.append(f"\nFULL IMAGE PHASE FRACTIONS:")
    lines.append("-" * 80)
    if convergence_results:
        full = convergence_results[-1]  # largest window = closest to full image
        lines.append(f"  Macro-porosity:  {full['macro_porosity']:.2f}%")
        lines.append(f"  Micro-porosity:  {full['micro_porosity']:.2f}%")
        lines.append(f"  Total porosity:  {full['total_porosity']:.2f}%")
        lines.append(f"  Solid fraction:  {full['solid_fraction']:.2f}%")
    
    lines.append(sep)
    
    for line in lines:
        print(line)
    
    report_path = os.path.join(OUTPUT_DIR, "rev_estimation_report.txt")
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))
    print(f"\n  Report saved: {report_path}")


def generate_plots(all_window_results, convergence_results):
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
    except ImportError:
        print("  matplotlib not available — skipping plots")
        return
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    sizes = [r['window_size_voxels'] for r in convergence_results]
    macro = [r['macro_porosity'] for r in convergence_results]
    micro = [r['micro_porosity'] for r in convergence_results]
    total = [r['total_porosity'] for r in convergence_results]
    
    ax.plot(sizes, macro, 'g-o', label='Macro-porosity', markersize=5)
    ax.plot(sizes, micro, 'r-s', label='Micro-porosity', markersize=5)
    ax.plot(sizes, total, 'b-^', label='Total porosity', markersize=5)
    
    ax.set_xlabel('Subvolume size (cubic voxels)', fontsize=11)
    ax.set_ylabel('Phase fraction (%)', fontsize=11)
    ax.set_title('Phase fraction convergence with subvolume size')
    ax.legend(fontsize=9, loc='right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "convergence_plot.png")
    plt.savefig(plot1_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot1_path}")
    
    fig, ax = plt.subplots(1, 1, figsize=(8, 5))
    
    ws_list = []
    cov_macro = []
    cov_micro = []
    cov_total = []
    
    for ws in sorted(all_window_results.keys()):
        results = all_window_results[ws]
        if len(results) < 2:
            continue
        
        macro_arr = np.array([r['macro_porosity'] for r in results])
        micro_arr = np.array([r['micro_porosity'] for r in results])
        total_arr = np.array([r['total_porosity'] for r in results])
        
        ws_list.append(ws)
        cov_macro.append(np.std(macro_arr) / np.mean(macro_arr) * 100 if np.mean(macro_arr) > 0 else 0)
        cov_micro.append(np.std(micro_arr) / np.mean(micro_arr) * 100 if np.mean(micro_arr) > 0 else 0)
        cov_total.append(np.std(total_arr) / np.mean(total_arr) * 100 if np.mean(total_arr) > 0 else 0)
    
    ax.plot(ws_list, cov_macro, 'g-o', label='Macro-porosity', markersize=5)
    ax.plot(ws_list, cov_micro, 'r-s', label='Micro-porosity', markersize=5)
    ax.plot(ws_list, cov_total, 'b-^', label='Total porosity', markersize=5)
    
    ax.axhline(y=5, color='grey', linestyle='--', alpha=0.5, label='5% CoV threshold')
    
    ax.set_xlabel('Window size (cubic voxels)', fontsize=11)
    ax.set_ylabel('Coefficient of Variation (%)', fontsize=11)
    ax.set_title('Spatial heterogeneity vs. subvolume size')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "heterogeneity_plot.png")
    plt.savefig(plot2_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot2_path}")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    ws_sorted = sorted(all_window_results.keys())
    ws_with_data = [ws for ws in ws_sorted if len(all_window_results[ws]) > 1]
    
    for idx, (phase, label, color) in enumerate([
        ('macro_porosity', 'Macro-porosity (%)', 'steelblue'),
        ('micro_porosity', 'Micro-porosity (%)', 'seagreen'),
        ('total_porosity', 'Total porosity (%)', 'indianred'),
    ]):
        data = [np.array([r[phase] for r in all_window_results[ws]]) 
                for ws in ws_with_data]
        
        bp = axes[idx].boxplot(data, labels=[str(ws) for ws in ws_with_data],
                               patch_artist=True)
        for patch in bp['boxes']:
            patch.set_facecolor(color)
            patch.set_alpha(0.5)
        
        axes[idx].set_xlabel('Window size (cubic voxels)', fontsize=10)
        axes[idx].set_ylabel(label, fontsize=10)
        axes[idx].tick_params(axis='x', rotation=45)
        axes[idx].grid(True, alpha=0.3)
    
    plt.suptitle('Phase fraction distribution across subvolumes', fontsize=12)
    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, "boxplots.png")
    plt.savefig(plot3_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {plot3_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    t0 = time.time()
    
    print("=" * 60)
    print("STATISTICAL REV ESTIMATION")
    print("Moving Window Analysis")
    print("=" * 60)
    
    im = load_image(INPUT_FILE, NX, NY, NZ, DTYPE)
    
    print("\nFull image phase fractions:")
    macro, micro, solid = compute_phase_fractions(im)
    print(f"  Macro-porosity: {macro:.2f}%")
    print(f"  Micro-porosity: {micro:.2f}%")
    print(f"  Total porosity: {macro + micro:.2f}%")
    print(f"  Solid fraction: {solid:.2f}%")
    
    print("\nConvergence analysis (centred subvolumes):")
    convergence_results = convergence_analysis(im, WINDOW_SIZES)
    
    moving_sizes = [200, 400, 600, 800, 1000]
    all_window_results = {}
    
    for ws in moving_sizes:
        if ws > min(NX, NY, NZ):
            continue
        
        stride = max(MIN_STRIDE, int(ws * STRIDE_FRACTION))
        print(f"\nMoving window: size={ws}, stride={stride}")
        results = moving_window_analysis(im, ws, stride)
        all_window_results[ws] = results
        
        total = np.array([r['total_porosity'] for r in results])
        print(f"  Total porosity: {np.mean(total):.2f}% ± {np.std(total):.2f}% "
              f"[{np.min(total):.2f}–{np.max(total):.2f}%]")
    
    print("\nExporting results...")
    export_results(all_window_results, convergence_results, OUTPUT_DIR)
    
    print_summary(all_window_results, convergence_results)
    
    print("\nGenerating plots...")
    generate_plots(all_window_results, convergence_results)
    
    elapsed = time.time() - t0
    print(f"\nDone — total time: {elapsed:.1f}s")
    print(f"All outputs in: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
