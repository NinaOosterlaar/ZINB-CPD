"""
Run sliding_ZINB_CPD_v3_SATAY.py on all strains from Data/combined_strains.
Processes all chromosomes for each strain with their corresponding density files.
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Thread-safe lock for printing
print_lock = Lock()


def safe_print(*args, **kwargs):
    """Thread-safe print function."""
    with print_lock:
        print(*args, **kwargs)


def read_count_data(csv_file):
    """Read count data from a CSV file."""
    df = pd.read_csv(csv_file)
    if "Value" in df.columns:
        return df["Value"].astype(float).to_numpy()
    if len(df.columns) < 2:
        raise ValueError(f"Expected at least two columns in {csv_file}")
    return df.iloc[:, 1].astype(float).to_numpy()


def remove_problematic_positions(data, chrom_name):
    """
    Remove known problematic positions by setting them to zero.
    
    Args:
        data: Array of count values
        chrom_name: Name of chromosome (e.g., 'ChrXV')
    
    Returns:
        cleaned_data: Data with problematic positions set to zero
        n_removed: Number of positions removed
    """
    data_cleaned = np.array(data, copy=True)
    n_removed = 0
    
    # Known problematic positions: (chromosome, position)
    problematic_positions = [
        ('ChrXV', 565596),
    ]
    
    for prob_chrom, prob_pos in problematic_positions:
        if chrom_name == prob_chrom and prob_pos < len(data_cleaned):
            if data_cleaned[prob_pos] != 0:
                data_cleaned[prob_pos] = 0
                n_removed += 1
    
    return data_cleaned, n_removed


def compute_global_outlier_threshold(strain_folder):
    """Compute 95th percentile threshold across all chromosomes in a strain (non-zero values only)."""
    all_data = []
    chromosome_files = sorted(strain_folder.glob("Chr*_distances.csv"))
    total_removed = 0
    
    for chrom_file in chromosome_files:
        try:
            # Extract chromosome name
            chrom_name = chrom_file.stem.replace("_distances", "")

            # Read data
            data = read_count_data(chrom_file)
            
            # Remove problematic positions
            data, n_removed = remove_problematic_positions(data, chrom_name)
            total_removed += n_removed
            
            all_data.extend(data)
        except Exception as e:
            print(f"    Warning: Could not read {chrom_file.name}: {e}")
    
    if not all_data:
        return None
    
    all_data = np.array(all_data)
    
    # Compute threshold only on non-zero values
    non_zero_data = all_data[all_data > 0]
    if len(non_zero_data) == 0:
        return None
    
    threshold = np.quantile(non_zero_data, 0.95)  # 95th percentile = top 5%
    n_outliers = np.sum(all_data > threshold)
    
    return threshold, n_outliers, len(all_data), total_removed


def process_strain(
    strain_folder,
    centromere_base,
    nucleosome_base,
    output_base,
    cpd_script,
    window_sizes,
    overlap,
    threshold_start,
    threshold_end,
    threshold_step,
    theta_block_size,
    n_workers,
    timeout_seconds,
):
    """
    Process a single strain: run CPD analysis on all chromosomes.
    
    Returns:
        (strain_name, n_processed, n_errors, n_skipped)
    """
    strain_name = strain_folder.name
    n_processed = 0
    n_errors = 0
    n_skipped = 0
    
    safe_print(f"\n{'='*80}")
    safe_print(f"Processing {strain_name}")
    safe_print(f"{'='*80}")
    
    # Find corresponding density files
    centromere_file = centromere_base / f"dataset-{strain_name}_combined_centromere_density_Boolean_True_bin_10000_absolute.csv"
    nucleosome_file = nucleosome_base / f"dataset-{strain_name}_combined_Boolean_True_nucleosome_density.csv"
    
    # Check if density files exist
    if not centromere_file.exists():
        safe_print(f"  ⚠ Warning: Centromere density file not found:")
        safe_print(f"    {centromere_file.name}")
        safe_print(f"  Skipping {strain_name}")
        return (strain_name, n_processed, n_errors, 1)
    
    if not nucleosome_file.exists():
        safe_print(f"  ⚠ Warning: Nucleosome density file not found:")
        safe_print(f"    {nucleosome_file.name}")
        safe_print(f"  Skipping {strain_name}")
        return (strain_name, n_processed, n_errors, 1)
    
    safe_print(f"  ✓ Centromere file: {centromere_file.name}")
    safe_print(f"  ✓ Nucleosome file: {nucleosome_file.name}")
    safe_print()
    
    # Find all chromosome files
    chromosome_files = sorted(strain_folder.glob("Chr*_distances.csv"))
    
    if not chromosome_files:
        safe_print(f"  ⚠ No chromosome files found in {strain_name}")
        return (strain_name, n_processed, n_errors, 1)
    
    # Compute global outlier threshold across all chromosomes for this strain
    safe_print(f"  Computing global outlier threshold across all chromosomes...")
    threshold_result = compute_global_outlier_threshold(strain_folder)
    
    if threshold_result is None:
        safe_print(f"  ⚠ Could not compute outlier threshold for {strain_name}")
        return (strain_name, n_processed, n_errors, 1)
    
    outlier_threshold, n_outliers, n_total, n_problematic = threshold_result
    if n_problematic > 0:
        safe_print(f"  Removed {n_problematic} problematic position(s) (set to zero)")
    safe_print(f"  Global 95th percentile (non-zero): {outlier_threshold:.1f}")
    safe_print(f"  Total outliers to cap: {n_outliers} ({100*n_outliers/n_total:.2f}%) across all chromosomes")
    safe_print()
    
    safe_print(f"  Processing {len(chromosome_files)} chromosomes:")
    safe_print(f"  Thresholds: {threshold_start} to {threshold_end} (step {threshold_step})")
    safe_print()
    
    # Process each chromosome
    for chrom_file in chromosome_files:
        # Extract chromosome name (e.g., "ChrI" from "ChrI_distances.csv")
        chrom_name = chrom_file.stem.replace("_distances", "")
        
        # Output folder: SATAY_CPD_results/CPD_SATAY_results/{strain_name}/{chromosome}/
        output_folder = output_base / strain_name / chrom_name
        
        # Build command - pass threshold range parameters
        cmd = [
            sys.executable,
            str(cpd_script),
            str(chrom_file),
            "--output_folder", str(output_folder),
            "--window_sizes", str(window_sizes[0]),
            "--overlap", str(overlap),
            "--theta_block_size", str(theta_block_size),
            "--threshold_start", str(threshold_start),
            "--threshold_end", str(threshold_end),
            "--threshold_step", str(threshold_step),
            "--outlier_threshold", str(outlier_threshold),
            "--n_workers", str(n_workers),
            "--nucleosome_file", str(nucleosome_file),
            "--centromere_file", str(centromere_file),
        ]
        
        # Run the command
        try:
            safe_print(f"    Processing {chrom_name}...")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT,
                timeout=timeout_seconds
            )
            
            if result.returncode == 0:
                # Print key diagnostic output (theta, outliers)
                if result.stdout:
                    for line in result.stdout.strip().split('\n'):
                        # Show lines about outlier removal and theta estimation
                        if any(keyword in line.lower() for keyword in 
                               ['outlier', 'theta', 'estimated', 'block', 'capped']):
                            safe_print(f"      {line}")
                safe_print(f"    ✓ {chrom_name} complete")
                n_processed += 1
            else:
                safe_print(f"    ✗ {chrom_name} failed (exit code {result.returncode})")
                n_errors += 1
                if result.stderr:
                    safe_print(f"      Error output:")
                    for line in result.stderr.strip().split('\n')[:10]:  # Show first 10 lines
                        safe_print(f"        {line}")
                    if len(result.stderr.strip().split('\n')) > 10:
                        safe_print(f"        ... (truncated)")
                
        except subprocess.TimeoutExpired:
            safe_print(f"    ✗ timeout after {timeout_seconds}s")
            n_errors += 1
        except Exception as e:
            safe_print(f"    ✗ {str(e)[:80]}")
            n_errors += 1
    
    safe_print(f"\n  Completed {strain_name}")
    
    return (strain_name, n_processed, n_errors, n_skipped)


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Run sliding_ZINB_CPD_v3_SATAY.py on all strains from Data/combined_strains."
    )
    parser.add_argument(
        "--threshold_start",
        type=float,
        default=5.0,
        help="Minimum threshold. Default: 5.0",
    )
    parser.add_argument(
        "--threshold_end",
        type=float,
        default=10.0,
        help="Maximum threshold. Default: 10.0",
    )
    parser.add_argument(
        "--threshold_step",
        type=float,
        default=0.5,
        help="Threshold step size. Default: 0.5",
    )
    parser.add_argument(
        "--n_strain_workers",
        type=int,
        default=1,
        help="Number of strains to process in parallel. Default: 1 (sequential)",
    )
    parser.add_argument(
        "--strains_data",
        type=Path,
        default=PROJECT_ROOT / "Data" / "combined_strains",
        help="Folder containing strain-level chromosome distance files.",
    )
    parser.add_argument(
        "--centromere_base",
        type=Path,
        default=PROJECT_ROOT / "Data_exploration" / "results" / "densities" / "centromere_strains" / "combined_Datasets_Boolean_True_bin_10000_absolute",
        help="Folder containing strain centromere density files.",
    )
    parser.add_argument(
        "--nucleosome_base",
        type=Path,
        default=PROJECT_ROOT / "Data_exploration" / "results" / "densities" / "nucleosome_strains" / "combined_Datasets_Boolean_True",
        help="Folder containing strain nucleosome density files.",
    )
    parser.add_argument(
        "--output_base",
        type=Path,
        default=PROJECT_ROOT / "SATAY_CPD_results" / "CPD_SATAY_results",
        help="Output folder for whole-genome SATAY CPD results.",
    )
    parser.add_argument(
        "--cpd_script",
        type=Path,
        default=PROJECT_ROOT / "Signal_processing" / "CPD_on_SATAY" / "sliding_ZINB_CPD_v3_SATAY.py",
        help="Path to the single-file CPD script.",
    )
    parser.add_argument("--window_size", type=int, default=100)
    parser.add_argument("--overlap", type=float, default=0.5)
    parser.add_argument("--theta_block_size", type=int, default=2000)
    parser.add_argument("--n_workers", type=int, default=1)
    parser.add_argument("--timeout_seconds", type=int, default=1800)
    args = parser.parse_args()
    
    # Paths
    strains_data = args.strains_data
    centromere_base = args.centromere_base
    nucleosome_base = args.nucleosome_base
    output_base = args.output_base
    cpd_script = args.cpd_script
    
    if not cpd_script.exists():
        print(f"Error: CPD script not found at {cpd_script}")
        return
    
    if not strains_data.exists():
        print(f"Error: Strains data folder not found at {strains_data}")
        return
    
    # Parameters
    window_sizes = [args.window_size]
    overlap = args.overlap
    threshold_start = args.threshold_start
    threshold_end = args.threshold_end
    threshold_step = args.threshold_step
    n_strain_workers = args.n_strain_workers
    theta_block_size = args.theta_block_size
    n_workers = args.n_workers
    timeout_seconds = args.timeout_seconds
    
    print("=" * 80)
    print("Running sliding_ZINB_CPD_v3_SATAY.py on all strains")
    print("=" * 80)
    print(f"Input data:   {strains_data}")
    print(f"Centromeres:  {centromere_base}")
    print(f"Nucleosomes:  {nucleosome_base}")
    print(f"Results:      {output_base}")
    print(f"Script:       {cpd_script}")
    print(f"Thresholds:   {threshold_start} to {threshold_end} (step {threshold_step})")
    print(f"Parallel:     {n_strain_workers} strain(s) at a time")
    print()
    
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    # Get all strain folders
    strain_folders = sorted([d for d in strains_data.iterdir() if d.is_dir() and d.name.startswith("strain_")])
    
    if not strain_folders:
        print("Error: No strain folders found in Data/combined_strains")
        return
    
    print(f"Found {len(strain_folders)} strains to process:\n")
    for strain_folder in strain_folders:
        print(f"  - {strain_folder.name}")
    print()
    
    # Process strains in parallel
    total_processed = 0
    total_skipped = 0
    total_errors = 0
    
    if n_strain_workers == 1:
        # Sequential processing
        for strain_folder in strain_folders:
            strain_name, n_proc, n_err, n_skip = process_strain(
                strain_folder,
                centromere_base,
                nucleosome_base,
                output_base,
                cpd_script,
                window_sizes,
                overlap,
                threshold_start,
                threshold_end,
                threshold_step,
                theta_block_size,
                n_workers,
                timeout_seconds,
            )
            total_processed += n_proc
            total_errors += n_err
            total_skipped += n_skip
    else:
        # Parallel processing
        print(f"Processing {len(strain_folders)} strains with {n_strain_workers} workers...")
        print()
        
        with ThreadPoolExecutor(max_workers=n_strain_workers) as executor:
            # Submit all strain processing jobs
            future_to_strain = {
                executor.submit(
                    process_strain,
                    strain_folder,
                    centromere_base,
                    nucleosome_base,
                    output_base,
                    cpd_script,
                    window_sizes,
                    overlap,
                    threshold_start,
                    threshold_end,
                    threshold_step,
                    theta_block_size,
                    n_workers,
                    timeout_seconds,
                ): strain_folder.name
                for strain_folder in strain_folders
            }
            
            # Collect results as they complete
            for future in as_completed(future_to_strain):
                strain_name = future_to_strain[future]
                try:
                    _, n_proc, n_err, n_skip = future.result()
                    total_processed += n_proc
                    total_errors += n_err
                    total_skipped += n_skip
                except Exception as exc:
                    safe_print(f"\n✗ {strain_name} generated an exception: {exc}")
                    total_skipped += 1
    
    # Final summary
    print("\n" + "=" * 80)
    print("✓ CPD Analysis Complete!")
    print("=" * 80)
    print(f"  Chromosomes processed successfully: {total_processed}")
    print(f"  Chromosomes with errors:            {total_errors}")
    print(f"  Strains skipped:                    {total_skipped}")
    print(f"\n  Results saved to: {output_base}")
    print("=" * 80)


if __name__ == "__main__":
    main()
