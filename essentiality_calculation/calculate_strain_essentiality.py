"""
Calculate essentiality levels for SATAY strain data.

Processes change point detection results for all strains and computes:
- Segment-level μ estimates (using fixed global theta)
- Standardized z-scores: z = (μ - μ̄) / σ_μ (computed across entire strain)

The z-score represents the essentiality level of each segment.
"""

import argparse
import os
import sys
from pathlib import Path
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial

import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

# Import reusable functions from pure_estimation
from Signal_processing.essentiality_calculation.pure_estimation import (
    read_count_data,
    parse_result_file,
    create_segments_with_mu_estimates,
    remove_top_quantile_outliers
)

# # List of strains to process
STRAINS = [
    'strain_FD',
    'strain_dnrp',
    'strain_yEK19',
    'strain_yEK23',
    'strain_yTW001',
    'strain_yWT03a',
    'strain_yWT04a',
    'strain_ylic137'
]


# All chromosomes in order
CHROMOSOMES = [
    "ChrI", "ChrII", "ChrIII", "ChrIV", "ChrV", "ChrVI",
    "ChrVII", "ChrVIII", "ChrIX", "ChrX", "ChrXI", "ChrXII",
    "ChrXIII", "ChrXIV", "ChrXV", "ChrXVI"
]
# CHROMOSOMES = ["ChrX"]


def add_zscore_to_segments(all_segments_dict, theta_dict, strain_name):
    """
    Add z-scores to all segments, computed per threshold across all chromosomes.
    
    For each threshold:
    - Collects μ values from all chromosomes at that threshold
    - Computes z = (μ - μ̄) / σ_μ where statistics are threshold-specific
    
    Parameters
    ----------
    all_segments_dict : dict
        Nested dict: {threshold: {chromosome: [segment_rows]}}
    theta_dict : dict
        Dict: {threshold: {chromosome: theta_global}}
    strain_name : str
        Name of the strain being processed
    """
    print(f"\n  Computing z-scores per threshold...")
    
    # Process each threshold separately
    for threshold, chrom_data in sorted(all_segments_dict.items()):
        # Collect all mu values for this threshold across all chromosomes
        # Apply log transformation: log(mu + 1) to handle skewed distributions
        threshold_mu_values = []
        threshold_log_mu_values = []
        for segments in chrom_data.values():
            for seg in segments:
                mu = seg["mu_estimate"]
                if not np.isnan(mu):
                    threshold_mu_values.append(mu)
                    threshold_log_mu_values.append(np.log(mu + 1))
        
        if len(threshold_log_mu_values) == 0:
            print(f"    Threshold {threshold:.1f}: No valid μ values")
            # Add NaN z-scores
            for segments in chrom_data.values():
                for seg in segments:
                    seg["mu_z_score"] = np.nan
            continue
        
        # Compute statistics on log-transformed values
        log_mu_mean = np.mean(threshold_log_mu_values)
        log_mu_std = np.std(threshold_log_mu_values, ddof=1) if len(threshold_log_mu_values) > 1 else 0.0
        
        # Get theta values for this threshold
        theta_values = list(theta_dict[threshold].values())
        theta_mean = np.mean(theta_values)
        theta_min = np.min(theta_values)
        theta_max = np.max(theta_values)
        
        # Add z-scores to all segments for this threshold (based on log-transformed values)
        for segments in chrom_data.values():
            for seg in segments:
                mu = seg["mu_estimate"]
                log_mu = np.log(mu + 1)
                if np.isnan(mu) or not np.isfinite(log_mu) or log_mu_std == 0.0:
                    seg["mu_z_score"] = np.nan
                else:
                    seg["mu_z_score"] = (log_mu - log_mu_mean) / log_mu_std
        
        # Print statistics including theta (showing both raw and log-transformed stats)
        if theta_min == theta_max:
            theta_str = f"θ={theta_mean:.6f}"
        else:
            theta_str = f"θ={theta_mean:.6f} (range: {theta_min:.6f}-{theta_max:.6f})"
        
        mu_mean_raw = np.mean(threshold_mu_values)
        print(f"    Threshold {threshold:5.1f}: {len(threshold_mu_values):4d} segments, "
              f"μ̄={mu_mean_raw:7.4f}, log(μ̄+1)={log_mu_mean:7.4f}, σ_log(μ)={log_mu_std:7.4f}, {theta_str}")


def process_strain(strain_name, base_data_folder, base_results_folder, 
                   output_subdir, thresholds, eps, tol, max_iter):
    """
    Process all chromosomes and specified thresholds for a single strain.
    
    Parameters
    ----------
    strain_name : str
        Name of the strain to process
    base_data_folder : str
        Path to data folder
    base_results_folder : str
        Path to results folder
    output_subdir : str
        Subfolder name for outputs
    thresholds : list of float or None
        List of thresholds to process. If None, processes all available thresholds.
    eps : float
        Numerical epsilon
    tol : float
        Convergence tolerance
    max_iter : int
        Maximum iterations
    
    Returns
    -------
    dict
        Summary statistics
    """
    print(f"\n{'='*80}")
    print(f"Processing {strain_name}")
    print(f"{'='*80}")
    
    strain_data_folder = os.path.join(base_data_folder, strain_name)
    strain_results_folder = os.path.join(base_results_folder, strain_name)
    
    if not os.path.isdir(strain_data_folder):
        print(f"  ERROR: Data folder not found: {strain_data_folder}")
        return None
    
    if not os.path.isdir(strain_results_folder):
        print(f"  ERROR: Results folder not found: {strain_results_folder}")
        return None
    
    # Store all segments before computing z-scores
    # Structure: {threshold: {chromosome: [segment_rows]}}
    all_segments = defaultdict(lambda: defaultdict(list))
    
    # Track theta values per threshold per chromosome
    # Structure: {threshold: {chromosome: theta_global}}
    theta_values = defaultdict(dict)
    
    # Track metadata for each file
    file_metadata = {}
    
    processed_files = 0
    skipped_files = 0
    
    # STEP 1: Load all chromosome data to compute global 99th percentile
    print(f"  Computing global outlier threshold across all chromosomes...")
    all_chrom_data = []
    chrom_data_cache = {}  # Store loaded data for reuse
    total_problematic = 0
    
    # Known problematic positions: (chromosome, position)
    problematic_positions = [
        ('ChrXV', 565596),
    ]
    
    for chrom in CHROMOSOMES:
        chrom_data_file = os.path.join(strain_data_folder, f"{chrom}_distances.csv")
        if os.path.exists(chrom_data_file):
            data = read_count_data(chrom_data_file).copy()
            
            # Remove problematic positions for this chromosome
            for prob_chrom, prob_pos in problematic_positions:
                if chrom == prob_chrom and prob_pos < len(data):
                    if data[prob_pos] != 0:
                        data[prob_pos] = 0
                        total_problematic += 1
            
            all_chrom_data.extend(data)
            chrom_data_cache[chrom] = data
    
    if not all_chrom_data:
        print(f"  ERROR: No chromosome data found for {strain_name}")
        return None
    
    # Compute global 99th percentile threshold (non-zero values only)
    all_chrom_data = np.array(all_chrom_data)
    non_zero_data = all_chrom_data[all_chrom_data > 0]
    
    if len(non_zero_data) == 0:
        print(f"  ERROR: No non-zero data found for {strain_name}")
        return None
    
    global_threshold = np.quantile(non_zero_data, 0.99)
    n_total = len(all_chrom_data)
    n_outliers = np.sum(all_chrom_data > global_threshold)
    
    if total_problematic > 0:
        print(f"  Removed {total_problematic} problematic position(s) (set to zero)")
    print(f"  Global 99th percentile (non-zero): {global_threshold:.1f}")
    print(f"  Total outliers to cap: {n_outliers} ({100*n_outliers/n_total:.2f}%) across all chromosomes")
    print()
    
    # STEP 2: Process each chromosome with global threshold
    # Process each chromosome
    for chrom in CHROMOSOMES:
        if chrom not in chrom_data_cache:
            print(f"  Skipping {chrom}: data file not found")
            continue
        
        # Get cached data
        data = chrom_data_cache[chrom]
        
        # Apply global threshold to this chromosome
        data_filtered = np.clip(data, None, global_threshold)
        n_chrom_outliers = np.sum(data > global_threshold)
        if n_chrom_outliers > 0:
            print(f"    {chrom}: capped {n_chrom_outliers} outliers ({100*n_chrom_outliers/len(data):.2f}%)")
        
        # Path to change point results
        chrom_results_folder = os.path.join(
            strain_results_folder, chrom, f"{chrom}_distances", "window100"
        )
        
        if not os.path.isdir(chrom_results_folder):
            print(f"  Skipping {chrom}: results folder not found")
            continue
        
        # Get all threshold files
        result_files = sorted([
            f for f in os.listdir(chrom_results_folder)
            if f.endswith('.txt') and f.startswith(f"{chrom}_distances_ws100_ov50_th")
        ])
        
        # Filter by specified thresholds if provided
        if thresholds is not None:
            filtered_files = []
            for f in result_files:
                # Extract threshold from filename
                threshold_str = f.split('_th')[1].replace('.txt', '')
                file_threshold = float(threshold_str)
                if file_threshold in thresholds:
                    filtered_files.append(f)
            result_files = filtered_files
        
        if not result_files:
            if thresholds is not None:
                print(f"  Skipping {chrom}: no result files found for specified thresholds")
            else:
                print(f"  Skipping {chrom}: no result files found")
            continue
        
        print(f"  Processing {chrom}: {len(result_files)} threshold files")
        
        # Process each threshold file
        for result_name in result_files:
            result_path = os.path.join(chrom_results_folder, result_name)
            
            try:
                change_points, theta_global = parse_result_file(result_path)
                
                # Extract threshold from filename
                # Format: {Chr}_distances_ws100_ov50_th{threshold}.txt
                threshold_str = result_name.split('_th')[1].replace('.txt', '')
                threshold = float(threshold_str)
                
                # Estimate segments (without z-scores) - reusing function from pure_estimation.py
                segment_rows = create_segments_with_mu_estimates(
                    data_filtered, change_points, theta_global,
                    eps=eps, tol=tol, max_iter=max_iter
                )
                
                # Store segments: {threshold: {chromosome: [segment_rows]}}
                all_segments[threshold][chrom] = segment_rows
                
                # Store theta value
                theta_values[threshold][chrom] = theta_global
                
                # Store metadata
                file_metadata[(chrom, threshold)] = {
                    'result_file': result_name,
                    'theta_global': theta_global,
                    'num_change_points': len(change_points),
                    'num_segments': len(segment_rows)
                }
                
                processed_files += 1
                
            except Exception as e:
                print(f"    ERROR processing {result_name}: {e}")
                skipped_files += 1
                continue
    
    if processed_files == 0:
        print(f"  No files processed for {strain_name}")
        return None
    
    print(f"\n  Processed {processed_files} files, skipped {skipped_files}")
    
    # Compute z-scores per threshold across all chromosomes
    add_zscore_to_segments(all_segments, theta_values, strain_name)
    
    # Save results
    print(f"\n  Saving results...")
    saved_files = 0
    
    for threshold in all_segments:
        for chrom, segment_rows in all_segments[threshold].items():
            metadata = file_metadata[(chrom, threshold)]
            
            # Create DataFrame
            segment_df = pd.DataFrame(segment_rows)
            
            # Add metadata columns
            segment_df.insert(0, 'strain', strain_name)
            segment_df.insert(1, 'chromosome', chrom)
            segment_df.insert(2, 'threshold', threshold)
            segment_df.insert(3, 'theta_global', metadata['theta_global'])
            segment_df.insert(4, 'num_change_points', metadata['num_change_points'])
            
            # Create output directory
            output_dir = os.path.join(
                base_results_folder, strain_name, chrom, 
                f"{chrom}_distances", "window100", output_subdir
            )
            os.makedirs(output_dir, exist_ok=True)
            
            # Save to CSV
            output_name = metadata['result_file'].replace('.txt', '_segment_mu.csv')
            output_path = os.path.join(output_dir, output_name)
            segment_df.to_csv(output_path, index=False)
            saved_files += 1
    
    print(f"  Saved {saved_files} segment files")
    
    # Compute summary statistics
    chromosomes_processed = set()
    for chrom_dict in all_segments.values():
        chromosomes_processed.update(chrom_dict.keys())
    
    summary = {
        'strain': strain_name,
        'processed_files': processed_files,
        'skipped_files': skipped_files,
        'saved_files': saved_files,
        'chromosomes_processed': len(chromosomes_processed),
        'thresholds_processed': len(all_segments)
    }
    
    return summary


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Calculate essentiality levels for SATAY strain data."
    )
    parser.add_argument(
        "--base_data_folder",
        type=str,
        default="Data/combined_strains",
        help="Folder containing strain count data.",
    )
    parser.add_argument(
        "--base_results_folder",
        type=str,
        default="SATAY_CPD_results/CPD_SATAY_results",
        help="Folder containing change point detection results.",
    )
    parser.add_argument(
        "--strains",
        type=str,
        nargs="+",
        default=STRAINS,
        help="List of strain names to process.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[3.0],
        help="List of thresholds to process (e.g., 0.5 1.0 1.5 ... 5.0). If not specified, processes all available thresholds.",
    )
    parser.add_argument(
        "--output_subdir",
        type=str,
        default="segment_mu",
        help="Subfolder for segment outputs.",
    )
    parser.add_argument(
        "--eps",
        type=float,
        default=1e-10,
        help="Numerical epsilon.",
    )
    parser.add_argument(
        "--tol",
        type=float,
        default=1e-6,
        help="Convergence tolerance for mu updates.",
    )
    parser.add_argument(
        "--max_iter",
        type=int,
        default=200,
        help="Maximum EM iterations per segment.",
    )
    parser.add_argument(
        "--summary_output",
        type=str,
        default="Signal_processing/results_new/essentiality_calculation/strain_essentiality_summary.csv",
        help="Path for summary statistics CSV.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=8,
        help="Number of parallel workers to use for processing strains (default: 4).",
    )
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Resolve paths
    base_data_folder = os.path.join(PROJECT_ROOT, args.base_data_folder)
    base_results_folder = os.path.join(PROJECT_ROOT, args.base_results_folder)
    summary_output = os.path.join(PROJECT_ROOT, args.summary_output)
    
    print("="*80)
    print("SATAY Strain Essentiality Calculation")
    print("="*80)
    print(f"Data folder: {base_data_folder}")
    print(f"Results folder: {base_results_folder}")
    print(f"Strains to process: {len(args.strains)}")
    for strain in args.strains:
        print(f"  - {strain}")
    if args.thresholds is not None:
        print(f"Thresholds to process: {sorted(args.thresholds)}")
    else:
        print(f"Thresholds: All available")
    print(f"Output subfolder: {args.output_subdir}")
    print(f"Parallel workers: {args.workers}")
    print(f"Summary output: {summary_output}")
    print("="*80)
    
    # Process each strain in parallel
    summaries = []
    
    # Create a partial function with fixed parameters
    process_func = partial(
        process_strain,
        base_data_folder=base_data_folder,
        base_results_folder=base_results_folder,
        output_subdir=args.output_subdir,
        thresholds=args.thresholds,
        eps=args.eps,
        tol=args.tol,
        max_iter=args.max_iter
    )
    
    # Use ProcessPoolExecutor for parallel processing
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        # Submit all strain processing tasks
        future_to_strain = {
            executor.submit(process_func, strain_name): strain_name
            for strain_name in args.strains
        }
        
        # Collect results as they complete
        for future in as_completed(future_to_strain):
            strain_name = future_to_strain[future]
            try:
                summary = future.result()
                if summary is not None:
                    summaries.append(summary)
            except Exception as exc:
                print(f"\n{'='*80}")
                print(f"ERROR: {strain_name} generated an exception: {exc}")
                print(f"{'='*80}\n")
    
    # Save summary statistics
    if summaries:
        summary_df = pd.DataFrame(summaries)
        os.makedirs(os.path.dirname(summary_output), exist_ok=True)
        summary_df.to_csv(summary_output, index=False)
        
        print("\n" + "="*80)
        print("SUMMARY")
        print("="*80)
        print(summary_df.to_string(index=False))
        print(f"\nSummary saved to: {summary_output}")
        print("="*80)
        print("\nDone!")
    else:
        print("\nNo strains were successfully processed.")


if __name__ == "__main__":
    main()
