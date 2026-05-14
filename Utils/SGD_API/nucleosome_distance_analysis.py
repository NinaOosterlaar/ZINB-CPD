"""
Calculate mean and standard deviation of distances between subsequent nucleosomes.

Analyzes nucleosome position data from SGD_API/nucleosome_data/2013/
Calculates the distance between consecutive nucleosome center positions.
"""

import json
import os
import numpy as np
from pathlib import Path

# Configuration
SCRIPT_DIR = Path(__file__).parent
DATA_DIR = SCRIPT_DIR / "nucleosome_data" / "2013"

# All yeast chromosomes (excluding exposure files)
CHROMOSOMES = [
    'ChrI', 'ChrII', 'ChrIII', 'ChrIV', 'ChrV', 'ChrVI', 'ChrVII', 'ChrVIII',
    'ChrIX', 'ChrX', 'ChrXI', 'ChrXII', 'ChrXIII', 'ChrXIV', 'ChrXV', 'ChrXVI'
]

def load_nucleosome_positions(chromosome):
    """Load nucleosome positions from JSON file.
    
    Args:
        chromosome: Chromosome name (e.g., 'ChrI')
        
    Returns:
        List of nucleosome data: [start, center, end, boolean_flag]
    """
    file_path = DATA_DIR / f"{chromosome}.json"
    
    if not file_path.exists():
        print(f"Warning: {file_path} not found")
        return None
    
    with open(file_path, 'r') as f:
        data = json.load(f)
    
    return data

def calculate_inter_nucleosome_distances(nucleosomes, exclude_top_percentile=0):
    """Calculate distances between consecutive nucleosome centers.
    
    Args:
        nucleosomes: List of [start, center, end, flag] for each nucleosome
        exclude_top_percentile: Exclude top X% of longest distances (default: 0)
        
    Returns:
        Array of distances between consecutive nucleosome centers
    """
    if nucleosomes is None or len(nucleosomes) < 2:
        return np.array([])
    
    # Extract center positions (index 1)
    centers = np.array([nuc[1] for nuc in nucleosomes])
    
    # Calculate distances: center[i+1] - center[i]
    distances = np.diff(centers)
    
    # Exclude top percentile if requested
    if exclude_top_percentile > 0 and len(distances) > 0:
        threshold = np.percentile(distances, 100 - exclude_top_percentile)
        distances = distances[distances <= threshold]
    
    return distances

def analyze_nucleosome_spacing(exclude_top_pct=5):
    """Analyze inter-nucleosome distances for all chromosomes.
    
    Args:
        exclude_top_pct: Exclude top X% of longest distances (default: 5)
    """
    
    print("="*80)
    print("NUCLEOSOME SPACING ANALYSIS")
    print("="*80)
    print(f"\nData directory: {DATA_DIR}")
    print(f"Analyzing {len(CHROMOSOMES)} chromosomes")
    print(f"Excluding top {exclude_top_pct}% of longest distances\n")
    
    # Store results
    chromosome_stats = {}
    all_distances = []
    
    # Analyze each chromosome
    for chrom in CHROMOSOMES:
        nucleosomes = load_nucleosome_positions(chrom)
        
        if nucleosomes is None:
            continue
        
        distances = calculate_inter_nucleosome_distances(nucleosomes, exclude_top_percentile=exclude_top_pct)
        
        if len(distances) == 0:
            print(f"{chrom}: No data")
            continue
        
        # Calculate statistics
        mean_dist = np.mean(distances)
        std_dist = np.std(distances)
        median_dist = np.median(distances)
        min_dist = np.min(distances)
        max_dist = np.max(distances)
        n_nucleosomes = len(nucleosomes)
        n_distances = len(distances)
        
        chromosome_stats[chrom] = {
            'mean': mean_dist,
            'std': std_dist,
            'median': median_dist,
            'min': min_dist,
            'max': max_dist,
            'n_nucleosomes': n_nucleosomes,
            'n_distances': n_distances
        }
        
        all_distances.extend(distances)
        
        print(f"{chrom} (n={n_nucleosomes} nucleosomes, {n_distances} distances):")
        print(f"  Mean:   {mean_dist:.2f} bp")
        print(f"  Std:    {std_dist:.2f} bp")
        print(f"  Median: {median_dist:.2f} bp")
        print(f"  Range:  [{min_dist:.0f}, {max_dist:.0f}] bp")
    
    # Combined statistics across all chromosomes
    print("\n" + "="*80)
    print("COMBINED STATISTICS (ALL CHROMOSOMES)")
    print("="*80)
    
    if len(all_distances) > 0:
        all_distances = np.array(all_distances)
        
        total_nucleosomes = sum(stats['n_nucleosomes'] for stats in chromosome_stats.values())
        total_distances = len(all_distances)
        
        mean_dist_combined = np.mean(all_distances)
        std_dist_combined = np.std(all_distances)
        median_dist_combined = np.median(all_distances)
        min_dist_combined = np.min(all_distances)
        max_dist_combined = np.max(all_distances)
        
        print(f"\nTotal nucleosomes: {total_nucleosomes}")
        print(f"Total distances (after excluding top {exclude_top_pct}%): {total_distances}")
        print(f"\nInter-nucleosome distance (center-to-center):")
        print(f"  Mean:   {mean_dist_combined:.2f} bp")
        print(f"  Std:    {std_dist_combined:.2f} bp")
        print(f"  Median: {median_dist_combined:.2f} bp")
        print(f"  Range:  [{min_dist_combined:.0f}, {max_dist_combined:.0f}] bp")
        
        # Percentiles
        print(f"\nPercentiles:")
        percentiles = [5, 25, 75, 95]
        for p in percentiles:
            val = np.percentile(all_distances, p)
            print(f"  {p}th:    {val:.2f} bp")
    else:
        print("\nNo data available!")
    
    return chromosome_stats, all_distances

if __name__ == "__main__":
    analyze_nucleosome_spacing()
