import json
import argparse
import os
import tempfile
import pandas as pd

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib"))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
from pathlib import Path

# Add parent directory to path for imports
import sys
sys.path.append(str(Path(__file__).resolve().parents[1]))
from Utils.plot_config import setup_plot_style

# Setup plotting style
setup_plot_style()

# Configuration
DEFAULT_GENES = [
    "PRO3", "CDC28", "SEC18", "KAR2", "RIB3", "POL2", "NOT1", "SUP35", "RNR1", "RNR2",
    "BEM1", "BEM2", "BEM3", "BEM4",
    "RSR1",      # BUD1 alias
    "CLA4", "BNI1", "CDC24", "CDC42", "GIC1", "GIC2",
    "SPA2", "PEA2",
    "MSB2", "SHO1", "STE20", "STE11", "STE7", "FUS3", "KSS1",
    "BUD2", "BUD3", "BUD4", "BUD5", "BUD6", "BUD7", "BUD8", "BUD9",
    "BUD10",     # AXL2 alias
    "BUD13", "BUD14", "BUD16", "BUD17", "BUD19", "BUD20", "BUD21", "BUD22", "BUD23", "BUD24",
    "AXL1", "AXL2",
    "CDC3", "CDC10", "CDC11", "CDC12",   # septins
    "RGA1", "RGA2",                       # Cdc42 GAPs
    "EXO70", "SEC3", "SEC4",              # exocytosis/polarized secretion
    "MYO2", "TPM1",                       # actin/polarized transport
    "RHO1", "RHO3", "RHO4",
    "BOI1", "BOI2",
    "SHE4", "SHE5",
    "FLO11"
]
genes = DEFAULT_GENES.copy()
protein_domain = "PF"
threshold = 3.0
strains = ["FD", "yEK19", "yEK23"]
window_size = 100
overlap = 50
padding_bp = 500  # Base pairs to show before and after gene
mu_z_threshold = 0.25  # muZ parameter for merged segments
segment_source = "merged"

# Paths
BASE_DIR = Path(__file__).resolve().parents[1]
gene_info_path = BASE_DIR / "Utils" / "SGD_API" / "architecture_info" / "yeast_genes_with_info.json"
strains_data_path = BASE_DIR / "results" / "CPD_segments"
count_data_path = BASE_DIR / "Data" / "combined_strains"
output_dir = BASE_DIR / "results" / "genes_overview_plots"


def load_gene_info():
    """Load gene information from JSON file."""
    with open(gene_info_path, 'r') as f:
        data = json.load(f)
    
    # Create a mapping from gene_name to full info
    gene_dict = {}
    for orf, info in data.items():
        if info['gene_name'] in genes:
            gene_dict[info['gene_name']] = {
                'orf': orf,
                'chromosome': info['location']['chromosome'],
                'start': info['location']['start'],
                'end': info['location']['end'],
                'essentiality': info['essentiality'],
                'protein_domains': {k: v for k, v in info['protein_domains'].items() 
                                   if k.startswith(protein_domain)}
            }
    return gene_dict


def load_strain_segments(strain, chromosome, threshold, window_size):
    """Load segment data for a specific strain, chromosome, and threshold."""
    # Convert chromosome format
    chr_short = chromosome.replace('Chromosome_', 'Chr')

    result_dir = (
        strains_data_path / f"strain_{strain}" / chr_short /
        f"{chr_short}_distances" / f"window{window_size}"
    )

    if segment_source == "merged":
        file_path = (
            result_dir / "merged_segments" /
            f"{chr_short}_th{threshold:.2f}_merged_segments_muZ{mu_z_threshold:.2f}.csv"
        )
    elif segment_source == "unmerged":
        overlap_percent = int(overlap * 100) if overlap <= 1 else int(overlap)
        file_path = (
            result_dir / "segment_mu" /
            f"{chr_short}_distances_ws{window_size}_ov{overlap_percent}_th{threshold:.2f}_segment_mu.csv"
        )
    else:
        raise ValueError(f"Unknown segment source: {segment_source}")
    
    if not file_path.exists():
        print(f"Warning: File not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    return df


def load_count_data(strain, chromosome):
    """Load raw count data for a specific strain and chromosome."""
    chr_short = chromosome.replace('Chromosome_', 'Chr')
    
    file_path = count_data_path / f"strain_{strain}" / f"{chr_short}_distances.csv"
    
    if not file_path.exists():
        print(f"Warning: Count data not found: {file_path}")
        return None
    
    df = pd.read_csv(file_path)
    return df


def calculate_global_percentiles(strains):
    """Calculate 95th percentile across all chromosomes for each strain."""
    percentiles = {}
    
    # Yeast chromosomes use Roman numerals
    roman_numerals = ['I', 'II', 'III', 'IV', 'V', 'VI', 'VII', 'VIII', 
                      'IX', 'X', 'XI', 'XII', 'XIII', 'XIV', 'XV', 'XVI']
    chromosomes = [f"Chr{num}" for num in roman_numerals]
    
    for strain in strains:
        print(f"Calculating global percentile for strain {strain}...")
        all_values = []
        
        strain_dir = count_data_path / f"strain_{strain}"
        if not strain_dir.exists():
            print(f"Warning: Strain directory not found: {strain_dir}")
            continue
        
        # Load all chromosome data
        for chrom in chromosomes:
            file_path = strain_dir / f"{chrom}_distances.csv"
            if file_path.exists():
                df = pd.read_csv(file_path)
                all_values.extend(df['Value'].values)
        
        if all_values:
            percentiles[strain] = np.percentile(all_values, 95)
        else:
            percentiles[strain] = None
    
    return percentiles


def get_overlapping_segments(segments_df, start, end):
    """Filter segments that overlap with the specified region."""
    if segments_df is None:
        return None
    
    # Segment overlaps if: segment_start < region_end AND segment_end > region_start
    mask = (segments_df['start_index'] < end) & (segments_df['end_index_exclusive'] > start)
    return segments_df[mask].copy()


def plot_gene_overview(gene_name, gene_info, strain_percentiles):
    """Create overview plot for a single gene showing protein domains and strain segments."""
    
    # Calculate display region
    gene_start = gene_info['start']
    gene_end = gene_info['end']
    display_start = gene_start - padding_bp
    display_end = gene_end + padding_bp
    
    # Create figure
    num_rows = len(strains) + 1  # +1 for gene annotation row
    fig, ax = plt.subplots(figsize=(16, num_rows * 0.8 + 2))
    
    # Y-axis positions for each row (more space for count data)
    y_positions = list(range(num_rows, 0, -1))
    y_spacing = 1.0  # Increased spacing to accommodate count data
    
    # Row 0 (top): Gene annotation with protein domains
    gene_y = y_positions[0] * y_spacing
    
    # Draw gene boundaries
    ax.axvline(gene_start, color='black', linestyle='--', linewidth=1.5, alpha=0.7, label='Gene boundaries')
    ax.axvline(gene_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
    
    # Draw gene as a horizontal line
    ax.plot([gene_start, gene_end], [gene_y, gene_y], color='black', linewidth=3, label='Gene')
    
    # Draw protein domains
    protein_domains = gene_info['protein_domains']
    domain_colors = {}
    
    # Generate random colors for each domain
    np.random.seed(42)  # For reproducibility
    cmap = plt.colormaps['tab20']
    
    for idx, (domain_id, domain_data) in enumerate(protein_domains.items()):
        color = cmap(idx % 20)
        domain_colors[domain_id] = color
        
        # Domains are in amino acid coordinates, need to convert to bp
        # Each amino acid is approximately 3 base pairs
        for start_aa, end_aa in zip(domain_data['start'], domain_data['end']):
            # Convert AA to BP (approximate)
            domain_start_bp = gene_start + (start_aa - 1) * 3
            domain_end_bp = gene_start + (end_aa * 3)
            
            # Ensure we don't go beyond gene boundaries
            domain_start_bp = max(gene_start, min(gene_end, domain_start_bp))
            domain_end_bp = max(gene_start, min(gene_end, domain_end_bp))
            
            rect = mpatches.Rectangle((domain_start_bp, gene_y - 0.2), 
                                     domain_end_bp - domain_start_bp, 0.4,
                                     linewidth=2, edgecolor='black', 
                                     facecolor=color, alpha=0.7)
            ax.add_patch(rect)
    
    # Collect all mu_z_scores to determine range for colormap
    all_mu_z_scores = []
    for strain in strains:
        segments_df = load_strain_segments(strain, gene_info['chromosome'], 
                                          threshold, window_size)
        if segments_df is not None:
            overlapping = get_overlapping_segments(segments_df, display_start, display_end)
            if overlapping is not None and len(overlapping) > 0:
                all_mu_z_scores.extend(overlapping['mu_z_score'].values)
    
    if all_mu_z_scores:
        # Create diverging colormap centered at 0 (RdBu: red for negative, blue for positive)
        # Fixed range for consistency across all figures
        vmin = -2
        vmax = 4
        norm = TwoSlopeNorm(vmin=vmin, vcenter=0, vmax=vmax)
        cmap_div = plt.cm.RdBu  # Red for negative, blue for positive
        
        # Clear and replot everything with proper colors
        ax.clear()
        
        # Redraw gene annotation
        ax.axvline(gene_start, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.axvline(gene_end, color='black', linestyle='--', linewidth=1.5, alpha=0.7)
        ax.plot([gene_start, gene_end], [gene_y, gene_y], color='black', linewidth=3)
        
        # Redraw protein domains
        for idx, (domain_id, domain_data) in enumerate(protein_domains.items()):
            color = domain_colors[domain_id]
            for start_aa, end_aa in zip(domain_data['start'], domain_data['end']):
                domain_start_bp = gene_start + (start_aa - 1) * 3
                domain_end_bp = gene_start + (end_aa * 3)
                domain_start_bp = max(gene_start, min(gene_end, domain_start_bp))
                domain_end_bp = max(gene_start, min(gene_end, domain_end_bp))
                
                rect = mpatches.Rectangle((domain_start_bp, gene_y - 0.2), 
                                         domain_end_bp - domain_start_bp, 0.4,
                                         linewidth=2, edgecolor='black', 
                                         facecolor=color, alpha=0.7)
                ax.add_patch(rect)
                
                # Add domain name as text on the domain
                domain_center_bp = (domain_start_bp + domain_end_bp) / 2
                domain_label = f"{domain_id}"
                ax.text(domain_center_bp, gene_y, domain_label, 
                       ha='center', va='center', fontsize=14, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                edgecolor='none', alpha=0.7))
        
        # Redraw strain segments with color and count data
        for idx, strain in enumerate(strains):
            strain_y = y_positions[idx + 1] * y_spacing
            
            segments_df = load_strain_segments(strain, gene_info['chromosome'], 
                                              threshold, window_size)
            
            if segments_df is not None:
                overlapping = get_overlapping_segments(segments_df, display_start, display_end)
                
                if overlapping is not None and len(overlapping) > 0:
                    for _, seg in overlapping.iterrows():
                        seg_start = max(display_start, seg['start_index'])
                        seg_end = min(display_end, seg['end_index_exclusive'])
                        mu_z = seg['mu_z_score']
                        
                        color = cmap_div(norm(mu_z))
                        rect = mpatches.Rectangle((seg_start, strain_y - 0.3), 
                                                 seg_end - seg_start, 0.6,
                                                 linewidth=2, edgecolor='black',
                                                 facecolor=color, alpha=0.9)
                        ax.add_patch(rect)
                        
                        # Add z-score text on the bar if the segment is wide enough
                        seg_width = seg_end - seg_start
                        if seg_width > (display_end - display_start) * 0.05:  # Only show if >5% of display width
                            seg_center = (seg_start + seg_end) / 2
                            ax.text(seg_center, strain_y, f"{mu_z:.1f}", 
                                   ha='center', va='center', fontsize=12,
                                   color='white' if abs(norm(mu_z) - 0.5) > 0.3 else 'black',
                                   fontweight='bold')
            
            # Add count data above the segments (scaled individually per strain)
            count_data = load_count_data(strain, gene_info['chromosome'])
            if count_data is not None:
                # Filter to display region
                region_data = count_data[
                    (count_data['Position'] >= display_start) & 
                    (count_data['Position'] <= display_end)
                ]
                
                
                if len(region_data) > 0:
                    # Use THIS strain's global 95th percentile to filter outliers
                    strain_global_percentile = strain_percentiles.get(strain)
                    
                    
                    if strain_global_percentile is not None and strain_global_percentile > 0:
                        # Filter out values above THIS strain's global 95th percentile
                        filtered_data = region_data[region_data['Value'] <= strain_global_percentile].copy()
                        
                        
                        if len(filtered_data) > 0:
                            # Normalize by THIS strain's LOCAL maximum in this region (independent scaling)
                            strain_local_max = filtered_data['Value'].max()
                            
                            if len(filtered_data) > 0:
                                strain_local_max = filtered_data['Value'].max()

                                if strain_local_max > 0:
                                    baseline = strain_y + 0.35

                                    # One horizontal zero-line per strain
                                    ax.hlines(
                                        baseline,
                                        display_start,
                                        display_end,
                                        color='darkgray',
                                        linewidth=0.8,
                                        alpha=0.5,
                                        zorder=8
                                    )

                                    # Plot only nonzero peaks
                                    nonzero_data = filtered_data[filtered_data['Value'] > 0].copy()

                                    if len(nonzero_data) > 0:
                                        normalized_counts = nonzero_data['Value'] / strain_local_max * 0.3

                                        ax.vlines(
                                            nonzero_data['Position'],
                                            baseline,
                                            baseline + normalized_counts,
                                            color='darkgray',
                                            linewidth=1.0,
                                            alpha=0.7,
                                            zorder=10
                                        )
    
    # Set axis properties
    ax.set_xlim(display_start, display_end)
    ax.set_ylim(0.5 * y_spacing, (num_rows + 0.5) * y_spacing)
    ax.set_xlabel('Genomic Position (bp)', fontsize=14)
    ax.set_yticks([y * y_spacing for y in y_positions])
    ax.set_yticklabels(['Gene\nAnnotation'] + strains, fontsize=12)
    ax.tick_params(axis='x', labelsize=12)
    
    # Title
    essential_status = "Essential" if gene_info['essentiality'] else "Non-essential"
    title = (f"{gene_name} ({gene_info['orf']}) - {gene_info['chromosome']}\n"
             f"{essential_status} ")
    ax.set_title(title, fontsize=18, fontweight='bold')
    
    # Add colorbar for mu_z_score
    if all_mu_z_scores:
        sm = plt.cm.ScalarMappable(cmap=cmap_div, norm=norm)
        sm.set_array([])
        cbar = plt.colorbar(sm, ax=ax, pad=0.02, fraction=0.046, aspect=20)
        cbar.set_label('μ z-score', rotation=270, labelpad=25, fontsize=14)
        cbar.ax.tick_params(labelsize=12)
    
    # Grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    
    plt.tight_layout()
    
    # Save figure
    output_file = output_dir / f"{gene_name}_{gene_info['orf']}_overview_{segment_source}.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    print(f"Saved: {output_file}")
    
    plt.close()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate overview plots for selected genes across SATAY CPD strains."
    )
    parser.add_argument(
        "--genes",
        nargs="+",
        default=None,
        help="Gene names to plot. If omitted, the default gene list defined in this file is used.",
    )
    parser.add_argument("--protein_domain", default=protein_domain, help="Protein-domain prefix to plot.")
    parser.add_argument("--threshold", type=float, default=threshold, help="CPD threshold used in segment filenames.")
    parser.add_argument("--strains", nargs="+", default=strains, help="Strains to include in the plot.")
    parser.add_argument("--window_size", type=int, default=window_size, help="Window size used in CPD results.")
    parser.add_argument("--overlap", type=float, default=overlap, help="CPD overlap used in unmerged segment filenames. Accepts either 0.5 or 50 for 50%%.")
    parser.add_argument("--padding_bp", type=int, default=padding_bp, help="Base pairs to show before and after each gene.")
    parser.add_argument("--mu_z_threshold", type=float, default=mu_z_threshold, help="Merged-segment muZ threshold.")
    parser.add_argument(
        "--segment_source",
        choices=["merged", "unmerged"],
        default=segment_source,
        help="Use merged segment files or raw segment_mu files for gene overview plots.",
    )
    parser.add_argument("--gene_info_path", default=str(gene_info_path), help="Path to yeast gene annotation JSON.")
    parser.add_argument("--strains_data_path", default=str(strains_data_path), help="Path to SATAY CPD result folders.")
    parser.add_argument("--count_data_path", default=str(count_data_path), help="Path to raw strain count folders.")
    parser.add_argument("--output_dir", default=str(output_dir), help="Folder where plots are written.")
    return parser.parse_args()


def apply_args(args):
    global genes, protein_domain, threshold, strains, window_size, overlap, padding_bp
    global mu_z_threshold, segment_source
    global gene_info_path, strains_data_path, count_data_path, output_dir

    genes = args.genes if args.genes is not None else DEFAULT_GENES.copy()
    protein_domain = args.protein_domain
    threshold = args.threshold
    strains = args.strains
    window_size = args.window_size
    overlap = args.overlap
    padding_bp = args.padding_bp
    mu_z_threshold = args.mu_z_threshold
    segment_source = args.segment_source
    gene_info_path = Path(args.gene_info_path)
    strains_data_path = Path(args.strains_data_path)
    count_data_path = Path(args.count_data_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)


def main(args=None):
    """Main function to generate all gene overview plots."""
    apply_args(parse_args() if args is None else args)

    print("Loading gene information...")
    gene_dict = load_gene_info()
    
    print(f"\nCalculating global 95th percentiles for all strains...")
    strain_percentiles = calculate_global_percentiles(strains)
    
    print(f"\nGenerating overview plots for {len(gene_dict)} genes...")
    print(f"Threshold: {threshold}")
    print(f"Strains: {', '.join(strains)}")
    print(f"Padding: ±{padding_bp} bp")
    if segment_source == "merged":
        print(f"Using merged segments with muZ threshold: {mu_z_threshold}")
    else:
        print(f"Using unmerged segment_mu files with overlap: {overlap}")
    print(f"Output directory: {output_dir}\n")
    
    for gene_name in genes:
        if gene_name in gene_dict:
            print(f"Processing {gene_name}...")
            plot_gene_overview(gene_name, gene_dict[gene_name], strain_percentiles)
        else:
            print(f"Warning: {gene_name} not found in gene info file")
    
    print("\nDone! All plots saved.")


if __name__ == "__main__":
    main()
