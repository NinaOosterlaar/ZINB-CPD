import json
import numpy as np 
import pandas as pd 
import os, sys
import argparse
import matplotlib.pyplot as plt 
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from Utils.plot_config import setup_plot_style, COLORS
from Utils.SGD_API.yeast_architecture import Nucleosomes

# Set up standardized plot style
setup_plot_style()
# import statsmodels.api as sm
# from statsmodels.gam.api import GLMGam, BSplines

chromosome_length = {
    "ChrI": 230218,
    "ChrII": 813184,
    "ChrIII": 316620,
    "ChrIV": 1531933,
    "ChrV": 576874,
    "ChrVI": 270161,
    "ChrVII": 1090940,
    "ChrVIII": 562643,
    "ChrIX": 439888,
    "ChrX": 745751,
    "ChrXI": 666816,
    "ChrXII": 1078171,
    "ChrXIII": 924431,
    "ChrXIV": 784333,
    "ChrXV": 1091291,
    "ChrXVI": 948066,
    "ChrM": 85779,          # mitochondrial genome (approx for S288C)
    "2micron": 6318         # 2-micron plasmid
}

mean_chrom_length = 754457.5  # average length of the 16 nuclear chromosomes



def process_single_dataset_centromere(strain_name, dataset_path, dataset_name, output_folder, bin=100, max_distance_global=None, min_distance_global=None, boolean=False):
    """Process a single dataset and save results immediately to save memory.
    
    Args:
        strain_name (str): Name of the strain
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        output_folder (str): Base output folder
        bin (int): Size of the sliding window for density calculation
        max_distance_global (int): Maximum distance to consider
        boolean (bool): If True, compute presence/absence density instead of counts
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    dataset_data = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_distances.csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        file_path = os.path.join(dataset_path, csv_file)
        dataset_data[chrom] = pd.read_csv(file_path)
    
    # Process each chromosome in this dataset
    for chrom in dataset_data:
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        if chrom == "ChrXV":
            dataset_data[chrom].loc[dataset_data[chrom]['Position'] == 565392, 'Value'] = 0
            
        df = dataset_data[chrom]

        if max_distance_global is not None:
            max_distance = max_distance_global
        else:
            max_distance = df['Centromere_Distance'].max()
        if min_distance_global is not None:
            min_distance = min_distance_global
        else:
            min_distance = df['Centromere_Distance'].min()
            
        # Create bins aligned around centromere (position 0) to ensure consistent bin centers across datasets
        # Find the range needed to cover all data
        data_range = max(abs(min_distance), abs(max_distance))
        
        # Calculate how many bins we need on each side of 0
        n_bins_each_side = int(np.ceil(data_range / bin)) + 1  # +1 for safety margin
        
        # Create bin centers that are multiples of the bin size: ..., -2*bin, -bin, 0, bin, 2*bin, ...
        bin_centers = np.arange(-n_bins_each_side * bin, (n_bins_each_side + 1) * bin, bin)
        
        # Create bin edges by shifting centers by half a bin size
        # For centers at ..., -bin, 0, bin, ..., edges will be at ..., -1.5*bin, -0.5*bin, 0.5*bin, 1.5*bin, ...
        bin_edges = bin_centers - bin/2
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin)  # Add final edge
        
        df['Distance_Bin'] = pd.cut(df['Centromere_Distance'], bins=bin_edges, right=False, include_lowest=True)

        if boolean:
            # Convert counts to presence/absence
            df['Value'] = df['Value'].apply(lambda x: 1 if x > 0 else 0)
        density = df.groupby('Distance_Bin')['Value'].sum().reset_index()
        density['Bin_Center'] = density['Distance_Bin'].apply(lambda x: x.left + bin / 2)
        
        density['Density_per_bp'] = density['Value'] / (bin)
        
        
        if density.empty:
            print(f"No valid density data for {chrom} in {strain_name}/{dataset_name} after filtering outliers. Skipping.")
            continue
        
        # Save to CSV immediately
        output_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_bin:{bin}_centromere_density.csv")
        density.to_csv(output_file, index=False)
    
    # Clear dataset from memory
    del dataset_data
    
    # Create plot for this dataset
    create_individual_chromosome_plots(strain_name, dataset_name, dataset_output_folder, boolean, bin)


def create_individual_chromosome_plots(strain_name, dataset_name, dataset_output_folder, boolean, bin=100):
    """Create individual clean bar plots for each chromosome."""
    
    # Get all chromosomes (I-XVI) and ensure consistent ordering
    all_chromosomes = [f"Chr{roman}" for roman in ["I", "II", "III", "IV", "V", "VI", "VII", "VIII", 
                                                  "IX", "X", "XI", "XII", "XIII", "XIV", "XV", "XVI"]]
    
    plots_created = 0
    
    for chrom in all_chromosomes:
        density_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:{boolean}_bin:{bin}_centromere_density.csv")
        
        if not os.path.exists(density_file):
            continue
        
        density = pd.read_csv(density_file)
        
        if density.empty:
            continue
        
        # Create individual plot for this chromosome
        fig, ax = plt.subplots(1, 1, figsize=(12, 6))
        
        # Extract x and y values from the density data
        x_values = density['Bin_Center']
        y_values = density['Density_per_bp']
        
        # Create bar plot with proper width
        bar_width = bin * 0.8  # Make bars slightly smaller than bin size for cleaner look
        
        bars = ax.bar(x_values, y_values, width=bar_width, 
                     color='steelblue', alpha=0.7, edgecolor='darkblue', linewidth=0.5)
        
        if boolean: 
            ax.set_title(f'Centromere Distance Insertion Rate - {chrom}\n{strain_name}/', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
            ax.set_ylabel('Insertion Rate', fontsize=12)
        else:
            ax.set_title(f'Centromere Distance Density - {chrom}\n{strain_name}/', 
                        fontsize=14, fontweight='bold')
            ax.set_xlabel('Distance from Centromere (bp)', fontsize=12)
            ax.set_ylabel('Density per bp', fontsize=12)

        # Add grid for better readability
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_axisbelow(True)
        
        # Format x-axis (now handling negative to positive range)
        min_distance = x_values.min()
        max_distance = x_values.max()
        distance_range = max_distance - min_distance
        
        if distance_range > 10000:
            # Show fewer ticks for large ranges
            tick_interval = int(distance_range / 8)  # About 8 ticks
            tick_interval = (tick_interval // 1000) * 1000  # Round to nearest 1000
            if tick_interval == 0:
                tick_interval = 1000
            # Create ticks that span from min to max distance
            tick_start = int(min_distance // tick_interval) * tick_interval
            tick_end = int(max_distance // tick_interval + 1) * tick_interval
            ax.set_xticks(np.arange(tick_start, tick_end + tick_interval, tick_interval))
            ax.tick_params(axis='x', rotation=45)
        
        # Add vertical line at x=0 (centromere position)
        ax.axvline(x=0, color='red', linestyle='-', alpha=0.8, linewidth=2, label='Centromere')
        
        # Set y-axis to start from 0
        ax.set_ylim(bottom=0)
        
        # Add some statistics to the plot
        mean_density = y_values.mean()
        max_density = y_values.max()
        ax.axhline(y=mean_density, color='red', linestyle='--', alpha=0.7, 
                  label=f'Mean: {mean_density:.3f}')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Tight layout for clean appearance
        plt.tight_layout()
        
        # Save individual plot
        plot_filename = f"{chrom}_centromere_density_Boolean_{boolean}_Bin{bin}.png"
        plt_file = os.path.join(dataset_output_folder, plot_filename)
        
        plt.savefig(plt_file, dpi=150, bbox_inches='tight', facecolor='white')
        plots_created += 1
        plt.close()


def density_from_centromere(input_folder, output_folder, bin=1000, max_distance_global=None, min_distance_global=None, boolean=False):
    """Memory-efficient version that processes one dataset at a time.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
        bin (int): Size of the sliding window for density calculation.
        max_distance_global (int): Maximum distance to consider globally.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_centromere(strain_name, dataset_path, dataset_name, output_folder, 
                                bin, max_distance_global, min_distance_global, boolean)


def density_from_nucleosome(input_folder, output_folder, boolean=False):
    """Compute density from nucleosome distances for all datasets in the input folder.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
        bin (int): Size of the sliding window for density calculation.
        max_distance_global (int): Maximum distance to consider globally.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    nucleosomes = Nucleosomes()
    nucleosomes_normalization = {}
    for chrom in chromosome_length.keys():
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        normalized_counts = nucleosomes.compute_exposure(chrom)
        nucleosomes_normalization[chrom] = normalized_counts

    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_nucleosome(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization, boolean)

    
def process_single_dataset_nucleosome(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization, boolean=False):
    """Process a single dataset for nucleosome distances and save results immediately to save memory.
    
    Args:
        strain_name (str): Name of the strain
        dataset_name (str): Name of the dataset
        output_folder (str): Path to the folder where the output CSV files will be saved.
        boolean (bool): If True, compute presence/absence density instead of counts.
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    counts = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_distances.csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        if chrom == "ChrM": continue
        file_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(file_path)
        counts[chrom] = {}

        for index, item in df.iterrows():
            if boolean and item['Value'] > 0: 
                value = 1 
            else: 
                value = item['Value']
                # Change the value of position 565392 in chromosome XV to 0
                if chrom == "ChrXV" and item['Position'] == 565392:
                    value = 0
            nucleosome_distance = item['Nucleosome_Distance']
            if nucleosome_distance in counts[chrom]:
                counts[chrom][nucleosome_distance] += value
            else:
                counts[chrom][nucleosome_distance] = value
        
        
        for distance in counts[chrom]:
            distance = int(distance)
            if distance in nucleosomes_normalization[chrom]:
                counts[chrom][distance] /= nucleosomes_normalization[chrom][distance]
            else:
                del counts[chrom][distance]
        for distance in nucleosomes_normalization[chrom]:
            if distance not in counts[chrom]:
                counts[chrom][distance] = 0

        # Save the processed counts to a CSV file
        output_file = os.path.join(dataset_output_folder, f"{chrom}_Boolean:_{boolean}_nucleosome_density.csv")
        with open(output_file, "w") as f:
            f.write("distance,density\n")  # Add header
            for dist, count in counts[chrom].items():
                f.write(f"{dist},{count}\n")
        create_nucleosome_plot(strain_name, dataset_name, dataset_output_folder, chrom, counts[chrom], boolean)

    # Clear counts from memory
    del counts


def create_nucleosome_plot(strain_name, dataset_name, dataset_output_folder, chrom, counts, boolean):
    """Create a plot for nucleosome distance density for a single chromosome.
    It should show each density value as a dot, and show a fitted polynomial line.
    """

    distances = np.array(list(counts.keys()))
    densities = np.array(list(counts.values()))
    
    # Create the plot
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))
    
    # Scatter plot of the raw data points
    ax.scatter(distances, densities, color='steelblue', alpha=0.6, edgecolor='darkblue', s=20, label='Data Points')
    

    coeffs = np.polyfit(distances, densities, deg=3)
    poly = np.poly1d(coeffs)
    x_fit = np.linspace(distances.min(), distances.max(), 500)
    y_fit = poly(x_fit)
    ax.plot(x_fit, y_fit, color='orange', linewidth=2, label='Fitted Polynomial (deg=3)')
    
    # Add the polynomial equation to the plot (simplified for readability)
    equation_text = f"y = {coeffs[0]:.2e}x³ + {coeffs[1]:.2e}x² + {coeffs[2]:.2e}x + {coeffs[3]:.2e}"
    ax.text(0.05, 0.95, equation_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5))

    if boolean:
        # Customize the plot
        ax.set_title(f'Nucleosome Distance Insertion Rate- {chrom}\n{strain_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance from Nearest Nucleosome (bp)', fontsize=12)
        ax.set_ylabel('Insertion Rate per bp', fontsize=12)
    else:
        # Customize the plot
        ax.set_title(f'Nucleosome Distance Density - {chrom}\n{strain_name}/{dataset_name}', 
                    fontsize=14, fontweight='bold')
        ax.set_xlabel('Distance from Nearest Nucleosome (bp)', fontsize=12)
        ax.set_ylabel('Density per bp', fontsize=12)

    # Add grid for better readability
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_axisbelow(True)
    ax.set_ylim(bottom=0)
    
    # Format x-axis
    max_distance = distances.max()
    if max_distance > 10000:
        # Show fewer ticks for large distances
        tick_interval = int(max_distance / 8)  # About 8 ticks
        tick_interval = (tick_interval // 1000) * 1000  # Round to nearest 1000
        if tick_interval == 0:
            tick_interval = 1000
        ax.set_xticks(np.arange(0, max_distance + tick_interval, tick_interval))
        ax.tick_params(axis='x', rotation=45)

    # Show legend
    ax.legend(loc='upper right', fontsize=12)

    # Save the plot
    output_file = os.path.join(dataset_output_folder, f"{chrom}_nucleosome_density.png")
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close(fig)

# ----------------- loader (unchanged) -----------------
def _load_nuc_density_tables(base_folder: str, boolean: bool) -> pd.DataFrame:
    rows = []
    suffix = f"_Boolean:_{boolean}_nucleosome_density.csv"

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith(suffix):
                continue
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            parts = os.path.normpath(root).split(os.sep)
            strain = parts[-2] if len(parts) >= 2 else "unknown_strain"
            dataset = parts[-1] if len(parts) >= 1 else "unknown_dataset"
            chrom = file.split("_")[0]

            df = pd.read_csv(path).rename(columns={"Distance":"distance","Density":"density"})
            if {"distance","density"} - set(df.columns):
                continue

            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df["density"]  = pd.to_numeric(df["density"], errors="coerce")
            df = df.dropna(subset=["distance","density"])
            df["chrom"]   = chrom
            df["strain"]  = strain
            df["dataset"] = dataset
            df["path"]    = path
            rows.append(df[["chrom","strain","dataset","distance","density","path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom","strain","dataset","distance","density","path"])
    return pd.concat(rows, ignore_index=True)

# ----------------- combiner (+ plotting) -----------------
def _combine_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool, min_distance=None, max_distance=None):
    """
    Writes one combined CSV (and PNG if plot=True) per group.
    CSV columns: distance, mean_density, sd_density, se_density, n_datasets
    """
    if df.empty:
        print("[combine] no data found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Filter by distance range if specified
    if min_distance is not None:
        df = df[df["distance"] >= min_distance]
    if max_distance is not None:
        df = df[df["distance"] <= max_distance]

    if df.empty:
        return

    if group_by != ["chrom"]:
        for chrom in chromosome_length:
            if chrom in df['chrom'].values:
                df.loc[df['chrom'] == chrom, 'density'] *= chromosome_length[chrom] / mean_chrom_length

    keys = group_by + ["distance"]
    combined = (df.groupby(keys, as_index=False)
                  .agg(mean_density=("density","mean"),
                       sd_density  =("density","std"),
                       n_datasets  =("density","size"),
                       se_density  =("density","sem")))
    combined = combined.sort_values(keys)

    # Write and plot per group
    for keys, sub in combined.groupby(group_by if group_by else [lambda _: True]):
        if not group_by:
            label = "ALL"
        else:
            if not isinstance(keys, tuple): keys = (keys,)
            label = "_".join(f"{col}-{val}" for col, val in zip(group_by, keys))

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)

        if plot:
            fig, ax = plt.subplots(figsize=(7,4))
            # main line
            ax.plot(sub["distance"], sub["mean_density"], label="Mean", color='black')
            # ribbon: ±2 SE (approximately 95% confidence interval)
            lo = sub["mean_density"] - 2 * sub["se_density"].fillna(0)
            hi = sub["mean_density"] + 2 * sub["se_density"].fillna(0)
            ax.fill_between(sub["distance"], lo, hi, alpha=0.15, label="±2 SE", color='black')

            ax.set_xlabel("Distance from nucleosome (bp)")
            ax.set_ylabel("Transposon Insertion Rate")
            ax.set_title(f"Combined nucleosome insertion rate — {label}")
            ax.legend(loc="best")
            ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
            ax.minorticks_on()  # enable minor ticks on both axes
            ax.set_ylim(bottom=0)

            fig.tight_layout()

            out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


def combine_nucleosome_data(data="All", boolean=False, plot=False, base_folder = "Data_exploration/results/densities/nucleosome", min_distance=None, max_distance=None):
    """
    Combine nucleosome density curves (distance,density) across folders.

    data:
      - "All":         one global curve
      - "Chromosomes": one curve per chromosome
      - "Strains":     one curve per strain
      - "Datasets":    one curve per dataset
    plot:
      - if True, saves a PNG next to each CSV
    min_distance:
      - if specified, only include distances >= min_distance
    max_distance:
      - if specified, only include distances <= max_distance
    """
    out_base = os.path.join(base_folder, f"combined_{data}_Boolean_{boolean}")
    os.makedirs(out_base, exist_ok=True)

    df = _load_nuc_density_tables(base_folder, boolean=boolean)
    if df.empty:
        print("[combine] no matching files found.")
        return

    tag = f"Boolean_{boolean}_nucleosome_density"

    if data == "All":
        _combine_curves(df, group_by=[], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Chromosomes":
        _combine_curves(df, group_by=["chrom"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Strains":
        _combine_curves(df, group_by=["strain"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Datasets":
        _combine_curves(df, group_by=["dataset"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    else:
        raise ValueError("data must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


# ---------- 1) Loader: read per-dataset centromere CSVs ----------
def _load_cen_density_tables(base_folder: str, boolean: bool = None, bin_size: int = None) -> pd.DataFrame:
    """
    Scans base_folder for centromere density files and returns one long DF with:
      ['chrom','strain','dataset','bin_size','boolean','Bin_Center','Density_per_bp','path']
    Filters by boolean and bin_size if specified.
    """
    rows = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith("_centromere_density.csv"):
                continue
            
            # Extract boolean and bin size from filename: ChrI_Boolean:True_bin:1000_centromere_density.csv
            file_parts = file.split("_")
            file_boolean = None
            file_bin_size = None
            
            for part in file_parts:
                if part.startswith("Boolean:"):
                    file_boolean = part.split(":")[1] == "True"
                elif part.startswith("bin:"):
                    file_bin_size = int(part.split(":")[1])
            
            # Skip files that don't match the filters
            if boolean is not None and file_boolean != boolean:
                continue
            if bin_size is not None and file_bin_size != bin_size:
                continue
                
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            root_parts = os.path.normpath(root).split(os.sep)
            strain  = root_parts[-2] if len(root_parts) >= 2 else "unknown_strain"
            dataset = root_parts[-1] if len(root_parts) >= 1 else "unknown_dataset"
            chrom   = file_parts[0]  # filename starts with 'chrX_...'

            df = pd.read_csv(path)
            # normalize column names
            df = df.rename(columns={
                "Distance": "Bin_Center",
                "distance": "Bin_Center",
                "Density": "Density_per_bp",
                "density": "Density_per_bp",
            })
            required = {"Bin_Center", "Density_per_bp"}
            if not required.issubset(df.columns):
                print(f"[skip] {path} missing {required - set(df.columns)}")
                continue

            df["Bin_Center"]    = pd.to_numeric(df["Bin_Center"], errors="coerce")
            df["Density_per_bp"] = pd.to_numeric(df["Density_per_bp"], errors="coerce")
            df = df.dropna(subset=["Bin_Center", "Density_per_bp"])

            df["chrom"]   = chrom
            df["strain"]  = strain
            df["dataset"] = dataset
            df["bin_size"] = file_bin_size
            df["boolean"] = file_boolean
            df["path"]    = path
            rows.append(df[["chrom","strain","dataset","bin_size","boolean","Bin_Center","Density_per_bp","path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom","strain","dataset","bin_size","boolean","Bin_Center","Density_per_bp","path"])
    return pd.concat(rows, ignore_index=True)


# ---------- 2) Combiner: unweighted mean ± SE, optional plotting ----------
def _combine_cen_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool, bin_size: int, absolute_distance: bool = False):
    """
    Unweighted means across datasets per Bin_Center (no exposure weighting).
    Writes one CSV (and PNG if plot=True) per group.
    CSV columns: Bin_Center, mean_density, sd_density, se_density, n_datasets
    absolute_distance: If True, convert Bin_Center to absolute values (overlapping left/right sides)
    """
    if df.empty:
        print("[centromere] no data found.")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Convert to absolute distance if requested
    if absolute_distance:
        df = df.copy()
        df["Bin_Center"] = df["Bin_Center"].abs()

    within_keys = group_by + ["Bin_Center"]
    combined = (df.groupby(within_keys, as_index=False).agg(
                mean_density=("Density_per_bp","mean"),
                sd_density  =("Density_per_bp","std"),
                n_datasets  =("Density_per_bp","size"),
                se_density  =("Density_per_bp","sem"),
            ))

    # Write and plot per group
    group_iter = [((), combined)] if not group_by else combined.groupby(group_by, dropna=False)

    for keys, sub in group_iter:
        label = "ALL" if not group_by else "_".join(
            f"{col}-{val}" for col, val in zip(group_by, keys if isinstance(keys, tuple) else (keys,))
        )

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)
        
        if not plot:
            continue
        
        sub_filtered = sub[sub["mean_density"] > 0]

        if len(sub_filtered) > 0:
            sub_sorted = sub_filtered.sort_values("Bin_Center").copy()
            fig, ax = plt.subplots(figsize=(7,4))
            ax.plot(sub_sorted["Bin_Center"], sub_sorted["mean_density"], label="Mean", color='black')
            # ribbon: ±2 SE (approximately 95% confidence interval)
            lo = sub_sorted["mean_density"] - 2 * sub_sorted["se_density"].fillna(0)
            hi = sub_sorted["mean_density"] + 2 * sub_sorted["se_density"].fillna(0)
            ax.fill_between(sub_sorted["Bin_Center"], lo, hi, alpha=0.15, label="±2 SE", color='black')
            # Only show centromere line at x=0 when using signed distances
            if not absolute_distance:
                ax.axvline(0, linestyle="--", linewidth=1, color="red", alpha=0.7, label="Centromere")
            ax.set_xlabel("Distance from centromere (bp)")
            ax.set_ylabel("Transposon Insertion Rate")
            ax.set_title(f"Centromere bias — {label}, Bin:{bin_size}")
            ax.legend(loc="best")
            ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
            ax.minorticks_on()
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


def combine_centromere_data(mode="All", boolean=None, bin_size=None, plot=True, absolute_distance=False, base_folder = "Data_exploration/results/densities/centromere"):
    """
    Combine signed centromere-distance curves (Bin_Center, Density_per_bp) with
    unweighted means across datasets (no exposure weighting), and plot if requested.

    mode:
      - "All":         one global curve
      - "Chromosomes": one curve per chromosome
      - "Strains":     one curve per strain
      - "Datasets":    one curve per dataset
    boolean: Filter by boolean value (True/False) - if None, includes all
    bin_size: Filter by bin size (e.g., 100, 1000) - if None, includes all
    absolute_distance: If True, use absolute distance (overlap left/right sides of centromere)
    """
    
    # Create descriptive folder name based on filters
    folder_parts = [f"combined_{mode}"]
    if boolean is not None:
        folder_parts.append(f"Boolean_{boolean}")
    if bin_size is not None:
        folder_parts.append(f"bin_{bin_size}")
    if absolute_distance:
        folder_parts.append("absolute")
    
    out_dir = os.path.join(base_folder, "_".join(folder_parts))
    os.makedirs(out_dir, exist_ok=True)

    df = _load_cen_density_tables(base_folder, boolean=boolean, bin_size=bin_size)
    if df.empty:
        print(f"[centromere] no matching files found for boolean={boolean}, bin_size={bin_size}.")
        return

    # Create descriptive tag for output files
    tag_parts = ["centromere_density"]
    if boolean is not None:
        tag_parts.append(f"Boolean_{boolean}")
    if bin_size is not None:
        tag_parts.append(f"bin_{bin_size}")
    if absolute_distance:
        tag_parts.append("absolute")
    tag = "_".join(tag_parts)

    if mode == "All":
        _combine_cen_curves(df, group_by=[], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Chromosomes":
        _combine_cen_curves(df, group_by=["chrom"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Strains":
        _combine_cen_curves(df, group_by=["strain"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Datasets":
        _combine_cen_curves(df, group_by=["dataset"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    else:
        raise ValueError("mode must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


# ========================================
# MEAN VALUE FUNCTIONS FOR NON-ZERO VALUES
# ========================================

def process_single_dataset_centromere_mean(strain_name, dataset_path, dataset_name, output_folder, bin=100, max_distance_global=None, min_distance_global=None):
    """Process a single dataset and compute mean of non-zero values by centromere distance.
    
    Args:
        strain_name (str): Name of the strain
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        output_folder (str): Base output folder
        bin (int): Size of the sliding window for mean calculation
        max_distance_global (int): Maximum distance to consider
        min_distance_global (int): Minimum distance to consider
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    dataset_data = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_distances.csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        file_path = os.path.join(dataset_path, csv_file)
        dataset_data[chrom] = pd.read_csv(file_path)
    
    # Process each chromosome in this dataset
    for chrom in dataset_data:
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        if chrom == "ChrXV":
            dataset_data[chrom].loc[dataset_data[chrom]['Position'] == 565392, 'Value'] = 0
            
        df = dataset_data[chrom]

        if max_distance_global is not None:
            max_distance = max_distance_global
        else:
            max_distance = df['Centromere_Distance'].max()
        if min_distance_global is not None:
            min_distance = min_distance_global
        else:
            min_distance = df['Centromere_Distance'].min()
            
        # Create bins aligned around centromere (position 0)
        data_range = max(abs(min_distance), abs(max_distance))
        n_bins_each_side = int(np.ceil(data_range / bin)) + 1
        bin_centers = np.arange(-n_bins_each_side * bin, (n_bins_each_side + 1) * bin, bin)
        bin_edges = bin_centers - bin/2
        bin_edges = np.append(bin_edges, bin_edges[-1] + bin)
        
        df['Distance_Bin'] = pd.cut(df['Centromere_Distance'], bins=bin_edges, right=False, include_lowest=True)

        # Filter only non-zero values
        df_nonzero = df[df['Value'] > 0].copy()
        
        # Remove top 5% percentile values (outliers)
        if len(df_nonzero) > 0:
            percentile_95 = df_nonzero['Value'].quantile(0.95)
            df_nonzero = df_nonzero[df_nonzero['Value'] <= percentile_95]
        
        # Compute mean of non-zero values per bin
        mean_data = df_nonzero.groupby('Distance_Bin')['Value'].agg(['mean', 'std', 'count']).reset_index()
        mean_data['Bin_Center'] = mean_data['Distance_Bin'].apply(lambda x: x.left + bin / 2)
        mean_data = mean_data.rename(columns={'mean': 'Mean_Nonzero', 'std': 'Std_Nonzero', 'count': 'Count_Nonzero'})
        
        if mean_data.empty:
            print(f"No valid mean data for {chrom} in {strain_name}/{dataset_name}. Skipping.")
            continue
        
        # Save to CSV immediately
        output_file = os.path.join(dataset_output_folder, f"{chrom}_bin:{bin}_centromere_mean.csv")
        mean_data[['Bin_Center', 'Mean_Nonzero', 'Std_Nonzero', 'Count_Nonzero']].to_csv(output_file, index=False)
    
    # Clear dataset from memory
    del dataset_data


def mean_from_centromere(input_folder, output_folder, bin=1000, max_distance_global=None, min_distance_global=None):
    """Compute mean of non-zero values from centromere distances for all datasets.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
        bin (int): Size of the sliding window for mean calculation.
        max_distance_global (int): Maximum distance to consider globally.
        min_distance_global (int): Minimum distance to consider globally.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    
    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_centromere_mean(strain_name, dataset_path, dataset_name, output_folder, 
                                bin, max_distance_global, min_distance_global)


def process_single_dataset_nucleosome_mean(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization):
    """Process a single dataset for nucleosome distances and compute mean of non-zero values.
    
    Args:
        strain_name (str): Name of the strain
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        output_folder (str): Path to the folder where the output CSV files will be saved.
        nucleosomes_normalization (dict): Normalization factors per chromosome
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    counts = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_distances.csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        if chrom == "ChrM": continue
        file_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Filter non-zero values only
        df_nonzero = df[df['Value'] > 0].copy()
        
        # Change the value of position 565392 in chromosome XV to 0
        if chrom == "ChrXV":
            df_nonzero = df_nonzero[df_nonzero['Position'] != 565392]
        
        # Remove top 5% percentile values (outliers)
        if len(df_nonzero) > 0:
            percentile_95 = df_nonzero['Value'].quantile(0.95)
            df_nonzero = df_nonzero[df_nonzero['Value'] <= percentile_95]
        
        # Group by nucleosome distance and compute mean
        mean_by_distance = df_nonzero.groupby('Nucleosome_Distance')['Value'].agg(['mean', 'std', 'count']).reset_index()
        
        counts[chrom] = {}
        for _, row in mean_by_distance.iterrows():
            distance = int(row['Nucleosome_Distance'])
            if distance in nucleosomes_normalization[chrom]:
                # Store mean and count info
                counts[chrom][distance] = {
                    'mean': row['mean'],
                    'std': row['std'] if not pd.isna(row['std']) else 0,
                    'count': row['count']
                }
        
        # Add zeros for distances with no non-zero values
        for distance in nucleosomes_normalization[chrom]:
            if distance not in counts[chrom]:
                counts[chrom][distance] = {'mean': 0, 'std': 0, 'count': 0}

        # Save the processed counts to a CSV file
        output_file = os.path.join(dataset_output_folder, f"{chrom}_nucleosome_mean.csv")
        with open(output_file, "w") as f:
            f.write("distance,mean_nonzero,std_nonzero,count_nonzero\n")
            for dist, data in counts[chrom].items():
                f.write(f"{dist},{data['mean']},{data['std']},{data['count']}\n")

    # Clear counts from memory
    del counts


def mean_from_nucleosome(input_folder, output_folder):
    """Compute mean of non-zero values from nucleosome distances for all datasets.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    nucleosomes = Nucleosomes()
    nucleosomes_normalization = {}
    for chrom in chromosome_length.keys():
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        normalized_counts = nucleosomes.compute_exposure(chrom)
        nucleosomes_normalization[chrom] = normalized_counts

    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_nucleosome_mean(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization)


def process_single_dataset_nucleosome_median(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization):
    """Process a single dataset for nucleosome distances and compute median of non-zero values.
    
    Args:
        strain_name (str): Name of the strain
        dataset_path (str): Path to the dataset folder
        dataset_name (str): Name of the dataset
        output_folder (str): Path to the folder where the output CSV files will be saved.
        nucleosomes_normalization (dict): Normalization factors per chromosome
    """
    # Create output folders
    strain_output_folder = os.path.join(output_folder, strain_name)
    dataset_output_folder = os.path.join(strain_output_folder, dataset_name)
    os.makedirs(dataset_output_folder, exist_ok=True)
    
    # Load only one dataset's data
    counts = {}
    csv_files = [f for f in os.listdir(dataset_path) if f.endswith("_distances.csv")]
    
    for csv_file in csv_files:
        chrom = csv_file.split("_")[0]
        if chrom == "ChrM": continue
        file_path = os.path.join(dataset_path, csv_file)
        df = pd.read_csv(file_path)
        
        # Filter non-zero values only
        df_nonzero = df[df['Value'] > 0].copy()
        
        # Change the value of position 565392 in chromosome XV to 0
        if chrom == "ChrXV":
            df_nonzero = df_nonzero[df_nonzero['Position'] != 565392]
        
        # Remove top 5% percentile values (outliers)
        if len(df_nonzero) > 0:
            percentile_95 = df_nonzero['Value'].quantile(0.95)
            df_nonzero = df_nonzero[df_nonzero['Value'] <= percentile_95]
        
        # Group by nucleosome distance and compute median
        median_by_distance = df_nonzero.groupby('Nucleosome_Distance')['Value'].agg(['median', 'std', 'count']).reset_index()
        
        counts[chrom] = {}
        for _, row in median_by_distance.iterrows():
            distance = int(row['Nucleosome_Distance'])
            if distance in nucleosomes_normalization[chrom]:
                # Store median and count info
                counts[chrom][distance] = {
                    'median': row['median'],
                    'std': row['std'] if not pd.isna(row['std']) else 0,
                    'count': row['count']
                }
        
        # Add zeros for distances with no non-zero values
        for distance in nucleosomes_normalization[chrom]:
            if distance not in counts[chrom]:
                counts[chrom][distance] = {'median': 0, 'std': 0, 'count': 0}

        # Save the processed counts to a CSV file
        output_file = os.path.join(dataset_output_folder, f"{chrom}_nucleosome_median.csv")
        with open(output_file, "w") as f:
            f.write("distance,median_nonzero,std_nonzero,count_nonzero\n")
            for dist, data in counts[chrom].items():
                f.write(f"{dist},{data['median']},{data['std']},{data['count']}\n")

    # Clear counts from memory
    del counts


def median_from_nucleosome(input_folder, output_folder):
    """Compute median of non-zero values from nucleosome distances for all datasets.
    
    Args:
        input_folder (str): Path to the folder containing distance CSV files (strain/dataset structure).
        output_folder (str): Path to the folder where the output CSV files will be saved.
    """
    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)
    nucleosomes = Nucleosomes()
    nucleosomes_normalization = {}
    for chrom in chromosome_length.keys():
        if chrom == "ChrM": continue  # Skip mitochondrial chromosome
        normalized_counts = nucleosomes.compute_exposure(chrom)
        nucleosomes_normalization[chrom] = normalized_counts

    # Collect all datasets without loading data
    datasets_to_process = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_to_process.append((strain_name, root, dataset_name))
    
    # Process each dataset individually
    for strain_name, dataset_path, dataset_name in datasets_to_process:
        process_single_dataset_nucleosome_median(strain_name, dataset_path, dataset_name, output_folder, nucleosomes_normalization)


# ========================================
# COMBINING AND PLOTTING MEAN VALUES
# ========================================

def _load_nuc_mean_tables(base_folder: str) -> pd.DataFrame:
    """Load nucleosome mean value tables from all datasets.
    
    Returns:
        DataFrame with columns: chrom, strain, dataset, distance, mean_nonzero, std_nonzero, count_nonzero, path
    """
    rows = []
    suffix = "_nucleosome_mean.csv"

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith(suffix):
                continue
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            parts = os.path.normpath(root).split(os.sep)
            strain = parts[-2] if len(parts) >= 2 else "unknown_strain"
            dataset = parts[-1] if len(parts) >= 1 else "unknown_dataset"
            chrom = file.split("_")[0]

            df = pd.read_csv(path)
            required = {"distance", "mean_nonzero"}
            if not required.issubset(df.columns):
                continue

            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df["mean_nonzero"] = pd.to_numeric(df["mean_nonzero"], errors="coerce")
            if "std_nonzero" in df.columns:
                df["std_nonzero"] = pd.to_numeric(df["std_nonzero"], errors="coerce")
            else:
                df["std_nonzero"] = np.nan
            if "count_nonzero" in df.columns:
                df["count_nonzero"] = pd.to_numeric(df["count_nonzero"], errors="coerce")
            else:
                df["count_nonzero"] = 1
                
            df = df.dropna(subset=["distance", "mean_nonzero"])
            df["chrom"] = chrom
            df["strain"] = strain
            df["dataset"] = dataset
            df["path"] = path
            rows.append(df[["chrom", "strain", "dataset", "distance", "mean_nonzero", "std_nonzero", "count_nonzero", "path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom", "strain", "dataset", "distance", "mean_nonzero", "std_nonzero", "count_nonzero", "path"])
    return pd.concat(rows, ignore_index=True)


def _combine_mean_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool, min_distance=None, max_distance=None):
    """
    Combine mean value curves across datasets.
    Writes one combined CSV (and PNG if plot=True) per group.
    CSV columns: distance, mean_of_means, sd_of_means, se_of_means, n_datasets
    """
    if df.empty:
        print("[combine_mean] no data found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Filter by distance range if specified
    if min_distance is not None:
        df = df[df["distance"] >= min_distance]
    if max_distance is not None:
        df = df[df["distance"] <= max_distance]

    if df.empty:
        return

    # Normalize by chromosome length if not grouping by chromosome
    if group_by != ["chrom"]:
        for chrom in chromosome_length:
            if chrom in df['chrom'].values:
                df.loc[df['chrom'] == chrom, 'mean_nonzero'] *= chromosome_length[chrom] / mean_chrom_length

    keys = group_by + ["distance"]
    combined = (df.groupby(keys, as_index=False)
                  .agg(mean_of_means=("mean_nonzero", "mean"),
                       sd_of_means=("mean_nonzero", "std"),
                       n_datasets=("mean_nonzero", "size"),
                       se_of_means=("mean_nonzero", "sem")))
    combined = combined.sort_values(keys)

    # Write and plot per group
    for keys, sub in combined.groupby(group_by if group_by else [lambda _: True]):
        if not group_by:
            label = "ALL"
        else:
            if not isinstance(keys, tuple): keys = (keys,)
            label = "_".join(f"{col}-{val}" for col, val in zip(group_by, keys))

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)

        if plot:
            # Filter out rows with mean=0 or very small means for cleaner plotting
            sub_filtered = sub[sub["mean_of_means"] > 0]
            
            if len(sub_filtered) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                # main line
                ax.plot(sub_filtered["distance"], sub_filtered["mean_of_means"], label="Mean", color='black')
                # ribbon: ±2 SE (approximately 95% confidence interval)
                lo = sub_filtered["mean_of_means"] - 2 * sub_filtered["se_of_means"].fillna(0)
                hi = sub_filtered["mean_of_means"] + 2 * sub_filtered["se_of_means"].fillna(0)
                ax.fill_between(sub_filtered["distance"], lo, hi, alpha=0.15, label="±2 SE", color='black')

                ax.set_xlabel("Distance from nucleosome (bp)")
                ax.set_ylabel("Mean Transposon Count ")
                ax.set_title(f"Mean transposon count at nucleosome distance — {label}")
                ax.legend(loc="best")
                ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
                ax.minorticks_on()
                ax.set_ylim(bottom=0)

                fig.tight_layout()

                out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)


def combine_nucleosome_mean_data(data="All", plot=False, base_folder="Data_exploration/results/means/nucleosome", min_distance=None, max_distance=None):
    """
    Combine nucleosome mean value curves across datasets and optionally plot.

    Args:
        data: "All", "Chromosomes", "Strains", or "Datasets"
        plot: if True, saves a PNG next to each CSV
        base_folder: base folder containing the mean data
        min_distance: if specified, only include distances >= min_distance
        max_distance: if specified, only include distances <= max_distance
    """
    out_base = os.path.join(base_folder, f"combined_{data}")
    os.makedirs(out_base, exist_ok=True)

    df = _load_nuc_mean_tables(base_folder)
    if df.empty:
        print("[combine_mean] no matching files found.")
        return

    tag = "nucleosome_mean"

    if data == "All":
        _combine_mean_curves(df, group_by=[], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Chromosomes":
        _combine_mean_curves(df, group_by=["chrom"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Strains":
        _combine_mean_curves(df, group_by=["strain"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Datasets":
        _combine_mean_curves(df, group_by=["dataset"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    else:
        raise ValueError("data must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


# ========================================
# COMBINING AND PLOTTING MEDIAN VALUES
# ========================================

def _load_nuc_median_tables(base_folder: str) -> pd.DataFrame:
    """Load nucleosome median value tables from all datasets.
    
    Returns:
        DataFrame with columns: chrom, strain, dataset, distance, median_nonzero, std_nonzero, count_nonzero, path
    """
    rows = []
    suffix = "_nucleosome_median.csv"

    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith(suffix):
                continue
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            parts = os.path.normpath(root).split(os.sep)
            strain = parts[-2] if len(parts) >= 2 else "unknown_strain"
            dataset = parts[-1] if len(parts) >= 1 else "unknown_dataset"
            chrom = file.split("_")[0]

            df = pd.read_csv(path)
            required = {"distance", "median_nonzero"}
            if not required.issubset(df.columns):
                continue

            df["distance"] = pd.to_numeric(df["distance"], errors="coerce")
            df["median_nonzero"] = pd.to_numeric(df["median_nonzero"], errors="coerce")
            if "std_nonzero" in df.columns:
                df["std_nonzero"] = pd.to_numeric(df["std_nonzero"], errors="coerce")
            else:
                df["std_nonzero"] = np.nan
            if "count_nonzero" in df.columns:
                df["count_nonzero"] = pd.to_numeric(df["count_nonzero"], errors="coerce")
            else:
                df["count_nonzero"] = 1
                
            df = df.dropna(subset=["distance", "median_nonzero"])
            df["chrom"] = chrom
            df["strain"] = strain
            df["dataset"] = dataset
            df["path"] = path
            rows.append(df[["chrom", "strain", "dataset", "distance", "median_nonzero", "std_nonzero", "count_nonzero", "path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom", "strain", "dataset", "distance", "median_nonzero", "std_nonzero", "count_nonzero", "path"])
    return pd.concat(rows, ignore_index=True)


def _combine_median_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool, min_distance=None, max_distance=None):
    """
    Combine median value curves across datasets.
    Writes one combined CSV (and PNG if plot=True) per group.
    CSV columns: distance, mean_of_medians, sd_of_medians, se_of_medians, n_datasets
    """
    if df.empty:
        print("[combine_median] no data found.")
        return

    os.makedirs(out_dir, exist_ok=True)

    # Filter by distance range if specified
    if min_distance is not None:
        df = df[df["distance"] >= min_distance]
    if max_distance is not None:
        df = df[df["distance"] <= max_distance]

    if df.empty:
        return

    # Normalize by chromosome length if not grouping by chromosome
    if group_by != ["chrom"]:
        for chrom in chromosome_length:
            if chrom in df['chrom'].values:
                df.loc[df['chrom'] == chrom, 'median_nonzero'] *= chromosome_length[chrom] / mean_chrom_length

    keys = group_by + ["distance"]
    combined = (df.groupby(keys, as_index=False)
                  .agg(mean_of_medians=("median_nonzero", "mean"),
                       sd_of_medians=("median_nonzero", "std"),
                       n_datasets=("median_nonzero", "size"),
                       se_of_medians=("median_nonzero", "sem")))
    combined = combined.sort_values(keys)

    # Write and plot per group
    for keys, sub in combined.groupby(group_by if group_by else [lambda _: True]):
        if not group_by:
            label = "ALL"
        else:
            if not isinstance(keys, tuple): keys = (keys,)
            label = "_".join(f"{col}-{val}" for col, val in zip(group_by, keys))

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)

        if plot:
            # Filter out rows with median=0 or very small medians for cleaner plotting
            sub_filtered = sub[sub["mean_of_medians"] > 0]
            
            if len(sub_filtered) > 0:
                fig, ax = plt.subplots(figsize=(7, 4))
                # main line
                ax.plot(sub_filtered["distance"], sub_filtered["mean_of_medians"], label="Mean of Medians", color='black')
                # ribbon: ±2 SE (approximately 95% confidence interval)
                lo = sub_filtered["mean_of_medians"] - 2 * sub_filtered["se_of_medians"].fillna(0)
                hi = sub_filtered["mean_of_medians"] + 2 * sub_filtered["se_of_medians"].fillna(0)
                ax.fill_between(sub_filtered["distance"], lo, hi, alpha=0.15, label="±2 SE", color='black')

                ax.set_xlabel("Distance from nucleosome (bp)")
                ax.set_ylabel("Median Transposon Count")
                ax.set_title(f"Median transposon count at nucleosome distance — {label}")
                ax.legend(loc="best")
                ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
                ax.minorticks_on()
                ax.set_ylim(bottom=0)

                fig.tight_layout()

                out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
                fig.savefig(out_png, dpi=150)
                plt.close(fig)


def combine_nucleosome_median_data(data="All", plot=False, base_folder="Data_exploration/results/medians/nucleosome", min_distance=None, max_distance=None):
    """
    Combine nucleosome median value curves across datasets and optionally plot.

    Args:
        data: "All", "Chromosomes", "Strains", or "Datasets"
        plot: if True, saves a PNG next to each CSV
        base_folder: base folder containing the median data
        min_distance: if specified, only include distances >= min_distance
        max_distance: if specified, only include distances <= max_distance
    """
    out_base = os.path.join(base_folder, f"combined_{data}")
    os.makedirs(out_base, exist_ok=True)

    df = _load_nuc_median_tables(base_folder)
    if df.empty:
        print("[combine_median] no matching files found.")
        return

    tag = "nucleosome_median"

    if data == "All":
        _combine_median_curves(df, group_by=[], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Chromosomes":
        _combine_median_curves(df, group_by=["chrom"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Strains":
        _combine_median_curves(df, group_by=["strain"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    elif data == "Datasets":
        _combine_median_curves(df, group_by=["dataset"], out_dir=out_base, tag=tag, plot=plot, min_distance=min_distance, max_distance=max_distance)
    else:
        raise ValueError("data must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


def _load_cen_mean_tables(base_folder: str, bin_size: int = None) -> pd.DataFrame:
    """
    Load centromere mean value tables and return one long DataFrame.
    Filters by bin_size if specified.
    """
    rows = []
    for root, dirs, files in os.walk(base_folder):
        for file in files:
            if not file.endswith("_centromere_mean.csv"):
                continue
            
            # Extract bin size from filename: ChrI_bin:10000_centromere_mean.csv
            file_parts = file.split("_")
            file_bin_size = None
            
            for part in file_parts:
                if part.startswith("bin:"):
                    file_bin_size = int(part.split(":")[1])
            
            # Skip files that don't match the filter
            if bin_size is not None and file_bin_size != bin_size:
                continue
                
            path = os.path.join(root, file)

            # infer metadata: .../<strain>/<dataset>/file.csv
            root_parts = os.path.normpath(root).split(os.sep)
            strain = root_parts[-2] if len(root_parts) >= 2 else "unknown_strain"
            dataset = root_parts[-1] if len(root_parts) >= 1 else "unknown_dataset"
            chrom = file_parts[0]

            df = pd.read_csv(path)
            required = {"Bin_Center", "Mean_Nonzero"}
            if not required.issubset(df.columns):
                print(f"[skip] {path} missing {required - set(df.columns)}")
                continue

            df["Bin_Center"] = pd.to_numeric(df["Bin_Center"], errors="coerce")
            df["Mean_Nonzero"] = pd.to_numeric(df["Mean_Nonzero"], errors="coerce")
            if "Std_Nonzero" in df.columns:
                df["Std_Nonzero"] = pd.to_numeric(df["Std_Nonzero"], errors="coerce")
            if "Count_Nonzero" in df.columns:
                df["Count_Nonzero"] = pd.to_numeric(df["Count_Nonzero"], errors="coerce")
            
            df = df.dropna(subset=["Bin_Center", "Mean_Nonzero"])

            df["chrom"] = chrom
            df["strain"] = strain
            df["dataset"] = dataset
            df["bin_size"] = file_bin_size
            df["path"] = path
            rows.append(df[["chrom", "strain", "dataset", "bin_size", "Bin_Center", "Mean_Nonzero", "Std_Nonzero", "Count_Nonzero", "path"]])

    if not rows:
        return pd.DataFrame(columns=["chrom", "strain", "dataset", "bin_size", "Bin_Center", "Mean_Nonzero", "Std_Nonzero", "Count_Nonzero", "path"])
    return pd.concat(rows, ignore_index=True)


def _combine_cen_mean_curves(df: pd.DataFrame, group_by: list, out_dir: str, tag: str, plot: bool, bin_size: int, absolute_distance: bool = False):
    """
    Combine centromere mean value curves across datasets.
    Writes one CSV (and PNG if plot=True) per group.
    CSV columns: Bin_Center, mean_of_means, sd_of_means, se_of_means, n_datasets
    """
    if df.empty:
        print("[centromere_mean] no data found.")
        return
    os.makedirs(out_dir, exist_ok=True)

    # Convert to absolute distance if requested
    if absolute_distance:
        df = df.copy()
        df["Bin_Center"] = df["Bin_Center"].abs()

    within_keys = group_by + ["Bin_Center"]
    combined = (df.groupby(within_keys, as_index=False).agg(
                mean_of_means=("Mean_Nonzero", "mean"),
                sd_of_means=("Mean_Nonzero", "std"),
                n_datasets=("Mean_Nonzero", "size"),
                se_of_means=("Mean_Nonzero", "sem"),
            ))

    # Write and plot per group
    group_iter = [((), combined)] if not group_by else combined.groupby(group_by, dropna=False)

    for keys, sub in group_iter:
        label = "ALL" if not group_by else "_".join(
            f"{col}-{val}" for col, val in zip(group_by, keys if isinstance(keys, tuple) else (keys,))
        )

        out_csv = os.path.join(out_dir, f"{label}_combined_{tag}.csv")
        sub.to_csv(out_csv, index=False)
        
        if not plot:
            continue
        
        sub_filtered = sub[sub["mean_of_means"] > 0]

        if len(sub_filtered) > 0:
            sub_sorted = sub_filtered.sort_values("Bin_Center").copy()
            fig, ax = plt.subplots(figsize=(7, 4))
            ax.plot(sub_sorted["Bin_Center"], sub_sorted["mean_of_means"], label="Mean", color='black')
            # ribbon: ±2 SE
            lo = sub_sorted["mean_of_means"] - 2 * sub_sorted["se_of_means"].fillna(0)
            hi = sub_sorted["mean_of_means"] + 2 * sub_sorted["se_of_means"].fillna(0)
            ax.fill_between(sub_sorted["Bin_Center"], lo, hi, alpha=0.15, label="±2 SE", color='black')
            # Only show centromere line at x=0 when using signed distances
            if not absolute_distance:
                ax.axvline(0, linestyle="--", linewidth=1, color="red", alpha=0.7, label="Centromere")
            ax.set_xlabel("Distance from centromere (bp)")
            ax.set_ylabel("Mean Transposon Count")
            ax.set_title(f"Mean transposon count at centromere distance — {label}, Bin:{bin_size}")
            ax.legend(loc="best")
            ax.grid(True, which='both', axis='both', alpha=0.4, linestyle='--')
            ax.minorticks_on()
            ax.set_ylim(bottom=0)
            fig.tight_layout()
            out_png = os.path.join(out_dir, f"{label}_combined_{tag}.png")
            fig.savefig(out_png, dpi=150)
            plt.close(fig)


def combine_centromere_mean_data(mode="All", bin_size=None, plot=True, absolute_distance=False, base_folder="Data_exploration/results/means/centromere"):
    """
    Combine centromere mean value curves across datasets and optionally plot.

    Args:
        mode: "All", "Chromosomes", "Strains", or "Datasets"
        bin_size: Filter by bin size (e.g., 100, 1000) - if None, includes all
        plot: if True, creates plots
        absolute_distance: If True, use absolute distance (overlap left/right sides of centromere)
        base_folder: base folder containing the mean data
    """
    # Create descriptive folder name based on filters
    folder_parts = [f"combined_{mode}"]
    if bin_size is not None:
        folder_parts.append(f"bin_{bin_size}")
    if absolute_distance:
        folder_parts.append("absolute")
    
    out_dir = os.path.join(base_folder, "_".join(folder_parts))
    os.makedirs(out_dir, exist_ok=True)

    df = _load_cen_mean_tables(base_folder, bin_size=bin_size)
    if df.empty:
        print(f"[centromere_mean] no matching files found for bin_size={bin_size}.")
        return

    # Create descriptive tag for output files
    tag_parts = ["centromere_mean"]
    if bin_size is not None:
        tag_parts.append(f"bin_{bin_size}")
    if absolute_distance:
        tag_parts.append("absolute")
    tag = "_".join(tag_parts)

    if mode == "All":
        _combine_cen_mean_curves(df, group_by=[], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Chromosomes":
        _combine_cen_mean_curves(df, group_by=["chrom"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Strains":
        _combine_cen_mean_curves(df, group_by=["strain"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    elif mode == "Datasets":
        _combine_cen_mean_curves(df, group_by=["dataset"], out_dir=out_dir, tag=tag, plot=plot, bin_size=bin_size, absolute_distance=absolute_distance)
    else:
        raise ValueError("mode must be one of: 'All', 'Chromosomes', 'Strains', 'Datasets'")


def parse_arguments():
    """Parse command line arguments for generating nucleosome and centromere bias plots."""
    parser = argparse.ArgumentParser(
        description="Generate nucleosome and centromere insertion-bias tables and plots."
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="Data/combined_strains",
        help="Folder containing distance-annotated CSV files. Use Data/combined_strains for strain-level plots or Data/distances_with_zeros_new for separate replicate/dataset plots."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data_exploration/results/densities",
        help="Base folder where bias tables and plots will be saved."
    )

    parser.add_argument(
        "--target",
        choices=["nucleosome", "centromere", "both"],
        default="both",
        help="Which bias analysis to run."
    )

    parser.add_argument(
        "--step",
        choices=["generate", "combine", "all"],
        default="all",
        help="Run raw density generation, combine generated density curves, or both."
    )

    parser.add_argument(
        "--metric",
        choices=["density", "mean", "both"],
        default="density",
        help="Compute insertion-rate densities, mean insertion counts, or both."
    )

    parser.add_argument(
        "--boolean",
        action="store_true",
        help="Convert insertion counts to presence/absence before calculating densities."
    )

    parser.add_argument(
        "--bin_size",
        type=int,
        default=10000,
        help="Centromere-distance bin size in base pairs."
    )

    parser.add_argument(
        "--combine_mode",
        choices=["All", "Chromosomes", "Strains", "Datasets"],
        default="All",
        help="How to group generated density curves when creating combined plots."
    )

    parser.add_argument(
        "--plot_combined",
        action="store_true",
        help="Create PNG plots for the combined bias curves."
    )

    parser.add_argument(
        "--absolute_centromere_distance",
        action="store_true",
        help="Combine centromere curves using absolute distance from the centromere."
    )

    parser.add_argument(
        "--min_nucleosome_distance",
        type=int,
        default=0,
        help="Minimum nucleosome distance included in combined nucleosome plots."
    )

    parser.add_argument(
        "--max_nucleosome_distance",
        type=int,
        default=458,
        help="Maximum nucleosome distance included in combined nucleosome plots."
    )

    return parser.parse_args()


def main():
    args = parse_arguments()

    density_nucleosome_output = os.path.join(args.output_dir, "density", "nucleosome")
    density_centromere_output = os.path.join(args.output_dir, "density", "centromere")
    mean_nucleosome_output = os.path.join(args.output_dir, "mean", "nucleosome")
    mean_centromere_output = os.path.join(args.output_dir, "mean", "centromere")

    run_nucleosome = args.target in ["nucleosome", "both"]
    run_centromere = args.target in ["centromere", "both"]
    run_generate = args.step in ["generate", "all"]
    run_combine = args.step in ["combine", "all"]
    run_density = args.metric in ["density", "both"]
    run_mean = args.metric in ["mean", "both"]

    if run_generate and run_density and run_nucleosome:
        density_from_nucleosome(
            input_folder=args.input_dir,
            output_folder=density_nucleosome_output,
            boolean=args.boolean
        )

    if run_generate and run_density and run_centromere:
        density_from_centromere(
            input_folder=args.input_dir,
            output_folder=density_centromere_output,
            bin=args.bin_size,
            boolean=args.boolean
        )

    if run_generate and run_mean and run_nucleosome:
        mean_from_nucleosome(
            input_folder=args.input_dir,
            output_folder=mean_nucleosome_output
        )

    if run_generate and run_mean and run_centromere:
        mean_from_centromere(
            input_folder=args.input_dir,
            output_folder=mean_centromere_output,
            bin=args.bin_size
        )

    if run_combine and run_density and run_nucleosome:
        combine_nucleosome_data(
            data=args.combine_mode,
            boolean=args.boolean,
            plot=args.plot_combined,
            base_folder=density_nucleosome_output,
            min_distance=args.min_nucleosome_distance,
            max_distance=args.max_nucleosome_distance
        )

    if run_combine and run_density and run_centromere:
        combine_centromere_data(
            mode=args.combine_mode,
            boolean=args.boolean,
            bin_size=args.bin_size,
            plot=args.plot_combined,
            absolute_distance=args.absolute_centromere_distance,
            base_folder=density_centromere_output
        )

    if run_combine and run_mean and run_nucleosome:
        combine_nucleosome_mean_data(
            data=args.combine_mode,
            plot=args.plot_combined,
            base_folder=mean_nucleosome_output,
            min_distance=args.min_nucleosome_distance,
            max_distance=args.max_nucleosome_distance
        )

    if run_combine and run_mean and run_centromere:
        combine_centromere_mean_data(
            mode=args.combine_mode,
            bin_size=args.bin_size,
            plot=args.plot_combined,
            absolute_distance=args.absolute_centromere_distance,
            base_folder=mean_centromere_output
        )


if __name__ == "__main__":
    main()
