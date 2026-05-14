import numpy as np
import pandas as pd
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
import re
import argparse
from tqdm import tqdm 
from Utils.SGD_API.yeast_architecture import Nucleosomes, Centromeres

chromosome_mapper = {
    "chrref|NC_001133|": "chrI",
    "chrref|NC_001134|": "chrII",
    "chrref|NC_001135|": "chrIII",
    "chrref|NC_001136|": "chrIV",
    "chrref|NC_001137|": "chrV",
    "chrref|NC_001138|": "chrVI",
    "chrref|NC_001139|": "chrVII",
    "chrref|NC_001140|": "chrVIII",
    "chrref|NC_001141|": "chrIX",
    "chrref|NC_001142|": "chrX",
    "chrref|NC_001143|": "chrXI",
    "chrref|NC_001144|": "chrXII",
    "chrref|NC_001145|": "chrXIII",
    "chrref|NC_001146|": "chrXIV",
    "chrref|NC_001147|": "chrXV",
    "chrref|NC_001148|": "chrXVI",
    "chrref|NC_001224|": "chrM",
}

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

def read_wig(file_path):
    """
    Reads a WIG file and returns a dict of DataFrames, one per chromosome.
    """
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"The file {file_path} does not exist.")
    
    current_chrom = None
    current_data = []
    
    chrom_data = {}

    with open(file_path, 'r') as file:
        for line in file:            
            if line.startswith('track'): continue  # skip metadata lines

            if line.startswith(('VariableStep', 'variableStep', 'fixedStep')):
                if current_chrom and current_data:
                    chrom_data[current_chrom] = pd.DataFrame(
                        current_data, columns=['Position', 'Value']
                    )
                    current_data = []
                
                raw_chrom = line.strip().split(sep="=")[1] # chrref|NC_001134| OR chrI
                current_chrom = chromosome_mapper.get(raw_chrom, raw_chrom).replace("chr", "Chr")
                continue
            
            # Parse data lines
            parts = line.strip().split()
            if len(parts) == 2 and current_chrom:
                position, value = parts
                current_data.append((int(position), int(value)))

    # Save last chromosome
    if current_chrom and current_data:
        chrom_data[current_chrom] = pd.DataFrame(
            current_data, columns=['Position', 'Value']
        )

    return chrom_data


def label_from_filename(fname: str) -> str:
    base = os.path.basename(fname)
    stem = os.path.splitext(base)[0]

    # cut before merged/FDDP tokens
    cut_positions = [p for p in (
        stem.find('_merged'), stem.find('_FDDP'),
        stem.find('-merged'), stem.find('-FDDP')
    ) if p != -1]
    head = stem[:min(cut_positions)] if cut_positions else stem

    # FD{strain}_{rep}
    m = re.search(r'(?i)\bFD(\d+)[-_](\d+)\b', head)
    if m:
        return f"FD{m.group(1)}_{m.group(2)}"

    # yLIC{strain}_{rep}
    m = re.search(r'(?i)\byLIC(\d+)[-_](\d+)\b', head)
    if m:
        return f"yLIC{m.group(1)}_{m.group(2)}"

    # dnrp{strain}-{repnum} ... {a|b}
    m_strain = re.search(r'(?i)\bdnrp(\d+)\b', head)
    if m_strain:
        strain = m_strain.group(1)
        # numeric replicate before _merged
        m_repnum = re.search(r'[._-](\d+)$', head)
        repnum = m_repnum.group(1) if m_repnum else None
        # a/b letter anywhere in full stem
        m_letter = re.search(r'(?i)(?:^|[-_])(a|b)(?:$|[-_])', stem)
        letter = m_letter.group(1).lower() if m_letter else None

        if repnum and letter:
            return f"dnrp{strain}-{repnum}-{letter}"  # UNIQUE
        if letter:
            return f"dnrp{strain}-{letter}"
        if repnum:
            return f"dnrp{strain}_{repnum}"
        return f"dnrp{strain}"

    # fallback: keep head
    return head


def compute_nucleosome_distance_array(nucleosome_obj, chrom):
    """Compute nearest nucleosome-center distance for every position in a chromosome."""
    length = chromosome_length[chrom]
    nucleosome_middles = np.array(nucleosome_obj.get_middles(chrom), dtype=np.int64)

    if nucleosome_middles.size == 0:
        return np.full(length, np.nan)

    nucleosome_middles.sort()
    positions = np.arange(1, length + 1, dtype=np.int64)
    insertion_points = np.searchsorted(nucleosome_middles, positions)

    left = np.full(length, np.iinfo(np.int64).max, dtype=np.int64)
    right = np.full(length, np.iinfo(np.int64).max, dtype=np.int64)

    has_left = insertion_points > 0
    has_right = insertion_points < nucleosome_middles.size
    left[has_left] = positions[has_left] - nucleosome_middles[insertion_points[has_left] - 1]
    right[has_right] = nucleosome_middles[insertion_points[has_right]] - positions[has_right]

    return np.minimum(left, right)


def compute_centromere_distance_array(centromere_obj, chrom):
    """Compute signed distance to centromere middle for every position in a chromosome."""
    length = chromosome_length[chrom]
    middle = centromere_obj.get_middle(chrom)

    if middle is None:
        return np.full(length, np.nan)

    return np.arange(1, length + 1, dtype=np.int64) - middle


def build_distance_dataframe(df, chrom, nucleosome_distances, centromere_distances, with_zeros):
    """Build one distance-annotated chromosome dataframe."""
    if chrom not in chromosome_length:
        raise ValueError(f"Unknown chromosome length for {chrom}")

    length = chromosome_length[chrom]
    observed = df.copy()
    observed["Position"] = observed["Position"].astype(np.int64)
    observed["Value"] = observed["Value"].astype(np.int64)
    observed = observed.groupby("Position", as_index=False)["Value"].sum()

    valid_positions = observed["Position"].between(1, length)
    invalid_count = len(observed) - int(valid_positions.sum())
    if invalid_count:
        print(f"Warning: skipping {invalid_count} out-of-range positions for {chrom}")
        observed = observed.loc[valid_positions]

    if with_zeros:
        positions = np.arange(1, length + 1, dtype=np.int64)
        values = np.zeros(length, dtype=np.int64)
        values[observed["Position"].to_numpy() - 1] = observed["Value"].to_numpy()

        distances_df = pd.DataFrame({
            "Position": positions,
            "Value": values,
            "Nucleosome_Distance": nucleosome_distances[chrom],
            "Centromere_Distance": centromere_distances[chrom],
        })
        return distances_df

    positions = observed["Position"].to_numpy()
    observed["Nucleosome_Distance"] = nucleosome_distances[chrom][positions - 1]
    observed["Centromere_Distance"] = centromere_distances[chrom][positions - 1]
    return observed[["Position", "Value", "Nucleosome_Distance", "Centromere_Distance"]].sort_values("Position")


def compute_distances(input_folder, output_folder, with_zeros = True):
    """For each signal in the SATAY wig file, compute its distance from the nearest nucleosome and centromere.

    Args:
        input_folder (str): Path to the folder containing SATAY wig files (including subfolders).
        output_folder (str): Path to the folder where the output CSV files will be saved.
    """

    # Create output folder if it doesn't exist
    os.makedirs(output_folder, exist_ok=True)

    nucleosome_obj = Nucleosomes()
    centromere_obj = Centromeres()
    nucleosome_distances = {}
    centromere_distances = {}
    
    for root, dirs, files in os.walk(input_folder):
        for wig_file in files:
            if not wig_file.endswith(".wig"): continue
            
            print(f"Processing wig file: {wig_file}")
            
            relative_path = os.path.relpath(root, input_folder)
            
            # Create an output folder structure that mirrors the input structure
            if relative_path == ".": wig_output_folder = os.path.join(output_folder, wig_file.replace(".wig", ""))
            else: wig_output_folder = os.path.join(output_folder, relative_path, wig_file.replace(".wig", ""))

            os.makedirs(wig_output_folder, exist_ok=True)

            # Read the wig file
            wig_file_path = os.path.join(root, wig_file)
            print(f"Processing wig file: {wig_file_path}")
            wig_data = read_wig(wig_file_path)

            for chrom in tqdm(wig_data, total=len(wig_data)):
                df = wig_data[chrom]
                if df.empty:
                    print(f"No data for {chrom} in {wig_file}. Skipping.")
                    continue

                if chrom not in nucleosome_distances:
                    nucleosome_distances[chrom] = compute_nucleosome_distance_array(nucleosome_obj, chrom)
                    centromere_distances[chrom] = compute_centromere_distance_array(centromere_obj, chrom)

                distances_df = build_distance_dataframe(
                    df=df,
                    chrom=chrom,
                    nucleosome_distances=nucleosome_distances,
                    centromere_distances=centromere_distances,
                    with_zeros=with_zeros
                )

                # Save each chromosome to its own file
                output_file = os.path.join(wig_output_folder, f"{chrom}_distances.csv")
                distances_df.to_csv(output_file, index=False)


def read_csv_file_with_distances(input_folder = "Data/distances_with_zeros"):
    """
    Reads CSV files from the input folder and organizes them into a nested dictionary structure.
    Parameters:
    input_folder : str
        Path to the folder containing CSV files.
    Returns:
    datasets : dict
        A dictionary where keys are dataset labels and values are dictionaries of chromosomes with their corresponding DataFrames.
    """
    # Collect all datasets without loading data
    datasets_paths = []
    
    for root, dirs, files in os.walk(input_folder):
        csv_files = [f for f in files if f.endswith(".csv")]
        if csv_files:  # Only process folders that contain CSV files
            path_parts = root.split("/")
            strain_name = path_parts[-2] if len(path_parts) >= 2 else "unknown_strain"
            dataset_name = path_parts[-1]
            datasets_paths.append((strain_name, root, dataset_name))
    
    datasets = {}
    for strain_name, dataset_path, dataset_name in datasets_paths:
        genome = {}
        for file in os.listdir(dataset_path):
            if file.endswith(".csv"):
                # Extract chromosome from filename (before first underscore)
                chrom = file.split("_")[0]
                # Only process valid chromosome files (Chr followed by roman numeral or number)
                if not chrom.startswith("Chr") or chrom == "ChrM":
                    continue
                file_path = os.path.join(dataset_path, file)
                df = pd.read_csv(file_path)
                genome[chrom] = df
                datasets_key = f"{label_from_filename(dataset_name)}"
                datasets[datasets_key] = genome
    return datasets



def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Process SATAY wig files and compute distances to nucleosomes and centromeres."
    )
    
    parser.add_argument(
        "--input_dir", 
        type=str, 
        required=True,
        help="Path to the folder containing SATAY wig files (including subfolders)"
    )
    
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="Data/distances_with_zeros",
        help="Path to the folder where the output CSV files will be saved (default: Data/distances_with_zeros)"
    )
    
    parser.add_argument(
        "--with_zeros", 
        action="store_true",
        help="Include positions with zero values in the output"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    compute_distances(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        with_zeros=args.with_zeros
    )
