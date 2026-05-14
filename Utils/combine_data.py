import numpy as np
import pandas as pd
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))) 
from Utils.reader import read_csv_file_with_distances
import random
import itertools
import shutil
import argparse

# Mapping of replicate names to their strain folders
replicate_to_strain = {
    "FD7": "strain_FD",
    "FD9": "strain_FD",
    "FD11": "strain_FD",
    "FD12": "strain_FD",
    "dnrp1-1": "strain_dnrp",
    "dnrp1-2": "strain_dnrp",
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
}


def get_strain_folder(dataset_name):
    """Determine the strain folder for a dataset."""
    # Check if it's a combined replicate
    if dataset_name in replicate_to_strain:
        return replicate_to_strain[dataset_name]
    
    # Try to infer from dataset name
    if dataset_name.startswith("FD"):
        return "strain_FD"
    elif dataset_name.startswith("dnrp"):
        return "strain_dnrp"
    elif dataset_name.startswith("yEK19"):
        return "strain_yEK19"
    elif dataset_name.startswith("yEK23"):
        return "strain_yEK23"
    elif dataset_name.startswith("yTW001"):
        return "strain_yTW001"
    elif dataset_name.startswith("yWT03"):
        return "strain_yWT03a"
    elif dataset_name.startswith("yWT04"):
        return "strain_yWT04a"
    elif dataset_name.startswith("yLIC") or dataset_name.startswith("ylic"):
        return "strain_ylic137"
    else:
        return "strain_unknown"


def combine_replicates(data, replicate_names, method="average", save=True, output_folder="Data/combined_replicates/"):
    """Combine replicate datasets by averaging or summing their data.
    Assumes replicate datasets have names containing replicate identifiers.
    Every dataset point in the dataset is combined using the specified method.
    
    Saves all datasets in the same folder structure as the original data:
    - Combined replicates (e.g., FD7, FD9) go into their strain folders (strain_FD)
    - Non-replicate datasets are copied as-is into their strain folders
    
    Args:
        data (Dictionary): Dictionary containing chromosome DataFrames for each dataset.
        replicate_names (list): List of replicate names to combine (e.g., ["FD7", "FD9"]).
        method (str): Method to combine replicates, either "average" or "sum".
        save (bool): Whether to save combined data to CSV files.
        output_folder (str): Base output folder path.
    Returns:
        new_data (Dictionary): Dictionary with combined replicate datasets.
    """
    new_data = {}
    datasets_to_remove = []
    
    # Step 1: Combine replicates
    for replicate_name in replicate_names:
        # Find all datasets that match this replicate name
        matching_datasets = [dataset for dataset in data if replicate_name in dataset]
        
        if not matching_datasets:
            print(f"No datasets found for replicate: {replicate_name}")
            continue
        
        print(f"Combining {len(matching_datasets)} datasets for replicate: {replicate_name}")
        combined_regions = {}
        
        for chrom in chromosome_length.keys():
            # Initialize a dictionary to accumulate values by position
            position_data = {}
            
            # Accumulate data from all matching datasets
            for dataset in matching_datasets:
                if chrom not in data[dataset]:
                    continue
                
                df = data[dataset][chrom]
                
                for _, row in df.iterrows():
                    pos = int(row['Position'])
                    value = row['Value']
                    nuc_dist = row['Nucleosome_Distance']
                    cent_dist = row['Centromere_Distance']
                    
                    if pos not in position_data:
                        position_data[pos] = {
                            'values': [],
                            'nucleosome_distance': nuc_dist,
                            'centromere_distance': cent_dist
                        }
                    position_data[pos]['values'].append(value)
            
            # Compute combined values for this chromosome
            combined_data = []
            for pos in sorted(position_data.keys()):
                values = position_data[pos]['values']
                
                if method == "average":
                    # Only consider non-zero values for averaging
                    non_zero_values = [v for v in values if v != 0]
                    
                    if len(non_zero_values) == 0:
                        # All values are zero
                        combined_value = 0
                    elif len(non_zero_values) == 1:
                        # Only one non-zero value, use it directly
                        combined_value = non_zero_values[0]
                    else:
                        # Two or more non-zero values, take the average
                        combined_value = np.mean(non_zero_values)
                elif method == "sum":
                    combined_value = np.sum(values)
                else:
                    raise ValueError(f"Unknown method: {method}")
                
                combined_data.append({
                    'Position': pos,
                    'Value': combined_value,
                    'Nucleosome_Distance': position_data[pos]['nucleosome_distance'],
                    'Centromere_Distance': position_data[pos]['centromere_distance']
                })
            
            # Convert to DataFrame
            if combined_data:
                combined_regions[chrom] = pd.DataFrame(combined_data)
            else:
                combined_regions[chrom] = pd.DataFrame(columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
        
        new_data[replicate_name] = combined_regions
        
        # Mark original replicate datasets for removal
        for dataset in matching_datasets:
            if dataset != replicate_name:
                datasets_to_remove.append(dataset)
    
    # Step 2: Remove original replicate datasets from data
    for dataset in datasets_to_remove:
        del data[dataset]
    
    # Step 3: Add combined data to the data dictionary
    data.update(new_data)
    
    # Step 4: Save all datasets if requested
    if save:
        os.makedirs(output_folder, exist_ok=True)
        
        # Save all datasets
        for dataset in data:
            strain_folder = get_strain_folder(dataset)
            dataset_folder = os.path.join(output_folder, strain_folder, dataset)
            os.makedirs(dataset_folder, exist_ok=True)
            
            for chrom in data[dataset]:
                output_path = os.path.join(dataset_folder, f"{chrom}_distances.csv")
                df = data[dataset][chrom]
                df.to_csv(output_path, index=False)
            
            print(f"Saved data for {dataset} to {dataset_folder}")
    
    return data


def combine_strain_datasets(input_folder, output_folder, method="average"):
    """
    Combine all biological replicates within each strain folder into a single dataset per strain.
    
    For example, in strain_FD folder with subfolders FD7, FD9, FD11, FD12,
    this will combine all of them into a single strain_FD dataset.
    
    Args:
        input_folder (str): Path to combined_replicates folder containing strain subfolders.
        output_folder (str): Path where to save the combined strain datasets.
        method (str): Method to combine datasets, either "average" or "sum".
    
    Returns:
        strain_data (dict): Dictionary with strain names as keys and chromosome DataFrames as values.
    """
    strain_data = {}
    
    # Iterate through each strain folder
    for strain_folder in sorted(os.listdir(input_folder)):
        strain_path = os.path.join(input_folder, strain_folder)
        
        if not os.path.isdir(strain_path) or strain_folder.startswith('.'):
            continue
        
        print(f"\nProcessing strain: {strain_folder}")
        
        # Get all replicate folders within this strain
        replicate_folders = [f for f in sorted(os.listdir(strain_path)) 
                           if os.path.isdir(os.path.join(strain_path, f)) and not f.startswith('.')]
        
        if not replicate_folders:
            print(f"  No replicate folders found in {strain_folder}")
            continue
        
        print(f"  Found {len(replicate_folders)} replicates: {replicate_folders}")
        
        # Initialize combined data for this strain
        combined_strain = {}
        
        # Process each chromosome
        for chrom in chromosome_length.keys():
            replicate_frames = []

            # Collect data from all replicates for this chromosome
            for replicate in replicate_folders:
                csv_file = os.path.join(strain_path, replicate, f"{chrom}_distances.csv")
                
                if not os.path.exists(csv_file):
                    print(f"  Warning: {csv_file} not found, skipping")
                    continue
                
                replicate_frames.append(pd.read_csv(csv_file))

            if replicate_frames:
                combined_df = pd.concat(replicate_frames, ignore_index=True)

                if method == "average":
                    non_zero_values = combined_df["Value"].where(combined_df["Value"] != 0)
                    combined_values = (
                        non_zero_values
                        .groupby(combined_df["Position"])
                        .mean()
                        .fillna(0.0)
                        .rename("Value")
                    )
                elif method == "sum":
                    combined_values = combined_df.groupby("Position")["Value"].sum()
                else:
                    raise ValueError(f"Unknown method: {method}")

                distances = combined_df.groupby("Position", as_index=True)[
                    ["Nucleosome_Distance", "Centromere_Distance"]
                ].first()

                combined_strain[chrom] = (
                    distances
                    .join(combined_values)
                    .reset_index()
                    [["Position", "Value", "Nucleosome_Distance", "Centromere_Distance"]]
                    .sort_values("Position")
                )
            else:
                combined_strain[chrom] = pd.DataFrame(columns=['Position', 'Value', 'Nucleosome_Distance', 'Centromere_Distance'])
        
        # Save this strain's combined data
        strain_data[strain_folder] = combined_strain
        
        # Save to CSV files
        output_strain_folder = os.path.join(output_folder, strain_folder)
        os.makedirs(output_strain_folder, exist_ok=True)
        
        for chrom, df in combined_strain.items():
            output_path = os.path.join(output_strain_folder, f"{chrom}_distances.csv")
            df.to_csv(output_path, index=False)
        
        print(f"  Saved combined {strain_folder} data to {output_strain_folder}")
    
    return strain_data

def parse_arguments():
    """Parse command line arguments for combining processed SATAY datasets."""
    parser = argparse.ArgumentParser(
        description="Combine distance-annotated SATAY replicate folders into strain-level datasets.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--input_dir",
        type=str,
        default="Data/distances_with_zeros",
        help="Folder containing strain folders with distance-annotated replicate datasets."
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="Data/combined_strains",
        help="Folder where final strain-level datasets will be saved."
    )

    parser.add_argument(
        "--method",
        choices=["average", "sum"],
        default="average",
        help="Method used to combine insertion counts."
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()

    combine_strain_datasets(
        input_folder=args.input_dir,
        output_folder=args.output_dir,
        method=args.method
    )
