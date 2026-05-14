import numpy as np
import os, sys
import matplotlib.pyplot as plt
import pandas as pd
import argparse
from concurrent.futures import ProcessPoolExecutor
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from Signal_processing.ZINB_MLE.estimate_ZINB import estimate_zinb
from Signal_processing.ZINB_MLE.EM import em_zinb_step
from Signal_processing.ZINB_MLE.log_likelihoods import zinb_log_likelihood


def load_density_lookup_tables(nucleosome_file, centromere_file):
    """Load the density lookup tables from CSV files.
    
    Returns:
        nucleosome_df: DataFrame with 'distance' and 'mean_density' columns
        centromere_df: DataFrame with 'Bin_Center' and 'mean_density' columns
    """
    nucleosome_df = pd.read_csv(nucleosome_file)
    centromere_df = pd.read_csv(centromere_file)
    return nucleosome_df, centromere_df


def interpolate_density(distance, lookup_df, distance_col, density_col='mean_density'):
    """Interpolate density value for a given distance using linear interpolation.
    
    Args:
        distance: The distance value to interpolate for
        lookup_df: DataFrame containing distance and mean_density columns
        distance_col: Name of the distance column 
        density_col: Name of the density column (default: 'mean_density')
    
    Returns:
        Interpolated mean_density value
    
    Raises:
        ValueError: If distance is outside the range of available data
    """
    distances = lookup_df[distance_col].values
    densities = lookup_df[density_col].values
    
    # Check bounds
    if distance < distances.min() or distance > distances.max():
        raise ValueError(f"Distance {distance} is outside the range [{distances.min()}, {distances.max()}]")
    
    # Use numpy's interp for linear interpolation
    return np.interp(distance, distances, densities)


def extract_change_points_from_scores(scores, window_size, overlap, threshold):
    """Derive change points for a given threshold from precomputed scores."""
    step_size = max(1, int(window_size * (1 - overlap)))
    change_points = []
    last_cp, last_score = -np.inf, 0.0

    for idx, score in enumerate(scores):
        if score > threshold:
            cp = idx * step_size + window_size
            if (cp - last_cp) >= window_size:
                change_points.append(cp)
                last_cp, last_score = cp, score
            elif score > last_score:
                change_points[-1] = cp
                last_cp, last_score = cp, score

    return change_points

def sliding_ZINB_CPD_v3(
    data,
    nucleosome_distances,
    centromere_distances,
    window_size,
    overlap,
    threshold=None,
    eps=1e-10,
    theta_global=None,
    tol=1e-6,
    max_iter=10,
    nucleosome_file="Data_exploration/results/densities/nucleosome_new/combined_All_Boolean_True/ALL_combined_Boolean_True_nucleosome_density.csv",
    centromere_file="Data_exploration/results/densities/centromere_new/combined_All_Boolean_True/ALL_combined_Boolean_True_centromere_density.csv",
    nucleosome_distance_col="distance",
    centromere_distance_col="Bin_Center",
    density_col="mean_density"
):
    data = np.asarray(data, dtype=np.float64)
    step_size = max(1, int(window_size * (1 - overlap)))
    n = len(data)
    max_nucl_distance = np.max(np.array([nucleosome_distances]))
    nucleosome_df, centromere_df = load_density_lookup_tables(nucleosome_file, centromere_file)

    # Create a lookup table for distance to mean density for nucleosomes
    distance_to_density = nucleosome_df.set_index(nucleosome_distance_col)[density_col]
    distance_to_density = distance_to_density.reindex(range(max_nucl_distance + 1), fill_value=0)

    # Create a lookup table for distance to mean density for centromeres
    centromere_distance_to_density = centromere_df.set_index(centromere_distance_col)[density_col]

    if theta_global is None or theta_global <= 0:
        theta_global = initialize_theta_global(data, eps=eps)

    print(theta_global)

    scores = []

    for start in range(0, n - 2 * window_size + 1, step_size):
        w1 = data[start : start + window_size]
        w2 = data[start + window_size : start + 2 * window_size]

        middle0 = start + window_size
        centr_dist_middle = abs(centromere_distances[middle0])  # Use absolute value for signed distances
        if centr_dist_middle > centromere_distance_to_density.index.max():
            centr_dist_middle = centromere_distance_to_density.index.max()

        # Version 3: pi0 from centromere-dependent saturation/density
        s0 = interpolate_density(
            centr_dist_middle,
            centromere_distance_to_density.reset_index(),
            centromere_distance_col,
            density_col
        )

        # Nucleosome-based scaling for pi1 and pi2
        nucl_dist0 = nucleosome_distances[start : start + 2 * window_size]
        nucl_dist1 = nucleosome_distances[start : start + window_size]
        nucl_dist2 = nucleosome_distances[start + window_size : start + 2 * window_size]

        temp0_nucl = distance_to_density.loc[nucl_dist0].mean()
        temp1_nucl = distance_to_density.loc[nucl_dist1].mean()
        temp2_nucl = distance_to_density.loc[nucl_dist2].mean()
        
        # s0 = 1 - pi0

        s1 = np.clip(s0 * (temp1_nucl / max(temp0_nucl, eps)), eps, 1 - eps)
        s2 = np.clip(s0 * (temp2_nucl / max(temp0_nucl, eps)), eps, 1 - eps)

        pi1 = 1 - s1
        pi2 = 1 - s2

        # Alternative model: separate mu per window
        mu1 = np.clip(np.mean(w1) / max(1 - pi1, eps), eps, None)
        mu2 = np.clip(np.mean(w2) / max(1 - pi2, eps), eps, None)

        # Null model: shared mu0 across both windows, but keep pi1 and pi2 fixed
        sum_y = np.sum(w1) + np.sum(w2)
        denom = window_size * (1 - pi1) + window_size * (1 - pi2)
        mu0 = np.clip(sum_y / max(denom, eps), eps, None)

        # Likelihoods
        ll0_w1 = zinb_log_likelihood(w1, mu0, theta_global, pi1, eps=eps)
        ll0_w2 = zinb_log_likelihood(w2, mu0, theta_global, pi2, eps=eps)
        ll0 = ll0_w1 + ll0_w2

        ll1 = zinb_log_likelihood(w1, mu1, theta_global, pi1, eps=eps)
        ll2 = zinb_log_likelihood(w2, mu2, theta_global, pi2, eps=eps)
        ll_alt = ll1 + ll2

        score = 2.0 * (ll_alt - ll0)
        scores.append(score)

    if threshold is None:
        return [], scores

    change_points = extract_change_points_from_scores(scores, window_size, overlap, threshold)

    return change_points, scores
    
def initialize_theta_global(data, eps=1e-10, theta_max=1000):
    results = estimate_zinb(data, eps=eps)
    theta_global = results['theta']
    print(f"Estimated global theta: {theta_global:.4f}")
    print(f"(Estimated global pi: {results['pi']:.4f}, mu: {results['mu']:.4f})")
    if theta_global >= theta_max:
        # Throw an error that the estimation of theta failed
        raise ValueError("Estimated global theta is very large, indicating a failure in estimation. ")
    return theta_global

def save_results(output_folder, dataset_name, change_points, scores, theta_global, window_size, overlap, threshold):  
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    output_file = os.path.join(output_folder, f"{dataset_name}_ws{window_size}_ov{int(overlap*100)}_th{threshold:.2f}.txt")  
    with open(output_file, "w") as f:
        for cp in change_points:
            f.write(f"{cp} \n")
        f.write(f"scores: {scores}\n")
        f.write(f"theta_global: {theta_global}\n")
        f.write(f"window_size: {window_size}, overlap: {overlap}, threshold: {threshold}\n")
    

def process_window_size(ws, data, nucleosome_distances, centromere_distances, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file, centromere_file):
    """Process all thresholds for a given window size."""
    window_output_folder = os.path.join(output_folder, f"window{ws}")

    print(f"Processing window size: {ws} (single sliding pass for all thresholds)")
    _, scores = sliding_ZINB_CPD_v3(
        data,
        nucleosome_distances,
        centromere_distances,
        ws,
        overlap,
        threshold=None,
        theta_global=theta_global,
        nucleosome_file=nucleosome_file,
        centromere_file=centromere_file,
    )

    for threshold in thresholds:
        print(f"Applying threshold {threshold:.2f} on precomputed scores (ws={ws})")
        change_points = extract_change_points_from_scores(scores, ws, overlap, threshold)
        save_results(window_output_folder, dataset_name, change_points, scores, theta_global, ws, overlap, threshold)
    return ws

def parse_arguments():
    parser = argparse.ArgumentParser(description="Apply a sliding window mean change point detection algorithm on discrete count data.")
    parser.add_argument("input_file", type=str, help="Path to the input CSV file containing the count data.")
    parser.add_argument("--output_folder", type=str, default="Signal_processing/results/sliding_mean/sliding_ZINB_CPD", help="Output folder for results.")
    parser.add_argument("--dataset_name", type=str, default="dataset", help="Name of the dataset being processed.")
    parser.add_argument("--n_workers", type=int, default=1, help="Number of parallel workers/CPUs to use.")
    parser.add_argument("--theta_global", type=float, default=0, help="Global theta value to use for all windows (if not provided, it will be estimated from the data).")
    return parser.parse_args()


if __name__ == "__main__":
    # Configuration
    base_data_folder = "Data/SATAY_synthetic"
    base_output_folder = "Signal_processing/results/version4"
    
    window_size = [100]
    overlap = 0.5
    thresholds = np.linspace(0, 15, 31)  # 16 thresholds from 0 to 15
    
    print(f"Thresholds: {thresholds}")
    print(f"Processing datasets 1-10 from {base_data_folder}")
    
    # Process each dataset (1-10)
    for dataset_num in range(1,11):
        print(f"\n{'='*60}")
        print(f"Processing dataset {dataset_num}")
        print(f"{'='*60}")
        
        # Build input file path and density file paths for this dataset
        dataset_folder = os.path.join(base_data_folder, str(dataset_num))
        input_file = os.path.join(dataset_folder, "SATAY_with_pi.csv")
        nucleosome_file = os.path.join(dataset_folder, "density_vs_distance_nucleosome_density.csv")
        centromere_file = os.path.join(dataset_folder, "density_vs_distance_centromere_density.csv")
        
        # Check if files exist
        if not os.path.exists(input_file):
            print(f"Warning: Input file not found: {input_file}")
            continue
        if not os.path.exists(nucleosome_file):
            print(f"Warning: Nucleosome density file not found: {nucleosome_file}")
            continue
        if not os.path.exists(centromere_file):
            print(f"Warning: Centromere density file not found: {centromere_file}")
            continue
        
        # Build output folder path
        output_folder = os.path.join(base_output_folder, str(dataset_num))
        dataset_name = f"dataset_{dataset_num}"
        
        # Create output folder if it doesn't exist
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        
        # Read data from CSV
        with open(input_file, "r") as f:
            lines = f.readlines()[1:]  # Skip header
            data = [int(float(line.strip().split(",")[1])) for line in lines]
            # Column indices: Position(0), Value(1), Centromere_distance(2), Nucleosome_distance(3)
            nucleosome_distance = [int(float(line.strip().split(",")[3])) for line in lines]
            centromere_distance = [int(float(line.strip().split(",")[2])) for line in lines]
        
        print(f"Loaded {len(data)} data points")
        
        # Estimate theta globally for this dataset
        theta_global = initialize_theta_global(data)
        print(f"Using global theta: {theta_global:.4f} for all window sizes and thresholds.")
        
        # Process different window sizes (currently just one: 100)
        n_workers = min(1, len(window_size))
        print(f"Using {n_workers} worker(s) to process {len(window_size)} window size(s) in parallel")
        
        with ProcessPoolExecutor(max_workers=n_workers) as executor:
            futures = [
                executor.submit(process_window_size, ws, data, nucleosome_distance, centromere_distance, overlap, thresholds, theta_global, output_folder, dataset_name, nucleosome_file=nucleosome_file, centromere_file=centromere_file)
                for ws in window_size
            ]
            for future in futures:
                ws_completed = future.result()
                print(f"Completed processing window size: {ws_completed}")
        
        print(f"Finished processing dataset {dataset_num}")
    
    print(f"\n{'='*60}")
    print("All datasets processed!")
    print(f"Results saved to: {base_output_folder}")
    print(f"{'='*60}")
        
