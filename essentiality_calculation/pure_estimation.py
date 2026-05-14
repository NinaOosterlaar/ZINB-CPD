import argparse
import os
import re
import sys

import numpy as np
import pandas as pd


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
	sys.path.append(PROJECT_ROOT)

from CPD_on_SATAY.ZINB_MLE.EM import em_zinb_step


RESULT_FILENAME_RE = re.compile(
	r"^(?P<dataset_name>.+)_ws(?P<window_size>\d+)_ov(?P<overlap_pct>\d+)_th(?P<threshold>\d+(?:\.\d+)?)\.txt$"
)


def resolve_path(path):
	if os.path.isabs(path):
		return path
	return os.path.join(PROJECT_ROOT, path)


def read_count_data(csv_file):
	df = pd.read_csv(csv_file)
	if "Value" in df.columns:
		return df["Value"].astype(float).to_numpy()
	if len(df.columns) < 2:
		raise ValueError(f"Expected at least two columns in {csv_file}")
	return df.iloc[:, 1].astype(float).to_numpy()


def remove_top_quantile_outliers(data, quantile=0.99, threshold=None):
	"""
	Remove outliers by capping non-zero values above the specified quantile.
	
	Args:
		data: Array of count values
		quantile: Values above this quantile will be capped (default: 0.99 for top 1%)
		threshold: Pre-computed threshold value (if None, computed from non-zero data)
		
	Returns:
		filtered_data: Data with outliers capped at the threshold value
		threshold: The threshold value used
		n_affected: Number of values that were capped
	"""
	data_array = np.asarray(data, dtype=np.float64)
	
	if threshold is None:
		# Compute threshold only on non-zero values
		non_zero_data = data_array[data_array > 0]
		if len(non_zero_data) > 0:
			threshold = np.quantile(non_zero_data, quantile)
		else:
			threshold = 0.0
	
	# Count how many values exceed the threshold (excluding zeros)
	n_affected = np.sum(data_array > threshold)
	
	# Cap values at the threshold (but keep zeros as zeros)
	filtered_data = np.clip(data_array, None, threshold)
	
	return filtered_data, threshold, n_affected


def parse_result_file(result_file):
	"""Parse change points and theta from a CPD result file."""
	change_points = []
	theta_global = None
	in_cp_section = True

	with open(result_file, "r") as f:
		for raw_line in f:
			line = raw_line.strip()
			if not line:
				continue

			if line.startswith("scores:"):
				in_cp_section = False
				continue

			if line.startswith("theta_global:"):
				theta_global = float(line.split(":", 1)[1].strip())
				continue

			if line.startswith("window_size:"):
				continue

			if in_cp_section:
				try:
					cp = int(float(line))
					change_points.append(cp)
				except ValueError:
					# Ignore malformed lines to keep processing robust.
					continue

	if theta_global is None:
		raise ValueError(f"Could not find theta_global in {result_file}")

	return change_points, theta_global


def parse_result_filename(filename):
	match = RESULT_FILENAME_RE.match(filename)
	if match is None:
		return {
			"dataset_name": None,
			"window_size": None,
			"overlap_pct": None,
			"threshold": None,
		}

	return {
		"dataset_name": match.group("dataset_name"),
		"window_size": int(match.group("window_size")),
		"overlap_pct": int(match.group("overlap_pct")),
		"threshold": float(match.group("threshold")),
	}


def find_window_folders(root_folder, window_size=None):
	"""Recursively find all folders starting with 'window' under root_folder.
	
	Args:
		root_folder: Root directory to search
		window_size: Optional window size to filter (e.g., 100 for 'window100')
	"""
	window_folders = []
	for dirpath, dirnames, _ in os.walk(root_folder):
		for dirname in dirnames:
			if dirname.startswith("window"):
				# If window_size is specified, only include matching folders
				if window_size is None or dirname == f"window{window_size}":
					window_folders.append(os.path.join(dirpath, dirname))
	return sorted(window_folders)


def estimate_segment_mu_fixed_theta(segment_data, theta_global, eps=1e-10, tol=1e-6, max_iter=200):
	"""Estimate segment mu and pi with fixed theta using EM-style updates."""
	segment_data = np.asarray(segment_data, dtype=np.float64)
	if len(segment_data) == 0:
		return np.nan, np.nan

	pi = float(np.clip(np.mean(segment_data == 0), eps, 1.0 - eps))
	mu = float(np.clip(np.mean(segment_data) / max(1.0 - pi, eps), eps, None))

	for _ in range(max_iter):
		update = em_zinb_step(segment_data, pi, mu, theta_global, eps=eps)
		pi_new = float(update["pi"])
		mu_new = float(update["mu"])

		if abs(mu_new - mu) <= tol * max(1.0, abs(mu)):
			pi, mu = pi_new, mu_new
			break

		pi, mu = pi_new, mu_new

	return mu, pi


def create_segments_with_mu_estimates(data, change_points, theta_global, eps=1e-10, tol=1e-6, max_iter=200):
	"""
	Create segments and estimate mu/pi for each, without computing z-scores.
	
	"""
	n = len(data)
	valid_cps = sorted({int(cp) for cp in change_points if 0 < int(cp) < n})
	boundaries = [0] + valid_cps + [n]

	segment_rows = []
	for idx in range(len(boundaries) - 1):
		start = boundaries[idx]
		end = boundaries[idx + 1]
		segment = data[start:end]

		mu_est, pi_est = estimate_segment_mu_fixed_theta(
			segment,
			theta_global,
			eps=eps,
			tol=tol,
			max_iter=max_iter,
		)

		segment_rows.append(
			{
				"segment_id": idx + 1,
				"start_index": start,
				"end_index_exclusive": end,
				"length": end - start,
				"raw_mean": float(np.mean(segment)) if len(segment) > 0 else np.nan,
				"mu_estimate": float(mu_est),
				"pi_estimate": float(pi_est),
			}
		)

	return segment_rows


def estimate_segments(data, change_points, theta_global, eps=1e-10, tol=1e-6, max_iter=200):
	"""
	Estimate segments with mu/pi and add z-scores.
	
	Z-scores are computed across all segments in this dataset.
	"""
	segment_rows = create_segments_with_mu_estimates(
		data, change_points, theta_global, eps=eps, tol=tol, max_iter=max_iter
	)

	# Calculate standardized z-scores for mu_estimate using weighted mean and std
	# Apply log transformation: log(mu + 1) to handle skewed distributions
	# z = (log(μ + 1) - log_μ̄_weighted) / σ_log_μ_weighted
	# Weights are based on segment lengths
	mu_values = np.array([row["mu_estimate"] for row in segment_rows])
	lengths = np.array([row["length"] for row in segment_rows])
	
	# Apply log transformation
	log_mu_values = np.log(mu_values + 1)
	# log_mu_values = np.log(np.maximum(mu_values, 1e-10))
	
	# Filter out NaN and inf values for statistics
	valid_mask = np.isfinite(log_mu_values)
	valid_log_mu = log_mu_values[valid_mask]
	valid_lengths = lengths[valid_mask]
	
	if len(valid_log_mu) > 0:
		# Compute weighted mean: Σ(log_mu_i * length_i) / Σ(length_i)
		total_length = np.sum(valid_lengths)
		log_mu_mean = np.sum(valid_log_mu * valid_lengths) / total_length
		
		# Compute weighted standard deviation: sqrt(Σ(length_i * (log_mu_i - log_mu_mean)²) / Σ(length_i))
		if len(valid_log_mu) > 1:
			weighted_variance = np.sum(valid_lengths * (valid_log_mu - log_mu_mean)**2) / total_length
			log_mu_std = np.sqrt(weighted_variance)
		else:
			log_mu_std = 0.0
		
		# Add z-score to each segment (based on log-transformed values)
		for i, row in enumerate(segment_rows):
			if not np.isfinite(log_mu_values[i]) or log_mu_std == 0.0:
				row["mu_z_score"] = np.nan
			else:
				row["mu_z_score"] = (log_mu_values[i] - log_mu_mean) / log_mu_std
	else:
		# All mu values are NaN or inf
		for row in segment_rows:
			row["mu_z_score"] = np.nan

	return segment_rows


def process_dataset(dataset_num, base_data_folder, base_results_folder, output_base_folder, output_subdir, window_size, eps, tol, max_iter, outlier_capping=True):
	dataset_data_file = os.path.join(base_data_folder, str(dataset_num), "SATAY_with_pi.csv")
	
	# Try both naming conventions: "dataset_X" and "X"
	dataset_results_folder = os.path.join(base_results_folder, str(dataset_num))
	if not os.path.isdir(dataset_results_folder):
		dataset_results_folder = os.path.join(base_results_folder, f"dataset_{dataset_num}")

	if not os.path.exists(dataset_data_file):
		print(f"Skipping dataset {dataset_num}: missing data file {dataset_data_file}")
		return 0

	if not os.path.isdir(dataset_results_folder):
		print(f"Skipping dataset {dataset_num}: missing result folder {dataset_results_folder}")
		return 0
		
	data = read_count_data(dataset_data_file)
	
	# Apply outlier capping if requested (default for real SATAY data)
	if outlier_capping:
		data, outlier_threshold, n_outliers = remove_top_quantile_outliers(data, quantile=0.99)
		if n_outliers > 0:
			print(f"  Dataset {dataset_num}: capped {n_outliers} values ({100*n_outliers/len(data):.2f}%) above threshold {outlier_threshold:.1f}")
		else:
			print(f"  Dataset {dataset_num}: no outliers detected (99th percentile = {outlier_threshold:.1f}")
	else:
		print(f"  Dataset {dataset_num}: outlier capping disabled (using raw count data)")
	
	processed_files = 0

	# Recursively find all window folders (handles nested strain/chromosome/region structure)
	# Filter by window_size if specified
	window_folders = find_window_folders(dataset_results_folder, window_size=window_size)

	for window_folder in window_folders:
		result_files = [
			name
			for name in sorted(os.listdir(window_folder))
			if name.endswith(".txt")
		]
		if not result_files:
			continue

		# Mirror the input folder structure in the output folder
		# Replace base_results_folder path with output_base_folder path
		relative_path = os.path.relpath(window_folder, dataset_results_folder)
		output_folder = os.path.join(output_base_folder, f"dataset_{dataset_num}", relative_path, output_subdir)
		os.makedirs(output_folder, exist_ok=True)

		for result_name in result_files:
			result_path = os.path.join(window_folder, result_name)
			change_points, theta_global = parse_result_file(result_path)
			file_meta = parse_result_filename(result_name)

			segment_rows = estimate_segments(
				data,
				change_points,
				theta_global,
				eps=eps,
				tol=tol,
				max_iter=max_iter,
			)

			segment_df = pd.DataFrame(segment_rows)
			segment_df.insert(0, "source_result_file", result_name)
			segment_df.insert(1, "dataset_num", int(dataset_num))
			segment_df.insert(2, "theta_global", float(theta_global))
			segment_df.insert(3, "num_change_points", int(len(change_points)))
			segment_df.insert(4, "dataset_name", file_meta["dataset_name"])
			segment_df.insert(5, "window_size", file_meta["window_size"])
			segment_df.insert(6, "overlap_pct", file_meta["overlap_pct"])
			segment_df.insert(7, "threshold", file_meta["threshold"])

			output_name = result_name.replace(".txt", "_segment_mu.csv")
			output_path = os.path.join(output_folder, output_name)
			segment_df.to_csv(output_path, index=False)
			processed_files += 1

	print(f"Dataset {dataset_num}: wrote segment estimates for {processed_files} threshold files")
	return processed_files


def parse_arguments():
	parser = argparse.ArgumentParser(
		description="Estimate segment-level mu from change point files using fixed global theta."
	)
	parser.add_argument(
		"--base_data_folder",
		type=str,
		default="Data/SATAY_synthetic",
		help="Folder containing SATAY synthetic datasets (1..10).",
	)
	parser.add_argument(
		"--base_results_folder",
		type=str,
		default="results/CPD_segments",
		help="Folder containing CPD outputs (input).",
	)
	parser.add_argument(
		"--output_base_folder",
		type=str,
		default="results/essentiality_score",
		help="Base folder where segment mu outputs will be written.",
	)
	parser.add_argument(
		"--datasets",
		type=int,
		nargs="+",
		default=list(range(1, 11)),
		help="Dataset numbers to process.",
	)
	parser.add_argument(
		"--output_subdir",
		type=str,
		default="segment_mu",
		help="Subfolder created inside each window folder for segment outputs.",
	)
	parser.add_argument(
		"--window_size",
		type=int,
		default=100,
		help="Window size to process (e.g., 100 for 'window100' folders). Set to None to process all.",
	)
	parser.add_argument("--eps", type=float, default=1e-10, help="Numerical epsilon.")
	parser.add_argument("--tol", type=float, default=1e-6, help="Convergence tolerance for mu updates.")
	parser.add_argument("--max_iter", type=int, default=200, help="Maximum EM iterations per segment.")
	parser.add_argument(
		"--outlier_capping",
		action="store_true",
		default=True,
		help="Apply outlier capping (top 1%% quantile) to count data. Default: True for real SATAY data. Use --no-outlier_capping for synthetic data.",
	)
	parser.add_argument(
		"--no-outlier_capping",
		dest="outlier_capping",
		action="store_false",
		help="Disable outlier capping for synthetic data where ground truth is known.",
	)
	return parser.parse_args()


def main():
	args = parse_arguments()
	base_data_folder = resolve_path(args.base_data_folder)
	base_results_folder = resolve_path(args.base_results_folder)
	output_base_folder = resolve_path(args.output_base_folder)

	total_files = 0
	for dataset_num in args.datasets:
		total_files += process_dataset(
			dataset_num=dataset_num,
			base_data_folder=base_data_folder,
			base_results_folder=base_results_folder,
			output_base_folder=output_base_folder,
			output_subdir=args.output_subdir,
			window_size=args.window_size,
			eps=args.eps,
			tol=args.tol,
			max_iter=args.max_iter,
			outlier_capping=args.outlier_capping,
		)

	print(f"Done. Generated segment-mu outputs for {total_files} result files.")


if __name__ == "__main__":
	main()
