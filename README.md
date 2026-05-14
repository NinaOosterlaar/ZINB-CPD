# ZINB-CPD

Tools for running zero-inflated negative binomial change point detection
(ZINB-CPD) workflows and related SATAY analysis utilities.

## Clone the repository

Clone the repository from GitHub and move into the project folder:

```bash
git clone https://github.com/NinaOosterlaar/ZINB-CPD.git
cd ZINB-CPD
```

## Install dependencies

This project uses Python. It is recommended to create a virtual environment
before installing the required libraries:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install the Python dependencies from `requirements.txt`:

```bash
pip install -r requirements.txt
```

The required third-party libraries are:

- `numpy`
- `pandas`
- `matplotlib`
- `tqdm`
- `sgd-rest`

## Step 1: compute distances from wiggle files

Start with the `.wig` files in `Data/wiggle_format`. The first processing step
computes the distance from each insertion position to the nearest nucleosome and
centromere.

Run:

```bash
python Utils/reader.py \
  --input_dir Data/wiggle_format \
  --output_dir Data/distances_with_zeros \
  --with_zeros
```

This creates distance-annotated CSV files in `Data/distances_with_zeros`. The
`--with_zeros` option includes positions without insertions as rows with
`Value = 0`, which is needed before combining strains.

## Step 2: combine strain datasets

After the distance files have been created, combine the replicate folders within
each strain into one strain-level dataset.

Run:

```bash
python Utils/combine_data.py \
  --input_dir Data/distances_with_zeros \
  --output_dir Data/combined_strains \
  --method average
```

This creates one folder per strain in `Data/combined_strains`, with one
distance-annotated CSV file per chromosome. The default `average` method averages
the non-zero insertion counts at each position. Use `--method sum` if you want
to sum insertion counts across replicates instead.

## Step 3: compute nucleosome and centromere bias

The ZINB-CPD algorithm uses centromere and nucleosome insertion bias to inform
the zero-inflation parameter.  To compute these biases, run:

```bash
python densities/densities.py \
  --input_dir Data/combined_strains \
  --output_dir results/bias_plots/combined_strains \
  --target both \
  --step All \
  --metric density \
  --boolean \
  --bin_size 10000 \
  --combine_mode Datasets \
  --plot_combined \
  --absolute_centromere_distance
```

For this workflow, `Data/combined_strains` is the input folder. The `--target both` option computes both nucleosome and centromere bias, and `--metric both` saves both the raw density and the smoothed density. The `--absolute_centromere_distance` option uses the absolute distance to the centromere instead of the signed distance, which is more appropriate for this dataset. The resulting density files are saved in `results/bias_plots/combined_strains/<strain>/`. `--combine_mode Datasets` computes the bias separately for each strain, which is recommended for this dataset since the strains have different insertion profiles. 

## Step 4: run ZINB-CPD on strains

After the distance files are available in `Data/combined_strains`, run the
ZINB-CPD algorithm on each strain. The runner first writes the required density
lookup files to `results/densities/<strain>/`, then passes those files to the
chromosome-level CPD jobs.

Run:

```bash
python CPD_on_SATAY/run_ZINB_CPD_SATAY_strains.py \
  --strains_data Data/combined_strains \
  --output_base results/CPD_segments \
  --n_strain_workers 3
```

This creates the CPD segment files in `results/CPD_segments`, with one folder per strain and one subfolder per chromosome, for example:

- `results/CPD_segments/<strain>/<chromosome>/...`


## Step 5: calculate segment essentiality scores

After the CPD segment files have been created, calculate segment-level
essentiality scores. This step estimates segment-level `mu` values with the
fixed `theta_global` from the CPD output, then adds a strain-level
`mu_z_score`.

Run:

```bash
python essentiality_calculation/calculate_strain_essentiality.py \
  --base_data_folder Data/combined_strains \
  --base_results_folder results/CPD_segments \
  --summary_output results/essentiality/strain_essentiality_summary.csv \
  --thresholds 3.0 \
  --workers 1
```

This writes scored segment files into each CPD result folder under
`segment_mu`, for example:

Use `--thresholds` to choose which CPD threshold files to score. The default is
`3.0`. 

## Step 6: merge similar neighboring segments

Finally, merge neighboring scored segments with very similar `mu_z_score`
values.

Run:

```bash
python essentiality_calculation/merge_segments.py \
  --base-dir results/CPD_segments \
  --input-th 3.0 \
  --merge-threshold 0.25
```

This writes merged segment files into `merged_segments` folders next to the
`segment_mu` folders. Use `--input-th` to match the threshold scored in Step 5,
and `--merge-threshold` to control how similar neighboring segments must be
before they are merged.

## Step 7: visualize genes

After segment scores have been merged, visualize selected genes with
`analyze/genes_plot.py`. If no genes are specified, the script uses the
`DEFAULT_GENES` list in `analyze/genes_plot.py`. You can either pass genes on
the command line with `--genes`, or edit `DEFAULT_GENES` in that file to change
the default set.

Run with the default genes:

```bash
python analyze/genes_plot.py \
  --strains  FD yEK19 yEK23 \
  --threshold 3.0 \
  --mu_z_threshold 0.25 \
  --strains_data_path results/CPD_segments \
  --count_data_path Data/combined_strains \
  --output_dir results/genes_overview_plots
```

Run for specific genes:

```bash
python analyze/genes_plot.py \
  --genes CDC28 SEC18 KAR2 \
  --strains FD yEK19 yEK23 \
  --threshold 3.0 \
  --mu_z_threshold 0.25 \
  --strains_data_path results/CPD_segments \
  --count_data_path Data/combined_strains \
  --output_dir results/genes_overview_plots
```

To plot the original, non-merged `segment_mu` segments instead, add
`--segment_source unmerged`. The output filenames end in
`_overview_unmerged.png`, so they can be written to the same output directory as
the merged plots.

```bash
python analyze/genes_plot.py \
  --strains FD yEK19 yEK23 \
  --threshold 3.0 \
  --segment_source unmerged \
  --strains_data_path results/CPD_segments \
  --count_data_path Data/combined_strains \
  --output_dir results/genes_overview_plots
```

The plots use gene annotations from
`Utils/SGD_API/architecture_info/yeast_genes_with_info.json`. That file only
contains genes that were previously retrieved from SGD, so update or regenerate
it when you want to plot genes that are missing or when you need current SGD
annotations. New yeast gene annotation files can be generated with the
`SGD_Genes` class in `Utils/SGD_API/yeast_genes.py`.
 
---


Currently, the results do not completely align with my original results from my other github repository: https://github.com/NinaOosterlaar/Transposon_Truths. I am unsure yet where the differences come from. The general results still seem valid. 
