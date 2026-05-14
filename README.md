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

