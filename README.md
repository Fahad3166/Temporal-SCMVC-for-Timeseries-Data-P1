# SCMVC Temporal Extensions for Multi-View Time-Series Clustering

This repository contains the P1 implementation and controlled follow-up experiments for adapting **Self-Weighted Contrastive Fusion for Deep Multi-View Clustering (SCMVC)** to multi-view time-series datasets.

The project includes:

- Original SCMVC-style MLP baseline on flattened temporal windows.
- TCN-SCMVC with temporal convolution encoders.
- LSTM-SCMVC with recurrent temporal encoders.
- Loaders for HAR, WISDM, MHEALTH, and PAMAP2.
- Controlled repeated-seed comparison results.

## Setup

Create and activate the conda environment:

```bash
conda create -n scmvc python=3.9
conda activate scmvc
```

Install dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scipy scikit-learn tqdm matplotlib
```

The experiments were run with:

- Python 3.9.25
- PyTorch 2.8.0
- NumPy 2.0.1
- scikit-learn 1.6.1
- Apple MPS backend on MacBook Air

## Dataset Placement

Datasets are not included in the GitHub repository. Download them from the original sources and place them as follows:

```text
HAR:
HAR_dataset/UCI HAR Dataset/

MHEALTH:
data/MHEALTHDATASET/

PAMAP2:
data/pamap2+physical+activity+monitoring/PAMAP2_Dataset/Protocol/

WISDM:
data/wisdm+smartphone+and+smartwatch+activity+and+biometrics+dataset/wisdm-dataset/
```

## Main Controlled Runners

Run all commands from the temporal code folder:

```bash
cd temporal_scmvc
```

### Original SCMVC Baseline on Flattened Windows

This runner uses the original SCMVC MLP encoder/decoder architecture. Each temporal view is flattened before being passed to SCMVC.

```bash
python train_temporal_mlp.py --dataset har --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50 --seed 0 --kmeans_n_init 10 --no_save --results_file ../p2_results/controlled_runs.csv
```

### TCN-SCMVC

```bash
python train_temporal_tcn.py --dataset har --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50 --seed 0 --kmeans_n_init 10 --no_save --results_file ../p2_results/controlled_runs.csv
```

### LSTM-SCMVC

```bash
python train_temporal_lstm.py --dataset har --max_samples 5000 --sample_strategy stratified --feature_dim 128 --high_feature_dim 64 --batch_size 128 --temperature 0.5 --pre_epochs 30 --con_epochs 50 --seed 0 --kmeans_n_init 10 --no_save --results_file ../p2_results/controlled_runs.csv
```

Use `--dataset wisdm`, `--dataset mhealth`, or `--dataset pamap2` for the other datasets. For WISDM, the reported controlled runs used:

```bash
--wisdm_max_files_per_view 20
```

## Controlled Evaluation Results

The controlled comparison used:

- Seeds: `0, 1, 2`
- Maximum samples: `5000`
- Stratified sampling
- Same temporal windows and labels for all three methods
- Same training epochs and clustering evaluation
- K-means `n_init=10`

Full details are in [P2_CONTROLLED_RESULTS.md](P2_CONTROLLED_RESULTS.md).

Summary:

| Dataset | Best ACC Method | ACC mean +/- std |
|---|---|---:|
| HAR | LSTM-SCMVC | 0.7211 +/- 0.0123 |
| WISDM | TCN-SCMVC | 0.3987 +/- 0.0081 |
| MHEALTH | LSTM-SCMVC | 0.5633 +/- 0.0122 |
| PAMAP2 | TCN-SCMVC | 0.4556 +/- 0.0049 |

PAMAP2 is mixed: TCN-SCMVC has the best ACC, but the flattened original SCMVC baseline has better NMI, ARI, and PUR.

## Result Files

```text
p1_results/controlled_summary.csv
p1_results/controlled_runs.csv
p1_results/controlled_runs_seeds_0_1_2.csv
p1_results/smoke.csv
```

## Notes

The project is currently a controlled architectural adaptation rather than a fully novel P2 model. The next step is to build on this reproducible baseline with stronger data audits, saved preprocessing indices, subject-leakage checks, and a more novel temporal fusion component.
