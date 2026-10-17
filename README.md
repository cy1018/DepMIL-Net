# DepMIL-Net

Reference implementation of **DepMIL-Net: Prototype-Guided Multiple Instance
Learning for Facial Video-Based Depression Severity Estimation**.

The public code contains the motion-aware Stage-I instance encoder and the
Stage-II severity-relevance calibrated ProtoDual aggregator. Dataset files,
labels, participant-level predictions, extracted features, and trained
checkpoints are not redistributed because access is governed by the respective
dataset providers.

## Environment

The implementation was tested with Python 3.10, PyTorch 2.8.0 and CUDA 12.6.

```bash
pip install -r requirements.txt
```

## Included implementation

- `models/depmil_net.py`: motion-aware representation learning, spatial
  attention pooling, multi-scale temporal aggregation, cumulative ordinal
  supervision, Transformer relation modelling, and ProtoDual aggregation.
- `datasets/depmil_dataset.py`: AVEC-style frame-folder and extracted-feature
  bag adapters.
- `run_depmil.py`: data validation, Stage-I training, feature extraction,
  Stage-II training, evaluation, and a synthetic model self-test.

## Data layout

Prepare the officially obtained and preprocessed facial frames locally:

```text
DATA_ROOT/
  label_dict.json
  Training/
    *_aligned/
      *.jpg
  Development/
    *_aligned/
      *.jpg
  Testing/
    *_aligned/
      *.jpg
```

No dataset content is included in this repository.

## Quick check

The synthetic check does not require dataset files:

```bash
python run_depmil.py self-test --gpu -1
```

Validate a locally prepared frame root:

```bash
python run_depmil.py data-check --data_root /path/to/data
```

Display the available training and evaluation options with:

```bash
python run_depmil.py --help
```

## Scope

This repository provides a reference implementation of the proposed method.
Paths, dataset access, and compute settings must be configured locally. The
repository does not include licensed datasets or experiment artifacts.
