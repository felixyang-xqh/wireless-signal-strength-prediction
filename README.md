# Wireless Signal Strength Prediction

Deep learning project for wireless signal strength prediction using multimodal feature fusion and regression modeling.

## Overview

This repository contains a PyTorch pipeline for point-level wireless signal strength prediction. The model combines:

- a four-channel scene patch encoder
- a numeric feature branch for transmitter and receiver coordinates
- CBAM attention on image features
- gated multimodal fusion
- a regression head for final signal strength prediction

The original course project also includes large datasets, processed samples, checkpoints, and presentation materials. Those artifacts are intentionally excluded here so the repository stays lightweight and recruiter-friendly.

## Model Highlights

- Input image patch: `64 x 64 x 4`
- Numeric input: normalized transmitter and receiver coordinates
- Image backbone: lightweight ConvNeXt-style CNN
- Attention: CBAM
- Fusion: gated fusion with concat ablation support
- Output: scalar regression target

## Repository Structure

```text
src/
  data/
  models/
  scripts/
  training/
requirements.txt
```

## Setup

```bash
python -m venv .venv
.venv\\Scripts\\activate
pip install -r requirements.txt
```

## Expected Data Layout

The repository does not include the RadioMapSeer dataset or processed training samples. The training scripts expect data arranged like this:

```text
data/
  processed/
    radiomapseer_point_residual_v6_400maps_multichannel/
      images/
      metadata/
        samples.csv
```

## Main Entry Points

```bash
python src/scripts/prepare_radiomapseer_point_dataset.py --radiomapseer-root <raw_dataset_dir> --output-root data/processed/radiomapseer_point_residual_v6_400maps_multichannel
python src/scripts/train_radiomapseer_regressor.py
python src/scripts/evaluate_radiomapseer_checkpoint.py --dataset-root <processed_dataset_dir> --checkpoint <checkpoint_path>
```

## Notes

- Large datasets and checkpoints are excluded from version control.
- Default script paths were normalized to English directory names for easier reuse.
- This repository focuses on code structure and modeling logic rather than bundled experiment artifacts.
