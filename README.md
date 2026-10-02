# Visual Localisation Noise Robustness

Systematic analysis of noise injection strategies for robust visual localisation, matching synthetic noise to real-world sensor degradation patterns.

[![DOI](https://img.shields.io/badge/DOI-10.70251%2FHYJR2348.43451460-blue)](https://doi.org/10.70251/HYJR2348.43451460)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-ResNet--18-red)

This repository contains the code and results for the paper:

> **Systematic Analysis of Noise Injection Strategies for Robust Visual Localisation: Matching Synthetic Noise to Real-World Sensor Degradation Patterns**
> Shamsiddinov Nodir. *American Journal of Student Research*, Vol. 4, No. 3, June 2026, pp. 451-460.
> DOI: [10.70251/HYJR2348.43451460](https://doi.org/10.70251/HYJR2348.43451460)

## Overview

Pose-regression models for visual localisation are usually trained on clean images, but real cameras suffer from sensor readout noise, low-light photon noise and motion blur. This project trains six ResNet-18 pose regressors, one baseline and five noise-augmented variants, and evaluates each under clean and corrupted test images to see which training noise gives the best robustness.

**Training strategies compared**

| Model | Noise injected during training |
|---|---|
| Baseline | None |
| Gaussian | Additive Gaussian noise, sigma = 10 |
| Poisson | Poisson shot noise, scale = 75 |
| Motion Blur | Linear motion blur, 5 x 5 kernel |
| Spatial | Spatially correlated Gaussian noise, sigma = 10, correlation length = 2 |
| Mixed | 5 x 5 motion blur followed by Poisson noise, scale = 200 |

All noise is applied on the fly to training batches. Test-time corruptions are applied to the held-out test images only.

## Key findings

Median translation error (metres), lower is better. Bold is the best value in each column. Full tables, including rotation error, are in the paper.

| Method | Clean | G-Low | G-Med | M-Blur | Corr | Mixed | G-Ext | P-Ext | Avg Noisy |
|---|---|---|---|---|---|---|---|---|---|
| Baseline (none) | **1.083** | 2.751 | 2.788 | 2.145 | 1.136 | 5.538 | 2.969 | 6.780 | 3.444 |
| Gaussian | 1.335 | **1.323** | **1.270** | 2.023 | 1.284 | 4.772 | **1.415** | 4.331 | **2.346** |
| Poisson | 4.002 | 4.159 | 4.154 | 4.122 | 4.001 | **1.715** | 4.128 | 2.526 | 3.544 |
| Motion Blur | 1.910 | 2.797 | 2.831 | **1.209** | 1.959 | 5.245 | 3.054 | 6.270 | 3.338 |
| Spatial | 1.297 | 2.726 | 2.632 | 2.307 | **1.167** | 5.127 | 2.603 | 5.627 | 3.170 |
| Mixed | 4.366 | 4.396 | 4.392 | 4.441 | 4.366 | 2.870 | 4.384 | **1.823** | 3.810 |

*G = Gaussian, Ext = extreme (out-of-distribution), Corr = spatially correlated noise, P = Poisson. Avg Noisy is the unweighted mean over all noisy conditions, including the two extreme conditions.*

Main takeaways from the paper:

- **No single augmentation wins everywhere.** Each strategy is best on the corruption it was trained for.
- **Gaussian augmentation** has the lowest average error across noisy conditions and is robust to extreme Gaussian noise (1.415 m vs 2.969 m for the baseline).
- **Poisson and mixed augmentation** are strongest in the realistic low-light "mixed" condition (1.715 m and 2.870 m vs 5.538 m for the baseline) and under extreme Poisson noise (mixed: 1.823 m vs 6.780 m).
- **Trade-off:** Poisson and Mixed models have much higher error on clean images (about 4.0 to 4.4 m vs 1.08 m for the baseline), so matching training noise to the expected deployment conditions matters.
- Rotation error follows the same trends as translation error.

## Method summary

- **Dataset:** [EuRoC MAV](https://doi.org/10.1177/0278364915620033), `MH_01_easy` (Machine Hall) sequence only. Images from both stereo cameras (cam0 and cam1) are combined and paired with motion-capture ground-truth poses.
- **Preprocessing:** resize to 224 x 224, ImageNet normalisation, random horizontal flip, random rotation (+/- 5 degrees), random resized crop (scale 0.9 to 1.0) and colour jitter.
- **Split:** 70% train, 15% validation, 15% test. The best checkpoint is selected by validation loss.
- **Model:** ImageNet-pretrained ResNet-18 with a frozen backbone and a small regression head (`Linear(512, 64)`, BatchNorm, ReLU, Dropout 0.7, `Linear(64, 6)`) that predicts the 6-DoF pose.
- **Loss:** Conservative Pose Loss, `L = ||t_pred - t_gt||_2 + beta * ||r_pred - r_gt||_2` with `beta = 10`.
- **Optimiser:** AdamW (learning rate 5e-4, weight decay 1e-3), ReduceLROnPlateau scheduler, early stopping (patience 10), up to 30 epochs, batch size 32.
- **Metrics:** median absolute translation error (metres) and median rotation error (degrees).

### Evaluation conditions

| Condition | Type | Parameters |
|---|---|---|
| clean | none | - |
| gaussian_low | Additive Gaussian | sigma = 5 |
| gaussian_medium | Additive Gaussian | sigma = 10 |
| motion_blur_5 | Linear motion blur | 5 x 5 kernel |
| correlated_noise | Spatially correlated Gaussian | sigma = 5, correlation length = 2 |
| mixed | Poisson then Gaussian | Poisson scale = 75, Gaussian sigma = 5 |
| gaussian_extreme (OOD) | Additive Gaussian | sigma = 20 |
| poisson_extreme (OOD) | Poisson shot noise | scale = 250 |

## Repository structure

```
.
├── research.ipynb        # Full pipeline: data loading, noise models, training, evaluation, plots
├── results/              # Evaluation outputs (tables, JSON, plots)
├── figures/              # Figures used in the paper
├── requirements.txt      # Python dependencies
├── CITATION.cff          # Citation metadata
├── LICENSE               # MIT License
└── README.md
```

## Getting started

### 1. Install dependencies

```bash
git clone https://github.com/nshamsiddinov1398-rgb/visual-localisation-noise-robustness.git
cd visual-localisation-noise-robustness
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

A CUDA-capable GPU is recommended. The experiments were run on an NVIDIA RTX 3050 (4 GB).

### 2. Download the data

Download the **Machine Hall 01 (easy)** sequence of the [EuRoC MAV dataset](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets) in ASL format and unzip it. The notebook expects this layout:

```
MH_01_easy/
└── mav0/
    ├── cam0/data/*.png
    ├── cam1/data/*.png
    └── state_groundtruth_estimate0/data.csv
```

The dataset is not included in this repository.

### 3. Run the notebook

Open `research.ipynb`, set `DATASET_PATH` to your local `MH_01_easy/mav0` folder, and run the cells in order:

```bash
jupyter notebook research.ipynb
```

Training saves each model's best checkpoint to `checkpoints/`, and evaluation writes `results.json`, `results_table.txt`, `summary_statistics.txt` and plots (`training_curves.png`, `robustness_comparison.png`, `standard_vs_extreme.png`) to the chosen results directory.

## Notes and limitations

These are important when interpreting or reusing the results.

- **Single sequence.** Only the well-lit, moderate-motion `MH_01_easy` sequence was used (hardware constraints). Results may not generalise to harder sequences, low-light scenes or outdoor environments.
- **Random frame-level split.** Frames from both cameras are shuffled and split randomly. At 20 Hz, neighbouring frames are nearly identical, so test images have close neighbours in the training set. Absolute errors are therefore likely optimistic. The comparison between training strategies is the main result, not the absolute numbers. A temporal split would be a stricter test.
- **Frozen backbone.** Only the small regression head is trained (roughly 33k parameters). Findings may differ for fully fine-tuned or larger models.
- **Rotation representation.** In the notebook, orientation is converted from the ground-truth quaternion to Euler angles (radians) for the loss.
- **Seeds and statistics.** The notebook uses a fixed seed of 42. The paper reports three-seed averages for the baseline and Gaussian models, while the other models use a single seed. No confidence intervals or significance tests are reported, so differences should be read as indicative trends.
- **Hand-picked noise parameters.** Noise strengths were chosen through pilot experiments, with no systematic sensitivity analysis.
- **Synthetic corruptions.** All degradations are simulated. Evaluation on real degraded footage (for example low-light sequences) is left to future work.
- **Extreme conditions.** `gaussian_extreme` and `poisson_extreme` are out-of-distribution stress tests, so rankings on the Avg Noisy column should be read with caution.

## Citation

If you use this work, please cite the paper (see also `CITATION.cff`):

```bibtex
@article{shamsiddinov2026noise,
  title   = {Systematic Analysis of Noise Injection Strategies for Robust Visual Localisation: Matching Synthetic Noise to Real-World Sensor Degradation Patterns},
  author  = {Shamsiddinov, Nodir},
  journal = {American Journal of Student Research},
  volume  = {4},
  number  = {3},
  pages   = {451--460},
  year    = {2026},
  doi     = {10.70251/HYJR2348.43451460}
}
```

## License

The code is released under the [MIT License](LICENSE). The paper is open access under the Creative Commons Attribution (CC BY) licence. The EuRoC dataset has its own terms, so please check the [dataset page](https://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets).

## Contact

Shamsiddinov Nodir, Academic Lyceum of Westminster International University in Tashkent, Uzbekistan.
