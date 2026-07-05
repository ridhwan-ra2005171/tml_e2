# Stolen Model Detection

This repository contains the code to reproduce our best leaderboard submission for the Stolen Model Detection task.

## Requirements

Install dependencies:

```bash
pip install torch torchvision pandas requests safetensors numpy
```

## Files Required

Place the following files in the same directory as `task_template.py`:

| File | Description |
|---|---|
| `target_model/weights.safetensors` | Target ResNet-18 weights |
| `target_model/train_main_idx.json` | Indices of the target's training samples in CIFAR-100 |
| `suspect_models/suspect_000.safetensors` … `suspect_359.safetensors` | 360 suspect model checkpoints |

All model files are available from `https://huggingface.co/SprintML/tml26_task2`. The script automatically resolves Git-LFS pointer files in-place, so a shallow clone without `git lfs` installed will still work.

CIFAR-100 is downloaded automatically on first run.

## How to Run

```bash
python task_template.py
```

This will:
1. Resolve any Git-LFS pointer files for the target and suspect models
2. Download CIFAR-100 (if not already present) and build two probe sets:
   - 2000 held-out test images (shared probe)
   - 2000 images from the target's exact training set (`train_main_idx.json`)
3. Load the target model and pre-compute five signals on the probe sets
4. Score all 360 suspect models on the same five signals
5. Rank-normalise scores to `[0, 1]`
6. Save results to `SUBMISSION.csv` and `SUBMISSION_diagnostics.csv`
7. Automatically submit to the leaderboard

## Method

For each suspect we compare it to the target along five complementary signals:

| Signal | What it captures | Weight |
|---|---|---|
| Layer-4 CKA | Deep representation similarity — robust to fine-tuning and pruning | 0.25 |
| Output agreement (1 − JSD) | Softmax-distribution match — catches distilled / extracted models | 0.25 |
| Logit correlation | Class-score structure preservation | 0.15 |
| Prediction agreement | Top-1 fidelity | 0.15 |
| Membership gap | Memorisation of the target's exact training samples | 0.20 |

The membership signal compares each suspect's confidence gap between the target's training samples and held-out test samples. A stolen or distilled model inherits the target's memorisation pattern and shows a similar gap; an independent model does not.

All signals are combined with a fixed weighted sum, then converted to a rank-percentile so the submission is a smooth continuous ranking, appropriate for the `TPR@5%FPR` evaluation metric.

## Expected Runtime

Scoring 360 models takes approximately **2–4 hours** on a single GPU, depending on GPU speed. No reference model training is required.

## Output

Results are saved to `SUBMISSION.csv` with columns `id` and `score`, and automatically submitted to the leaderboard. A `SUBMISSION_diagnostics.csv` with all per-signal values is also written for inspection.
