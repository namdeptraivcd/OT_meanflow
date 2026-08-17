# OT_meanflow

MNIST sanity checks for comparing original MeanFlow against OT-Geodesic MeanFlow variants.

## Methods

- `meanflow`: original MeanFlow objective, random noise-data pairing, JVP target.
- `ot_gmf_hard`: hard balanced mini-batch OT assignment, direct displacement target, no JVP.
- `ot_gmf_sinkhorn_hard`: low-entropy Sinkhorn plan, hard balanced rounding, direct displacement target, no JVP.

All three use the same small two-time U-Net: `model(z_t, r, t) -> u`.

## Kaggle Quick Run

```bash
bash scripts/run_all_quick.sh
```

This trains the MNIST classifier, then runs all three methods for `STEPS=20000`.

For a full run:

```bash
bash scripts/train_classifier.sh
bash scripts/run_meanflow.sh
bash scripts/run_ot_hard.sh
bash scripts/run_ot_sinkhorn.sh
```

The training scripts use:

```bash
torchrun --standalone --nproc_per_node=2
```

so two T4 GPUs are used with distributed data parallel training.

## Resume

Each run writes:

```text
outputs/<method>/checkpoints/latest.pt
outputs/<method>/checkpoints/step_XXXXXXX.pt
outputs/<method>/logs.csv
outputs/<method>/eval.csv
outputs/<method>/eval/*.png
```

To resume:

```bash
METHOD=meanflow STEPS=100000 bash scripts/resume_method.sh
METHOD=ot_gmf_hard STEPS=100000 bash scripts/resume_method.sh
METHOD=ot_gmf_sinkhorn_hard STEPS=100000 bash scripts/resume_method.sh
```

## Compare

```bash
bash scripts/compare_results.sh
```

Optional time-to-threshold summary:

```bash
bash scripts/compare_results.sh --fid-threshold 50 --entropy-threshold 0.9
```

Primary metrics:

- generated grids in `outputs/<method>/eval/`
- feature-FID from a small MNIST classifier
- generated classifier confidence
- class entropy
- seconds per step
- peak GPU memory
