# Transfer pilot: diagnostics after the locked analysis (not part of the locked reading)

The locked result (report.md, three folds pooled) stands: hse_tok is worse than hse_const on JSD AV and partial AV,
no difference on JSD V. The two observations below describe the configuration; they do not change that reading.

## 1. hse_tok fold 2: training destabilised right after warm-up

Per-window training CE, recovered from the running averages printed in the Kaggle log (8 clips per step, warm-up
= 18 steps):

| clips (optimizer steps) | CE AV | CE V |
|---|---|---|
| 48-200 (6-25) | 1.09 | 1.31 |
| 200-400 (25-50) | 1.25 | 1.55 |
| 400-800 (50-100) | **1.62** | **1.73** |
| 800-1200 (100-150) | 1.19 | 1.41 |
| 1200-2400 (150-300) | 1.09-1.15 | 1.30-1.34 |
| 2400-4800 (300-600) | 1.14-1.30 | 1.28-1.45 |

Epoch means: fold 2 AV 1.214 / V 1.385; the other five runs AV 0.97-1.01 / V 1.21-1.24 (state.json). The spike
starts at peak learning rate and the loss does not return to the level of the other runs. Logs of folds 0-1 were
not inspected; their epoch means show no comparable rise.

## 2. All folds: the inserted tokens grew about 100x relative to the text embeddings

`analysis/transfer_projector_check.py` (trained projector of hse_tok, held-out clips of its fold):

| | fold 0 | fold 1 | fold 2 |
|---|---|---|---|
| token RMS at initialisation | 0.015 | 0.014 | 0.014 |
| token RMS after training | 1.28 | 1.19 | 1.42 |
| between-clip SD | 1.08 | 0.94 | 1.22 |
| max abs token value | 32 | 42 | 34 |

Text-token embedding RMS: 0.0137. The fixed scale controlled the initial size only; with projector LR 1e-3 and
no constraint on the output, the tokens left the scale of the embeddings in every fold. This is a candidate
explanation for the fold 2 instability and for the large drops under permuted / zero input, not an established cause.
