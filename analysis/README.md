# Conflict × subjectivity analyses

Code for the project on cross-modal conflict and annotator disagreement, built on the
CH-SIMS v2(s) clips re-annotated in EmoS.

| File | What it does |
| --- | --- |
| `chsims_emos.py` | Joins CH-SIMS v2 `meta.csv` (unimodal labels `label_T/A/V`) with the EmoS Basic-7 annotations, one row per clip, and adds conflict columns |
| `pilot_conflict_disagreement.py` | Pilot: annotator agreement with vs. without cross-modal conflict |
| `preprocess_chsims.py` | Extracts frames and 16 kHz audio from the raw clips and writes `manifest.csv` |
| `../kaggle/benchmark_qwen25_omni.ipynb` | Speed and memory test of Qwen2.5-Omni-7B on Kaggle (2× T4) |
| `../kaggle/run_qwen25_omni.ipynb` | Full run: label logits for all clips × 4 conditions, plus verbalized and per-modality generation on the test split; resumable |
| `../kaggle/rq3_lora_hard.ipynb`, `../kaggle/rq3_lora_soft.ipynb` | RQ3: LoRA fine-tuning on the train split with hard labels (`ds_label`) or soft labels (the 3 annotators' votes), then label logits on the test split; resumable |
| `rq_analysis.py` | RQ1/RQ2 tables from the full run: `python analysis/rq_analysis.py --runs results/omni_runs --out results/rq_analysis` |
| `rq3_analysis.py` | RQ3 comparison on the test split: zero-shot, LoRA hard, LoRA soft, each with and without temperature scaling: `python analysis/rq3_analysis.py --out results/rq3_analysis` |
| `../kaggle/build_notebooks.py` | Builds all Kaggle notebooks from shared cells: edit cells there, then run it |

Requirements: Python 3.9+, `pandas`, `openpyxl`, and `ffmpeg`/`ffprobe` on `PATH` for preprocessing.

## Workflow

1. **Pilot** (labels only, no video): `python analysis/pilot_conflict_disagreement.py`
2. **Get the videos**: download CH-SIMS v2(s) from the official links in
   [thuiar/ch-sims-v2](https://github.com/thuiar/ch-sims-v2). Clips sit at `Raw/<video_id>/<clip_id>.mp4`.
3. **Preprocess** (CPU):
   ```bash
   python analysis/preprocess_chsims.py --raw-dir /path/to/ch-simsv2s/Raw --out-dir /path/to/chsims_mm --limit 20  # check first
   python analysis/preprocess_chsims.py --raw-dir /path/to/ch-simsv2s/Raw --out-dir /path/to/chsims_mm
   ```
   Defaults: 2 fps, at most 16 frames per clip, longest side 448 px. Every clip should end with
   `status = ok`; `no_audio` and `missing_video` rows are kept in the manifest for inspection.
4. **Upload** `/path/to/chsims_mm` to Kaggle as a **private** dataset. CH-SIMS v2 media must not be redistributed.
5. **Speed test**: open `kaggle/benchmark_qwen25_omni.ipynb` on Kaggle, add the dataset as input,
   set *Accelerator = GPU T4 x2* and *Internet = on*, then run all.
6. **Full run**: same settings with `kaggle/run_qwen25_omni.ipynb`, using *Save Version → Save & Run All*.
   It stops after `TIME_BUDGET_H` hours; to continue, add the notebook's own output as an input and
   commit again. Results: `omni_runs/scores.jsonl` and `omni_runs/generations.jsonl`.
7. **RQ3 fine-tuning**: `kaggle/rq3_lora_hard.ipynb` and `kaggle/rq3_lora_soft.ipynb`, each as its own Kaggle
   notebook with the same settings. Each trains for 2 epochs, keeps the epoch with the lowest validation loss,
   then scores the test split (4 conditions) and the valid split (T+A+V). Resume works as in step 6.
   Results: `rq3_<arm>/scores.jsonl` (same format as `omni_runs/scores.jsonl`, `model` = `<model>+lora-<arm>`)
   and `rq3_<arm>/state.json` (training history). Copy both folders to `results/rq3/`, then run `rq3_analysis.py`.

## Note on clip ids

`meta.csv` stores clip ids as strings (`"0001"`, `"00002_0"`); the EmoS workbooks store them as
integers, so `"00002_0"` became `20` and can collide with a real clip 20. `chsims_emos.load_joined`
joins on video, clip number and text to avoid this. One clip (`aqgy4_0009/00026_0`) is stored as
`000260.mp4`; the preprocessing script handles it.
