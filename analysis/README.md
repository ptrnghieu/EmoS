# Conflict × subjectivity analyses

Code for the project on cross-modal conflict and annotator disagreement, built on the
CH-SIMS v2(s) clips re-annotated in EmoS.

| File | What it does |
| --- | --- |
| `chsims_emos.py` | Joins CH-SIMS v2 `meta.csv` (unimodal labels `label_T/A/V`) with the EmoS Basic-7 annotations, one row per clip, and adds conflict columns |
| `pilot_conflict_disagreement.py` | Pilot: annotator agreement with vs. without cross-modal conflict |
| `preprocess_chsims.py` | Extracts frames and 16 kHz audio from the raw clips and writes `manifest.csv` |
| `../kaggle/benchmark_qwen25_omni.ipynb` | Speed and memory test of Qwen2.5-Omni-7B on Kaggle (2× T4) |

Requirements: Python 3.10+, `pandas`, `openpyxl`, and `ffmpeg`/`ffprobe` on `PATH` for preprocessing.

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

## Note on clip ids

`meta.csv` stores clip ids as strings (`"0001"`, `"00002_0"`); the EmoS workbooks store them as
integers, so `"00002_0"` became `20` and can collide with a real clip 20. `chsims_emos.load_joined`
joins on video, clip number and text to avoid this. One clip (`aqgy4_0009/00026_0`) is stored as
`000260.mp4`; the preprocessing script handles it.
