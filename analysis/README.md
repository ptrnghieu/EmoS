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
| `rq1_models.py` | RQ1 models: ordinal regression with a random intercept per video (binary conflict and conflict type), pairwise conflict-type tests with Holm correction, mixed models for the continuous disagreement measures (Basic-7 entropy, GoEmotions entropy, 1 − ds_confidence), robustness checks: `python analysis/rq1_models.py --out results/rq1_models` |
| `bci_analysis.py` | Bayesian causal inference (Koerding et al., 2007): human disagreement by T/A/V valence spread (non-monotonicity test), and a training-free combiner over each model's T+A+V and T/A/V answers, with an oracle that uses the human per-modality labels: `python analysis/bci_analysis.py --out results/bci_analysis` |
| `rq_conflict_definitions_and_errors.py` | Follow-ups to the related-work review: RQ1 under four conflict definitions (ours and Du et al.'s), and how many model errors match at least one annotator, without and with conflict: `python analysis/rq_conflict_definitions_and_errors.py --out results/rq_conflict_definitions_and_errors` |
| `rq2_compare_models.py` | RQ2 across models: Qwen2.5-Omni-7B, MiniCPM-o 2.6, Gemma 4 E4B-it and Phi-4-multimodal (those with results) per condition, with paired bootstrap differences against Qwen and against humans: `python analysis/rq2_compare_models.py --out results/rq2_compare_models` |
| `rq4_analysis.py` | RQ4 causal-inference fusion on the test split, variants A (RQ3 soft) to D, against the criteria fixed before the runs (primary: Spearman of model vs human entropy, D minus A, per seed), plus the gate pi and the per-modality polarity heads: `python analysis/rq4_analysis.py --out results/rq4_analysis` |
| `preprocess_cremad.py` | Download CREMA-D (GitHub LFS media), extract frames and 16 kHz audio, write a manifest with the raters' votes per condition (audio only, video only, audio-visual) and a split by actor; run on Kaggle through `kaggle/prep_cremad.ipynb` |
| `cremad_human.py` | M1 on CREMA-D: perceived conflict (audio-only vs video-only raters) and within-modality ambiguity against the audio-visual raters' disagreement, average and clip level, with a Dirichlet-multinomial ceiling: `python analysis/cremad_human.py --out results/cremad_human` |
| `cremad_ambiguity_diagnostic.py` | Does visual ambiguity explain audio-visual disagreement beyond conflict, emotion class, perceived intensity and actor? Out-of-sample (10-fold CV over actors) and actor fixed-effects models; ambiguity vs low intensity: `python analysis/cremad_ambiguity_diagnostic.py --out results/cremad_ambiguity_diagnostic` |
| `cremad_models.py` | M2 on CREMA-D: zero-shot models against the raters of each condition (accuracy, JSD, clip-level entropy vs human entropy per condition, response to perceived conflict, which modality the AV answer follows): `python analysis/cremad_models.py --out results/cremad_models` |
| `probe_analysis.py` | Stage 1 probe: a softmax probe on the saved hidden states (RQ3 soft adapter and base model) predicting the 3-vote distribution; is it better than the model's output entropy at ranking human disagreement (criterion fixed in the docstring): `python analysis/probe_analysis.py --out results/probe_analysis` |
| `rq3_analysis.py` | RQ3 comparison on the test split: zero-shot, LoRA hard, LoRA soft, each with and without temperature scaling: `python analysis/rq3_analysis.py --out results/rq3_analysis` |
| `../kaggle/run_minicpm_o.ipynb` | RQ2 with a second model: MiniCPM-o 2.6 label scoring, all clips × 4 conditions, same record format; resumable |
| `../kaggle/run_gemma4.ipynb` | RQ2 with a third family: Gemma 4 E4B-it label scoring, same design; records the largest activation per item (fp16 check) |
| `../kaggle/run_phi4mm.ipynb` | RQ2 with a fourth family: Phi-4-multimodal-instruct label scoring, same design; the model's own LoRA switching by input type |
| `../kaggle/rq4_ci_{B,C,D}.ipynb` | RQ4, causal-inference fusion (fine-tuned): four readings per clip (T+A+V, T, A, V), per-modality polarity heads, and B: heads only, C: fixed mixture, D: per-clip common-cause gate; variant A is the RQ3 soft arm |
| `../kaggle/run_emotion_llama.ipynb` | RQ2 with an emotion specialist: Emotion-LLaMA (Llama-2) with the authors' code in its own Python 3.9 environment; six labels (no Disgust, worried read as Fear); runs `../kaggle/score_emotion_llama.py` |
| `../kaggle/build_notebooks.py` | Builds all Kaggle notebooks from shared cells: edit cells there, then run it |

Requirements: Python 3.9+, `pandas`, `openpyxl`, and `ffmpeg`/`ffprobe` on `PATH` for preprocessing; `scipy` and `statsmodels` for `rq1_models.py`.

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
8. **Second model**: `kaggle/run_minicpm_o.ipynb` with the same settings (pins `transformers==4.48.3`, see the notebook).
   Results: `minicpm_runs/scores.jsonl`; copy to `results/minicpm_runs/`, then run `rq_analysis.py --runs results/minicpm_runs --out results/rq_analysis_minicpm` and `rq2_compare_models.py`.
9. **Further families**: `kaggle/run_gemma4.ipynb` (transformers 5.20.1) and `kaggle/run_phi4mm.ipynb` (transformers 4.48.2, peft 0.13.2),
   each first with `LIMIT = 10` (check the fp16 activation table), then the full run. Copy the results to
   `results/gemma4_runs/` and `results/phi4mm_runs/`, run `rq_analysis.py` on each, then `rq2_compare_models.py`.
   `kaggle/run_emotion_llama.ipynb` the same way (results in `emollama_runs/`); its records have no Disgust logit,
   so it appears only in the six-label tables of `rq2_compare_models.py` (Disgust removed from every model and the humans).
10. **RQ4 causal-inference fusion**: `kaggle/rq4_ci_B.ipynb`, `rq4_ci_C.ipynb`, `rq4_ci_D.ipynb`, first with `LIMIT = 10`,
   then the full run (two sessions each; resume as in step 6). Copy each run's last output folder `rq4_<variant>[_seed<N>]/`
   (`scores.jsonl`, `state.json`) to `results/rq4/`, then run `rq4_analysis.py`.
11. **Stage 1 probe**: `kaggle/probe_hidden.ipynb`, with the RQ3 soft notebook's output (seed 0, `rq3_soft/best/adapter.pt`)
   attached as a second input; `LIMIT = 10` first, then the full run (about 2-3 hours). Copy `probe_hidden/` to
   `results/probe_hidden/`, then run `probe_analysis.py`.
12. **CREMA-D** (M1-M3): `kaggle/prep_cremad.ipynb` (CPU) writes `cremad_mm/`. M2: `kaggle/cremad_qwen.ipynb` and
    `kaggle/cremad_minicpm.ipynb` with `cremad_mm/` attached, `LIMIT = 10` first, then the full run; copy
    `cremad_qwen/` and `cremad_minicpm/` to `results/`.
    M3: `kaggle/cremad_lora.ipynb` (LoRA on the audio-visual vote distribution, test actors scored under AV, A, V);
    copy `cremad_lora/` to `results/` and run `cremad_models.py --runs` with it and the zero-shot runs (the clips
    common to all runs are the test actors').
13. **Stage 1** (CREMA-D, per-condition supervision): `kaggle/stage1_cremad.ipynb`, one committed version per
    (`ARM`, `PHASE`, `FOLD`, `SEED`, lambdas); copy each `stage1_*` output folder to `results/stage1/`.

## Note on clip ids

`meta.csv` stores clip ids as strings (`"0001"`, `"00002_0"`); the EmoS workbooks store them as
integers, so `"00002_0"` became `20` and can collide with a real clip 20. `chsims_emos.load_joined`
joins on video, clip number and text to avoid this. One clip (`aqgy4_0009/00026_0`) is stored as
`000260.mp4`; the preprocessing script handles it.
