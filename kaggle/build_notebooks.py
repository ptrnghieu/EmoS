"""Builds the Kaggle notebooks in this folder from shared cells, so they stay in sync.

    python kaggle/build_notebooks.py

Writes kaggle/benchmark_qwen25_omni.ipynb (speed test), kaggle/run_qwen25_omni.ipynb (full run) and
kaggle/rq3_lora_{hard,soft}.ipynb (RQ3 fine-tuning, one notebook per arm).
Edit the cells here, not in the .ipynb files.
"""

import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def md(src):
    return {"cell_type": "markdown", "metadata": {}, "source": src.strip("\n")}


def code(src):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n")}


def write(path, cells):
    cells = [dict(c) for c in cells]  # shared cells are reused across notebooks
    nb = {"cells": cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 5}
    for i, c in enumerate(nb["cells"]):
        c["id"] = f"cell-{i:02d}"
        c["source"] = c["source"].splitlines(keepends=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
        f.write("\n")
    print("wrote", path.relative_to(HERE.parent), len(cells), "cells")


# ---- shared cells ------------------------------------------------------------------------

INSTALL = code(r"""
# Versions these notebooks were written against (transformers 5.x API: `dtype=`, thinker class).
%pip install -q "transformers==5.20.1" "qwen-omni-utils==0.0.9" audioread librosa
""")

COMMON_CONFIG = r"""
import glob
import json
import os
import re
import shutil
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen2.5-Omni-7B")  # env override: local tests with a tiny model
LOAD_IN_4BIT = False         # True: 4-bit NF4 weights (needs bitsandbytes), if fp16 does not fit
MAX_MEMORY = {0: "12GiB", 1: "14GiB"}  # leave headroom on GPU 0 for activations
FRAME_MAX_PIXELS = 448 * 252           # per-frame pixel cap passed to the Qwen video processor
DRY_RUN = os.environ.get("DRY_RUN") == "1"  # build inputs only, no model (for local checks)
SEED = 0

# Folder made by analysis/preprocess_chsims.py; found automatically under /kaggle/input.
DATA_DIR = os.environ.get("CHSIMS_MM_DIR")
if DATA_DIR is None:
    hits = glob.glob("/kaggle/input/**/manifest.csv", recursive=True)
    assert hits, "No manifest.csv under /kaggle/input: attach the preprocessed dataset."
    DATA_DIR = os.path.dirname(hits[0])
DATA_DIR = Path(DATA_DIR)
"""

COMMON_REPORT = r"""
OUT_DIR.mkdir(parents=True, exist_ok=True)
print("data:", DATA_DIR, "| out:", OUT_DIR, "| dry run:", DRY_RUN)
print("free disk on / (GB):", round(shutil.disk_usage("/").free / 1e9, 1))
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i}", torch.cuda.get_device_name(i), round(torch.cuda.get_device_properties(i).total_memory / 2**30, 1), "GiB")
"""

LOAD_MODEL = code(r"""
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
model = None
if not DRY_RUN:
    kwargs = dict(dtype=torch.float16, device_map="auto", max_memory=MAX_MEMORY, attn_implementation="sdpa")
    if LOAD_IN_4BIT:
        from transformers import BitsAndBytesConfig
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.float16,
            llm_int8_skip_modules=["audio_tower", "visual", "lm_head"])
    t0 = time.time()
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs).eval()
    print(f"loaded in {time.time() - t0:.0f} s")
    for i in range(torch.cuda.device_count()):
        print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
""")

PROMPTS = code(r"""
LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
SYSTEM = ("You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of "
          "perceiving auditory and visual inputs, as well as generating text and speech.")
DESCRIBE = {
    "TAV": "the video frames, the audio track and the transcript",
    "T": "only the transcript",
    "A": "only the audio track",
    "V": "only the video frames, without sound",
}
CONDITIONS = list(DESCRIBE)
ASK = {
    # one label; scored from the logits of its first token
    "label": f"Which emotion does the speaker express? Choose exactly one of: {', '.join(LABELS)}.\nAnswer with the label only.",
    # verbalized distribution
    "distribution": ("Imagine 100 people watched this clip. Estimate how many of them would choose each emotion "
                     f"from: {', '.join(LABELS)}. Reply with JSON only, mapping each label to a count; the counts sum to 100."),
    # RQ3: judge each modality on its own, then aggregate
    "per_modality": (
        "Judge each modality on its own before deciding:\n"
        "- text: what the words alone suggest;\n"
        "- audio: what the tone of voice alone suggests;\n"
        "- video: what the face and body alone suggest.\n"
        "Then say whether the modalities conflict, and estimate how 100 people who watched the whole clip "
        "would split across the emotions.\n"
        f"Use only these labels: {', '.join(LABELS)}. Reply with JSON only, in this form:\n"
        '{"text": {"cue": "<a few words>", "label": "<label>"}, "audio": {"cue": "...", "label": "..."}, '
        '"video": {"cue": "...", "label": "..."}, "conflict": true or false, "distribution": {"<label>": <count>, ...}}'),
}

label_first_ids = [processor.tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_first_ids)) == len(LABELS), "two labels share a first token"
# The checkpoint's generation_config sets no end-of-turn token, so without this the model keeps
# writing ("Human: ...") until max_new_tokens.
STOP_IDS = processor.tokenizer.convert_tokens_to_ids(["<|im_end|>", "<|endoftext|>"])


def build_messages(row, condition, ask="label"):
    content = []
    if "V" in condition:
        frames = sorted(glob.glob(str(DATA_DIR / row["frames_dir"] / "*.jpg")))
        content.append({"type": "video", "video": [f"file://{f}" for f in frames],
                        "sample_fps": float(row["sample_fps"]), "max_pixels": FRAME_MAX_PIXELS})
    if "A" in condition:
        content.append({"type": "audio", "audio": str(DATA_DIR / row["audio_path"])})
    text = f"You are given {DESCRIBE[condition]} of a speaker in a scene from a Chinese film or TV series.\n"
    if "T" in condition:
        text += f"Transcript: {row['text']}\n"
    content.append({"type": "text", "text": text + ASK[ask]})
    return [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": content}]


def prepare(messages):
    text = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    audios, images, videos, video_kwargs = process_mm_info(messages, use_audio_in_video=False, return_video_kwargs=True)
    # qwen-omni-utils returns fps as a per-video list; the transformers 5.x processor wants one number.
    fps = video_kwargs.pop("fps", [])
    if fps:
        video_kwargs["fps"] = float(fps[0])
    inputs = processor(text=text, audio=audios, images=images, videos=videos, return_tensors="pt",
                       padding=True, use_audio_in_video=False, **video_kwargs)
    if model is not None:
        inputs = {k: (v.to(model.device, dtype=model.dtype) if torch.is_floating_point(v) else v.to(model.device))
                  if torch.is_tensor(v) else v for k, v in inputs.items()}
    return inputs


def parse_json(text):
    match = re.search(r"\{.*\}", text, flags=re.S)
    try:
        return json.loads(match.group(0)) if match else None
    except (json.JSONDecodeError, ValueError):
        return None
""")


# ---- benchmark notebook ------------------------------------------------------------------

BENCHMARK = [
    md(r"""
# Speed test: Qwen2.5-Omni-7B on CH-SIMS v2 (Kaggle, 2× T4)

Measures how long Qwen2.5-Omni-7B takes per clip under the four input conditions of the
proposal (T+A+V, T only, A only, V only), how much GPU memory it uses, and extrapolates the
GPU-hours for the full plan. It runs on 50 clips from the test split, half with cross-modal
conflict and half without.

**Before running**
1. *Settings → Accelerator*: **GPU T4 x2**. *Settings → Internet*: **on** (pip install + model download, about 22.4 GB).
2. *Add input*: your **private** dataset made by `analysis/preprocess_chsims.py` (it holds `manifest.csv`, `frames/`, `audio/`).
3. Run all. Results land in `/kaggle/working/`.

Only the "thinker" (the text-generating part) of Qwen2.5-Omni is loaded; the speech parts are not needed.
T4 has no bf16 and no FlashAttention-2, so the model runs in fp16 with `sdpa` attention.
"""),
    INSTALL,
    code(COMMON_CONFIG + r"""
N_CLIPS = 50                 # clips in the speed test (half conflict, half not)
SPLIT = "test"               # CH-SIMS v2 split to sample from
N_GEN_CLIPS = 10             # clips for the generation (verbalized distribution) timing
N_SAMPLE_CLIPS = 5           # clips for the sampling timing (10 samples each)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working"))
""" + COMMON_REPORT),
    md(r"""
## Pick the clips

Only clips whose frames and audio were both extracted (`status == "ok"`).
"""),
    code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
usable = manifest[manifest["status"] == "ok"]
pool = usable[usable["mode"] == SPLIT]
if len(pool) < N_CLIPS:
    print(f"only {len(pool)} usable '{SPLIT}' clips; sampling from all splits")
    pool = usable
per_group = N_CLIPS // 2
clips = pd.concat([
    g.sample(min(len(g), per_group), random_state=SEED) for _, g in pool.groupby("conflict")
]).reset_index(drop=True)
print(len(clips), "clips | conflict:", clips["conflict"].value_counts().to_dict(),
      "| frames per clip (median):", clips["n_frames"].median())
"""),
    md("## Load the processor and the thinker model"),
    LOAD_MODEL,
    md(r"""
## Prompts and label scoring

Each condition shows the model a different subset of the clip. The answer is scored from the
logits of the first answer token: the 7 labels start with 7 different tokens
(Ang / Dis / Fear / Joy / Neutral / Sad / Sur), so one forward pass gives a distribution over the labels.
`label_mass` is the share of the full next-token probability that falls on those 7 tokens;
if it is low, the model wants to answer with something else and the prompt needs work.
"""),
    PROMPTS,
    code(r"""
@torch.inference_mode()
def score(row, condition):
    t0 = time.perf_counter()
    inputs = prepare(build_messages(row, condition))
    t1 = time.perf_counter()
    out = {"uid": row["uid"], "condition": condition, "n_tokens": int(inputs["input_ids"].shape[1]),
           "prep_s": t1 - t0}
    if model is None:
        return out
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    torch.cuda.synchronize()
    out["gpu_s"] = time.perf_counter() - t1
    out["finite"] = bool(torch.isfinite(logits).all())
    probs = torch.softmax(logits, dim=-1)[label_first_ids].cpu().numpy()
    out["label_mass"] = float(probs.sum())
    probs = probs / max(probs.sum(), 1e-12)
    out.update({f"p_{l}": float(p) for l, p in zip(LABELS, probs)})
    out["pred"] = LABELS[int(np.argmax(probs))]
    return out
"""),
    md(r"""
## Run the speed test

Two warm-up clips first (not timed), then every clip under every condition. A clip that runs
out of memory is recorded with `error = "oom"` and the run carries on.
"""),
    code(r"""
for _, row in clips.head(2).iterrows():
    score(row, "TAV")

records = []
for i, row in clips.iterrows():
    for condition in CONDITIONS:
        if model is not None:
            for d in range(torch.cuda.device_count()):
                torch.cuda.reset_peak_memory_stats(d)
        try:
            rec = score(row, condition)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec = {"uid": row["uid"], "condition": condition, "error": "oom"}
        if model is not None:
            rec["peak_gib"] = max(torch.cuda.max_memory_allocated(d) for d in range(torch.cuda.device_count())) / 2**30
        rec.update(conflict=row["conflict"], ds_label=row["ds_label"], n_frames=row["n_frames"], duration=row["duration"])
        records.append(rec)
    if (i + 1) % 10 == 0:
        print(f"{i + 1}/{len(clips)} clips")

results = pd.DataFrame(records)
results.to_csv(OUT_DIR / "benchmark_scoring.csv", index=False)
results.head()
"""),
    md(r"""
## Generation timing

Two kinds of generation the proposal needs: a verbalized label distribution (JSON, up to 96 new
tokens) and 10 sampled answers per clip at temperature 0.7. For one-word answers, sampling is
close to drawing from the label distribution scored above, so the sampled run may turn out unnecessary.
"""),
    code(r"""
@torch.inference_mode()
def timed_generate(row, ask, **gen_kwargs):
    inputs = prepare(build_messages(row, "TAV", ask=ask))
    t0 = time.perf_counter()
    out = model.generate(**inputs, eos_token_id=STOP_IDS, pad_token_id=STOP_IDS[1], **gen_kwargs)
    torch.cuda.synchronize()
    new = out[0, inputs["input_ids"].shape[1]:]
    return time.perf_counter() - t0, processor.tokenizer.decode(new, skip_special_tokens=True)


gen_records = []
if model is not None:
    for _, row in clips.head(N_GEN_CLIPS).iterrows():
        seconds, text = timed_generate(row, "distribution", max_new_tokens=96, do_sample=False)
        gen_records.append({"uid": row["uid"], "kind": "verbalized", "gen_s": seconds,
                            "parsed": parse_json(text) is not None, "output": text})
    for _, row in clips.head(N_SAMPLE_CLIPS).iterrows():
        for _ in range(10):
            seconds, text = timed_generate(row, "label", max_new_tokens=8, do_sample=True, temperature=0.7)
            gen_records.append({"uid": row["uid"], "kind": "sample", "gen_s": seconds, "output": text})
gen = pd.DataFrame(gen_records)
gen.to_csv(OUT_DIR / "benchmark_generation.csv", index=False)
gen.head()
"""),
    md(r"""
## Summary and GPU-hour estimate for the full plan

Plan sizes from the proposal: label scoring for all 4,403 clips × 4 conditions; verbalized
distribution and 10 samples for the 1,036 test clips (T+A+V only). `prep_s` runs on CPU but
still holds the GPU session, so it is counted.
"""),
    code(r"""
N_ALL, N_TEST = 4403, 1036
ok = results[results.get("error").isna()] if "error" in results else results
summary = ok.groupby("condition").agg(
    clips=("uid", "size"), tokens_median=("n_tokens", "median"), prep_s=("prep_s", "mean"),
    **({"gpu_s": ("gpu_s", "mean"), "peak_gib": ("peak_gib", "max"), "label_mass": ("label_mass", "mean"),
        "nonfinite": ("finite", lambda x: int((~x.astype(bool)).sum()))} if model is not None else {}),
).round(3)
print(summary.to_string())
if "error" in results:
    print("OOM:", int((results["error"] == "oom").sum()), "of", len(results))

if model is not None:
    per_clip = (summary["prep_s"] + summary["gpu_s"]).sum()  # all 4 conditions for one clip
    hours = {"scoring, 4,403 clips x 4 conditions": N_ALL * per_clip / 3600}
    if len(gen):
        tav_prep = summary.loc["TAV", "prep_s"]
        verbal = gen[gen["kind"] == "verbalized"]["gen_s"].mean() + tav_prep
        sample = gen[gen["kind"] == "sample"]["gen_s"].mean() * 10 + tav_prep
        hours["verbalized distribution, 1,036 test clips"] = N_TEST * verbal / 3600
        hours["10 samples, 1,036 test clips"] = N_TEST * sample / 3600
        print("verbalized outputs parsed as JSON:", f"{gen[gen['kind'] == 'verbalized']['parsed'].mean():.0%}")
    hours["total"] = sum(hours.values())
    print(pd.Series(hours).round(1).to_string())
    pd.Series(hours, name="gpu_hours").to_csv(OUT_DIR / "benchmark_hours.csv")
    ok_tav = ok[ok["condition"] == "TAV"]
    print("sanity, T+A+V accuracy vs ds_label on these clips:", f"{(ok_tav['pred'] == ok_tav['ds_label']).mean():.0%}")
    summary.to_csv(OUT_DIR / "benchmark_summary.csv")
"""),
]


# ---- full-run notebook -------------------------------------------------------------------

RUN = [
    md(r"""
# Full run: Qwen2.5-Omni-7B on CH-SIMS v2 (Kaggle, 2× T4)

Runs everything the proposal needs from this model, in one notebook:

| Task | Clips | What is saved |
| --- | --- | --- |
| `score` | all 4,403 × 4 conditions (T+A+V, T, A, V) | logits of the 7 labels' first tokens + log-sum-exp over the vocabulary |
| `verbalized` | test split (1,036), T+A+V | the model's JSON distribution over the labels (RQ2) |
| `per_modality` | test split (1,036), T+A+V | per-modality labels, conflict flag and distribution (RQ3 prompt) |

Estimated total from the speed test: about 7 GPU-hours (scoring about 3 h; generation is an
estimate, see the proposal). Results are appended line by line to `/kaggle/working/omni_runs/*.jsonl`.

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. Use **Save Version → Save & Run All (Commit)** so it runs in the background without the browser open.
3. It stops by itself after `TIME_BUDGET_H` hours. **To continue**, add this notebook's own output
   (*Add input → Your Work → this notebook*) as an input and commit again: finished items are found
   under `/kaggle/input` and skipped, and the new output holds old + new records.
4. For a quick check first, set `LIMIT = 10`.
"""),
    INSTALL,
    code(COMMON_CONFIG + r"""
TASKS = ["score", "verbalized", "per_modality"]
GEN_SPLIT = "test"                     # split used for the generation tasks
MAX_NEW_TOKENS = {"verbalized": 96, "per_modality": 320}
TIME_BUDGET_H = 11.0                   # stop starting new items after this many hours
LIMIT = None                           # e.g. 10 for a quick check; None = all clips
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/omni_runs"))
T_START = time.time()
""" + COMMON_REPORT),
    md("## Clips"),
    code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
if LIMIT:
    clips = pd.concat([clips[clips["mode"] == GEN_SPLIT].head(LIMIT), clips[clips["mode"] != GEN_SPLIT].head(LIMIT)])
gen_clips = clips[clips["mode"] == GEN_SPLIT]
print(len(clips), "clips to score |", len(gen_clips), f"'{GEN_SPLIT}' clips for generation")
"""),
    md("## Load the processor and the thinker model"),
    LOAD_MODEL,
    md("## Prompts"),
    PROMPTS,
    md(r"""
## Resume

Every record is one JSON line keyed by (model, uid, condition, task). Records from earlier runs,
found anywhere under `/kaggle/input` or already in `OUT_DIR`, are copied into this run's output
and skipped below.
"""),
    code(r"""
FILES = {"score": OUT_DIR / "scores.jsonl", "gen": OUT_DIR / "generations.jsonl"}


def key(rec):
    return (rec["model"], rec["uid"], rec["condition"], rec["task"])


done = set()
for kind, path in FILES.items():
    sources = sorted(set(glob.glob(f"/kaggle/input/**/omni_runs/{path.name}", recursive=True)) | ({str(path)} if path.exists() else set()))
    records = {}
    for src in sources:
        with open(src, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    rec = json.loads(line)
                    if "error" not in rec:
                        records[key(rec)] = rec
    with open(path, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done |= set(records)
    print(f"{path.name}: {len(records)} finished records from {len(sources)} file(s)")


def save(kind, rec):
    with open(FILES[kind], "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def out_of_time():
    return (time.time() - T_START) / 3600 > TIME_BUDGET_H
"""),
    md(r"""
## Run

Order: all label scoring first, then the two generation tasks. A clip that fails (for example out of
memory) is saved with an `error` field and retried on the next run.
"""),
    code(r"""
@torch.inference_mode()
def run_score(row, condition):
    t0 = time.perf_counter()
    inputs = prepare(build_messages(row, condition, ask="label"))
    t1 = time.perf_counter()
    rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": "score",
           "n_tokens": int(inputs["input_ids"].shape[1]), "prep_s": round(t1 - t0, 3)}
    if model is None:
        return rec
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    rec["gpu_s"] = round(time.perf_counter() - t1, 3)
    rec["finite"] = bool(torch.isfinite(logits).all())
    rec["lse"] = float(torch.logsumexp(logits, dim=-1))  # p(label) = exp(logit - lse)
    rec.update({f"logit_{l}": float(v) for l, v in zip(LABELS, logits[label_first_ids].tolist())})
    return rec


@torch.inference_mode()
def run_generate(row, task):
    ask = "distribution" if task == "verbalized" else task
    inputs = prepare(build_messages(row, "TAV", ask=ask))
    t0 = time.perf_counter()
    rec = {"model": MODEL_ID, "uid": row["uid"], "condition": "TAV", "task": task}
    if model is None:
        return rec
    out = model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS[task], do_sample=False,
                         eos_token_id=STOP_IDS, pad_token_id=STOP_IDS[1])
    new = out[0, inputs["input_ids"].shape[1]:]
    rec.update(gen_s=round(time.perf_counter() - t0, 3), n_new_tokens=int(len(new)),
               output=processor.tokenizer.decode(new, skip_special_tokens=True))
    rec["parsed"] = parse_json(rec["output"]) is not None
    return rec


jobs = []
if "score" in TASKS:
    jobs += [("score", row, c) for _, row in clips.iterrows() for c in CONDITIONS]
for task in ("verbalized", "per_modality"):
    if task in TASKS:
        jobs += [(task, row, "TAV") for _, row in gen_clips.iterrows()]
todo = [j for j in jobs if (MODEL_ID, j[1]["uid"], j[2], j[0]) not in done]
print(f"{len(jobs)} items, {len(jobs) - len(todo)} already done, {len(todo)} to run")

if DRY_RUN:  # check that inputs build for a few items of every task, then stop
    for task in TASKS:
        for job in [j for j in todo if j[0] == task][:2]:
            rec = run_score(job[1], job[2]) if task == "score" else run_generate(job[1], task)
            print(task, job[2], rec.get("n_tokens", "inputs ok"))
    todo = []

n_done, stopped_early = 0, False
for task, row, condition in todo:
    if out_of_time():
        stopped_early = True
        break
    try:
        rec = run_score(row, condition) if task == "score" else run_generate(row, task)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": task, "error": "oom"}
    save("score" if task == "score" else "gen", rec)
    n_done += 1
    if n_done % 200 == 0:
        print(f"{n_done}/{len(todo)} | {(time.time() - T_START) / 3600:.2f} h elapsed | last task: {task}")

print(f"finished {n_done} items this run", "(stopped at the time budget: commit again to continue)" if stopped_early else "")
"""),
    md("## Progress and quick checks"),
    code(r"""
def read(path):
    return pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]) if path.exists() else pd.DataFrame()


scores, gens = read(FILES["score"]), read(FILES["gen"])
expected = {"score": len(clips) * len(CONDITIONS), "verbalized": len(gen_clips), "per_modality": len(gen_clips)}
for task, n in expected.items():
    df = scores if task == "score" else gens
    got = 0 if df.empty else int(((df["task"] == task) & df.get("error", pd.Series(index=df.index, dtype=object)).isna()).sum())
    print(f"{task:13s} {got:6d} / {n} done")

if not scores.empty and "lse" in scores:
    s = scores.dropna(subset=["lse"])
    mass = np.exp(s[[f"logit_{l}" for l in LABELS]].sub(s["lse"], axis=0)).sum(axis=1)
    print("score: non-finite rows:", int((~s["finite"].astype(bool)).sum()), "| mean label mass by condition:",
          mass.groupby(s["condition"]).mean().round(3).to_dict())
if not gens.empty and "parsed" in gens:
    g = gens.dropna(subset=["parsed"])
    print("generation: JSON parse rate by task:", g.groupby("task")["parsed"].mean().round(3).to_dict(),
          "| mean seconds:", g.groupby("task")["gen_s"].mean().round(2).to_dict())
"""),
]


# ---- RQ3 fine-tuning notebooks (one per arm) ----------------------------------------------

ARM_TEXT = {
    "hard": "**hard labels**: cross-entropy with the dataset's single label `ds_label` (the usual way to fine-tune)",
    "soft": ("**soft labels**: cross-entropy with the 3 annotators' vote distribution, e.g. 2× Joy + 1× Neutral → "
             "Joy 2/3, Neutral 1/3 (same minimum as KL divergence to that distribution)"),
}


def rq3_cells(arm):
    return [
        md(rf"""
# RQ3: LoRA fine-tuning of Qwen2.5-Omni-7B with {arm} labels (Kaggle, 2× T4)

One of the two fine-tuned arms of RQ3. This notebook trains with {ARM_TEXT[arm]}.
The other arm is the same notebook with `ARM = "{'soft' if arm == 'hard' else 'hard'}"`.

**What it does**
1. Adds small LoRA adapters (rank 16) to the attention layers of the language model; the audio and
   vision encoders and all original weights stay frozen.
2. Trains on the CH-SIMS v2 `train` split (2,722 clips, T+A+V input) for 2 epochs. The loss uses the
   same 7 first-token logits that the zero-shot run scored, so the outputs are directly comparable.
3. After each epoch, checks the `valid` split (647 clips) and keeps the epoch with the lowest
   validation loss of this arm.
4. With the best adapter, scores the `test` split under all 4 conditions (T+A+V, T, A, V) and the
   `valid` split under T+A+V (for temperature scaling), in the same format as `omni_runs/scores.jsonl`.

Estimate from the zero-shot timings: about 3–4 GPU-hours. The first progress lines print the
measured speed and a projection.

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. Optional quick check: set `LIMIT = 10` and run all interactively (a few minutes after the model download).
   Do **not** attach that run's output later; the full run ignores it anyway because the clip counts differ.
3. **Save Version → Save & Run All (Commit)**.
4. If it stops at `TIME_BUDGET_H`, add this notebook's own output as an input (*Add input → Your Work*)
   and commit again: training resumes from the last checkpoint, scoring skips finished clips.
5. Results: `/kaggle/working/rq3_{arm}/` with `scores.jsonl`, `state.json` (training history) and the best adapter.
"""),
        code(r"""
# peft added for LoRA; the rest as in the other notebooks.
%pip install -q "transformers==5.20.1" "qwen-omni-utils==0.0.9" "peft==0.21.2" audioread librosa
"""),
        code(COMMON_CONFIG + rf"""
ARM = "{arm}"                  # "hard": ds_label; "soft": distribution of the 3 annotators
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
# Attention projections of the language model only: the audio tower also has q/k/v_proj layers.
LORA_TARGETS = r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
LR = 1e-4
EPOCHS = 2
GRAD_ACCUM = 8               # clips per optimizer step (one clip per forward pass)
WARMUP = 0.03                # share of optimizer steps with a rising learning rate
CKPT_EVERY = 200             # save a checkpoint every this many training clips (multiple of GRAD_ACCUM)
TRAIN_MAX_MEMORY = {{0: "10GiB", 1: "14GiB"}}
TIME_BUDGET_H = 11.0         # stop starting new work after this many hours (Kaggle stops at 12)
LIMIT = None                 # e.g. 10 for a quick check: 10 clips per split
TAG = f"{{MODEL_ID}}+lora-{{ARM}}"  # the `model` field of the saved records
OUT_DIR = Path(os.environ.get("OUT_DIR", f"/kaggle/working/rq3_{{ARM}}"))
T_START = time.time()
assert ARM in ("hard", "soft") and CKPT_EVERY % GRAD_ACCUM == 0
""" + COMMON_REPORT),
        md(r"""
## Load the model and add LoRA adapters

Same thinker model as the zero-shot run, in fp16. The adapters are kept in fp32 and the loss is
scaled (`GradScaler`) so small fp16 gradients do not round to zero. Gradient checkpointing trades
compute for memory. The language-model head only computes the last position, the one the label is read from.
"""),
        code(r"""
from peft import LoraConfig, get_peft_model, get_peft_model_state_dict, set_peft_model_state_dict
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ON_GPU = torch.cuda.is_available()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
kwargs = dict(dtype=torch.float16 if ON_GPU else torch.float32, attn_implementation="sdpa")
if ON_GPU:
    kwargs.update(device_map="auto", max_memory=TRAIN_MAX_MEMORY)
t0 = time.time()
base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs)
print(f"loaded in {time.time() - t0:.0f} s")
base.config.use_cache = False
base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
base.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:],) + args[1:])

torch.manual_seed(SEED)
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                        target_modules=LORA_TARGETS, bias="none"))
trainable = [p for p in model.parameters() if p.requires_grad]
n_lora = sum(name.endswith("lora_A.default") for name, _ in model.named_modules())
print(f"LoRA on {n_lora} layers | trainable params: {sum(p.numel() for p in trainable) / 1e6:.1f} M |",
      "adapter dtype:", trainable[0].dtype)
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
"""),
        md("## Prompts (same as the zero-shot run)"),
        PROMPTS,
        md(r"""
## Clips and targets

`target` is what the loss compares against: the 3 annotators' votes as a distribution over the 7
labels (soft arm), or a one-hot vector on `ds_label` (hard arm). `human` is always the vote
distribution, used for the validation metrics.
"""),
        code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
splits = {m: clips[clips["mode"] == m].reset_index(drop=True) for m in ("train", "valid", "test")}
if LIMIT:
    splits = {m: df.head(LIMIT) for m, df in splits.items()}


def targets(df):
    votes = df[["ann1", "ann2", "ann3"]].to_numpy()
    human = np.stack([(votes == l).sum(axis=1) for l in LABELS], axis=1) / votes.shape[1]
    hard = np.eye(len(LABELS))[[LABELS.index(l) for l in df["ds_label"]]]
    return human, (human if ARM == "soft" else hard)


for m, df in splits.items():
    human, target = targets(df)
    df["human"], df["target"] = list(human), list(target)
    print(f"{m:5s} {len(df):5d} clips | majority labels: {df['ds_label'].value_counts().to_dict()}")
train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
"""),
        md(r"""
## Loss and validation

`label_logits` runs the T+A+V prompt and returns the logits of the 7 labels' first tokens. The loss is
the cross-entropy between `target` and the softmax over those 7 logits. Validation also reports the
Jensen–Shannon divergence to the annotators' distribution (lower is closer), accuracy against
`ds_label`, and the model's mean entropy.
"""),
        code(r"""
def label_logits(row, condition="TAV"):
    inputs = prepare(build_messages(row, condition, ask="label"))
    out = model(**inputs, use_cache=False)
    return out.logits[0, -1, label_first_ids].float(), inputs


def loss_fn(logits, row):
    target = torch.tensor(row["target"], dtype=torch.float32, device=logits.device)
    return -(target * torch.log_softmax(logits, dim=-1)).sum()


def jsd(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.where(a > 0, a * np.log(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None)), 0).sum(axis=-1)
    return (kl(p, m) + kl(q, m)) / 2


@torch.inference_mode()
def validate(df):
    model.eval()
    probs, losses = [], []
    for _, row in df.iterrows():
        logits, _ = label_logits(row)
        losses.append(float(loss_fn(logits, row)))
        probs.append(torch.softmax(logits, dim=-1).cpu().numpy())
    model.train()
    probs, human = np.array(probs), np.stack(df["human"])
    return {"val_loss": float(np.mean(losses)), "val_jsd": float(jsd(probs, human).mean()),
            "val_acc": float((np.array(LABELS)[probs.argmax(1)] == df["ds_label"].to_numpy()).mean()),
            "val_entropy": float(-(probs * np.log(np.clip(probs, 1e-12, None))).sum(1).mean())}
"""),
        md(r"""
## Checkpoints and resume

Everything needed to continue sits in `OUT_DIR`: the latest adapter and optimizer state (`last/`),
the best adapter so far (`best/`), and `state.json` (position in the data, training history).
On start, the most recent `rq3_<arm>` folder found under `/kaggle/input` is copied in, as long as it
was made with the same clip counts (so a `LIMIT` test run is never resumed by the full run).
"""),
        code(r"""
optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=0.0)
steps_per_epoch = -(-len(train_df) // GRAD_ACCUM)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = max(1, int(WARMUP * total_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min((s + 1) / warmup_steps, max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps))))
scaler = torch.amp.GradScaler("cuda", enabled=ON_GPU)
SIZES = {m: len(df) for m, df in splits.items()}
state = {"arm": ARM, "sizes": SIZES, "epoch": 0, "pos": 0, "opt_step": 0, "loss_sum": 0.0, "loss_n": 0,
         "oom": 0, "history": [], "best_epoch": None, "best_val_loss": None, "train_done": False}


def read_state(folder):
    path = Path(folder) / "state.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_checkpoint():
    tmp = OUT_DIR / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    torch.save(get_peft_model_state_dict(model), tmp / "adapter.pt")
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict()}, tmp / "optim.pt")
    shutil.rmtree(OUT_DIR / "last", ignore_errors=True)
    tmp.rename(OUT_DIR / "last")
    state["saved_at"] = time.time()
    (OUT_DIR / "state.json").write_text(json.dumps(state, indent=1))


def out_of_time():
    return (time.time() - T_START) / 3600 > TIME_BUDGET_H


if read_state(OUT_DIR) is None:
    found = [(s, Path(p).parent) for p in glob.glob(f"/kaggle/input/**/rq3_{ARM}/state.json", recursive=True)
             if (s := read_state(Path(p).parent))]
    usable = [(s, d) for s, d in found if s.get("sizes") == SIZES]
    if len(usable) < len(found):
        print(f"ignoring {len(found) - len(usable)} earlier run(s) made with different clip counts")
    if usable:
        src = max(usable, key=lambda x: x[0].get("saved_at", 0))[1]
        shutil.copytree(src, OUT_DIR, dirs_exist_ok=True)
        print("copied earlier run from", src)

saved = read_state(OUT_DIR)
if saved and saved.get("sizes") == SIZES:
    state.update(saved)
    if (OUT_DIR / "last" / "adapter.pt").exists():
        set_peft_model_state_dict(model, torch.load(OUT_DIR / "last" / "adapter.pt", map_location="cpu"))
        opt = torch.load(OUT_DIR / "last" / "optim.pt", map_location="cpu")
        optimizer.load_state_dict(opt["optimizer"])
        scheduler.load_state_dict(opt["scheduler"])
        scaler.load_state_dict(opt["scaler"])
    print(f"resuming: epoch {state['epoch']}, clip {state['pos']}/{len(train_df)}, optimizer step {state['opt_step']},",
          "training done" if state["train_done"] else "")
else:
    print(f"fresh start: {total_steps} optimizer steps ({EPOCHS} epochs x {steps_per_epoch})")
"""),
        md(r"""
## Train

One clip per forward pass; gradients are summed over `GRAD_ACCUM` clips before each optimizer step.
Each epoch sees the training clips in a fixed shuffled order (seeded by the epoch), so a resumed run
continues exactly where the last one stopped. A clip that runs out of GPU memory is skipped and counted.
"""),
        code(r"""
def train():
    model.train()
    t_run, n_run = time.time(), 0
    while state["epoch"] < EPOCHS:
        epoch = state["epoch"]
        order = np.random.default_rng(SEED + epoch).permutation(len(train_df))
        while state["pos"] < len(order):
            if out_of_time():
                save_checkpoint()
                return False
            batch = order[state["pos"]:state["pos"] + GRAD_ACCUM]
            for i in batch:
                row = train_df.iloc[i]
                try:
                    logits, _ = label_logits(row)
                    loss = loss_fn(logits, row)
                    scaler.scale(loss / len(batch)).backward()
                    state["loss_sum"] += loss.item()
                    state["loss_n"] += 1
                except torch.cuda.OutOfMemoryError:
                    torch.cuda.empty_cache()
                    state["oom"] += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["pos"] += len(batch)
            state["opt_step"] += 1
            n_run += len(batch)
            if state["pos"] % CKPT_EVERY == 0 or state["pos"] == len(order):
                save_checkpoint()
            if n_run in (48, 200) or state["opt_step"] % 50 == 0:
                sec = (time.time() - t_run) / n_run
                left = (EPOCHS * len(train_df) - epoch * len(train_df) - state["pos"]) * sec / 3600
                peak = max([torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())] or [0])
                print(f"epoch {epoch} clip {state['pos']}/{len(order)} | loss {state['loss_sum'] / max(1, state['loss_n']):.3f}"
                      f" | {sec:.2f} s/clip, about {left:.1f} h of training left | peak {peak:.1f} GiB | OOM {state['oom']}")
        if not any(h["epoch"] == epoch for h in state["history"]):
            t0 = time.time()
            metrics = validate(valid_df)
            entry = {"epoch": epoch, "train_loss": state["loss_sum"] / max(1, state["loss_n"]), **metrics,
                     "val_minutes": round((time.time() - t0) / 60, 1), "session_hours": round((time.time() - T_START) / 3600, 2)}
            state["history"].append(entry)
            print(entry)
            if state["best_val_loss"] is None or metrics["val_loss"] < state["best_val_loss"]:
                state["best_epoch"], state["best_val_loss"] = epoch, metrics["val_loss"]
                (OUT_DIR / "best").mkdir(exist_ok=True)
                torch.save(get_peft_model_state_dict(model), OUT_DIR / "best" / "adapter.pt")
        state.update(epoch=epoch + 1, pos=0, loss_sum=0.0, loss_n=0)
        save_checkpoint()
    state["train_done"] = True
    save_checkpoint()
    return True


finished = state["train_done"] or train()
print("training finished, best epoch:", state["best_epoch"]) if finished else print(
    "stopped at the time budget: add this notebook's output as an input and commit again")
"""),
        md(r"""
## Score the test split with the best adapter

Same record format as the zero-shot `omni_runs/scores.jsonl`, with `model` set to `TAG`, so the
analysis script can compare the arms clip by clip. Valid clips are scored under T+A+V for
temperature scaling.
"""),
        code(r"""
SCORES = OUT_DIR / "scores.jsonl"


def key(rec):
    return (rec["model"], rec["uid"], rec["condition"], rec["task"])


done, n_new = set(), 0
if finished:
    model.eval()
    set_peft_model_state_dict(model, torch.load(OUT_DIR / "best" / "adapter.pt", map_location="cpu"))
    records = {}
    if SCORES.exists():
        for line in open(SCORES, encoding="utf-8"):
            if line.strip() and "error" not in (rec := json.loads(line)):
                records[key(rec)] = rec
    with open(SCORES, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done = set(records)
    jobs = [(row, c, "test") for _, row in test_df.iterrows() for c in CONDITIONS]
    jobs += [(row, "TAV", "valid") for _, row in valid_df.iterrows()]
    todo = [j for j in jobs if (TAG, j[0]["uid"], j[1], "score") not in done]
    print(f"{len(jobs)} items, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for row, condition, split in todo:
        if out_of_time():
            print("stopped at the time budget: add this notebook's output as an input and commit again")
            break
        rec = {"model": TAG, "uid": row["uid"], "condition": condition, "task": "score", "split": split}
        try:
            with torch.inference_mode():
                t0 = time.perf_counter()
                inputs = prepare(build_messages(row, condition, ask="label"))
                t1 = time.perf_counter()
                logits = model(**inputs, use_cache=False).logits[0, -1].float()
            rec.update(n_tokens=int(inputs["input_ids"].shape[1]), prep_s=round(t1 - t0, 3),
                       gpu_s=round(time.perf_counter() - t1, 3), finite=bool(torch.isfinite(logits).all()),
                       lse=float(torch.logsumexp(logits, dim=-1)))
            rec.update({f"logit_{l}": float(v) for l, v in zip(LABELS, logits[label_first_ids].tolist())})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            rec["error"] = "oom"
        with open(SCORES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
    print(f"scored {n_new} items this run")
"""),
        md("## Summary"),
        code(r"""
print(pd.DataFrame(state["history"]).round(4).to_string(index=False) if state["history"] else "no finished epoch yet")
if SCORES.exists():
    s = pd.DataFrame([json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip()])
    s = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()].merge(
        clips[["uid", "ds_label", "conflict"]], on="uid")
    p = np.exp(s[[f"logit_{l}" for l in LABELS]].to_numpy() - s[["lse"]].to_numpy())
    s["label_mass"] = p.sum(1)
    p = p / p.sum(1, keepdims=True)
    s["pred"] = np.array(LABELS)[p.argmax(1)]
    s["entropy"] = -(p * np.log(np.clip(p, 1e-12, None))).sum(1)
    s["correct"] = s["pred"] == s["ds_label"]
    print(s.groupby(["split", "condition"]).agg(n=("uid", "size"), accuracy=("correct", "mean"),
                                                 label_mass=("label_mass", "mean")).round(3).to_string())
    tav = s[(s["split"] == "test") & (s["condition"] == "TAV")]
    print("test T+A+V, mean entropy by conflict:", tav.groupby("conflict")["entropy"].mean().round(3).to_dict(),
          "| share predicted Neutral:", tav.groupby("conflict")["pred"].apply(lambda x: (x == "Neutral").mean()).round(3).to_dict())
"""),
    ]


if __name__ == "__main__":
    write(HERE / "benchmark_qwen25_omni.ipynb", BENCHMARK)
    write(HERE / "run_qwen25_omni.ipynb", RUN)
    for arm in ("hard", "soft"):
        write(HERE / f"rq3_lora_{arm}.ipynb", rq3_cells(arm))
