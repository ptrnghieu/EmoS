"""Builds the Kaggle notebooks in this folder from shared cells, so they stay in sync.

    python kaggle/build_notebooks.py

Writes kaggle/benchmark_qwen25_omni.ipynb (speed test), kaggle/run_qwen25_omni.ipynb (full run),
kaggle/rq3_lora_{hard,soft}.ipynb (RQ3 fine-tuning, one notebook per arm), and the RQ2 label scoring
with further model families: kaggle/run_minicpm_o.ipynb (MiniCPM-o 2.6), kaggle/run_gemma4.ipynb (Gemma 4 E4B-it),
kaggle/run_phi4mm.ipynb (Phi-4-multimodal-instruct), kaggle/run_emotion_llama.ipynb (Emotion-LLaMA,
which runs kaggle/score_emotion_llama.py in its own Python 3.9 environment).
RQ4 (causal-inference fusion, fine-tuned): kaggle/rq4_ci_{B,C,D}.ipynb, one notebook per variant.
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
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "manifest.csv"))]  # no ** : media folders make it slow
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
    sources = sorted(set(h for k in range(6) for h in glob.glob("/kaggle/input/" + "*/" * k + f"omni_runs/{path.name}")) | ({str(path)} if path.exists() else set()))
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

Measured on Kaggle: about 6 GPU-hours to the end of training and validation, about 7 with scoring.
The first progress lines print the measured speed and a projection.

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. Optional quick check: set `LIMIT = 10` and run all interactively (a few minutes after the model download).
   Do **not** attach that run's output later; the full run ignores it anyway because the clip counts differ.
3. **Save Version → Save & Run All (Commit)**.
4. If it stops at `TIME_BUDGET_H`, add this notebook's own output as an input (*Add input → Your Work*)
   and commit again: training resumes from the last checkpoint, scoring skips finished clips.
5. Results: `/kaggle/working/rq3_{arm}/` with `scores.jsonl`, `state.json` (training history) and the best adapter.

**Extra seeds.** `SEED` sets the LoRA initialisation, the order of the training clips and dropout.
The first run used `SEED = 0`. For the extra runs set `SEED = 1`, then `SEED = 2`, each in its own committed
version: results go to `rq3_{arm}_seed1/` and `rq3_{arm}_seed2/`, and their records are tagged
`...+lora-{arm}-seed1` / `-seed2`. To resume one of them, attach the output of the version that ran that seed.
"""),
        code(r"""
# peft added for LoRA; the rest as in the other notebooks.
%pip install -q "transformers==5.20.1" "qwen-omni-utils==0.0.9" "peft==0.21.2" audioread librosa
# Kaggle ships torchao 0.10; peft 0.21 raises an ImportError for torchao < 0.16 even when it is not used.
%pip uninstall -y -q torchao
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
SEED = int(os.environ.get("SEED", 0))  # 0 = the first run; 1 and 2 = the extra seeds, one notebook version each
SUFFIX = f"_seed{{SEED}}" if SEED else ""  # seed 0 keeps the first run's names
RUN_NAME = f"rq3_{{ARM}}{{SUFFIX}}"
TAG = f"{{MODEL_ID}}+lora-{{ARM}}{{SUFFIX.replace('_', '-')}}"  # the `model` field of the saved records
OUT_DIR = Path(os.environ.get("OUT_DIR", f"/kaggle/working/{{RUN_NAME}}"))
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
On start, the most recent folder of this run (`rq3_<arm>`, or `rq3_<arm>_seed<N>` for an extra seed) found
under `/kaggle/input` is copied in, as long as it was made with the same clip counts and seed (so a `LIMIT`
test run is never resumed by the full run, and seeds never mix).
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
state = {"arm": ARM, "seed": SEED, "sizes": SIZES, "epoch": 0, "pos": 0, "opt_step": 0, "loss_sum": 0.0, "loss_n": 0,
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
    found = [(s, Path(p).parent) for p in (h for k in range(6) for h in glob.glob("/kaggle/input/" + "*/" * k + f"{RUN_NAME}/state.json"))
             if (s := read_state(Path(p).parent))]
    usable = [(s, d) for s, d in found if s.get("sizes") == SIZES and s.get("seed", 0) == SEED]
    if len(usable) < len(found):
        print(f"ignoring {len(found) - len(usable)} earlier run(s) made with different clip counts or seed")
    if usable:
        src = max(usable, key=lambda x: x[0].get("saved_at", 0))[1]
        shutil.copytree(src, OUT_DIR, dirs_exist_ok=True)
        print("copied earlier run from", src)

saved = read_state(OUT_DIR)
if saved and saved.get("sizes") == SIZES and saved.get("seed", 0) == SEED:
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
        # seed 0 keeps the first run's order (generator seed = epoch); other seeds use 1000 * SEED + epoch,
        # so no two runs share an epoch's order
        order = np.random.default_rng(1000 * SEED + epoch).permutation(len(train_df))
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


# ---- RQ2 label scoring with further models: shared cells -------------------------------

SCORE_CLIPS = [
    md("## Clips"),
    code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
if LIMIT:
    clips = pd.concat([clips[clips["mode"] == "test"].head(LIMIT), clips[clips["mode"] != "test"].head(LIMIT)])
print(len(clips), "clips to score")
"""),
]


def score_cells(run_dir, run_score):
    """Resume, run and progress cells of a label-scoring notebook; `run_score(row, condition)` is model-specific."""
    return [
        md(r"""
## Resume

Records are keyed by (model, uid, condition, task). Earlier records found under `/kaggle/input` or in
`OUT_DIR` are copied into this run's output and skipped below.
"""),
        code(r"""
SCORES = OUT_DIR / "scores.jsonl"


def key(rec):
    return (rec["model"], rec["uid"], rec["condition"], rec["task"])


sources = sorted(set(h for k in range(6) for h in glob.glob("/kaggle/input/" + "*/" * k + "{run_dir}/scores.jsonl")) | ({str(SCORES)} if SCORES.exists() else set()))
records = {}
for src in sources:
    for line in open(src, encoding="utf-8"):
        if line.strip() and "error" not in (rec := json.loads(line)):
            records[key(rec)] = rec
with open(SCORES, "w", encoding="utf-8") as f:
    for rec in records.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
done = set(records)
print(f"{len(records)} finished records from {len(sources)} file(s)")


def out_of_time():
    return (time.time() - T_START) / 3600 > TIME_BUDGET_H
""".replace("{run_dir}", run_dir)),
        md(r"""
## Run

All clips under T+A+V first, then T, A and V, so a run cut short still has the main condition complete.
A clip that fails (for example out of memory) is saved with an `error` field and retried on the next run.
"""),
        code(run_score.strip("\n") + "\n\n" + r"""
jobs = [(row, c) for c in CONDITIONS for _, row in clips.iterrows()]
todo = [j for j in jobs if (MODEL_ID, j[0]["uid"], j[1], "score") not in done]
print(f"{len(jobs)} items, {len(jobs) - len(todo)} already done, {len(todo)} to run")

n_done, stopped_early, t_run = 0, False, time.time()
for row, condition in todo:
    if out_of_time():
        stopped_early = True
        break
    try:
        rec = run_score(row, condition)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": "score", "error": "oom"}
    with open(SCORES, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_done += 1
    if n_done in (20, 100) or n_done % 500 == 0:
        sec = (time.time() - t_run) / n_done
        peak = max([torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())] or [0])
        print(f"{n_done}/{len(todo)} | {sec:.2f} s/item, about {(len(todo) - n_done) * sec / 3600:.1f} h left | "
              f"peak {peak:.1f} GiB | last: {condition}")

print(f"finished {n_done} items this run", "(stopped at the time budget: commit again to continue)" if stopped_early else "")
"""),
        md("## Progress and quick checks"),
        code(r"""
s = pd.DataFrame([json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip()])
ok = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()] if len(s) else s
print(f"{len(ok)} / {len(clips) * len(CONDITIONS)} items done |", "OOM:", int((s.get("error") == "oom").sum()) if "error" in s else 0)
if len(ok):
    p = np.exp(ok[[f"logit_{l}" for l in LABELS]].to_numpy() - ok[["lse"]].to_numpy())
    ok = ok.assign(label_mass=p.sum(1), pred=np.array(LABELS)[p.argmax(1)]).merge(clips[["uid", "ds_label"]], on="uid")
    print(ok.groupby("condition").agg(n=("uid", "size"), label_mass=("label_mass", "mean"),
                                      accuracy=("pred", lambda x: (x == ok.loc[x.index, "ds_label"]).mean()),
                                      nonfinite=("finite", lambda x: int((~x.astype(bool)).sum())),
                                      gpu_s=("gpu_s", "mean"), prep_s=("prep_s", "mean")).round(3).to_string())
"""),
    ]


# ---- second model: MiniCPM-o 2.6, label scoring only ------------------------------------

MINICPM = [
    md(r"""
# RQ2, second model: MiniCPM-o 2.6 on CH-SIMS v2 (Kaggle, 2× T4)

Repeats the label scoring of the Qwen2.5-Omni-7B run with a second omni model from another family:
all 4,403 clips × 4 conditions (T+A+V, T, A, V), the same prompt and the same record format
(logits of the 7 labels' first tokens + log-sum-exp), written to `/kaggle/working/minicpm_runs/scores.jsonl`.
Only label scoring is run; the generation tasks of the Qwen run are left out.

How the inputs follow the model's own usage (its `chat` method, which this notebook reproduces up to
the first answer token):
- **T+A+V**: MiniCPM-o's omni format, one unit per second of the clip = one frame + that second of audio,
  then the transcript and the question.
- **V**: the same one-frame-per-second images, no audio. **A**: the whole audio. **T**: the transcript.
- System prompt: the model's own omni prompt. Images use one slice each (64 tokens per frame).
- When audio is present the model's chat template adds speech tokens before the answer
  (`<|spk_bos|><|spk|><|spk_eos|><|tts_bos|>`); the label is read from the next text token, as `chat` would generate it.

**Versions**: the model card asks for `transformers==4.44.2`, whose `tokenizers` has no build for the
Python 3.13 that Kaggle runs. This notebook pins `transformers==4.48.3`, checked locally with the model's
code on a small random-weight copy: the scored logits match those of the model's own `generate`.
The speech-output part (TTS) is not loaded. T4 has no bf16, so the model runs in fp16, split over both GPUs.

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. First set `LIMIT = 10` and run all interactively: check the load report, the `label mass` and the
   timing lines, and send them over before the full run.
3. Full run: `LIMIT = None`, **Save Version → Save & Run All (Commit)**. It stops after `TIME_BUDGET_H`;
   to continue, add this notebook's output as an input and commit again.
"""),
    code(r"""
# transformers 4.48.3: newest checked with MiniCPM-o's remote code; 4.44.2 (model card) cannot install on Python 3.13.
%pip install -q "transformers==4.48.3" accelerate librosa soundfile
"""),
    code(r"""
import glob
import json
import math
import os
import shutil
import time
from copy import deepcopy
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "openbmb/MiniCPM-o-2_6")  # env override: local tests with a tiny model
SPLIT_LAYER = 13             # LLM layers 0..12 on GPU 0 (with the vision and audio encoders), the rest on GPU 1
TIME_BUDGET_H = 11.0         # stop starting new items after this many hours
LIMIT = None                 # e.g. 10 for a quick check: 10 test clips + 10 other clips
SR = 16000
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/minicpm_runs"))
T_START = time.time()

DATA_DIR = os.environ.get("CHSIMS_MM_DIR")
if DATA_DIR is None:
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "manifest.csv"))]  # no ** : media folders make it slow
    assert hits, "No manifest.csv under /kaggle/input: attach the preprocessed dataset."
    DATA_DIR = os.path.dirname(hits[0])
DATA_DIR = Path(DATA_DIR)
DRY_RUN = False
""" + COMMON_REPORT),
    *SCORE_CLIPS,
    md(r"""
## Load the model

Without the speech-output part. On GPU the layers are placed by hand: the vision and audio encoders
must sit with the token embeddings, because the model's code writes their outputs into the embedding
tensor directly.
"""),
    code(r"""
from transformers import AutoConfig, AutoModel, AutoTokenizer

ON_GPU = torch.cuda.is_available()
kwargs = dict(trust_remote_code=True, attn_implementation="sdpa", init_vision=True, init_audio=True, init_tts=False,
              torch_dtype=torch.float16 if ON_GPU else torch.float32)
if ON_GPU:
    n_layers = AutoConfig.from_pretrained(MODEL_ID, trust_remote_code=True).num_hidden_layers
    device_map = {m: 0 for m in ("vpm", "resampler", "apm", "audio_avg_pooler", "audio_projection_layer",
                                 "llm.model.embed_tokens")}
    device_map.update({f"llm.model.layers.{i}": 0 if i < SPLIT_LAYER else 1 for i in range(n_layers)})
    device_map.update({"llm.model.norm": 1, "llm.model.rotary_emb": 1, "llm.lm_head": 1})
    kwargs["device_map"] = device_map
t0 = time.time()
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, trust_remote_code=True)
model = AutoModel.from_pretrained(MODEL_ID, **kwargs).eval()
processor = model.processor
print(f"loaded in {time.time() - t0:.0f} s | first parameter on {model.device}, dtype {model.dtype}")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
"""),
    md(r"""
## Prompts and label scoring

Same wording as the Qwen run. `prepare` follows `MiniCPMO.chat` up to the processor call;
`label_logits` runs the embedding steps of `MiniCPMO.generate` and one forward pass of the language model.
"""),
    code(r"""
LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
DESCRIBE = {
    "TAV": "the video frames, the audio track and the transcript",
    "T": "only the transcript",
    "A": "only the audio track",
    "V": "only the video frames, without sound",
}
CONDITIONS = list(DESCRIBE)
ASK = f"Which emotion does the speaker express? Choose exactly one of: {', '.join(LABELS)}.\nAnswer with the label only."
label_first_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_first_ids)) == len(LABELS), "two labels share a first token"


def per_second_frames(row, n_units):
    # frames were extracted at row.sample_fps; take the one nearest the middle of each second
    frames = sorted(glob.glob(str(DATA_DIR / row["frames_dir"] / "*.jpg")))
    fps = float(row["sample_fps"])
    return [Image.open(frames[min(len(frames) - 1, int(round((i + 0.5) * fps)))]).convert("RGB") for i in range(n_units)]


def build_messages(row, condition):
    audio = librosa.load(str(DATA_DIR / row["audio_path"]), sr=SR, mono=True)[0] if "A" in condition else None
    n_units = max(1, math.ceil(float(row["duration"])))
    text = f"You are given {DESCRIBE[condition]} of a speaker in a scene from a Chinese film or TV series.\n"
    if "T" in condition:
        text += f"Transcript: {row['text']}\n"
    text += ASK
    if condition == "TAV":
        content = []
        for i, img in enumerate(per_second_frames(row, n_units)):
            content += ["<unit>", img, audio[i * SR:(i + 1) * SR]]
        content.append(text)
    elif condition == "V":
        content = per_second_frames(row, n_units) + [text]
    elif condition == "A":
        content = [audio, text]
    else:
        content = [text]
    return [model.get_sys_prompt(mode="omni", language="en"), {"role": "user", "content": content}]


def prepare(messages, omni):
    images, audios, audio_parts, use_tts = [], [], [], False
    messages = deepcopy(messages)
    for i, msg in enumerate(messages):
        parts = []
        for c in msg["content"]:
            if isinstance(c, Image.Image):
                images.append(c)
                parts.append("(<image>./</image>)")
            elif isinstance(c, np.ndarray):  # audio
                audios.append(c)
                audio_parts.append(i)
                parts.append("(<audio>./</audio>)")
                use_tts = True
            else:
                parts.append(c)
        msg["content"] = ("" if omni else "\n").join(parts)
    prompt = processor.tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True,
        chat_template=model.default_tts_chat_template if use_tts else None)
    inputs = processor([prompt], [images], [audios], [audio_parts], max_slice_nums=1, use_image_id=False,
                       chunk_input=True, return_tensors="pt", max_length=32768).to(model.device)
    inputs.pop("image_sizes")  # the model casts pixel values and audio features to its own dtype
    return inputs


@torch.inference_mode()
def label_logits(inputs):
    emb, _ = model.get_vllm_embedding(inputs)
    emb = model.get_omni_embedding(inputs, input_embeddings=emb, chunk_length=model.config.audio_chunk_length)
    return model.llm(inputs_embeds=emb, attention_mask=inputs["attention_mask"], use_cache=False).logits[0, -1].float()
"""),
    *score_cells("minicpm_runs", r"""
def run_score(row, condition):
    t0 = time.perf_counter()
    inputs = prepare(build_messages(row, condition), omni=condition == "TAV")
    t1 = time.perf_counter()
    logits = label_logits(inputs)
    rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": "score",
           "n_tokens": int(inputs["input_ids"].shape[1]), "prep_s": round(t1 - t0, 3),
           "gpu_s": round(time.perf_counter() - t1, 3), "finite": bool(torch.isfinite(logits).all()),
           "lse": float(torch.logsumexp(logits, dim=-1))}
    rec.update({f"logit_{l}": float(v) for l, v in zip(LABELS, logits[label_first_ids].tolist())})
    return rec
"""),

]


# ---- third model: Gemma 4 E4B-it, label scoring only ------------------------------------

GEMMA4 = [
    md(r"""
# RQ2, third model: Gemma 4 E4B-it on CH-SIMS v2 (Kaggle, T4)

Repeats the label scoring of the Qwen2.5-Omni-7B and MiniCPM-o 2.6 runs with a model from a third family
(Google): all 4,403 clips × 4 conditions (T+A+V, T, A, V), the same question and the same record format
(logits of the 7 labels' first tokens + log-sum-exp), written to `/kaggle/working/gemma4_runs/scores.jsonl`.

How the inputs follow the model card:
- The frames go in as one video (all extracted frames, with their real timestamps), **before** the text;
  the audio goes **after** the text, as the card recommends. 70 visual tokens per frame (the processor default for video).
- No system prompt; thinking is off (`enable_thinking=False`), so the answer starts right after `<|turn>model`.
- Audio longer than 30 s is cut at 30 s by the processor (the model's limit).

**Precision.** T4 has no bf16, and the checkpoint is bf16. The language model runs in fp16 and the vision and
audio encoders in fp32. Every record stores `max_abs_act`, the largest activation seen in the language model
for that item: fp16 overflows at 65,504, so the quick check at the end flags any item that comes close.

**Memory.** Everything sits on GPU 0 (about 11.5 GiB) except the per-layer embedding table (2.8 B parameters,
a lookup only), which stays in CPU memory.

**Versions.** transformers 5.20.1, as for Qwen. In 5.19 and 5.20 `Gemma4ForConditionalGeneration.forward`
does not pass `num_frames_per_video` on to the video encoder, so `label_logits` runs the video encoder itself
and hands its output to the model. This was checked locally on a small random-weight copy: the logits match
transformers 5.18.0 (where the plain call still works) exactly for all four conditions.

**Running it**
1. *Accelerator*: **GPU T4 x2** (only one is used); *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. First set `LIMIT = 10` and run all interactively: check the load report, the `label mass`, the activation check
   and the timing lines, and send them over before the full run.
3. Full run: `LIMIT = None`, **Save Version → Save & Run All (Commit)**. It stops after `TIME_BUDGET_H`;
   to continue, add this notebook's output as an input and commit again.
"""),
    code(r"""
# transformers 5.20.1: the version of the Qwen run; Gemma 4 needs >= 5.5.
%pip install -q "transformers==5.20.1" accelerate librosa soundfile
"""),
    code(r"""
import glob
import json
import os
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "google/gemma-4-E4B-it")  # env override: local tests with a tiny model
TIME_BUDGET_H = 11.0         # stop starting new items after this many hours
LIMIT = None                 # e.g. 10 for a quick check: 10 test clips + 10 other clips
SR = 16000
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/gemma4_runs"))
T_START = time.time()

DATA_DIR = os.environ.get("CHSIMS_MM_DIR")
if DATA_DIR is None:
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "manifest.csv"))]  # no ** : media folders make it slow
    assert hits, "No manifest.csv under /kaggle/input: attach the preprocessed dataset."
    DATA_DIR = os.path.dirname(hits[0])
DATA_DIR = Path(DATA_DIR)
DRY_RUN = False
""" + COMMON_REPORT),
    *SCORE_CLIPS,
    md(r"""
## Load the model

Language model in fp16, vision and audio encoders in fp32. All on GPU 0 except the per-layer embedding
table, which stays on the CPU: hooks move the token ids to it and its output back to the GPU.
Forward hooks record the largest activation of every decoder layer (before its output scale) and of every
MLP's inner layer.
"""),
    code(r"""
from transformers import AutoModelForMultimodalLM, AutoProcessor
from transformers.video_utils import VideoMetadata

DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"
LLM_DTYPE = torch.float16 if DEVICE != "cpu" else torch.float32
ENCODERS = ("vision_tower", "embed_vision", "audio_tower", "embed_audio")
PLE = "model.language_model.embed_tokens_per_layer"

t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID)
tokenizer = processor.tokenizer
model = AutoModelForMultimodalLM.from_pretrained(MODEL_ID, dtype=LLM_DTYPE).eval()  # default: sdpa where supported
for name in ENCODERS:
    getattr(model.model, name).float()
ENC_DTYPE = model.model.audio_tower.dtype
for name, p in model.named_parameters():  # tied weights (lm_head = embed_tokens) are listed once
    if not name.startswith(PLE):
        p.data = p.data.to(DEVICE)
for name, module in model.named_modules():
    if not name.startswith(PLE):
        for b, buf in list(module.named_buffers(recurse=False)):
            module._buffers[b] = buf.to(DEVICE)
ple = model.get_submodule(PLE)
ple.register_forward_pre_hook(lambda m, args: (args[0].to(ple.weight.device),))
ple.register_forward_hook(lambda m, args, out: out.to(DEVICE))

ACT = {"max": 0.0}


def track(x):
    ACT["max"] = max(ACT["max"], float(x.detach().abs().amax()))


for layer in model.model.language_model.layers:
    layer.register_forward_hook(lambda m, args, out: track(out / m.layer_scalar.abs().clamp(min=1e-6)))
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, args: track(args[0]))

print(f"loaded in {time.time() - t0:.0f} s | language model {LLM_DTYPE}, encoders {ENC_DTYPE} | attention:",
      {n: getattr(model.config, n)._attn_implementation for n in ("text_config", "vision_config", "audio_config")})
print("layer output scales:", np.round([float(l.layer_scalar) for l in model.model.language_model.layers], 3).tolist())
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
"""),
    md(r"""
## Prompts and label scoring

Same wording as the Qwen and MiniCPM runs. The chat template places `<|video|>` and `<|audio|>` where the
content items are; the processor expands them (frames are labelled `mm:ss`).
"""),
    code(r"""
LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
DESCRIBE = {
    "TAV": "the video frames, the audio track and the transcript",
    "T": "only the transcript",
    "A": "only the audio track",
    "V": "only the video frames, without sound",
}
CONDITIONS = list(DESCRIBE)
ASK = f"Which emotion does the speaker express? Choose exactly one of: {', '.join(LABELS)}.\nAnswer with the label only."
label_first_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_first_ids)) == len(LABELS), "two labels share a first token"


def prepare(row, condition):
    content, kwargs = [], {}
    if "V" in condition:
        frames = sorted(glob.glob(str(DATA_DIR / row["frames_dir"] / "*.jpg")))
        kwargs["videos"] = [np.stack([np.asarray(Image.open(f).convert("RGB")) for f in frames])]
        kwargs["video_metadata"] = [VideoMetadata(total_num_frames=len(frames), fps=float(row["sample_fps"]),
                                                  frames_indices=list(range(len(frames))))]
        kwargs["do_sample_frames"] = False  # frames were already sampled by preprocess_chsims.py
        content.append({"type": "video"})
    text = f"You are given {DESCRIBE[condition]} of a speaker in a scene from a Chinese film or TV series.\n"
    if "T" in condition:
        text += f"Transcript: {row['text']}\n"
    content.append({"type": "text", "text": text + ASK})
    if "A" in condition:
        kwargs["audio"] = [librosa.load(str(DATA_DIR / row["audio_path"]), sr=SR, mono=True)[0]]
        content.append({"type": "audio"})
    prompt = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
    inputs = processor(text=prompt, return_tensors="pt", **kwargs)
    return {k: v.to(DEVICE, dtype=ENC_DTYPE) if torch.is_floating_point(v) else v.to(DEVICE) for k, v in inputs.items()}


@torch.inference_mode()
def label_logits(inputs):
    inputs = dict(inputs)
    if "pixel_values_videos" in inputs:
        # transformers 5.19-5.20 does not pass num_frames_per_video on to the video encoder: run it here
        video = model.model.get_video_features(inputs.pop("pixel_values_videos"), inputs.pop("video_position_ids"),
                                               inputs.pop("num_frames_per_video"), return_dict=True)
        inputs["mm_encoder_outputs"] = {"video": video}
    return model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()


inputs = prepare(clips.iloc[0], "TAV")
shown = tokenizer.decode(inputs["input_ids"][0])
for t in ("<|video|>", "<|audio|>"):
    shown = shown.replace(t, "")
print(inputs["input_ids"].shape[1], "tokens; prompt without the placeholder tokens:\n", shown)
"""),
    *score_cells("gemma4_runs", r"""
def run_score(row, condition):
    t0 = time.perf_counter()
    inputs = prepare(row, condition)
    t1 = time.perf_counter()
    ACT["max"] = 0.0
    logits = label_logits(inputs)
    rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": "score",
           "n_tokens": int(inputs["input_ids"].shape[1]), "prep_s": round(t1 - t0, 3),
           "gpu_s": round(time.perf_counter() - t1, 3), "finite": bool(torch.isfinite(logits).all()),
           "max_abs_act": ACT["max"], "lse": float(torch.logsumexp(logits, dim=-1))}
    rec.update({f"logit_{l}": float(v) for l, v in zip(LABELS, logits[label_first_ids].tolist())})
    return rec
"""),
    md(r"""
## fp16 check

fp16 overflows above 65,504. Items whose largest activation passes 30,000 are too close: if there are
any, do not start the full run.
"""),
    code(r"""
if len(ok):
    print(ok.groupby("condition")["max_abs_act"].describe(percentiles=[0.5, 0.99]).round(1).to_string())
    near = ok[(ok["max_abs_act"] > 30000) | ~ok["finite"].astype(bool)]
    print(f"items near fp16 overflow or not finite: {len(near)}")
"""),
]


# ---- fourth model: Phi-4-multimodal-instruct, label scoring only ------------------------

PHI4MM = [
    md(r"""
# RQ2, fourth model: Phi-4-multimodal-instruct on CH-SIMS v2 (Kaggle, 2× T4)

Repeats the label scoring of the other runs with a model from a fourth family (Microsoft Phi): all 4,403 clips
× 4 conditions (T+A+V, T, A, V), the same question and the same record format (logits of the 7 labels'
first tokens + log-sum-exp), written to `/kaggle/working/phi4mm_runs/scores.jsonl`.

How the inputs follow the model card (the model's own code and processor are used unchanged):
- Prompt `<|user|><|image_1|>…<|image_k|><|audio_1|>{text}<|end|><|assistant|>`: frames first, then audio, then text
  (the card's vision-speech format); no system prompt. All extracted frames go in as separate images
  (a 448-px frame becomes one crop plus the global view, about 430 tokens).
- The model switches its LoRA adapters by input type, as in its own `forward`: images (with or without audio) →
  vision adapter, and the audio then goes through the projection trained for that mode; audio only → speech adapter;
  text only → no adapter. The card notes that in vision-speech training the audio mostly carried a spoken query,
  which matters for reading the T+A+V results.

**Precision.** T4 has no bf16; the checkpoint is bf16. The whole model runs in fp16, as the Qwen and MiniCPM
runs (with fp32 encoders, clips with many frames ran out of memory in the first check). Each record stores
`max_abs_act` (the largest activation in the language model) so fp16 overflow (above 65,504) is visible, and
`top_token`, the most likely next token over the whole vocabulary.

**Versions.** `transformers==4.48.2` and `peft==0.13.2`, as the model card pins; attention `sdpa` (no flash-attn on T4).
Checked locally with the model's code on a small random-weight copy: the scored logits match those of the
model's own `generate` in each condition.

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. First set `LIMIT = 10` and run all interactively: check the load report, the `label mass`, the activation check
   and the timing lines, and send them over before the full run.
3. Full run: `LIMIT = None`, **Save Version → Save & Run All (Commit)**. It stops after `TIME_BUDGET_H`;
   to continue, add this notebook's output as an input and commit again.
"""),
    code(r"""
# Pins from the model card. peft is needed by the model's own code (its LoRA adapters).
%pip install -q "transformers==4.48.2" "peft==0.13.2" backoff accelerate librosa soundfile
"""),
    code(r"""
import glob
import json
import os
import shutil
import time
from pathlib import Path

import librosa
import numpy as np
import pandas as pd
import torch
from PIL import Image

MODEL_ID = os.environ.get("MODEL_ID", "microsoft/Phi-4-multimodal-instruct")  # env override: local tests with a tiny model
REVISION = os.environ.get("REVISION", "93f923e1a7727d1c4f446756212d9d3e8fcc5d81")  # main on 2026-10-18
SPLIT_LAYER = 8              # language-model layers 0..7 on GPU 0 (with the encoders), the rest on GPU 1
TIME_BUDGET_H = 11.0         # stop starting new items after this many hours
LIMIT = None                 # e.g. 10 for a quick check: 10 test clips + 10 other clips
SR = 16000
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/phi4mm_runs"))
T_START = time.time()

DATA_DIR = os.environ.get("CHSIMS_MM_DIR")
if DATA_DIR is None:
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "manifest.csv"))]  # no ** : media folders make it slow
    assert hits, "No manifest.csv under /kaggle/input: attach the preprocessed dataset."
    DATA_DIR = os.path.dirname(hits[0])
DATA_DIR = Path(DATA_DIR)
DRY_RUN = False
""" + COMMON_REPORT),
    *SCORE_CLIPS,
    md(r"""
## Load the model

All in fp16. On GPU the layers are placed by hand: the encoders sit with the token embeddings (the model writes their outputs into the embedding
tensor), and the output layer shares its weights with the token embeddings, so it stays on GPU 0 too.
"""),
    code(r"""
from transformers import AutoConfig, AutoModelForCausalLM, AutoProcessor

ON_GPU = torch.cuda.is_available()
LLM_DTYPE = torch.float16 if ON_GPU else torch.float32
kwargs = dict(revision=REVISION, trust_remote_code=True, torch_dtype=LLM_DTYPE, _attn_implementation="sdpa")
if ON_GPU:
    n_layers = AutoConfig.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True).num_hidden_layers
    device_map = {"model.embed_tokens": 0, "model.embed_tokens_extend": 0, "lm_head": 0}
    device_map.update({f"model.layers.{i}": 0 if i < SPLIT_LAYER else 1 for i in range(n_layers)})
    device_map["model.norm"] = 1
    kwargs["device_map"] = device_map
t0 = time.time()
processor = AutoProcessor.from_pretrained(MODEL_ID, revision=REVISION, trust_remote_code=True)
tokenizer = processor.tokenizer
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, **kwargs).eval()
DEVICE = model.model.embed_tokens.weight.device

ACT = {"max": 0.0}


def track(x):
    ACT["max"] = max(ACT["max"], float(x.detach().abs().amax()))


for layer in model.model.layers:
    layer.register_forward_hook(lambda m, args, out: track(out[0] if isinstance(out, tuple) else out))
    layer.mlp.down_proj.register_forward_pre_hook(lambda m, args: track(args[0]))

print(f"loaded in {time.time() - t0:.0f} s | language model {LLM_DTYPE}, encoders {model.model.embed_tokens_extend.image_embed.img_projection[0].weight.dtype}")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
"""),
    md(r"""
## Prompts and label scoring

Same wording as the other runs. The processor sets `input_mode` (0 text, 1 vision, 2 speech, 3 vision + speech),
from which the model picks its LoRA adapter.
"""),
    code(r"""
LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
DESCRIBE = {
    "TAV": "the video frames, the audio track and the transcript",
    "T": "only the transcript",
    "A": "only the audio track",
    "V": "only the video frames, without sound",
}
CONDITIONS = list(DESCRIBE)
ASK = f"Which emotion does the speaker express? Choose exactly one of: {', '.join(LABELS)}.\nAnswer with the label only."
label_first_ids = [tokenizer.encode(l, add_special_tokens=False)[0] for l in LABELS]
assert len(set(label_first_ids)) == len(LABELS), "two labels share a first token"


def prepare(row, condition):
    images, audios, tags = None, None, ""
    if "V" in condition:
        frames = sorted(glob.glob(str(DATA_DIR / row["frames_dir"] / "*.jpg")))
        images = [Image.open(f).convert("RGB") for f in frames]
        tags += "".join(f"<|image_{i + 1}|>" for i in range(len(images)))
    if "A" in condition:
        audios = [(librosa.load(str(DATA_DIR / row["audio_path"]), sr=SR, mono=True)[0], SR)]
        tags += "<|audio_1|>"
    text = f"You are given {DESCRIBE[condition]} of a speaker in a scene from a Chinese film or TV series.\n"
    if "T" in condition:
        text += f"Transcript: {row['text']}\n"
    prompt = f"<|user|>{tags}{text}{ASK}<|end|><|assistant|>"
    return processor(text=prompt, images=images, audios=audios, return_tensors="pt").to(DEVICE)


@torch.inference_mode()
def label_logits(inputs):
    return model(**inputs, num_logits_to_keep=1).logits[0, -1].float()


import re

inputs = prepare(clips.iloc[0], "TAV")
shown = re.sub(r"((<\|endoftext1[01]\|>)\2*)", lambda m: f"[{m.group(1).count(m.group(2))} x {m.group(2)}]",
               tokenizer.decode(inputs["input_ids"][0]))  # image (10) and audio (11) placeholder runs
print(inputs["input_ids"].shape[1], "tokens | input_mode", inputs["input_mode"].tolist(), "| prompt:\n", shown)
"""),
    md(r"""
## Memory check

The clip with the most frames (then the longest audio) under T+A+V, the most memory-hungry item of the run.
If this cell runs out of memory, lower `SPLIT_LAYER` and restart.
"""),
    code(r"""
longest = manifest[manifest["status"] == "ok"].sort_values(["n_frames", "duration"]).iloc[-1]
for d in range(torch.cuda.device_count()):
    torch.cuda.reset_peak_memory_stats(d)
t0 = time.perf_counter()
inputs = prepare(longest, "TAV")
logits = label_logits(inputs)
print(f"{longest['uid']}: {longest['n_frames']} frames, {longest['duration']:.1f} s, {inputs['input_ids'].shape[1]} tokens, "
      f"{time.perf_counter() - t0:.1f} s | peak GiB:",
      [round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())])
del inputs, logits
torch.cuda.empty_cache()
"""),
    *score_cells("phi4mm_runs", r"""
def run_score(row, condition):
    t0 = time.perf_counter()
    inputs = prepare(row, condition)
    t1 = time.perf_counter()
    ACT["max"] = 0.0
    logits = label_logits(inputs)
    rec = {"model": MODEL_ID, "uid": row["uid"], "condition": condition, "task": "score",
           "n_tokens": int(inputs["input_ids"].shape[1]), "input_mode": int(inputs["input_mode"][0]),
           "prep_s": round(t1 - t0, 3), "gpu_s": round(time.perf_counter() - t1, 3),
           "finite": bool(torch.isfinite(logits).all()), "max_abs_act": ACT["max"],
           "top_token": tokenizer.decode([int(logits.argmax())]),
           "lse": float(torch.logsumexp(logits, dim=-1))}
    rec.update({f"logit_{l}": float(v) for l, v in zip(LABELS, logits[label_first_ids].tolist())})
    return rec
"""),
    md(r"""
## fp16 check

fp16 overflows above 65,504. Items whose largest activation passes 30,000 are too close: if there are
any, do not start the full run.
"""),
    code(r"""
if len(ok):
    print(ok.groupby("condition")["max_abs_act"].describe(percentiles=[0.5, 0.99]).round(1).to_string())
    near = ok[(ok["max_abs_act"] > 30000) | ~ok["finite"].astype(bool)]
    print(f"items near fp16 overflow or not finite: {len(near)}")
"""),
]


# ---- fifth model: Emotion-LLaMA (emotion specialist), label scoring only -----------------

EMOLLAMA_PINS = [  # environment.yaml of the repository, plus the README's extra installs; unpinned ones fixed here
    "huggingface-hub==0.18.0", "pyyaml==6.0", "regex==2022.10.31", "tokenizers==0.13.2", "tqdm==4.64.1",
    "transformers==4.30.0", "timm==0.6.13", "webdataset==0.2.48", "omegaconf==2.3.0", "opencv-python==4.7.0.72",
    "decord==0.6.0", "peft==0.2.0", "accelerate==0.20.3", "iopath==0.1.10", "matplotlib==3.7.0", "psutil==5.9.4",
    "wandb==0.16.0", "moviepy==1.0.3", "soundfile==0.12.1", "sentencepiece==0.2.2", "numpy==1.26.4",
    "pandas==2.3.3", "setuptools==69.5.1",
]

EMOLLAMA = [
    md(r"""
# RQ2, emotion specialist: Emotion-LLaMA on CH-SIMS v2 (Kaggle, 2× T4)

Label scoring with Emotion-LLaMA (Cheng et al., NeurIPS 2024; Llama-2-7B-chat + LoRA, EVA-CLIP image
encoder, HuBERT audio features), the one non-Qwen model trained for multimodal emotion recognition.
All 4,403 clips × 4 conditions, written to `/kaggle/working/emollama_runs/scores.jsonl` in the format of the other runs.

**Environment.** The repository needs Python 3.9 (torch 2.0, transformers 4.30, peft 0.2.0); Kaggle runs 3.13.
This notebook builds a separate Python 3.9 environment with `uv`, installs the repository's pins, and runs
`score_emotion_llama.py` in it. The model is built by the authors' own code (demo config), unchanged.

**What the model sees** (details in the script's docstring):
- the clip's first extracted frame (the demo uses the first frame), HuBERT features of the audio (as the demo),
  zeros for the FaceMAE and VideoMAE features (as the demo: the authors publish no extraction for raw clips);
- the evaluation instruction of the authors' dataset code, with the Chinese transcript;
- the checkpoint's six labels in the order of its training instruction: happy, sad, neutral, angry, worried, surprise.
  "worried" is recorded as Fear (the authors' zero-shot alias). **There is no Disgust**: `logit_Disgust` is empty,
  and the comparison with humans uses the six labels.
- T, A and V alone were never seen in training (T: no media; A: only the three feature tokens; V: audio feature zero).

**Weights** (pinned revisions): the demo checkpoint `Emoation_LLaMA.pth` from the authors' Hugging Face Space,
Llama-2-7b-chat from the mirror the Space itself uses, `TencentGameMate/chinese-hubert-large`, and EVA-CLIP-g,
which the repository's code downloads. Licences: the repository is CC BY-NC 4.0 plus the Llama 2 licence; research use.

**Precision and memory.** As in the authors' evaluation config: Llama in fp16 (`low_resource: False`), split over
the two GPUs; EVA in fp16; HuBERT in fp32. The script checks that every tensor of the checkpoint was loaded
(the authors' loader uses `strict=False`).

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. First set `LIMIT = 10` and run all interactively (about 25 GB of downloads): check the checkpoint line,
   the label tokens, the prompts, the `label mass` and the timing lines, and send them over before the full run.
3. Full run: `LIMIT = None`, **Save Version → Save & Run All (Commit)**. It stops after `TIME_BUDGET_H`;
   to continue, add this notebook's output as an input and commit again.
"""),
    code(r"""
import glob
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TIME_BUDGET_H = 11.0         # stop starting new items after this many hours (counted from the start of the notebook)
LIMIT = None                 # e.g. 10 for a quick check: 10 test clips + 10 other clips
SPLIT_LAYER = 14             # Llama layers 0..13 on GPU 0 (with EVA), the rest on GPU 1 (with HuBERT)
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/emollama_runs"))
WORK = Path(os.environ.get("EMOLLAMA_WORK", "/tmp/emollama"))  # environment, code, weights: not part of the output
ENV_PY = Path(os.environ.get("EMOLLAMA_PY", WORK / "env" / "bin" / "python"))
REPO = Path(os.environ.get("EMOLLAMA_REPO", WORK / "Emotion-LLaMA"))
REPO_COMMIT = "6f8d05087b2a91e445ad986545977fda7eb1b6a4"           # main on 2026-10-18
LLAMA_REPO, LLAMA_REV = "ZebangCheng/Emotion-LLaMA", "2ba523d3543da46b9f8b6dab308f2a56298c9e37"
SPACE_REPO, SPACE_REV = "ZebangCheng/Emotion-LLaMA", "9cc18338a896d6aa6a3090bbf37055bf93909b73"
HUBERT_REPO, HUBERT_REV = "TencentGameMate/chinese-hubert-large", "90cb660492214f687e60f5ca509b20edae6e75bd"
MODEL_NAME = "ZebangCheng/Emotion-LLaMA (Emoation_LLaMA.pth)"
T_START = time.time()

DATA_DIR = os.environ.get("CHSIMS_MM_DIR")
if DATA_DIR is None:
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "manifest.csv"))]  # no ** : media folders make it slow
    assert hits, "No manifest.csv under /kaggle/input: attach the preprocessed dataset."
    DATA_DIR = os.path.dirname(hits[0])
DATA_DIR = Path(DATA_DIR)
DRY_RUN = False
WORK.mkdir(parents=True, exist_ok=True)


def sh(cmd):
    print("$", cmd, flush=True)
    subprocess.run(cmd, shell=True, check=True)
""" + COMMON_REPORT),
    md("## Python 3.9 environment and the repository"),
    code(r"""
PINS = """ + repr(EMOLLAMA_PINS) + r"""
if not ENV_PY.exists():
    sh(f"{sys.executable} -m pip install -q uv")
    sh(f"{sys.executable} -m uv venv -q --python 3.9 {ENV_PY.parent.parent}")  # uv fetches a Python 3.9 build
    sh(f"{sys.executable} -m uv pip install -q --python {ENV_PY} torch==2.0.0 torchvision==0.15.1 torchaudio==2.0.1 "
       "--index-url https://download.pytorch.org/whl/cu118")
    sh(f"{sys.executable} -m uv pip install -q --python {ENV_PY} " + " ".join(f'"{p}"' for p in PINS))
sh(f"{ENV_PY} -c 'import sys, torch, transformers, peft; print(sys.version.split()[0], torch.__version__, "
   "transformers.__version__, peft.__version__, \"cuda:\", torch.cuda.is_available())'")
if not REPO.exists():
    sh(f"git clone -q https://github.com/ZebangCheng/Emotion-LLaMA.git {REPO}")
    sh(f"git -C {REPO} checkout -q {REPO_COMMIT}")
sh(f"git -C {REPO} log -1 --format='%H %cd'")
"""),
    md("## Weights"),
    code(r"""
from huggingface_hub import hf_hub_download, snapshot_download

if os.environ.get("EMOLLAMA_LOCAL"):  # local test with small stand-ins
    LLAMA, CKPT, HUBERT = os.environ["EMOLLAMA_LLAMA"], os.environ["EMOLLAMA_CKPT"], os.environ["EMOLLAMA_HUBERT"]
else:
    t0 = time.time()
    LLAMA = snapshot_download(LLAMA_REPO, revision=LLAMA_REV, local_dir=WORK / "llama", allow_patterns=[
        "config.json", "generation_config.json", "model.safetensors.index.json", "model-*.safetensors",
        "tokenizer.model", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"])
    CKPT = hf_hub_download(SPACE_REPO, "Emoation_LLaMA.pth", repo_type="space", revision=SPACE_REV, local_dir=WORK / "ckpt")
    HUBERT = snapshot_download(HUBERT_REPO, revision=HUBERT_REV, local_dir=WORK / "hubert",
                               allow_patterns=["config.json", "preprocessor_config.json", "pytorch_model.bin"])
    print(f"downloaded in {time.time() - t0:.0f} s | free disk on / (GB): {shutil.disk_usage('/').free / 1e9:.1f}")
print(LLAMA, CKPT, HUBERT, sep="\n")
"""),
    md(r"""
## Scoring script

Runs in the Python 3.9 environment. Kept in this repository as `kaggle/score_emotion_llama.py`; the cell
below writes the same file.
"""),
    code("%%writefile score_emotion_llama.py\n" + (HERE / "score_emotion_llama.py").read_text(encoding="utf-8")),
    md("## Run"),
    code(r"""
cmd = [str(ENV_PY), "-u", "score_emotion_llama.py", "--repo", str(REPO), "--data-dir", str(DATA_DIR),
       "--out-dir", str(OUT_DIR), "--llama", str(LLAMA), "--ckpt", str(CKPT), "--hubert", str(HUBERT),
       "--model-name", MODEL_NAME, "--split-layer", str(SPLIT_LAYER), "--limit", str(LIMIT or 0),
       "--time-budget-h", str(TIME_BUDGET_H - (time.time() - T_START) / 3600),
       "--resume-glob", "/kaggle/input/**/emollama_runs/scores.jsonl"]
if os.environ.get("EMOLLAMA_CPU_TEST"):
    cmd.append("--cpu-test")
proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
for line in proc.stdout:
    if "%|" not in line:  # leave out download progress bars
        print(line, end="", flush=True)
assert proc.wait() == 0, "the scoring script failed: see the lines above"
"""),
    md(r"""
## Progress and quick checks

Six labels: `label_mass` is the probability the model puts on the first tokens of its six labels;
accuracy is against the majority label, on all clips and on clips whose majority label is not Disgust.
"""),
    code(r"""
SIX = ["Anger", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
s = pd.DataFrame([json.loads(l) for l in open(OUT_DIR / "scores.jsonl", encoding="utf-8") if l.strip()])
ok = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()] if len(s) else s
n_items = int((manifest["status"] == "ok").sum()) * 4 if not LIMIT else None
print(f"{len(ok)} items done" + (f" of {n_items}" if n_items else ""), "| OOM:", int((s.get("error") == "oom").sum()) if "error" in s else 0)
if len(ok):
    p = np.exp(ok[[f"logit_{l}" for l in SIX]].to_numpy(dtype=float) - ok[["lse"]].to_numpy())
    ok = ok.assign(label_mass=p.sum(1), pred=np.array(SIX)[p.argmax(1)]).merge(manifest[["uid", "ds_label"]], on="uid")
    ok["hit"] = ok["pred"] == ok["ds_label"]
    print(ok.groupby("condition").agg(n=("uid", "size"), label_mass=("label_mass", "mean"), accuracy=("hit", "mean"),
                                      accuracy_no_disgust=("hit", lambda x: x[ok.loc[x.index, "ds_label"] != "Disgust"].mean()),
                                      nonfinite=("finite", lambda x: int((~x.astype(bool)).sum())),
                                      gpu_s=("gpu_s", "mean"), prep_s=("prep_s", "mean"), tokens=("n_tokens", "mean")).round(3).to_string())
    print(ok.groupby("condition")["pred"].value_counts(normalize=True).unstack().round(2).to_string())
"""),
]


# ---- RQ4: causal-inference fusion (fine-tuned), one notebook per variant -------------------

CI_TEXT = {
    "B": "**B, unimodal heads only**: the RQ3 soft-label loss on the T+A+V answer, plus a head per modality that "
         "predicts that modality's polarity (CH-SIMS v2 per-modality labels). No fusion.",
    "C": "**C, fixed mixture**: B plus fusion of the T+A+V answer with the T, A, V answers, with one learned weight "
         "for the fused reading that is the same for every clip.",
    "D": "**D, causal inference**: C, but the weight of the fused reading (the probability of a common cause) is "
         "inferred per clip from the per-modality polarity heads and the model's state.",
}


def rq4_cells(variant):
    return [
        md(rf"""
# RQ4: causal-inference fusion on Qwen2.5-Omni-7B, variant {variant} (Kaggle, 2× T4)

Inspired by Bayesian causal inference in multisensory perception (Körding et al., 2007): an observer infers
whether the cues share a cause and fuses them only then. Per clip the model reads the clip four times, under
T+A+V, T, A and V (one LoRA, shared). Heads on the last hidden state of the T, A and V readings predict each
modality's polarity. The output is
`Q = pi * P(T+A+V) + (1 - pi) * (w_T P(T) + w_A P(A) + w_V P(V))`,
where `pi` is the weight of the fused reading. The loss is the cross-entropy between `Q` and the 3 annotators'
vote distribution, plus `AUX_WEIGHT` times the cross-entropy of the polarity heads.

This notebook trains {CI_TEXT[variant]}
Variant A is the RQ3 soft-label arm (already run). The other variants are the same notebook with
`VARIANT = "B"`, `"C"` or `"D"`.

Same LoRA, prompt, splits, learning rate, epochs and checkpointing as RQ3; the heads (well under 1 M parameters)
are trained in fp32 at `HEAD_LR`. Four readings per training clip make a run about twice as long as RQ3
(estimate: about 12 GPU-hours to the end of training, so two sessions).

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset.
2. First set `LIMIT = 10` and run all interactively; send the progress lines (speed, peak memory, OOM count).
   Do **not** attach that run's output later.
3. **Save Version → Save & Run All (Commit)**. When it stops at `TIME_BUDGET_H`, add this notebook's own output
   as an input and commit again: training resumes from the last checkpoint, scoring skips finished clips.
4. Results: `/kaggle/working/rq4_{variant}/` (or `rq4_{variant}_seed<N>/`) with `scores.jsonl`, `state.json` and the
   best adapter and heads. `scores.jsonl` holds the four readings (`task = score`) and the fused output
   (`task = fused`, logits = log Q) for the test and valid splits.

**Seeds.** As in RQ3: `SEED = 0`, then `1` and `2`, each in its own committed version.
"""),
        code(r"""
# peft added for LoRA; the rest as in the other notebooks.
%pip install -q "transformers==5.20.1" "qwen-omni-utils==0.0.9" "peft==0.21.2" audioread librosa
# Kaggle ships torchao 0.10; peft 0.21 raises an ImportError for torchao < 0.16 even when it is not used.
%pip uninstall -y -q torchao
"""),
        code(COMMON_CONFIG + rf"""
VARIANT = "{variant}"            # B: polarity heads only; C: + fixed mixture; D: + per-clip causal inference
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05
LORA_TARGETS = r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
LR = 1e-4                      # LoRA, as in RQ3
HEAD_LR = 1e-3                 # polarity heads and gate
AUX_WEIGHT = 1.0               # weight of the per-modality polarity loss
EPOCHS = 2
GRAD_ACCUM = 8
WARMUP = 0.03
CKPT_EVERY = 200
TRAIN_MAX_MEMORY = {{0: "10GiB", 1: "14GiB"}}
TIME_BUDGET_H = 11.0
LIMIT = None                   # e.g. 10 for a quick check: 10 clips per split
SEED = int(os.environ.get("SEED", 0))
SUFFIX = f"_seed{{SEED}}" if SEED else ""
RUN_NAME = f"rq4_{{VARIANT}}{{SUFFIX}}"
TAG = f"{{MODEL_ID}}+ci-{{VARIANT}}{{SUFFIX.replace('_', '-')}}"
OUT_DIR = Path(os.environ.get("OUT_DIR", f"/kaggle/working/{{RUN_NAME}}"))
READINGS = ("TAV", "T", "A", "V")
T_START = time.time()
assert VARIANT in ("B", "C", "D") and CKPT_EVERY % GRAD_ACCUM == 0
""" + COMMON_REPORT),
        md(r"""
## Load the model, add LoRA and the heads

As in RQ3, plus a hook that keeps the last hidden state (the input of the language-model head) for the heads.
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
HIDDEN = {}


def keep_last(module, args):
    HIDDEN["h"] = args[0][:, -1]
    return (args[0][:, -1:],) + args[1:]


base.lm_head.register_forward_pre_hook(keep_last)

torch.manual_seed(SEED)
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                        target_modules=LORA_TARGETS, bias="none"))
lora_params = [p for p in model.parameters() if p.requires_grad]
HEAD_DEVICE = base.lm_head.weight.device


class CIHeads(torch.nn.Module):
    # polarity head per modality (negative / neutral / positive), gate for the fused reading, mixture weights
    def __init__(self, d, n_labels):
        super().__init__()
        self.norm = torch.nn.LayerNorm(d, elementwise_affine=False)  # raw hidden states are large: keep the heads unsaturated
        self.pol = torch.nn.ModuleDict({m: torch.nn.Linear(d, 3) for m in "TAV"})
        self.proj = torch.nn.Linear(d, 16)
        self.gate = torch.nn.Sequential(torch.nn.Linear(9 + 3 + 4 + 16, 32), torch.nn.GELU(), torch.nn.Linear(32, 1))
        self.gate_const = torch.nn.Parameter(torch.zeros(1))   # variant C: one gate logit for every clip
        self.mix = torch.nn.Parameter(torch.zeros(3))          # weights of the T, A, V readings


heads = CIHeads(base.config.text_config.hidden_size if hasattr(base.config, "text_config") else base.config.hidden_size,
                7).to(HEAD_DEVICE, torch.float32)
print(f"LoRA trainable params: {sum(p.numel() for p in lora_params) / 1e6:.1f} M | heads: "
      f"{sum(p.numel() for p in heads.parameters()) / 1e3:.0f} k on {HEAD_DEVICE}")
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB")
"""),
        md("## Prompts (same as the zero-shot run)"),
        PROMPTS,
        md(r"""
## Clips and targets

`human`: the 3 annotators' vote distribution (the target, as in the RQ3 soft arm). `polarity`: the class
(negative, neutral, positive) of each modality's CH-SIMS v2 label, the target of the polarity heads.
"""),
        code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
splits = {m: clips[clips["mode"] == m].reset_index(drop=True) for m in ("train", "valid", "test")}
if LIMIT:
    splits = {m: df.head(LIMIT) for m, df in splits.items()}
for m, df in splits.items():
    votes = df[["ann1", "ann2", "ann3"]].to_numpy()
    df["human"] = list(np.stack([(votes == l).sum(axis=1) for l in LABELS], axis=1) / votes.shape[1])
    for mod in "TAV":
        df[f"pol_{mod}"] = (np.sign(df[f"label_{mod}"]).astype(int) + 1).to_numpy()  # 0 negative, 1 neutral, 2 positive
    print(f"{m:5s} {len(df):5d} clips | conflict share {df['conflict'].mean():.2f}")
train_df, valid_df, test_df = splits["train"], splits["valid"], splits["test"]
"""),
        md(r"""
## Readings, fusion and loss

`read` runs the four prompts and keeps, for each, the 7 label logits and the last hidden state. `fuse` builds
`Q`. Gate features (variant D): the three polarity distributions, their pairwise L1 distances, the entropy
of each reading's label distribution, and a 16-dimensional projection of the T+A+V hidden state.
"""),
        code(r"""
def read(row):
    # per reading: the 7 label logits, the last hidden state, and the log-sum-exp over the whole vocabulary
    out = {}
    for c in READINGS:
        inputs = prepare(build_messages(row, c, ask="label"))
        logits = model(**inputs, use_cache=False).logits[0, -1].float()
        out[c] = (logits[label_first_ids].to(HEAD_DEVICE), HIDDEN["h"][0].to(HEAD_DEVICE, torch.float32),
                  float(torch.logsumexp(logits.detach(), dim=-1)))
    return out


def ent(p):
    return -(p * torch.log(p.clamp_min(1e-12))).sum(-1, keepdim=True)


def fuse(out):
    P = {c: torch.softmax(out[c][0], dim=-1) for c in READINGS}
    pol = {m: torch.log_softmax(heads.pol[m](heads.norm(out[m][1])), dim=-1) for m in "TAV"}
    if VARIANT == "B":
        return P["TAV"], None, pol
    w = torch.softmax(heads.mix, dim=0)
    separate = w[0] * P["T"] + w[1] * P["A"] + w[2] * P["V"]
    if VARIANT == "C":
        pi = torch.sigmoid(heads.gate_const)
    else:
        q = {m: pol[m].exp() for m in "TAV"}
        feats = torch.cat([q["T"], q["A"], q["V"],
                           (q["T"] - q["A"]).abs().sum(-1, keepdim=True), (q["T"] - q["V"]).abs().sum(-1, keepdim=True),
                           (q["A"] - q["V"]).abs().sum(-1, keepdim=True),
                           ent(P["TAV"]), ent(P["T"]), ent(P["A"]), ent(P["V"]), heads.proj(heads.norm(out["TAV"][1]))])
        pi = torch.sigmoid(heads.gate(feats))
    return pi * P["TAV"] + (1 - pi) * separate, pi, pol


def losses(row, Q, pol):
    human = torch.tensor(row["human"], dtype=torch.float32, device=HEAD_DEVICE)
    main = -(human * torch.log(Q.clamp_min(1e-8))).sum()
    aux = sum(-pol[m][int(row[f"pol_{m}"])] for m in "TAV") / 3
    return main, aux


def jsd(p, q):
    m = (p + q) / 2
    kl = lambda a, b: np.where(a > 0, a * np.log(np.clip(a, 1e-12, None) / np.clip(b, 1e-12, None)), 0).sum(axis=-1)
    return (kl(p, m) + kl(q, m)) / 2


@torch.inference_mode()
def validate(df):
    model.eval()
    Qs, mains, auxs, pis, pol_hits = [], [], [], [], []
    for _, row in df.iterrows():
        Q, pi, pol = fuse(read(row))
        main, aux = losses(row, Q, pol)
        mains.append(float(main)); auxs.append(float(aux))
        Qs.append(Q.cpu().numpy()); pis.append(float(pi) if pi is not None else np.nan)
        pol_hits.append([int(pol[m].argmax()) == int(row[f"pol_{m}"]) for m in "TAV"])
    model.train()
    Qs, human, conflict = np.array(Qs), np.stack(df["human"]), df["conflict"].to_numpy() == 1
    pis = np.array(pis)
    return {"val_loss": float(np.mean(mains)), "val_aux": float(np.mean(auxs)), "val_jsd": float(jsd(Qs, human).mean()),
            "val_acc": float((np.array(LABELS)[Qs.argmax(1)] == df["ds_label"].to_numpy()).mean()),
            "val_entropy": float(-(Qs * np.log(np.clip(Qs, 1e-12, None))).sum(1).mean()),
            "val_polarity_acc": [round(float(x), 3) for x in np.mean(pol_hits, axis=0)],
            "val_pi_no_conflict": float(np.nanmean(pis[~conflict])) if (~conflict).any() and not np.isnan(pis).all() else None,
            "val_pi_conflict": float(np.nanmean(pis[conflict])) if conflict.any() and not np.isnan(pis).all() else None}
"""),
        md(r"""
## Checkpoints and resume

As in RQ3; the heads are saved next to the adapter (`heads.pt`).
"""),
        code(r"""
optimizer = torch.optim.AdamW([{"params": lora_params, "lr": LR}, {"params": list(heads.parameters()), "lr": HEAD_LR}],
                              weight_decay=0.0)
steps_per_epoch = -(-len(train_df) // GRAD_ACCUM)
total_steps = EPOCHS * steps_per_epoch
warmup_steps = max(1, int(WARMUP * total_steps))
scheduler = torch.optim.lr_scheduler.LambdaLR(
    optimizer, lambda s: min((s + 1) / warmup_steps, max(0.0, (total_steps - s) / max(1, total_steps - warmup_steps))))
scaler = torch.amp.GradScaler("cuda", enabled=ON_GPU)
SIZES = {m: len(df) for m, df in splits.items()}
state = {"variant": VARIANT, "seed": SEED, "sizes": SIZES, "epoch": 0, "pos": 0, "opt_step": 0, "loss_sum": 0.0,
         "aux_sum": 0.0, "loss_n": 0, "oom": 0, "history": [], "best_epoch": None, "best_val_loss": None, "train_done": False}


def read_state(folder):
    path = Path(folder) / "state.json"
    return json.loads(path.read_text()) if path.exists() else None


def save_weights(folder):
    torch.save(get_peft_model_state_dict(model), folder / "adapter.pt")
    torch.save(heads.state_dict(), folder / "heads.pt")


def load_weights(folder):
    set_peft_model_state_dict(model, torch.load(folder / "adapter.pt", map_location="cpu"))
    heads.load_state_dict(torch.load(folder / "heads.pt", map_location="cpu"))


def save_checkpoint():
    tmp = OUT_DIR / "last.tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir()
    save_weights(tmp)
    torch.save({"optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                "scaler": scaler.state_dict()}, tmp / "optim.pt")
    shutil.rmtree(OUT_DIR / "last", ignore_errors=True)
    tmp.rename(OUT_DIR / "last")
    state["saved_at"] = time.time()
    (OUT_DIR / "state.json").write_text(json.dumps(state, indent=1))


def out_of_time():
    return (time.time() - T_START) / 3600 > TIME_BUDGET_H


def same_run(s):
    return s.get("sizes") == SIZES and s.get("seed", 0) == SEED and s.get("variant") == VARIANT


if read_state(OUT_DIR) is None:
    found = [(s, Path(p).parent) for p in (h for k in range(6) for h in glob.glob("/kaggle/input/" + "*/" * k + f"{RUN_NAME}/state.json"))
             if (s := read_state(Path(p).parent))]
    usable = [(s, d) for s, d in found if same_run(s)]
    if len(usable) < len(found):
        print(f"ignoring {len(found) - len(usable)} earlier run(s) made with different clip counts, seed or variant")
    if usable:
        src = max(usable, key=lambda x: x[0].get("saved_at", 0))[1]
        shutil.copytree(src, OUT_DIR, dirs_exist_ok=True)
        print("copied earlier run from", src)

saved = read_state(OUT_DIR)
if saved and same_run(saved):
    state.update(saved)
    if (OUT_DIR / "last" / "adapter.pt").exists():
        load_weights(OUT_DIR / "last")
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

As in RQ3: one clip (four readings) per forward step, gradients summed over `GRAD_ACCUM` clips, a fixed
shuffled order per epoch, the best epoch chosen by the validation loss (cross-entropy of `Q` to the votes).
"""),
        code(r"""
def train():
    model.train()
    t_run, n_run = time.time(), 0
    while state["epoch"] < EPOCHS:
        epoch = state["epoch"]
        order = np.random.default_rng(1000 * SEED + epoch).permutation(len(train_df))
        while state["pos"] < len(order):
            if out_of_time():
                save_checkpoint()
                return False
            batch = order[state["pos"]:state["pos"] + GRAD_ACCUM]
            for i in batch:
                row = train_df.iloc[i]
                try:
                    Q, _, pol = fuse(read(row))
                    main, aux = losses(row, Q, pol)
                    scaler.scale((main + AUX_WEIGHT * aux) / len(batch)).backward()
                    state["loss_sum"] += main.item()
                    state["aux_sum"] += aux.item()
                    state["loss_n"] += 1
                except torch.cuda.OutOfMemoryError:
                    optimizer.zero_grad(set_to_none=True)  # a partial backward may have left gradients
                    torch.cuda.empty_cache()
                    state["oom"] += 1
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(lora_params + list(heads.parameters()), 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()
            state["pos"] += len(batch)
            state["opt_step"] += 1
            n_run += len(batch)
            if state["pos"] % CKPT_EVERY == 0 or state["pos"] == len(order):
                save_checkpoint()
            if n_run <= 2 * GRAD_ACCUM or n_run in (48, 200) or state["opt_step"] % 50 == 0:
                sec = (time.time() - t_run) / n_run
                left = (EPOCHS * len(train_df) - epoch * len(train_df) - state["pos"]) * sec / 3600
                peak = max([torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())] or [0])
                n = max(1, state["loss_n"])
                print(f"epoch {epoch} clip {state['pos']}/{len(order)} | loss {state['loss_sum'] / n:.3f} aux {state['aux_sum'] / n:.3f}"
                      f" | {sec:.2f} s/clip, about {left:.1f} h of training left | peak {peak:.1f} GiB | OOM {state['oom']}")
        if not any(h["epoch"] == epoch for h in state["history"]):
            t0 = time.time()
            metrics = validate(valid_df)
            n = max(1, state["loss_n"])
            entry = {"epoch": epoch, "train_loss": state["loss_sum"] / n, "train_aux": state["aux_sum"] / n, **metrics,
                     "val_minutes": round((time.time() - t0) / 60, 1), "session_hours": round((time.time() - T_START) / 3600, 2),
                     "peak_gib": [round(torch.cuda.max_memory_allocated(d) / 2**30, 1) for d in range(torch.cuda.device_count())]}
            state["history"].append(entry)
            print(entry)
            if state["best_val_loss"] is None or metrics["val_loss"] < state["best_val_loss"]:
                state["best_epoch"], state["best_val_loss"] = epoch, metrics["val_loss"]
                (OUT_DIR / "best").mkdir(exist_ok=True)
                save_weights(OUT_DIR / "best")
        state.update(epoch=epoch + 1, pos=0, loss_sum=0.0, aux_sum=0.0, loss_n=0)
        save_checkpoint()
    state["train_done"] = True
    save_checkpoint()
    return True


finished = state["train_done"] or train()
print("training finished, best epoch:", state["best_epoch"]) if finished else print(
    "stopped at the time budget: add this notebook's output as an input and commit again")
"""),
        md(r"""
## Score the test and valid splits with the best adapter and heads

Per clip, the four readings are saved as `task = score` records (as in RQ3) and the fused output as one
`task = fused` record (condition `TAV`, logits = log Q, so a softmax returns Q), with `pi` and the three
polarity distributions.
"""),
        code(r"""
SCORES = OUT_DIR / "scores.jsonl"


def key(rec):
    return (rec["model"], rec["uid"], rec["condition"], rec["task"])


done, n_new = set(), 0
if finished:
    model.eval()
    load_weights(OUT_DIR / "best")
    records = {}
    if SCORES.exists():
        for line in open(SCORES, encoding="utf-8"):
            if line.strip() and "error" not in (rec := json.loads(line)):
                records[key(rec)] = rec
    with open(SCORES, "w", encoding="utf-8") as f:
        for rec in records.values():
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    done = set(records)
    jobs = [(row, "test") for _, row in test_df.iterrows()] + [(row, "valid") for _, row in valid_df.iterrows()]
    todo = [j for j in jobs if (TAG, j[0]["uid"], "TAV", "fused") not in done]
    print(f"{len(jobs)} clips, {len(jobs) - len(todo)} already done, {len(todo)} to run")
    for row, split in todo:
        if out_of_time():
            print("stopped at the time budget: add this notebook's output as an input and commit again")
            break
        new = []
        try:
            with torch.inference_mode():
                t0 = time.perf_counter()
                out = read(row)
                Q, pi, pol = fuse(out)
            for c in READINGS:
                new.append({"model": TAG, "uid": row["uid"], "condition": c, "task": "score", "split": split,
                            "lse": out[c][2], **{f"logit_{l}": float(v) for l, v in zip(LABELS, out[c][0].tolist())}})
            logq = torch.log(Q.clamp_min(1e-12))
            new.append({"model": TAG, "uid": row["uid"], "condition": "TAV", "task": "fused", "split": split,
                        "gpu_s": round(time.perf_counter() - t0, 3), "lse": 0.0, "pi": float(pi) if pi is not None else None,
                        **{f"pol_{m}": [round(float(x), 4) for x in pol[m].exp().tolist()] for m in "TAV"},
                        **{f"logit_{l}": float(v) for l, v in zip(LABELS, logq.tolist())}})
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            new = [{"model": TAG, "uid": row["uid"], "condition": "TAV", "task": "fused", "split": split, "error": "oom"}]
        with open(SCORES, "a", encoding="utf-8") as f:
            for rec in new:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        n_new += 1
    print(f"scored {n_new} clips this run")
"""),
        md("## Summary"),
        code(r"""
print(pd.DataFrame(state["history"]).round(4).to_string(index=False) if state["history"] else "no finished epoch yet")
if SCORES.exists():
    s = pd.DataFrame([json.loads(l) for l in open(SCORES, encoding="utf-8") if l.strip()])
    s = s[(s["task"] == "fused") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()].merge(
        clips[["uid", "ds_label", "conflict", "label_T", "label_A", "label_V"]], on="uid")
    if len(s):
        q = np.exp(s[[f"logit_{l}" for l in LABELS]].to_numpy())
        s["pred"], s["entropy"] = np.array(LABELS)[q.argmax(1)], -(q * np.log(np.clip(q, 1e-12, None))).sum(1)
        for m in "TAV":
            s[f"pol_hit_{m}"] = np.stack(s[f"pol_{m}"]).argmax(1) == (np.sign(s[f"label_{m}"]).astype(int) + 1)
        print(s.groupby(["split", "conflict"]).agg(n=("uid", "size"), accuracy=("pred", lambda x: (x == s.loc[x.index, "ds_label"]).mean()),
                                                    entropy=("entropy", "mean"), pi=("pi", "mean"), pol_T=("pol_hit_T", "mean"),
                                                    pol_A=("pol_hit_A", "mean"), pol_V=("pol_hit_V", "mean")).round(3).to_string())
"""),
    ]


# ---- Stage 1 probe: hidden states of the RQ3 soft LoRA model and of the base model ---------

PROBE = [
    md(r"""
# Hidden states for the disagreement probe (Qwen2.5-Omni-7B, Kaggle, 2× T4)

Does the model hold more information about which clips the annotators disagree on than its output entropy
shows? This notebook reads every clip (train, valid, test) once under T+A+V with the prompt of the other
runs and saves, at the last position (the one the label is read from), the hidden state after several
layers of the language model. Each clip is read twice: with the RQ3 soft-label LoRA adapter (`A`) and without
it (`base`, the zero-shot model). The 7 label logits are saved too, to check against the earlier runs.
A linear probe is then trained on these states on a CPU (`analysis/probe_analysis.py`).

**Running it**
1. *Accelerator*: **GPU T4 x2**; *Internet*: **on**; *Add input*: the preprocessed `chsims-mm` dataset **and the
   output of the RQ3 soft-label notebook (seed 0)**, which holds `rq3_soft/best/adapter.pt`.
2. First set `LIMIT = 10` and run all interactively (about 5 minutes), then **Save & Run All (Commit)** with
   `LIMIT = None` (about 2-3 hours). If it stops at `TIME_BUDGET_H`, add this notebook's own output as an input
   and commit again: finished clips are skipped.
3. Results: `/kaggle/working/probe_hidden/`: `shard_*.npz` (uids, hidden states, label logits) and `meta.json`.
   About 350 MB; download the folder and copy it to `results/probe_hidden/`.
"""),
    code(r"""
# peft added for LoRA; the rest as in the other notebooks.
%pip install -q "transformers==5.20.1" "qwen-omni-utils==0.0.9" "peft==0.21.2" audioread librosa
# Kaggle ships torchao 0.10; peft 0.21 raises an ImportError for torchao < 0.16 even when it is not used.
%pip uninstall -y -q torchao
"""),
    code(COMMON_CONFIG + r"""
LORA_R, LORA_ALPHA, LORA_DROPOUT = 16, 32, 0.05  # as in the RQ3 notebooks
LORA_TARGETS = r"^model\.layers\.\d+\.self_attn\.(q|k|v|o)_proj$"
SHARD = 250                  # clips per saved shard
TIME_BUDGET_H = 11.0
LIMIT = None                 # e.g. 10 for a quick check: 10 clips per split
OUT_DIR = Path(os.environ.get("OUT_DIR", "/kaggle/working/probe_hidden"))
T_START = time.time()

ADAPTER = os.environ.get("ADAPTER")
if ADAPTER is None:
    hits = [h for k in range(6) for h in sorted(glob.glob("/kaggle/input/" + "*/" * k + "rq3_soft/best/adapter.pt"))]
    assert hits, "No rq3_soft/best/adapter.pt under /kaggle/input: attach the output of the RQ3 soft notebook (seed 0)."
    ADAPTER = hits[0]
print("adapter:", ADAPTER)
""" + COMMON_REPORT),
    md(r"""
## Load the model and the RQ3 soft adapter

Inference only: fp16, no gradient checkpointing. Forward hooks keep the last position of the residual stream
after each layer in `LAYERS` (a quarter, half, three quarters of the way, and the last layer) and of the final
norm (the input of the language-model head).
"""),
    code(r"""
from peft import LoraConfig, get_peft_model, set_peft_model_state_dict
from qwen_omni_utils import process_mm_info
from transformers import Qwen2_5OmniProcessor, Qwen2_5OmniThinkerForConditionalGeneration

ON_GPU = torch.cuda.is_available()
processor = Qwen2_5OmniProcessor.from_pretrained(MODEL_ID)
kwargs = dict(dtype=torch.float16 if ON_GPU else torch.float32, attn_implementation="sdpa")
if ON_GPU:
    kwargs.update(device_map="auto", max_memory=MAX_MEMORY)
t0 = time.time()
base = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(MODEL_ID, **kwargs).eval()
print(f"loaded in {time.time() - t0:.0f} s")
base.lm_head.register_forward_pre_hook(lambda module, args: (args[0][:, -1:],) + args[1:])
model = get_peft_model(base, LoraConfig(r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                                        target_modules=LORA_TARGETS, bias="none"))
result = set_peft_model_state_dict(model, torch.load(ADAPTER, map_location="cpu"))
assert not [k for k in result.unexpected_keys if "lora" in k], result.unexpected_keys
model.eval()

layers = base.model.layers
n = len(layers)
LAYERS = sorted({max(1, n // 4), max(1, n // 2), max(1, 3 * n // 4), n})  # hidden state after layer i (1-based)
HIDDEN = {}
for i in LAYERS:
    layers[i - 1].register_forward_hook(lambda mod, args, out, i=i: HIDDEN.__setitem__(
        f"layer{i}", (out[0] if isinstance(out, tuple) else out)[0, -1].float().cpu()))
base.model.norm.register_forward_hook(lambda mod, args, out: HIDDEN.__setitem__("norm", out[0, -1].float().cpu()))
print("layers:", n, "| saved:", LAYERS, "+ norm | hidden size:", base.config.text_config.hidden_size
      if hasattr(base.config, "text_config") else base.config.hidden_size)
"""),
    md("## Prompts (same as the other runs)"),
    PROMPTS,
    md("## Clips"),
    code(r"""
manifest = pd.read_csv(DATA_DIR / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
order = {"test": 0, "valid": 1, "train": 2}  # test first, so a short run still covers the evaluation split
clips = clips.sort_values("mode", key=lambda m: m.map(order), kind="stable").reset_index(drop=True)
if LIMIT:
    clips = clips.groupby("mode", sort=False).head(LIMIT).reset_index(drop=True)
print(clips["mode"].value_counts().to_dict())
"""),
    md(r"""
## Read every clip twice and save shards

Per clip and model (`A`, `base`): the hidden states and the 7 label logits. Shards found under `/kaggle/input`
(an earlier run of this notebook) or in `OUT_DIR` are kept and their clips skipped.
"""),
    code(r"""
for p in (h for k in range(6) for h in glob.glob("/kaggle/input/" + "*/" * k + "probe_hidden/shard_*.npz")):
    if not (OUT_DIR / Path(p).name).exists():
        shutil.copy(p, OUT_DIR / Path(p).name)
done = set()
for p in sorted(OUT_DIR.glob("shard_*.npz")):
    done |= set(np.load(p, allow_pickle=False)["uid"].tolist())
todo = clips[~clips["uid"].isin(done)]
print(f"{len(clips)} clips, {len(done & set(clips['uid']))} already done, {len(todo)} to run")


@torch.inference_mode()
def read(inputs):
    HIDDEN.clear()
    logits = model(**inputs, use_cache=False).logits[0, -1].float()
    return {k: v.numpy().astype(np.float16) for k, v in HIDDEN.items()}, logits[label_first_ids].cpu().numpy()


buf, n_oom, t_run = [], 0, time.time()


def flush():
    if not buf:
        return
    k = len(list(OUT_DIR.glob("shard_*.npz")))
    arrays = {"uid": np.array([b["uid"] for b in buf])}
    for key in buf[0]:
        if key != "uid":
            arrays[key] = np.stack([b[key] for b in buf])
    np.savez(OUT_DIR / f"shard_{k:03d}.npz", **arrays)
    buf.clear()


for j, (_, row) in enumerate(todo.iterrows()):
    if (time.time() - T_START) / 3600 > TIME_BUDGET_H:
        print("stopped at the time budget: add this notebook's output as an input and commit again")
        break
    try:
        inputs = prepare(build_messages(row, "TAV", ask="label"))
        rec = {"uid": row["uid"]}
        for name in ("A", "base"):
            if name == "A":
                hidden, logits = read(inputs)
            else:
                with model.disable_adapter():
                    hidden, logits = read(inputs)
            rec.update({f"{name}_{k}": v for k, v in hidden.items()})
            rec[f"{name}_logits"] = logits
        buf.append(rec)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        n_oom += 1
    if len(buf) == SHARD:
        flush()
    if j + 1 in (5, 50) or (j + 1) % 500 == 0:
        sec = (time.time() - t_run) / (j + 1)
        peak = max([torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())] or [0])
        print(f"{j + 1}/{len(todo)} | {sec:.2f} s/clip, about {(len(todo) - j - 1) * sec / 3600:.1f} h left | peak {peak:.1f} GiB | OOM {n_oom}")
flush()
(OUT_DIR / "meta.json").write_text(json.dumps({"model": MODEL_ID, "adapter": ADAPTER, "layers": LAYERS, "labels": LABELS,
                                               "oom_this_run": n_oom}, indent=1))
"""),
    md("## Check"),
    code(r"""
total = {}
for p in sorted(OUT_DIR.glob("shard_*.npz")):
    z = np.load(p, allow_pickle=False)
    for u in z["uid"].tolist():
        total[u] = True
    keys = [k for k in z.files if k != "uid"]
print(f"{len(total)} of {len(clips)} clips saved | arrays per clip:", {k: z[k].shape[1:] for k in keys})
print(clips[clips["uid"].isin(total)]["mode"].value_counts().to_dict())
"""),
]


if __name__ == "__main__":
    write(HERE / "benchmark_qwen25_omni.ipynb", BENCHMARK)
    write(HERE / "run_qwen25_omni.ipynb", RUN)
    write(HERE / "run_minicpm_o.ipynb", MINICPM)
    write(HERE / "run_gemma4.ipynb", GEMMA4)
    write(HERE / "run_phi4mm.ipynb", PHI4MM)
    write(HERE / "run_emotion_llama.ipynb", EMOLLAMA)
    for arm in ("hard", "soft"):
        write(HERE / f"rq3_lora_{arm}.ipynb", rq3_cells(arm))
    for variant in ("B", "C", "D"):
        write(HERE / f"rq4_ci_{variant}.ipynb", rq4_cells(variant))
    write(HERE / "probe_hidden.ipynb", PROBE)
