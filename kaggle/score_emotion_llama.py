"""Emotion-LLaMA label scoring on CH-SIMS v2, inside the repository's own Python 3.9 environment.

Run by kaggle/run_emotion_llama.ipynb (which also builds the environment and downloads the weights).
The model is built with the authors' code (`minigpt4`, demo config) and scored like the other RQ2 runs:
one forward pass, logits of the first token of each label, written as JSON lines.

How the inputs follow the authors' code:
- Image: the clip's first extracted frame through the model's eval image processor (448 × 448), as in the
  demo, which takes the first frame of the video.
- Audio: HuBERT-large (Chinese) last hidden layer, averaged over time, as in the demo.
- FaceMAE and VideoMAE features: zeros, as in the demo (the authors publish no extraction for raw clips).
- Prompt: the evaluation instruction of the training dataset code,
  `<video><VideoHere></video> <feature><FeatureHere></feature> The person in video says: {text}.  [emotion] Please
  determine which emotion label in the video represents: {labels}. ` in the Llama-2 chat wrapper
  `<s>[INST] ... [/INST]`, embedded with the model's `get_context_emb`.
- Labels: the checkpoint's six, in the order of its training instruction (happy, sad, neutral, angry, worried,
  surprise); "worried" is read as Fear, the alias the authors use for zero-shot evaluation. There is no Disgust.

Conditions the model was not trained on (A, T, V) are built as close to the trained input as possible:
- V: image, audio feature set to zeros, no transcript line.
- A: no image tokens (the 256 patch tokens and the class token are left out), only the three feature tokens.
- T: no media placeholders, only the transcript line and the question.
"""

import argparse
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf
import torch
from PIL import Image

LABELS = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]  # record format of the other runs
MODEL_LABELS = ["happy", "sad", "neutral", "angry", "worried", "surprise"]  # training instruction order
AS_LABEL = {"happy": "Joy", "sad": "Sadness", "neutral": "Neutral", "angry": "Anger", "worried": "Fear",
            "surprise": "Surprise"}
CONDITIONS = ["TAV", "T", "A", "V"]
INSTRUCTION = "Please determine which emotion label in the video represents: {}.".format(", ".join(MODEL_LABELS))
SR = 16000


def parse_args():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--repo", required=True, help="Emotion-LLaMA repository checkout")
    p.add_argument("--data-dir", required=True, help="folder with manifest.csv, frames/ and audio/")
    p.add_argument("--out-dir", required=True)
    p.add_argument("--llama", required=True, help="Llama-2-7b-chat-hf folder")
    p.add_argument("--ckpt", required=True, help="Emotion-LLaMA checkpoint (.pth)")
    p.add_argument("--hubert", required=True, help="chinese-hubert-large folder")
    p.add_argument("--model-name", default="ZebangCheng/Emotion-LLaMA (Emoation_LLaMA.pth)")
    p.add_argument("--resume-glob", default="", help="earlier scores.jsonl files to copy in, e.g. /kaggle/input/**/emollama_runs/scores.jsonl")
    p.add_argument("--split-layer", type=int, default=14, help="Llama layers below this on cuda:0, the rest on cuda:1")
    p.add_argument("--limit", type=int, default=0, help="e.g. 10: 10 test clips + 10 other clips")
    p.add_argument("--time-budget-h", type=float, default=11.0)
    p.add_argument("--cpu-test", action="store_true", help="local check on CPU: fp32 everywhere")
    return p.parse_args()


args = parse_args()
T_START = time.time()
sys.path.insert(0, args.repo)
os.chdir(args.repo)

from minigpt4.common.config import Config  # noqa: E402
from minigpt4.common.registry import registry  # noqa: E402
from minigpt4.datasets.builders import *  # noqa: E402,F401,F403  (registers the builders, as app.py does)
from minigpt4.models import *  # noqa: E402,F401,F403
from minigpt4.processors import *  # noqa: E402,F401,F403
from minigpt4.tasks import *  # noqa: E402,F401,F403
from transformers import HubertModel, Wav2Vec2FeatureExtractor  # noqa: E402

# ---- model ------------------------------------------------------------------------------

options = [f"model.ckpt={args.ckpt}", f"model.llama_model={args.llama}", "model.low_resource=False"]
if args.cpu_test:
    options.append("model.vit_precision=fp32")  # fp16 convolutions do not run on the CPU
cfg = Config(argparse.Namespace(cfg_path="eval_configs/demo.yaml", options=options))
model_cfg = cfg.model_cfg
t0 = time.time()
model = registry.get_model_class(model_cfg.arch).from_config(model_cfg).eval()
vis_cfg = cfg.datasets_cfg.feature_face_caption.vis_processor.train
vis_processor = registry.get_processor_class(vis_cfg.name).from_config(vis_cfg)

# from_config loads the checkpoint with strict=False, so check that every tensor of it really arrived
ckpt = torch.load(args.ckpt, map_location="cpu")["model"]
own = model.state_dict()
n_layers = model.llama_model.base_model.model.config.num_hidden_layers
expected = [k for k in ckpt if ".layers." not in k or int(k.split(".layers.")[1].split(".")[0]) < n_layers]
missing = [k for k in expected if k not in own]
differ = [k for k in expected if k in own and not torch.equal(own[k].float(), ckpt[k].float())]
print(f"checkpoint: {len(expected)} tensors expected, {len(missing)} missing, {len(differ)} different", flush=True)
assert not missing and not differ, (missing[:5], differ[:5])
del ckpt, own

if args.cpu_test:
    model.llama_model.float()  # fp16 matmuls do not run on the CPU in torch 2.0
    inner = model.llama_model.base_model.model
    inner.lm_head = inner.lm_head[0]  # peft's output wrapper casts its input back to fp16
    DEVICE = AUDIO_DEVICE = torch.device("cpu")
else:
    from accelerate import dispatch_model

    DEVICE, AUDIO_DEVICE = torch.device("cuda:0"), torch.device("cuda:1")
    llama = model.llama_model.base_model.model  # LlamaForCausalLM inside the LoRA wrapper
    device_map = {"model.embed_tokens": 0, "model.norm": 1, "lm_head": 1}
    device_map.update({f"model.layers.{i}": 0 if i < args.split_layer else 1 for i in range(n_layers)})
    dispatch_model(llama, device_map=device_map)
    for name in ("visual_encoder", "ln_vision", "llama_proj", "feats_llama_proj1", "feats_llama_proj2",
                 "feats_llama_proj3", "cls_tk_llama_proj", "attention"):
        getattr(model, name).to(DEVICE)
assert model.device == DEVICE, model.device  # maybe_autocast and get_context_emb follow this device

feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained(args.hubert)
hubert = HubertModel.from_pretrained(args.hubert).eval().to(AUDIO_DEVICE)  # fp32, as in the demo
tokenizer = model.llama_tokenizer
label_first_ids = [tokenizer(w, add_special_tokens=False).input_ids[0] for w in MODEL_LABELS]
assert len(set(label_first_ids)) == len(MODEL_LABELS), label_first_ids
print(f"loaded in {time.time() - t0:.0f} s | label first tokens:",
      {w: tokenizer.convert_ids_to_tokens(i) for w, i in zip(MODEL_LABELS, label_first_ids)}, flush=True)
for i in range(torch.cuda.device_count()):
    print(f"cuda:{i} allocated after load: {torch.cuda.memory_allocated(i) / 2**30:.1f} GiB", flush=True)


# ---- inputs -----------------------------------------------------------------------------

@torch.no_grad()
def audio_feature(row):
    samples, sr = sf.read(str(Path(args.data_dir) / row["audio_path"]), dtype="float32")
    assert sr == SR and samples.ndim == 1, (sr, samples.shape)
    input_values = feature_extractor(samples, sampling_rate=sr, return_tensors="pt").input_values.to(AUDIO_DEVICE)
    hidden = hubert(input_values, output_hidden_states=True).hidden_states
    feat = torch.stack(hidden)[[-1]].sum(dim=0)  # last layer, (1, T, 1024): the demo's own lines
    return feat.mean(dim=1, keepdim=True).cpu()  # (1, 1, 1024)


def first_frame(row):
    frames = sorted(glob.glob(str(Path(args.data_dir) / row["frames_dir"] / "*.jpg")))
    return vis_processor(Image.open(frames[0]).convert("RGB")).unsqueeze(0).to(DEVICE)


def feature_tokens(video_features):
    # the three feature tokens exactly as in MiniGPTv2.encode_img, without the image tokens
    with model.maybe_autocast():
        f = video_features.to(DEVICE)
        split = torch.split(f, 1, dim=1)
        return torch.stack([model.feats_llama_proj1(split[0].squeeze(1)), model.feats_llama_proj2(split[1].squeeze(1)),
                            model.feats_llama_proj3(split[2].squeeze(1))], dim=1)


def instruction(row, condition):
    line = f"The person in video says: {row['text']}. " if "T" in condition else ""
    media = "<video><VideoHere></video> <feature><FeatureHere></feature> " if condition != "T" else ""
    return "<s>[INST] " + media + "{} [{}] {} ".format(line, "emotion", INSTRUCTION) + " [/INST]"


@torch.no_grad()
def context_embedding(row, condition):
    text = instruction(row, condition)
    if condition == "T":  # no placeholder: the same tokenization as get_context_emb's first segment
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True).input_ids.to(DEVICE)
        return model.embed_tokens(ids), text
    zeros = torch.zeros([1, 2, 1024])  # FaceMAE and VideoMAE, zero as in the demo
    audio = audio_feature(row) if "A" in condition else torch.zeros([1, 1, 1024])
    video_features = torch.cat((zeros, audio), dim=1)
    if "V" in condition:
        media, _ = model.encode_img(first_frame(row), video_features)
    else:
        media = feature_tokens(video_features)
    return model.get_context_emb(text, [media]), text


@torch.no_grad()
def label_logits(embs):
    with model.maybe_autocast():
        out = model.llama_model(inputs_embeds=embs, use_cache=False)
    return out.logits[0, -1].float()


# ---- run --------------------------------------------------------------------------------

out_dir = Path(args.out_dir)
out_dir.mkdir(parents=True, exist_ok=True)
SCORES = out_dir / "scores.jsonl"
manifest = pd.read_csv(Path(args.data_dir) / "manifest.csv", dtype={"video_id": str, "clip_id": str})
clips = manifest[manifest["status"] == "ok"].reset_index(drop=True)
if args.limit:
    clips = pd.concat([clips[clips["mode"] == "test"].head(args.limit), clips[clips["mode"] != "test"].head(args.limit)])
print(len(clips), "clips to score", flush=True)

row0 = clips.iloc[0]
print("prompt (T+A+V):", repr(instruction(row0, "TAV")), flush=True)
print("prompt (T):", repr(instruction(row0, "T")), flush=True)

# the T+A+V and V prompts must equal what the authors' evaluation builds (dataset format + conversation template)
from minigpt4.conversation.conversation import CONV_VISION_minigptv2  # noqa: E402
from minigpt4.evaluation.prompting import prepare_conversation_texts  # noqa: E402

conv = CONV_VISION_minigptv2.copy()
conv.system = ""
for condition, line in (("TAV", f"The person in video says: {row0['text']}. "), ("V", "")):
    dataset_text = "<video><VideoHere></video> <feature><FeatureHere></feature> {} [{}] {} ".format(line, "emotion", INSTRUCTION)
    assert prepare_conversation_texts([dataset_text], conv)[0] == instruction(row0, condition), condition

# the label read from the logits must match the first word of the authors' own greedy answer (T+A+V)
for _, row in clips.head(3).iterrows():
    embs, _ = context_embedding(row, "TAV")
    logits = label_logits(embs)
    video_features = torch.cat((torch.zeros([1, 2, 1024]), audio_feature(row)), dim=1)
    answer = model.generate(first_frame(row), video_features, [instruction(row, "TAV")], max_new_tokens=8)[0]
    print(f"{row['uid']}: authors' generate -> {answer!r} | top token {tokenizer.convert_ids_to_tokens(int(logits.argmax()))!r} | "
          f"top of the six: {MODEL_LABELS[int(logits[label_first_ids].argmax())]}", flush=True)


def key(rec):
    return (rec["model"], rec["uid"], rec["condition"], rec["task"])


sources = sorted(set(glob.glob(args.resume_glob, recursive=True) if args.resume_glob else []) | ({str(SCORES)} if SCORES.exists() else set()))
records = {}
for src in sources:
    for line in open(src, encoding="utf-8"):
        if line.strip() and "error" not in (rec := json.loads(line)):
            records[key(rec)] = rec
with open(SCORES, "w", encoding="utf-8") as f:
    for rec in records.values():
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
done = set(records)
print(f"{len(records)} finished records from {len(sources)} file(s)", flush=True)

jobs = [(row, c) for c in CONDITIONS for _, row in clips.iterrows()]
todo = [j for j in jobs if (args.model_name, j[0]["uid"], j[1], "score") not in done]
print(f"{len(jobs)} items, {len(jobs) - len(todo)} already done, {len(todo)} to run", flush=True)

n_done, stopped_early, t_run = 0, False, time.time()
for row, condition in todo:
    if (time.time() - T_START) / 3600 > args.time_budget_h:
        stopped_early = True
        break
    rec = {"model": args.model_name, "uid": row["uid"], "condition": condition, "task": "score"}
    try:
        t0 = time.perf_counter()
        embs, _ = context_embedding(row, condition)
        t1 = time.perf_counter()
        logits = label_logits(embs)
        rec.update({"n_tokens": int(embs.shape[1]), "prep_s": round(t1 - t0, 3), "gpu_s": round(time.perf_counter() - t1, 3),
                    "finite": bool(torch.isfinite(logits).all()), "lse": float(torch.logsumexp(logits, dim=-1))})
        rec.update({f"logit_{l}": None for l in LABELS})
        rec.update({f"logit_{AS_LABEL[w]}": float(logits[i]) for w, i in zip(MODEL_LABELS, label_first_ids)})
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        rec["error"] = "oom"
    with open(SCORES, "a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    n_done += 1
    if n_done in (20, 100) or n_done % 500 == 0:
        sec = (time.time() - t_run) / n_done
        peak = max([torch.cuda.max_memory_allocated(d) / 2**30 for d in range(torch.cuda.device_count())] or [0])
        print(f"{n_done}/{len(todo)} | {sec:.2f} s/item, about {(len(todo) - n_done) * sec / 3600:.1f} h left | "
              f"peak {peak:.1f} GiB | last: {condition}", flush=True)

print(f"finished {n_done} items this run", "(stopped at the time budget: commit again to continue)" if stopped_early else "", flush=True)
