"""Download CREMA-D, extract frames and 16 kHz audio, and write a manifest with the raters' votes per condition.

CREMA-D (Cao et al., 2014; https://github.com/CheyneyComputerScience/CREMA-D, ODbL 1.0): 7,442 clips of 91 actors
reading 12 fixed sentences, each rated by about 10 people in each of three conditions: audio only, video only,
audio-visual (different people per condition). Six labels: Anger, Disgust, Fear, Happy, Neutral, Sad.

Media come from the repository's Git LFS store through media.githubusercontent.com (no clone needed); the
per-rater responses (finishedResponses.csv) and actor table from raw.githubusercontent.com.

Output (paths relative to <out>, same layout as preprocess_chsims.py so the run notebooks can read it):
    <out>/frames/<actor>/<clip>/NNN.jpg     frames from the presentation video (VideoFlash/*.flv)
    <out>/audio/<actor>/<clip>.wav          mono 16 kHz from AudioWAV/*.wav
    <out>/manifest.csv                      one row per clip: actor, sentence, intended emotion and level,
                                            split (by actor), votes per condition, paths, status

Votes: `{cond}_{label}` counts and `{cond}_n` raters for cond in a (audio only), v (video only), av (audio-visual);
labels in LABELS order. Split: actors shuffled with a fixed seed, 70/10/20 % of actors to train/valid/test,
stratified by sex. Clips the dataset notes as broken are kept with status `known_bad`.

Usage (needs ffmpeg/ffprobe on PATH):
    python analysis/preprocess_cremad.py --out-dir /data/cremad_mm
    python analysis/preprocess_cremad.py --out-dir /tmp/cremad_mm --limit 20   # quick check
"""

from __future__ import annotations

import argparse
import io
import json
import shutil
import subprocess
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "CheyneyComputerScience/CREMA-D"
MEDIA = f"https://media.githubusercontent.com/media/{REPO}/master"
RAW = f"https://raw.githubusercontent.com/{REPO}/master"
LABELS = ["Anger", "Disgust", "Fear", "Happy", "Neutral", "Sad"]
CODE = {"A": "Anger", "D": "Disgust", "F": "Fear", "H": "Happy", "N": "Neutral", "S": "Sad"}
CONDITIONS = {1: "a", 2: "v", 3: "av"}  # queryType in finishedResponses.csv
SENTENCES = {"IEO": "It's eleven o'clock.", "TIE": "That is exactly what happened.", "IOM": "I'm on my way to the meeting.",
             "IWW": "I wonder what this is about.", "TAI": "The airplane is almost full.",
             "MTI": "Maybe tomorrow it will be cold.", "IWL": "I would like a new alarm clock.",
             "ITH": "I think I have a doctor's appointment.", "DFA": "Don't forget a jacket.",
             "ITS": "I think I've seen this before.", "TSI": "The surface is slick.",
             "WSI": "We'll stop in a couple of minutes."}
INTENDED = {"ANG": "Anger", "DIS": "Disgust", "FEA": "Fear", "HAP": "Happy", "NEU": "Neutral", "SAD": "Sad"}
LEVEL = {"LO": "low", "MD": "medium", "HI": "high", "XX": "unspecified"}
# probe and extract_frames as in preprocess_chsims.py (copied so this file runs on its own, e.g. on Kaggle)


def probe(path: Path) -> tuple[float, bool]:
    """Return (duration in seconds, whether the file has an audio stream)."""
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration:stream=codec_type", "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    )
    info = json.loads(out.stdout)
    duration = float(info.get("format", {}).get("duration", 0.0))
    has_audio = any(s.get("codec_type") == "audio" for s in info.get("streams", []))
    return duration, has_audio


def extract_frames(src: Path, dst: Path, fps: float, max_frames: int, max_side: int, duration: float) -> tuple[int, float]:
    """Sample frames at `fps`, lowered so a clip never yields more than `max_frames`."""
    if duration > 0:
        fps = min(fps, max_frames / duration)
    tmp = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    scale = f"scale={max_side}:{max_side}:force_original_aspect_ratio=decrease"
    cmd = ["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vf", f"fps={fps:.4f},{scale}", "-q:v", "3", str(tmp / "%03d.jpg")]
    subprocess.run(cmd, check=True)
    frames = sorted(tmp.glob("*.jpg"))
    if not frames:  # clip shorter than one sampling interval: keep its first frame
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vf", scale, "-frames:v", "1",
                        "-q:v", "3", str(tmp / "001.jpg")], check=True)
        frames = sorted(tmp.glob("*.jpg"))
    for extra in frames[max_frames:]:
        extra.unlink()
    shutil.rmtree(dst, ignore_errors=True)
    tmp.rename(dst)
    return min(len(frames), max_frames), fps


KNOWN_BAD = {"1076_MTI_NEU_XX", "1076_MTI_SAD_XX", "1064_TIE_SAD_XX", "1064_IEO_DIS_MD"}  # per the dataset README


def fetch(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 ** k)


def votes_table(responses: pd.DataFrame) -> pd.DataFrame:
    r = responses.drop_duplicates(["localid", "clipName", "queryType"])
    r = r[r.respEmo.isin(CODE)]
    counts = r.assign(label=r.respEmo.map(CODE)).groupby(["clipName", "queryType", "label"]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=LABELS, fill_value=0)
    parts = []
    for q, cond in CONDITIONS.items():
        c = counts.xs(q, level="queryType")
        c.columns = [f"{cond}_{l}" for l in LABELS]
        c[f"{cond}_n"] = c.sum(axis=1)
        parts.append(c)
    return pd.concat(parts, axis=1).fillna(0).astype(int)


def split_actors(actors: pd.DataFrame, seed: int = 0) -> dict[int, str]:
    rng = np.random.default_rng(seed)
    out = {}
    for _, group in actors.groupby("Sex"):
        ids = rng.permutation(group.ActorID.to_numpy())
        n_train, n_valid = round(0.7 * len(ids)), round(0.1 * len(ids))
        out.update({int(a): "train" for a in ids[:n_train]})
        out.update({int(a): "valid" for a in ids[n_train:n_train + n_valid]})
        out.update({int(a): "test" for a in ids[n_train + n_valid:]})
    return out


def clip_table() -> pd.DataFrame:
    """One row per clip: actor, sentence, intended emotion and level, split by actor, votes per condition."""
    responses = pd.read_csv(io.BytesIO(fetch(f"{RAW}/finishedResponses.csv")), low_memory=False)
    actors = pd.read_csv(io.BytesIO(fetch(f"{RAW}/VideoDemographics.csv")))
    names = pd.read_csv(io.BytesIO(fetch(f"{RAW}/SentenceFilenames.csv")))["Filename"]

    clips = pd.DataFrame({"uid": names})
    parts = clips.uid.str.split("_", expand=True)
    clips["actor"] = parts[0].astype(int)
    clips["sentence_code"], clips["text"] = parts[1], parts[1].map(SENTENCES)
    clips["intended"], clips["level"] = parts[2].map(INTENDED), parts[3].map(LEVEL)
    clips["video_id"] = clips.actor.astype(str)  # grouping key for video-level bootstraps (one actor = one "video")
    clips["mode"] = clips.actor.map(split_actors(actors))
    clips = clips.merge(actors.rename(columns={"ActorID": "actor"}), on="actor", how="left")
    return clips.merge(votes_table(responses), left_on="uid", right_index=True, how="left")


def process_clip(row: dict, out_dir: Path, args: argparse.Namespace) -> dict:
    clip, actor = row["uid"], str(row["actor"])
    frames_rel, audio_rel = Path("frames") / actor / clip, Path("audio") / actor / f"{clip}.wav"
    result = {"uid": clip, "frames_dir": str(frames_rel), "audio_path": str(audio_rel), "n_frames": 0,
              "sample_fps": 0.0, "duration": 0.0, "status": "known_bad" if clip in KNOWN_BAD else "ok"}
    try:
        audio = out_dir / audio_rel
        if not audio.exists():
            audio.parent.mkdir(parents=True, exist_ok=True)
            tmp = audio.with_name(audio.stem + ".dl.wav")
            tmp.write_bytes(fetch(f"{MEDIA}/AudioWAV/{clip}.wav"))
            subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(tmp), "-ac", "1", "-ar", "16000",
                            "-c:a", "pcm_s16le", str(audio)], check=True, capture_output=True, text=True)
            tmp.unlink()
        frames_dir = out_dir / frames_rel
        existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        video = out_dir / "tmp_video" / f"{clip}.flv"
        if not existing or not (out_dir / frames_rel / "duration.txt").exists():
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(fetch(f"{MEDIA}/VideoFlash/{clip}.flv"))
            duration, _ = probe(video)
            result["n_frames"], result["sample_fps"] = extract_frames(
                video, frames_dir, args.fps, args.max_frames, args.max_side, duration)
            (frames_dir / "duration.txt").write_text(f"{duration} {result['sample_fps']}")
            video.unlink()
            result["duration"] = duration
        else:
            duration, fps = map(float, (frames_dir / "duration.txt").read_text().split())
            result.update(n_frames=len(existing), duration=duration, sample_fps=fps)
        if result["n_frames"] == 0 and result["status"] == "ok":
            result["status"] = "no_frames"
    except Exception as err:  # network or ffmpeg error: recorded, the clip is retried on the next run
        result["status"] = f"error: {getattr(err, 'stderr', None) or err}"[:200]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0, help="target sampling rate (frames per second)")
    parser.add_argument("--max-frames", type=int, default=16, help="cap on frames per clip")
    parser.add_argument("--max-side", type=int, default=448, help="longest side of saved frames, in pixels")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N clips")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    clips = clip_table()
    if args.limit:
        clips = clips.groupby("mode").head(max(1, args.limit // 3)).reset_index(drop=True)

    rows = clips.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: process_clip(r, args.out_dir, args), rows))
    manifest = clips.merge(pd.DataFrame(results), on="uid", how="left")
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    print(manifest["status"].str.slice(0, 40).value_counts().to_string())
    print(manifest.groupby("mode").agg(actors=("actor", "nunique"), clips=("uid", "size")).to_string())
    print(f"frames per clip: median {manifest['n_frames'].median():.0f}, max {manifest['n_frames'].max()} | "
          f"duration median {manifest['duration'].median():.2f} s | raters per condition (median): "
          f"{ {c: int(manifest[f'{c}_n'].median()) for c in CONDITIONS.values()} }")
    print(f"wrote {args.out_dir / 'manifest.csv'} ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
