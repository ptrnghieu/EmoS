"""Extract frames and 16 kHz audio from CH-SIMS v2(s) clips and write a manifest for model runs.

Input:  the CH-SIMS v2(s) "Raw" folder (Raw/<video_id>/<clip_id>.mp4), downloaded from the
        official links in https://github.com/thuiar/ch-sims-v2.
Output: <out>/frames/<video_id>/<clip_id>/NNN.jpg   (at most --max-frames per clip)
        <out>/audio/<video_id>/<clip_id>.wav        (mono, 16 kHz, 16-bit PCM)
        <out>/manifest.csv                          (one row per clip: labels, conflict, paths)

Paths in the manifest are relative to <out>, so the folder can be uploaded as a (private)
Kaggle dataset as is. Re-running skips clips whose outputs already exist.

Usage (from the repository root, needs ffmpeg/ffprobe on PATH):
    python analysis/preprocess_chsims.py --raw-dir /data/ch-simsv2s/Raw --out-dir /data/chsims_mm
    python analysis/preprocess_chsims.py ... --limit 20   # quick check on a few clips
"""

import argparse
import json
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pandas as pd

from chsims_emos import load_joined


def find_clip(raw_dir: Path, video_id: str, clip_id: str) -> Path | None:
    # One clip ("00026_0") is stored without the underscore ("000260.mp4").
    for name in (clip_id, clip_id.replace("_", "")):
        path = raw_dir / video_id / f"{name}.mp4"
        if path.exists():
            return path
    return None


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


def extract_audio(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    tmp = dst.with_name(dst.stem + ".tmp.wav")
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000",
                    "-c:a", "pcm_s16le", str(tmp)], check=True)
    tmp.rename(dst)


def process_clip(row: dict, raw_dir: Path, out_dir: Path, args: argparse.Namespace) -> dict:
    frames_rel = Path("frames") / row["video_id"] / row["clip_id"]
    audio_rel = Path("audio") / row["video_id"] / f"{row['clip_id']}.wav"
    result = {"uid": row["uid"], "frames_dir": str(frames_rel), "audio_path": "", "n_frames": 0,
              "sample_fps": 0.0, "duration": 0.0, "has_audio": False, "status": "ok"}
    src = find_clip(raw_dir, row["video_id"], row["clip_id"])
    if src is None:
        result["status"] = "missing_video"
        return result
    try:
        result["duration"], result["has_audio"] = probe(src)
        frames_dir = out_dir / frames_rel
        existing = sorted(frames_dir.glob("*.jpg")) if frames_dir.exists() else []
        if existing:
            result["n_frames"] = len(existing)
            result["sample_fps"] = min(args.fps, args.max_frames / result["duration"]) if result["duration"] else args.fps
        else:
            result["n_frames"], result["sample_fps"] = extract_frames(
                src, frames_dir, args.fps, args.max_frames, args.max_side, result["duration"])
        if result["has_audio"]:
            if not (out_dir / audio_rel).exists():
                extract_audio(src, out_dir / audio_rel)
            result["audio_path"] = str(audio_rel)
        else:
            result["status"] = "no_audio"
    except subprocess.CalledProcessError as err:
        result["status"] = f"ffmpeg_error: {err.stderr or err}"[:200]
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--raw-dir", type=Path, required=True, help="CH-SIMS v2(s) Raw folder")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--fps", type=float, default=2.0, help="target sampling rate (frames per second)")
    parser.add_argument("--max-frames", type=int, default=16, help="cap on frames per clip")
    parser.add_argument("--max-side", type=int, default=448, help="longest side of saved frames, in pixels")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N clips")
    args = parser.parse_args()

    clips = load_joined()
    if args.limit:
        clips = clips.head(args.limit)
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = clips.to_dict("records")
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: process_clip(r, args.raw_dir, args.out_dir, args), rows))

    manifest = clips.merge(pd.DataFrame(results), on="uid", how="left")
    manifest.to_csv(args.out_dir / "manifest.csv", index=False)
    print(manifest["status"].value_counts().to_string())
    print(f"frames per clip: median {manifest['n_frames'].median():.0f}, max {manifest['n_frames'].max()}")
    print(f"wrote {args.out_dir / 'manifest.csv'} ({len(manifest)} clips)")


if __name__ == "__main__":
    main()
