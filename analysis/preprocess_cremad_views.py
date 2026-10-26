"""Extra visual views of CREMA-D for the input ablation of the visual probe (experiment A, folded into B).

The original frames (preprocess_cremad.py, view A0: 2 fps, longest side 448) stay as they are. This script adds:
    A1  frames_4fps/<actor>/<clip>/NNN.jpg   4 fps (twice the frames), longest side 448, same cap of 16 frames
    A2  faces/<actor>/<clip>/NNN.jpg         face crops at A0's timestamps (same sampling rate per clip), cut from
                                             the source frames (480x360) and resized to FACE_SIDE x FACE_SIDE
A3 (full frames + face crops) needs no new files: the run notebook passes A0's frames and A2's crops as two videos.

Face crops. OpenCV's frontal-face Haar cascade on every frame; the largest face per frame; one box per clip (median of
the per-frame boxes, so the crop does not jitter between frames), widened by MARGIN around its centre to keep chin and
forehead, made square, and shifted to stay inside the frame. Clips with no detected face in any frame get a central
square crop and status `face_fallback`. FACE_SIDE = 336, so a crop is 12 x 12 patches of 28 pixels after the model's
resizing, about the same number of visual tokens as a full frame of A0 (a 448x336 frame under the 448*252 pixel cap):
the face gets the token budget the whole frame had. The source video is 480x360, so the crop is upsampled; it adds
tokens per face, not new pixel detail.

Output (paths relative to <out>): the folders above, views.csv (one row per clip: paths, frame counts, rates, face
box, frames with a detected face, status) and face_samples.jpg (48 random crops, for a visual check).

Usage (needs ffmpeg/ffprobe on PATH and opencv-python; downloads the videos again):
    python analysis/preprocess_cremad_views.py --manifest /data/cremad_mm/manifest.csv --out-dir /data/cremad_views
"""

from __future__ import annotations

import argparse
import shutil
import subprocess
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from preprocess_cremad import MEDIA, extract_frames, fetch, probe

FACE_SIDE = 336
MARGIN = 1.6
FPS_A1 = 4.0
MAX_FRAMES, MAX_SIDE = 16, 448  # as for A0
CASCADE = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")


def face_box(frames: list[np.ndarray]) -> tuple[tuple[int, int, int], int]:
    """(centre x, centre y, side) of the square crop for a clip, and the number of frames with a detected face."""
    h, w = frames[0].shape[:2]
    found = []
    for f in frames:
        gray = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
        faces = CASCADE.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(h // 6, h // 6))
        if len(faces):
            x, y, bw, bh = max(faces, key=lambda b: b[2] * b[3])
            found.append((x + bw / 2, y + bh / 2, max(bw, bh)))
    if found:
        cx, cy, side = np.median(np.array(found), 0)
    else:
        cx, cy, side = w / 2, h / 2, 0.6 * h / MARGIN
    side = int(round(min(MARGIN * side, h, w)))
    cx = int(round(min(max(cx, side / 2), w - side / 2)))
    cy = int(round(min(max(cy, side / 2), h - side / 2)))
    return (cx, cy, side), len(found)


def crop(frame: np.ndarray, box: tuple[int, int, int]) -> np.ndarray:
    cx, cy, side = box
    x0, y0 = cx - side // 2, cy - side // 2
    return cv2.resize(frame[y0:y0 + side, x0:x0 + side], (FACE_SIDE, FACE_SIDE), interpolation=cv2.INTER_CUBIC)


def faces_at(video: Path, dst: Path, fps: float) -> tuple[int, tuple[int, int, int], int]:
    """Source-resolution frames at `fps` (A0's rate for this clip), cropped to one face box per clip."""
    tmp = dst.with_name(dst.name + ".src")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-vf", f"fps={fps:.4f}",
                    "-q:v", "2", str(tmp / "%03d.png")], check=True)
    paths = sorted(tmp.glob("*.png"))[:MAX_FRAMES]
    if not paths:  # clip shorter than one sampling interval: its first frame, as in extract_frames
        subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-i", str(video), "-frames:v", "1",
                        str(tmp / "001.png")], check=True)
        paths = sorted(tmp.glob("*.png"))
    frames = [cv2.imread(str(p)) for p in paths]
    box, n_found = face_box(frames)
    out = dst.with_name(dst.name + ".tmp")
    shutil.rmtree(out, ignore_errors=True)
    out.mkdir(parents=True)
    for k, f in enumerate(frames, 1):
        cv2.imwrite(str(out / f"{k:03d}.jpg"), crop(f, box), [cv2.IMWRITE_JPEG_QUALITY, 92])
    shutil.rmtree(dst, ignore_errors=True)
    out.rename(dst)
    shutil.rmtree(tmp)
    return len(frames), box, n_found


def process_clip(row: dict, out_dir: Path) -> dict:
    clip, actor = row["uid"], str(row["actor"])
    rel4, relf = Path("frames_4fps") / actor / clip, Path("faces") / actor / clip
    result = {"uid": clip, "frames4_dir": str(rel4), "faces_dir": str(relf), "status": "ok"}
    done = out_dir / relf / "box.txt"
    try:
        if done.exists():  # finished in an earlier run
            n4, fps4, nf, cx, cy, side, found = done.read_text().split()
        else:
            video = out_dir / "tmp_video" / f"{clip}.flv"
            video.parent.mkdir(parents=True, exist_ok=True)
            video.write_bytes(fetch(f"{MEDIA}/VideoFlash/{clip}.flv"))
            duration, _ = probe(video)
            n4, fps4 = extract_frames(video, out_dir / rel4, FPS_A1, MAX_FRAMES, MAX_SIDE, duration)
            nf, (cx, cy, side), found = faces_at(video, out_dir / relf, float(row["sample_fps"]))
            video.unlink()
            done.write_text(f"{n4} {fps4} {nf} {cx} {cy} {side} {found}")
        result.update(n_frames4=int(n4), sample_fps4=float(fps4), n_faces=int(nf), face_box=f"{cx} {cy} {side}",
                      face_frames_detected=int(found))
        if int(nf) != int(row["n_frames"]):
            result["status"] = f"frame_count_mismatch: {nf} vs {row['n_frames']}"
        elif int(found) == 0:
            result["status"] = "face_fallback"
    except Exception as err:  # network or ffmpeg error: recorded, the clip is retried on the next run
        result["status"] = f"error: {getattr(err, 'stderr', None) or err}"[:200]
    return result


def contact_sheet(out_dir: Path, views: pd.DataFrame, n: int = 48, seed: int = 0) -> None:
    ok = views[views.status.isin(["ok", "face_fallback"])]
    pick = ok.sample(min(n, len(ok)), random_state=seed)
    tiles = []
    for _, r in pick.iterrows():
        frames = sorted((out_dir / r.faces_dir).glob("*.jpg"))
        img = cv2.resize(cv2.imread(str(frames[len(frames) // 2])), (112, 112))
        if r.status == "face_fallback":
            cv2.rectangle(img, (0, 0), (111, 111), (0, 0, 255), 3)
        tiles.append(img)
    tiles += [np.zeros_like(tiles[0])] * (-len(tiles) % 8)
    rows = [np.hstack(tiles[i:i + 8]) for i in range(0, len(tiles), 8)]
    cv2.imwrite(str(out_dir / "face_samples.jpg"), np.vstack(rows))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True, help="manifest.csv of preprocess_cremad.py (A0)")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None, help="only process the first N clips")
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(1)

    manifest = pd.read_csv(args.manifest)
    clips = manifest[manifest.status == "ok"]
    if args.limit:
        clips = clips.head(args.limit)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(lambda r: process_clip(r, args.out_dir), clips.to_dict("records")))
    views = pd.DataFrame(results)
    views.to_csv(args.out_dir / "views.csv", index=False)
    shutil.rmtree(args.out_dir / "tmp_video", ignore_errors=True)
    print(views.status.str.slice(0, 40).value_counts().to_string())
    good = views[views.status.isin(["ok", "face_fallback"])]
    print(f"A1 frames per clip: median {good.n_frames4.median():.0f}, max {good.n_frames4.max()} | "
          f"face crops per clip: median {good.n_faces.median():.0f} | frames with a detected face: "
          f"{good.face_frames_detected.sum() / good.n_faces.sum():.1%} | face crop side (source px): median "
          f"{good.face_box.str.split().str[2].astype(int).median():.0f}")
    if len(good):
        contact_sheet(args.out_dir, good)
    print(f"wrote {args.out_dir / 'views.csv'} ({len(views)} clips)")


if __name__ == "__main__":
    main()
