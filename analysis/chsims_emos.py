"""Join CH-SIMS v2(s) metadata with the EmoS Basic-7 re-annotation, one row per clip.

Key quirk: meta.csv stores clip ids as strings ("0001", "00002_0"), while the EmoS
workbooks store them as integers, so "00002_0" became 20 and can collide with a real
clip 20 of the same video. We therefore join on (video_id, int(clip without "_"), text).
"""

from pathlib import Path

import numpy as np
import pandas as pd

CHSIMS_DIR = Path(__file__).resolve().parents[1] / "datasets/data-final/datasets/ch-simsv2s"
EMOS_BATCHES = {
    # batch name -> (workbook, per-annotator columns in that workbook)
    "first": ("First-2500/analysis_outputs/basic7/annotated_with_ds.xlsx", ["a1", "a2", "a3"]),
    "last": ("Last-2500/analysis_outputs_after2500/basic7/annotated_with_ds.xlsx", ["cp", "ljx", "wqf"]),
}
BASIC7 = ["Anger", "Disgust", "Fear", "Joy", "Neutral", "Sadness", "Surprise"]


def clip_key(clip_id) -> int:
    return int(str(clip_id).replace("_", ""))


def load_meta(chsims_dir: Path = CHSIMS_DIR) -> pd.DataFrame:
    meta = pd.read_csv(chsims_dir / "meta.csv", dtype={"video_id": str, "clip_id": str})
    meta["clip_key"] = meta["clip_id"].map(clip_key)
    meta["uid"] = meta["video_id"] + "__" + meta["clip_id"]
    return meta


def load_emos(chsims_dir: Path = CHSIMS_DIR) -> pd.DataFrame:
    frames = []
    for batch, (path, annotators) in EMOS_BATCHES.items():
        df = pd.read_excel(chsims_dir / path)
        df = df.rename(columns={a: f"ann{i + 1}" for i, a in enumerate(annotators)})
        df["batch"] = batch
        df["clip_key"] = df["clip_id"].map(clip_key)
        frames.append(df[["video_id", "clip_key", "text", "ann1", "ann2", "ann3",
                          "ds_label", "ds_confidence", "unique_labels", "batch"]])
    emos = pd.concat(frames, ignore_index=True)
    emos["video_id"] = emos["video_id"].astype(str)
    return emos


def add_conflict_columns(df: pd.DataFrame) -> pd.DataFrame:
    sign_t, sign_a, sign_v = (np.sign(df[c]).astype(int) for c in ("label_T", "label_A", "label_V"))
    sign_m = np.sign(df["label"]).astype(int)
    df["conflict"] = ((sign_t != sign_a) | (sign_t != sign_v) | (sign_a != sign_v)).astype(int)
    for name, sign in (("T", sign_t), ("A", sign_a), ("V", sign_v)):
        df[f"dev_{name}"] = (sign != sign_m).astype(int)
    df["n_deviating"] = df[["dev_T", "dev_A", "dev_V"]].sum(axis=1)
    return df


def load_joined(chsims_dir: Path = CHSIMS_DIR) -> pd.DataFrame:
    """One row per CH-SIMS v2(s) clip that has EmoS labels, with conflict columns added."""
    meta, emos = load_meta(chsims_dir), load_emos(chsims_dir)
    df = meta.merge(emos, on=["video_id", "clip_key", "text"], how="inner", validate="one_to_one")
    if len(df) != len(meta):
        missing = meta[~meta["uid"].isin(df["uid"])]
        raise ValueError(f"{len(missing)} meta.csv clips have no EmoS row, e.g. {missing['uid'].head(3).tolist()}")
    return add_conflict_columns(df)
