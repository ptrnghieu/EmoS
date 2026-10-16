"""Pilot: does cross-modal conflict in CH-SIMS v2 go with annotator disagreement in EmoS?

Joins CH-SIMS v2 unimodal sentiment labels (label_T / label_A / label_V, from meta.csv)
with the EmoS Basic-7 re-annotation (3 annotators per clip) and reports how often the
three annotators agree, split by whether the modalities conflict.

Usage (from the repository root):
    python analysis/pilot_conflict_disagreement.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1] / "datasets/data-final/datasets/ch-simsv2s"
META = ROOT / "meta.csv"
EMOS_BATCHES = {
    "first": ROOT / "First-2500/analysis_outputs/basic7/annotated_with_ds.xlsx",
    "last": ROOT / "Last-2500/analysis_outputs_after2500/basic7/annotated_with_ds.xlsx",
}


def load() -> pd.DataFrame:
    meta = pd.read_csv(META, dtype={"video_id": str})
    emos = pd.concat(
        [pd.read_excel(path).assign(batch=name) for name, path in EMOS_BATCHES.items()]
    )
    for df in (meta, emos):
        df["video_id"] = df["video_id"].astype(str)
        df["clip_id"] = df["clip_id"].astype(int)
    df = emos.merge(meta, on=["video_id", "clip_id"], suffixes=("", "_meta"))

    sign_t, sign_a, sign_v = (np.sign(df[c]).astype(int) for c in ("label_T", "label_A", "label_V"))
    sign_m = np.sign(df["label"]).astype(int)
    df["conflict"] = ((sign_t != sign_a) | (sign_t != sign_v) | (sign_a != sign_v)).astype(int)
    df["n_deviating"] = (sign_t != sign_m).astype(int) + (sign_a != sign_m).astype(int) + (sign_v != sign_m).astype(int)
    for name, sign in (("T", sign_t), ("A", sign_a), ("V", sign_v)):
        df[f"dev_{name}"] = (sign != sign_m).astype(int)
    return df


def agreement_table(df: pd.DataFrame, by: str) -> pd.DataFrame:
    return df.groupby(by).agg(
        n=("unique_labels", "size"),
        all_agree=("unique_labels", lambda x: (x == 1).mean()),
        all_differ=("unique_labels", lambda x: (x == 3).mean()),
        mean_unique_labels=("unique_labels", "mean"),
        mean_ds_confidence=("ds_confidence", "mean"),
    ).round(3)


def chi_square(table: pd.DataFrame) -> tuple[float, int]:
    observed = table.to_numpy(dtype=float)
    expected = observed.sum(1, keepdims=True) * observed.sum(0, keepdims=True) / observed.sum()
    stat = ((observed - expected) ** 2 / expected).sum()
    dof = (observed.shape[0] - 1) * (observed.shape[1] - 1)
    return round(float(stat), 1), dof


def main() -> None:
    pd.set_option("display.width", 160)
    df = load()
    print(f"clips after join: {len(df)}  |  videos: {df.video_id.nunique()}  |  conflict rate: {df.conflict.mean():.3f}\n")

    print("Agreement by conflict (0 = all modalities share a polarity):")
    print(agreement_table(df, "conflict"), "\n")
    stat, dof = chi_square(pd.crosstab(df.conflict, df.unique_labels))
    print(f"chi-square conflict x unique_labels: {stat} (df = {dof})\n")

    print("Agreement by number of modalities deviating from the multimodal label:")
    print(agreement_table(df, "n_deviating"), "\n")

    print("Agreement by annotator batch and conflict:")
    print(agreement_table(df, ["batch", "conflict"]), "\n")

    print("Share of clips where each modality deviates from the multimodal label:")
    print(df[["dev_T", "dev_A", "dev_V"]].mean().round(3).to_string(), "\n")

    single = df[df.n_deviating == 1].copy()
    single["deviating"] = np.select([single.dev_T == 1, single.dev_A == 1], ["T", "A"], default="V")
    print("Clips with exactly one deviating modality:")
    print(agreement_table(single, "deviating"))


if __name__ == "__main__":
    main()
