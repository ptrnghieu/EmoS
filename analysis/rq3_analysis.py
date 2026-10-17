"""RQ3 analysis: does fine-tuning on the annotators' vote distribution make the model's uncertainty
track human disagreement under cross-modal conflict? Compares, on the CH-SIMS v2 test split:

    zero-shot            Qwen2.5-Omni-7B as is (results/omni_runs/scores.jsonl)
    LoRA hard            fine-tuned on ds_label (results/rq3/rq3_hard/scores.jsonl)
    LoRA soft            fine-tuned on the 3-vote distribution (results/rq3/rq3_soft/scores.jsonl)
    ... + TS             each of the above with temperature scaling: one temperature per arm, chosen
                         on the valid split (T+A+V) to minimise the JSD to the human distribution,
                         then applied to every condition of the test split

All arms are scored on the same clips from the 7 label first-token logits. Confidence intervals are
95% percentile bootstraps over videos; every statistic uses the same resampled videos, so the paired
differences between arms have proper intervals.

Usage (from the repository root):
    python analysis/rq3_analysis.py --out results/rq3_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq_analysis import CONDITIONS, LABELS, POLARITY, auroc, entropy, jsd, macro_f1, markdown, odd_one_out, spearman, load_humans

N_BOOT = 1000
TEMPERATURES = np.linspace(0.5, 8, 151)
RUNS = {"zero-shot": "results/omni_runs/scores.jsonl",
        "LoRA hard": "results/rq3/rq3_hard/scores.jsonl",
        "LoRA soft": "results/rq3/rq3_soft/scores.jsonl"}


def softmax(z: np.ndarray, t: float = 1.0) -> np.ndarray:
    z = z / t
    e = np.exp(z - z.max(axis=1, keepdims=True))
    return e / e.sum(axis=1, keepdims=True)


def load_logits(path: str) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s["task"] == "score") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    return s.set_index(["uid", "condition"])[[f"logit_{l}" for l in LABELS]]


def fmt(point: float, samples: np.ndarray) -> str:
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return f"{point:.3f} [{lo:.3f}, {hi:.3f}]"


class Bootstrap:
    """Row-index resamples of the test clips, drawn by video once and shared by every statistic."""

    def __init__(self, videos: np.ndarray, n: int, seed: int = 0):
        rng = np.random.default_rng(seed)
        groups = [np.flatnonzero(videos == v) for v in pd.unique(videos)]
        self.samples = [np.concatenate([groups[i] for i in rng.integers(0, len(groups), len(groups))]) for _ in range(n)]
        self.all = np.arange(len(videos))

    def __call__(self, stat) -> str:
        return fmt(stat(self.all), np.array([stat(idx) for idx in self.samples]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/rq3_analysis"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    h = load_humans()
    test = h[h["mode"] == "test"].reset_index(drop=True)
    valid = h[h["mode"] == "valid"].reset_index(drop=True)
    H, Hv = np.stack(test.human_dist), np.stack(valid.human_dist)
    conflict = test.conflict.to_numpy() == 1
    disagree = test.all_agree.to_numpy() == 0
    human_ent = test.human_entropy.to_numpy()
    gold = test.ds_label.to_numpy()
    boot = Bootstrap(test.video_id.to_numpy(), N_BOOT)

    # ---- distributions of every arm ------------------------------------------------------------
    dist, temps = {}, []  # dist[arm][condition] -> (n_test, 7), rows in `test` order
    for name, path in RUNS.items():
        logits = load_logits(path)
        z_valid = logits.loc[[(u, "TAV") for u in valid.uid]].to_numpy()
        t_best = min(TEMPERATURES, key=lambda t: jsd(softmax(z_valid, t), Hv).mean())
        temps.append({"arm": name, "temperature chosen on valid": round(float(t_best), 2),
                      "valid JSD at T=1": jsd(softmax(z_valid), Hv).mean(),
                      "valid JSD at chosen T": jsd(softmax(z_valid, t_best), Hv).mean()})
        for arm, t in ((name, 1.0), (f"{name} + TS", t_best)):
            dist[arm] = {c: softmax(logits.loc[[(u, c) for u in test.uid]].to_numpy(), t) for c in CONDITIONS}
    arms = list(dist)

    def stats(P):
        ent = entropy(P)
        pred = np.array(LABELS)[P.argmax(1)]
        return {
            "JSD to humans": lambda i: jsd(P[i], H[i]).mean(),
            "accuracy": lambda i: (pred[i] == gold[i]).mean(),
            "accuracy, all 3 agree": lambda i: (pred[i] == gold[i])[~disagree[i]].mean(),
            "entropy change on conflict": lambda i: ent[i][conflict[i]].mean() - ent[i][~conflict[i]].mean(),
            "Spearman(entropy, human entropy)": lambda i: spearman(ent[i], human_ent[i]),
            "AUROC: entropy flags disagreement": lambda i: auroc(ent[i], disagree[i]),
        }

    tables = {}

    # ---- main table: T+A+V ---------------------------------------------------------------------
    human_change = lambda i: human_ent[i][conflict[i]].mean() - human_ent[i][~conflict[i]].mean()
    rows = [{"arm": "humans (reference)", "entropy change on conflict": boot(human_change),
             "mean entropy": human_ent.mean()}]
    for arm in arms:
        P = dist[arm]["TAV"]
        pred = np.array(LABELS)[P.argmax(1)]
        row = {"arm": arm, **{k: boot(f) for k, f in stats(P).items()},
               "macro-F1": macro_f1(gold, pred), "mean entropy": entropy(P).mean(),
               "JSD, no conflict": jsd(P, H)[~conflict].mean(), "JSD, conflict": jsd(P, H)[conflict].mean()}
        rows.append(row)
    tables["rq3_main_tav"] = pd.DataFrame(rows)

    # ---- paired differences --------------------------------------------------------------------
    pairs = [("LoRA soft", "LoRA hard"), ("LoRA soft + TS", "LoRA hard + TS"), ("LoRA soft", "LoRA hard + TS"),
             ("LoRA soft", "zero-shot + TS"), ("LoRA hard", "zero-shot"), ("LoRA soft", "zero-shot")]
    rows = []
    for a, b in pairs:
        sa, sb = stats(dist[a]["TAV"]), stats(dist[b]["TAV"])
        rows.append({"comparison": f"{a} minus {b}",
                     **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i)) for k in sa}})
    tables["rq3_paired_differences_tav"] = pd.DataFrame(rows)

    # ---- every condition -----------------------------------------------------------------------
    rows = []
    for arm in arms:
        for c in CONDITIONS:
            P = dist[arm][c]
            s = stats(P)
            rows.append({"arm": arm, "condition": c, "accuracy": s["accuracy"](boot.all),
                         "JSD to humans": s["JSD to humans"](boot.all), "mean entropy": entropy(P).mean(),
                         "entropy change on conflict": boot(s["entropy change on conflict"]),
                         "share predicted Neutral": (P.argmax(1) == LABELS.index("Neutral")).mean()})
    tables["rq3_by_condition"] = pd.DataFrame(rows)

    # ---- Neutral collapse ----------------------------------------------------------------------
    neutral = LABELS.index("Neutral")
    rows = [{"who": "humans: majority label is Neutral", "no conflict": (gold[~conflict] == "Neutral").mean(),
             "conflict": (gold[conflict] == "Neutral").mean()},
            {"who": "humans: share of Neutral votes", "no conflict": H[~conflict, neutral].mean(),
             "conflict": H[conflict, neutral].mean()}]
    for arm in arms:
        P = dist[arm]["TAV"]
        rows.append({"who": f"{arm}: predicts Neutral", "no conflict": (P[~conflict].argmax(1) == neutral).mean(),
                     "conflict": (P[conflict].argmax(1) == neutral).mean()})
        rows.append({"who": f"{arm}: probability on Neutral", "no conflict": P[~conflict, neutral].mean(),
                     "conflict": P[conflict, neutral].mean()})
    tables["rq3_neutral"] = pd.DataFrame(rows)

    # ---- following the odd modality when it flips polarity (T+A+V) ------------------------------
    single = odd_one_out(test).set_index("uid")
    flips = single[(single.odd_sign != 0) & (single.majority_sign != 0)]
    pos = {u: i for i, u in enumerate(test.uid)}
    in_flip = np.zeros(len(test), bool)
    in_flip[[pos[u] for u in flips.index]] = True
    odd = np.array([flips.odd.get(u, "") for u in test.uid])
    odd_sign = np.array([flips.odd_sign.get(u, 9) for u in test.uid])
    major = np.array([flips.majority_sign.get(u, 9) for u in test.uid])
    votes = test[["ann1", "ann2", "ann3"]].apply(lambda col: col.map(POLARITY)).to_numpy(float)  # NaN: Surprise

    def follow_rates(pol: np.ndarray, n: np.ndarray) -> dict:
        """pol: per clip, (n answers x polarity); for the model n = 1. Pooled rate over answers."""
        out = {"answers": int(n[in_flip].sum())}
        for m in "TAV":
            out[f"clips, odd {m}"] = int((in_flip & (odd == m)).sum())
        for m in "TAV":
            sel = lambda i, m=m: i[in_flip[i] & (odd[i] == m)]
            out[f"follows odd {m}"] = boot(lambda i, s=sel: np.nansum(pol[s(i)] == odd_sign[s(i), None]) / max(1, n[s(i)].sum()))
        sel = lambda i: i[in_flip[i]]
        out["follows the other two"] = boot(lambda i: np.nansum(pol[sel(i)] == major[sel(i), None]) / max(1, n[sel(i)].sum()))
        out["answers Neutral"] = boot(lambda i: np.nansum(pol[sel(i)] == 0) / max(1, n[sel(i)].sum()))
        return out

    rows = [{"who": "humans (3 annotators pooled)", **follow_rates(votes, (~np.isnan(votes)).sum(1))}]
    for arm in arms:
        pol = np.array([POLARITY.get(l, np.nan) for l in np.array(LABELS)[dist[arm]["TAV"].argmax(1)]])[:, None]
        rows.append({"who": arm, **follow_rates(pol, (~np.isnan(pol)).sum(1))})
    tables["rq3_follows_odd_modality_flips"] = pd.DataFrame(rows)

    # ---- training history and temperatures -----------------------------------------------------
    rows = []
    for arm in ("hard", "soft"):
        state = json.loads(Path(f"results/rq3/rq3_{arm}/state.json").read_text())
        rows += [{"arm": f"LoRA {arm}", "best": h_["epoch"] == state["best_epoch"],
                  **{k: v for k, v in h_.items() if k != "session_hours"}} for h_ in state["history"]]
    tables["rq3_training_history"] = pd.DataFrame(rows)
    tables["rq3_temperatures"] = pd.DataFrame(temps)

    lines = ["# RQ3 results: LoRA fine-tuning with hard vs soft labels (Qwen2.5-Omni-7B, CH-SIMS v2 test split)\n",
             f"Generated by `analysis/rq3_analysis.py`. {len(test)} test clips ({conflict.sum()} with conflict). "
             f"CIs: 95% video-level bootstrap ({N_BOOT} resamples, shared across arms). "
             f"Odd-modality table: {in_flip.sum()} test clips where one modality flips polarity.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
