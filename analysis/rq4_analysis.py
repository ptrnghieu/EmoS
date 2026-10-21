"""RQ4 analysis: does causal-inference fusion make the model's uncertainty track human disagreement clip by
clip? Variants (one LoRA on Qwen2.5-Omni-7B, CH-SIMS v2 test split), each adding one step to the previous:

    A  LoRA soft (the RQ3 soft arm)                       results/rq3/rq3_soft[_seed<N>]/scores.jsonl
    B  A + a polarity head per modality, no fusion        results/rq4/rq4_B[_seed<N>]/scores.jsonl
    C  B + fusion with one weight pi for every clip       results/rq4/rq4_C[...]
    D  B + fusion with pi inferred per clip               results/rq4/rq4_D[...]

The output of A is its T+A+V answer; the output of B, C and D is the `fused` record (Q; for B, Q is the T+A+V
answer). Criteria fixed before the runs (proposal, 20/10/2026):
    primary     Spearman between the model's entropy and human entropy, D above A (paired video-level
                bootstrap on the test split, consistent across seeds)
    secondary   JSD to the 3-vote distribution, AUROC for clips where the annotators disagree, whether the
                T, A, V readings flag conflict clips, following of the odd modality closer to the humans'
    constraint  accuracy on clips where all 3 annotators agree does not drop below A's
No temperature scaling: the criteria are on the raw outputs. Clips missing from any seed-0 run (OOM) are
dropped from every arm so all comparisons use the same clips.

Usage (from the repository root):
    python analysis/rq4_analysis.py --out results/rq4_analysis
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import LABELS, POLARITY, auroc, entropy, following, jsd, load_humans, markdown, odd_one_out, spearman

ARMS = ["A", "B", "C", "D"]
PAIRS = [("D", "A"), ("B", "A"), ("C", "B"), ("D", "C"), ("D", "B")]  # D minus A is the primary comparison
LOGITS = [f"logit_{l}" for l in LABELS]


def run_dir(arm: str, seed: int, rq4_dir: Path) -> Path:
    suffix = f"_seed{seed}" if seed else ""
    return Path(f"results/rq3/rq3_soft{suffix}") if arm == "A" else rq4_dir / f"rq4_{arm}{suffix}"


def load_run(path: Path) -> dict:
    """Readings (uid x condition -> 7 label logits), the output distribution per uid, and, for RQ4 runs, pi and
    the polarity heads' distributions."""
    s = pd.DataFrame([json.loads(l) for l in open(path / "scores.jsonl", encoding="utf-8") if l.strip()])
    s = s[s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    readings = s[s.task == "score"].set_index(["uid", "condition"])[LOGITS]
    fused = s[s.task == "fused"].set_index("uid")
    if len(fused):
        out = fused[LOGITS]  # log Q
        extra = fused[["pi", "pol_T", "pol_A", "pol_V"]]
    else:
        out = readings.xs("TAV", level="condition")
        extra = None
    return {"readings": readings, "out": out, "extra": extra}


def seeds_of(arm: str, rq4_dir: Path) -> list[int]:
    base = run_dir(arm, 0, rq4_dir)
    found = [0] if (base / "scores.jsonl").exists() else []
    found += sorted(int(p.name.rsplit("seed", 1)[1]) for p in base.parent.glob(f"{base.name}_seed*")
                    if (p / "scores.jsonl").exists())
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=Path("results/rq4_analysis"))
    parser.add_argument("--rq4-dir", type=Path, default=Path("results/rq4"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    seeds = {arm: seeds_of(arm, args.rq4_dir) for arm in ARMS}
    runs = {(arm, s): load_run(run_dir(arm, s, args.rq4_dir)) for arm in ARMS for s in seeds[arm]}
    arms0 = [arm for arm in ARMS if 0 in seeds[arm]]
    assert "A" in arms0 and len(arms0) > 1, f"need A and at least one RQ4 variant at seed 0, found {seeds}"

    h = load_humans()
    test_all = h[h["mode"] == "test"]
    keep = set(test_all.uid)
    for arm in arms0:
        r = runs[(arm, 0)]
        keep &= set(r["out"].index) & {u for u, c in r["readings"].index if c in ("T", "A", "V")}
    test = test_all[test_all.uid.isin(keep)].reset_index(drop=True)
    dropped = len(test_all) - len(test)
    H = np.stack(test.human_dist)
    conflict = test.conflict.to_numpy() == 1
    disagree = test.all_agree.to_numpy() == 0
    human_ent = test.human_entropy.to_numpy()
    gold = test.ds_label.to_numpy()
    spread = np.ptp(test[["label_T", "label_A", "label_V"]].to_numpy(), axis=1)
    boot = Bootstrap(test.video_id.to_numpy(), N_BOOT)

    def output(arm, seed):
        return softmax(runs[(arm, seed)]["out"].loc[test.uid].to_numpy(float))

    def stats(P):
        ent = entropy(P)
        pred = np.array(LABELS)[P.argmax(1)]
        return {
            "Spearman(entropy, human entropy)": lambda i: spearman(ent[i], human_ent[i]),
            "JSD to humans": lambda i: jsd(P[i], H[i]).mean(),
            "AUROC: entropy flags disagreement": lambda i: auroc(ent[i], disagree[i]),
            "entropy change on conflict": lambda i: ent[i][conflict[i]].mean() - ent[i][~conflict[i]].mean(),
            "accuracy": lambda i: (pred[i] == gold[i]).mean(),
            "accuracy, all 3 agree": lambda i: (pred[i] == gold[i])[~disagree[i]].mean(),
        }

    tables = {}

    # ---- main table, seed 0 --------------------------------------------------------------------
    human_change = lambda i: human_ent[i][conflict[i]].mean() - human_ent[i][~conflict[i]].mean()
    rows = [{"arm": "humans (reference)", "entropy change on conflict": boot(human_change), "mean entropy": human_ent.mean()}]
    for arm in arms0:
        P = output(arm, 0)
        rows.append({"arm": arm, **{k: boot(f) for k, f in stats(P).items()}, "mean entropy": entropy(P).mean()})
    tables["rq4_main_tav_seed0"] = pd.DataFrame(rows)

    # ---- paired differences, seed 0 ------------------------------------------------------------
    rows = []
    for a, b in PAIRS:
        if a in arms0 and b in arms0:
            sa, sb = stats(output(a, 0)), stats(output(b, 0))
            rows.append({"comparison": f"{a} minus {b}" + (" (primary)" if (a, b) == ("D", "A") else ""),
                         **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i)) for k in sa}})
    tables["rq4_paired_differences_seed0"] = pd.DataFrame(rows)

    # ---- seeds: point values, and the primary comparison per seed ------------------------------
    keys = list(stats(output("A", 0)))
    rows = []
    for arm in ARMS:
        points = []
        for s in seeds[arm]:
            r = runs[(arm, s)]
            if not set(test.uid) <= set(r["out"].index):
                rows.append({"arm": arm, "seed": str(s), "note": f"{len(set(test.uid) - set(r['out'].index))} test clips missing"})
                continue
            st = stats(output(arm, s))
            points.append({k: st[k](boot.all) for k in keys})
            rows.append({"arm": arm, "seed": str(s), **points[-1]})
        if len(points) > 1:
            v = pd.DataFrame(points)
            rows.append({"arm": arm, "seed": f"mean ± SD ({len(v)} seeds)",
                         **{k: f"{v[k].mean():.3f} ± {v[k].std(ddof=1):.3f}" for k in keys}})
    tables["rq4_seeds"] = pd.DataFrame(rows)

    rows = []
    for a, b in PAIRS:
        for s in sorted(set(seeds[a]) & set(seeds[b])):
            if not all(set(test.uid) <= set(runs[(x, s)]["out"].index) for x in (a, b)):
                continue
            sa, sb = stats(output(a, s)), stats(output(b, s))
            rows.append({"comparison": f"{a} minus {b}", "seed": s,
                         **{k: boot(lambda i, k=k: sa[k](i) - sb[k](i))
                            for k in ("Spearman(entropy, human entropy)", "JSD to humans", "accuracy, all 3 agree")}})
    tables["rq4_paired_differences_by_seed"] = pd.DataFrame(rows)

    # ---- the gate: is the fused reading trusted less on conflict clips? ------------------------
    rows = []
    for arm in ("C", "D"):
        for s in seeds[arm]:
            extra = runs[(arm, s)]["extra"]
            if extra is None or not set(test.uid) <= set(extra.index):
                continue
            pi = extra.loc[test.uid, "pi"].to_numpy(float)
            row = {"arm": arm, "seed": s, "pi, no conflict": pi[~conflict].mean(), "pi, conflict": pi[conflict].mean(),
                   "pi SD across clips": pi.std()}
            if pi.std() < 1e-6:  # variant C: one pi for every clip, nothing to rank
                rows.append(row)
                continue
            rows.append({**row,
                         "pi: conflict minus no conflict": boot(lambda i: pi[i][conflict[i]].mean() - pi[i][~conflict[i]].mean()),
                         "AUROC: 1 - pi flags conflict": boot(lambda i: auroc(1 - pi[i], conflict[i])),
                         "AUROC: 1 - pi flags disagreement": boot(lambda i: auroc(1 - pi[i], disagree[i])),
                         "Spearman(1 - pi, human entropy)": boot(lambda i: spearman(1 - pi[i], human_ent[i])),
                         "Spearman(1 - pi, T/A/V label spread)": boot(lambda i: spearman(1 - pi[i], spread[i]))})
    tables["rq4_gate"] = pd.DataFrame(rows)

    # ---- per-modality readings: do they see the conflict? (seed 0) -----------------------------
    pol_true = {m: np.sign(test[f"label_{m}"].to_numpy()).astype(int) + 1 for m in "TAV"}  # 0 neg, 1 neutral, 2 pos
    rows = []
    for arm in arms0:
        r = runs[(arm, 0)]
        P = {c: softmax(r["readings"].loc[[(u, c) for u in test.uid]].to_numpy(float)) for c in "TAV"}
        label_jsd = (jsd(P["T"], P["A"]) + jsd(P["T"], P["V"]) + jsd(P["A"], P["V"])) / 3
        rows.append({"arm": arm, "signal": "T, A, V label readings: mean pairwise JSD",
                     "AUROC: flags conflict": boot(lambda i: auroc(label_jsd[i], conflict[i])),
                     "Spearman with T/A/V label spread": spearman(label_jsd, spread),
                     "Spearman with human entropy": spearman(label_jsd, human_ent)})
        if r["extra"] is None:
            continue
        q = {m: np.stack(r["extra"].loc[test.uid, f"pol_{m}"].to_numpy()) for m in "TAV"}
        head_l1 = (np.abs(q["T"] - q["A"]).sum(1) + np.abs(q["T"] - q["V"]).sum(1) + np.abs(q["A"] - q["V"]).sum(1)) / 3
        rows.append({"arm": arm, "signal": "polarity heads: mean pairwise L1",
                     "AUROC: flags conflict": boot(lambda i: auroc(head_l1[i], conflict[i])),
                     "Spearman with T/A/V label spread": spearman(head_l1, spread),
                     "Spearman with human entropy": spearman(head_l1, human_ent),
                     **{f"head accuracy {m}": (q[m].argmax(1) == pol_true[m]).mean() for m in "TAV"},
                     **{f"head accuracy {m}, conflict": (q[m].argmax(1) == pol_true[m])[conflict].mean() for m in "TAV"}})
    tables["rq4_conflict_detection_seed0"] = pd.DataFrame(rows)

    # ---- following the odd modality when it flips polarity (seed 0) ----------------------------
    single = odd_one_out(test)
    flips = single[(single.odd_sign != 0) & (single.majority_sign != 0)]
    pooled = pd.concat([flips.assign(_pol=flips[f"ann{i}"].map(POLARITY)) for i in (1, 2, 3)], ignore_index=True)
    human_row = following(pooled["_pol"], pooled)
    rows = [{"who": "humans (3 annotators pooled)", **human_row}]
    rate_keys = [f"follows odd {m}" for m in "TAV"]
    for arm in arms0:
        pred = pd.Series(np.array(LABELS)[output(arm, 0).argmax(1)], index=test.uid)
        sub = flips.assign(_pol=flips.uid.map(pred).map(POLARITY))
        row = following(sub["_pol"], sub)
        rows.append({"who": arm, **row, "sum |arm - humans| over follows odd T, A, V":
                     sum(abs(row[k] - human_row[k]) for k in rate_keys)})
    tables["rq4_follows_odd_modality_flips_seed0"] = pd.DataFrame(rows)

    # ---- training history ----------------------------------------------------------------------
    rows = []
    for (arm, s) in runs:
        state_path = run_dir(arm, s, args.rq4_dir) / "state.json"
        if arm == "A" or not state_path.exists():
            continue
        state = json.loads(state_path.read_text())
        rows += [{"arm": arm, "seed": s, "best": e["epoch"] == state["best_epoch"], "OOM clips (training)": state.get("oom"),
                  **{k: v for k, v in e.items() if k not in ("session_hours", "val_minutes")}} for e in state["history"]]
    tables["rq4_training_history"] = pd.DataFrame(rows)

    lines = ["# RQ4 results: causal-inference fusion (Qwen2.5-Omni-7B, CH-SIMS v2 test split)\n",
             f"Generated by `analysis/rq4_analysis.py`. {len(test)} test clips ({conflict.sum()} with conflict"
             f"{f'; {dropped} dropped because a seed-0 run has no score for them' if dropped else ''}). "
             f"Seeds found: {', '.join(f'{a}: {seeds[a] or 'none'}' for a in ARMS)}. "
             f"CIs: 95% video-level bootstrap ({N_BOOT} resamples, shared across arms). "
             f"Odd-modality table: {len(flips)} test clips where one modality flips polarity.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
