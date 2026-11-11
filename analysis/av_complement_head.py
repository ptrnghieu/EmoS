"""Does HSEmotion add information useful for the audio-visual votes beyond Qwen's own readings? (Design approved
11/11/2026, before any result.)

Protocol (CPU only; avoids stacking leakage). For each Stage 1 checkpoint k (s1_noI, outer fold k) only the actors of
its held-out fold k are used: none of them was seen by that checkpoint. These actors are split into 3 head folds
(by actor, stratified by sex, fixed seed); a head is fitted on two and applied to the third, in turn, so every clip
gets one held-out prediction. Weight decay and epoch are chosen on an inner actor split inside the head's training
part (15% of its actors), then the head is refitted on the whole training part. Train and test features of every fit
come from the same checkpoint. Predictions of the three checkpoints are pooled; bootstrap over actors.
Limit: the heads see fewer actors than a stacking system trained on all Stage 1 training actors would; this tests
whether HSE is complementary on actors new to Qwen, not such a system. A negative result is read with that limit.

Target: the AV raters' vote distribution. Feature sets (logits centred over the six classes per reading and clip,
then every dimension standardised with the head's training statistics):
    R0  Qwen AV logits                       R1  R0 + HSEmotion mean and SD of the A2 face crops (2560-d)
    R2  Qwen AV + V + A logits               R3  R2 + HSEmotion
Heads, loss, weight-decay grid and epoch selection as analysis/hsemotion_head.py (linear main, MLP 128 sensitivity).
s1_noI's own AV prediction is reported as a reference only.

Reading (fixed before the run). Main comparison R1 - R0 (linear); secondary R3 - R2. Main metric JSD AV; partial
Spearman AV (given the intended emotion) is reported separately, never pooled into one success rule.
    JSD clearly lower           -> HSE adds benefit for the fit of the AV distribution in this protocol
    partial clearly higher      -> HSE adds benefit for the AV disagreement ranking
    one better, the other worse -> trade-off, not an overall improvement
    R1 beats R0, R3 not R2      -> no HSE benefit shown once Qwen's A/V logits are added; this does NOT show that the
                                   information is mostly redundant (data size and head limits are alternatives)
    both unclear                -> no benefit found with these data and readouts; look at the CI widths before level 2
A difference in significance between the two comparisons does not show that their improvements differ.

Usage (from the repository root; needs torch):
    python analysis/av_complement_head.py --out results/av_complement_head
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from combined_head import hsemotion_pooled
from cremad_human import counts, jsd
from hsemotion_head import INNER_VAL_SHARE, MAX_EPOCHS, WEIGHT_DECAYS, train
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, SPLIT_SEED, norm_entropy, outer_folds, partial_spearman

N_HEAD, N_BOOT = 3, 1000
SETS = {"R0": (("AV",), False), "R1": (("AV",), True), "R2": (("AV", "V", "A"), False), "R3": (("AV", "V", "A"), True)}
KINDS = ("linear", "mlp")
LCOLS = [f"logit_{l}" for l in LABELS]


def read_logits(path: Path, cond: str) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    z = s.drop_duplicates("uid", keep="last").set_index("uid")[LCOLS]
    return z.sub(z.mean(1), axis=0)  # centre over the six classes: drops the offset softmax ignores


def head_folds(actors: pd.Series, k: int) -> dict[int, int]:
    """3 head folds over the held-out actors of checkpoint k, stratified by sex, fixed seed."""
    rng = np.random.default_rng(SPLIT_SEED + 100 + k)
    out = {}
    for _, group in actors.groupby(actors):
        for j, a in enumerate(rng.permutation(group.index.to_numpy())):
            out[int(a)] = j % N_HEAD
    return out


def fit_predict(kind, X, P, tr, te, actors, seed):
    """Choose (weight decay, epoch) on an inner actor split of the training part, refit on all of it, predict te."""
    tr_act = sorted(set(actors[tr]))
    inner = set(np.random.default_rng(seed).permutation(tr_act)[:max(1, round(INNER_VAL_SHARE * len(tr_act)))])
    val = tr & np.isin(actors, list(inner))
    fit = tr & ~val
    mu, sd = X[fit].mean(0), X[fit].std(0) + 1e-6
    best = None
    for wd in WEIGHT_DECAYS:
        _, curve = train(kind, (X[fit] - mu) / sd, P[fit], wd, MAX_EPOCHS, (X[val] - mu) / sd, P[val])
        ep = int(np.argmin(curve)) + 1
        if best is None or curve[ep - 1] < best[0]:
            best = (curve[ep - 1], wd, ep)
    mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
    head, _ = train(kind, (X[tr] - mu) / sd, P[tr], best[1], best[2])
    with torch.no_grad():
        return torch.softmax(head(torch.from_numpy((X[te] - mu) / sd)), -1).numpy(), best


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--features", type=Path, default=Path("results/hsemotion_features/features.npz"))
    parser.add_argument("--out", type=Path, default=Path("results/av_complement_head"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    hse = hsemotion_pooled(args.features)
    rows, grid, sizes = [], [], []
    for k in range(N_OUTER):
        run = args.stage1 / f"stage1_s1_noI_f{k}_s0"
        st = json.loads((run / "state.json").read_text())
        assert (st["arm"], st["fold"], st["phase"], st["train_done"]) == ("s1_noI", k, "outer", True), st
        lg = {c: read_logits(run / "scores.jsonl", c) for c in ("AV", "V", "A")}
        uids = [u for u in lg["AV"].index if u in df.index and df.loc[u, "fold"] == k and df.loc[u, "av_n"] > 0
                and u in hse and all(u in lg[c].index for c in ("V", "A"))]
        d = df.loc[uids]
        P = counts(d, "av")
        P = (P / P.sum(1, keepdims=True)).astype(np.float32)
        actors = d.actor.to_numpy()
        hf = d.actor.map(head_folds(d.groupby("actor")["Sex"].first(), k)).to_numpy()
        H = np.stack([hse[u] for u in uids]).astype(np.float32)
        feats = {name: np.hstack([lg[c].loc[uids].to_numpy(np.float32) for c in conds] + ([H] if use_hse else []))
                 for name, (conds, use_hse) in SETS.items()}
        pred = {(n, kind): np.zeros_like(P) for n in SETS for kind in KINDS}
        for j in range(N_HEAD):
            te, tr = hf == j, hf != j
            for n, X in feats.items():
                for kind in KINDS:
                    p, (score, wd, ep) = fit_predict(kind, X, P, tr, te, actors, SPLIT_SEED + 200 + 10 * k + j)
                    pred[(n, kind)][te] = p
                    grid.append({"checkpoint": k, "head fold": j, "features": n, "head": kind, "weight decay": wd,
                                 "epoch": ep, "inner JSD": round(score, 4)})
        sizes.append({"checkpoint": k, "clips": len(uids), "actors": len(set(actors)),
                      "actors per head fold": "/".join(str(len(set(actors[hf == j]))) for j in range(N_HEAD))})
        base = softmax(lg["AV"].loc[uids].to_numpy(float))
        for i, u in enumerate(uids):
            rec = {"uid": u, "checkpoint": k, "actor": int(actors[i]), "emotion": d.intended.iloc[i], "head_fold": int(hf[i]),
                   "p_votes": P[i].tolist(), "s1_noI": base[i].tolist()}
            rec.update({f"{n}_{kind}": pred[(n, kind)][i].tolist() for n in SETS for kind in KINDS})
            rows.append(rec)

    R = pd.DataFrame(rows)
    pd.DataFrame([{**{c: r[c] for c in ("uid", "checkpoint", "head_fold")}, **{f"{a}_{l}": v for a in ["s1_noI"] + [f"{n}_{kd}" for n in SETS for kd in KINDS]
                                                                                for l, v in zip(LABELS, r[a])}} for _, r in R.iterrows()]).to_csv(args.out / "predictions.csv", index=False)
    Pv = np.stack(R.p_votes)
    hp, emo, ckpt, act = norm_entropy(Pv), R.emotion.to_numpy(), R.checkpoint.to_numpy(), R.actor.to_numpy()
    top = Pv == Pv.max(1, keepdims=True)
    Q = {a: np.stack(R[a]) for a in ["s1_noI"] + [f"{n}_{kd}" for kd in KINDS for n in SETS]}

    def metrics(q):
        j, hit, hq = jsd(q, Pv), top[np.arange(len(q)), q.argmax(1)], norm_entropy(q)
        return {"JSD AV": lambda i: j[i].mean(), "top label among raters' top": lambda i: hit[i].mean(),
                "Spearman(H(q), A^AV)": lambda i: spearman(hq[i], hp[i]),
                "partial given emotion": lambda i: partial_spearman(hq[i], hp[i], emo[i]), "mean H(q)": lambda i: hq[i].mean()}

    boot = Bootstrap(act, N_BOOT)
    tables = {"sizes": pd.DataFrame(sizes)}
    tables["readings"] = pd.DataFrame([{"predictor": a, **{m: boot(f) for m, f in metrics(q).items()},
                                        **{f"JSD ckpt {k}": round(float(jsd(q, Pv)[ckpt == k].mean()), 4) for k in range(N_OUTER)},
                                        "human mean H": round(float(hp.mean()), 3)} for a, q in Q.items()])
    pairs = [(f"R1_{kd}", f"R0_{kd}") for kd in KINDS] + [(f"R3_{kd}", f"R2_{kd}") for kd in KINDS] + [(f"R2_{kd}", f"R0_{kd}") for kd in KINDS]
    prows, verdict = [], {}
    for new, ref in pairs:
        a, b = metrics(Q[new]), metrics(Q[ref])
        rec = {"comparison": f"{new} minus {ref}"}
        for m in a:
            stat = lambda i, m=m: a[m](i) - b[m](i)
            samples = np.array([stat(i) for i in boot.samples])
            point = stat(boot.all)
            lo, hi = np.nanpercentile(samples, [2.5, 97.5])
            rec[m] = fmt(point, samples)
            verdict[(new, ref, m)] = (point, lo, hi)
        for k in range(N_OUTER):
            sel = np.flatnonzero(ckpt == k)
            bk = Bootstrap(act[sel], N_BOOT)
            rec[f"JSD ckpt {k}"] = bk(lambda i: jsd(Q[new][sel], Pv[sel])[i].mean() - jsd(Q[ref][sel], Pv[sel])[i].mean())
            rec[f"partial ckpt {k}"] = bk(lambda i: partial_spearman(norm_entropy(Q[new][sel])[i], hp[sel][i], emo[sel][i])
                                          - partial_spearman(norm_entropy(Q[ref][sel])[i], hp[sel][i], emo[sel][i]))
        prows.append(rec)
    tables["paired"] = pd.DataFrame(prows)

    def status(new, ref, m, lower_is_better):
        point, lo, hi = verdict[(new, ref, m)]
        if (hi < 0) if lower_is_better else (lo > 0):
            return "clearly better"
        if (lo > 0) if lower_is_better else (hi < 0):
            return "clearly worse"
        return "CI contains 0"

    vrows = []
    for new, ref in pairs[:4]:
        sj, sp = status(new, ref, "JSD AV", True), status(new, ref, "partial given emotion", False)
        if sj == "clearly better" and sp == "clearly better":
            text = "HSE adds benefit for both the AV distribution fit and the AV disagreement ranking in this protocol"
        elif "clearly worse" in (sj, sp) and "clearly better" in (sj, sp):
            text = "trade-off: one metric better, the other worse; not an overall improvement"
        elif sj == "clearly better":
            text = "HSE adds benefit for the fit of the AV distribution in this protocol"
        elif sp == "clearly better":
            text = "HSE adds benefit for the AV disagreement ranking in this protocol"
        elif "clearly worse" in (sj, sp):
            text = "worse on at least one metric, better on none"
        else:
            text = "no benefit found with these data and readouts (check the CI widths)"
        vrows.append({"comparison": f"{new} minus {ref}", "role": "main" if new == "R1_linear" else ("secondary" if new == "R3_linear" else "MLP sensitivity"),
                      "JSD AV": "{:.4f} [{:.4f}, {:.4f}]".format(*verdict[(new, ref, "JSD AV")]), "JSD status": sj,
                      "partial AV": "{:.4f} [{:.4f}, {:.4f}]".format(*verdict[(new, ref, "partial given emotion")]), "partial status": sp,
                      "reading": text})
    tables["reading"] = pd.DataFrame(vrows)

    erows = []
    for new, ref in (("R1_linear", "R0_linear"), ("R3_linear", "R2_linear")):
        jn, jr = jsd(Q[new], Pv), jsd(Q[ref], Pv)
        hn, hr = norm_entropy(Q[new]), norm_entropy(Q[ref])
        for e in sorted(set(emo)):
            sel = np.flatnonzero(emo == e)
            be = Bootstrap(act[sel], N_BOOT)
            erows.append({"comparison": f"{new} minus {ref}", "intended emotion": e, "clips": len(sel),
                          "JSD difference": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean()),
                          "within-class Spearman difference": be(lambda i: spearman(hn[sel][i], hp[sel][i]) - spearman(hr[sel][i], hp[sel][i]))})
    tables["by_emotion_paired"] = pd.DataFrame(erows)
    tables["selection"] = pd.DataFrame(grid)

    lines = ["# Does HSEmotion add information for the AV votes beyond Qwen's readings? (level 1, CPU)\n",
             "Generated by `analysis/av_complement_head.py`. Heads fitted and evaluated within the held-out actors of each "
             "s1_noI checkpoint (3 head folds by actor; inner split inside the head's training part). Reading rules: see the "
             "docstring (fixed before the run). Limit: heads see fewer actors than a full stacking system would.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        if name != "selection":
            lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    sel = tables["selection"].groupby(["features", "head"]).agg(wd_at_edge=("weight decay", lambda s: f"{(s == max(WEIGHT_DECAYS)).mean():.0%} at 1.0, {(s == min(WEIGHT_DECAYS)).mean():.0%} at 1e-4"),
                                                                epoch_at_edge=("epoch", lambda s: f"{(s == MAX_EPOCHS).mean():.0%} at {MAX_EPOCHS}")).reset_index()
    lines += ["## selection at the grid edges\n", markdown(sel), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
