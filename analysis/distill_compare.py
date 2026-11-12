"""V distillation pilot: reading fixed before the runs (11/11/2026).

Runs: results/distill/distill_<arm>_f<k>_s0/ (kaggle/distill_cremad.ipynb), arms cont (human votes) and distill
(V target 2/3 votes + 1/3 cross-fitted HSE-only teacher), both continued one epoch from s1_noI of fold k with
1/2 CE_AV + 1/2 CE_V. Held-out actors of each fold, three folds pooled, paired bootstrap over actors.

Transfer (distill minus cont, V reading): JSD V clearly lower -> better fit of the V vote distribution; a better
disagreement ranking needs partial V clearly higher as well.
AV preservation, margins set by the project (not derived from the expected CI widths), distill minus cont:
    delta JSD AV: upper 95% bound < +0.002;   delta partial AV: lower 95% bound > -0.02
  per metric: "meets the margin" (whole CI on the acceptable side) / "not shown" (CI crosses the margin) /
  "degradation beyond the margin" (whole CI on the wrong side). Not meeting a margin is not by itself a proven trade-off.
Overall:
    V better, AV meets both margins       -> V transfer useful, AV preserved (relative to cont)
    V and AV both clearly better           -> further evidence that the benefit reaches AV
    V better, AV degradation beyond margin -> trade-off; not the joint goal
    V better, AV margins not shown         -> V transfer shown; AV preservation not shown
    V not better                           -> this configuration did not transfer the teacher's benefit
Both arms are also read against s1_noI (cont and distill alike may have moved from the start).
Retained share of the teacher's V benefit (secondary): (JSD_cont - JSD_distill) / (JSD_cont - JSD_teacher), teacher =
mean of the five cross-fitted teachers on the held-out actors (an ensemble reference). Reported only when the
denominator's bootstrap interval is entirely above 0; otherwise the absolute JSD differences are reported.

Usage (from the repository root):
    python analysis/distill_compare.py --runs results/distill --teacher results/distill_teacher --out results/distill_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, fmt, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

N_BOOT = 1000
ARMS = ("cont", "distill")
KEY = {"AV": "av", "V": "v", "A": "a"}
MARGIN_JSD, MARGIN_PARTIAL = 0.002, -0.02
ROOT = Path(__file__).resolve().parents[1]
REFS = {"HSEmotion + linear head (all train actors)": "hsemotion_head/linear", "HSEmotion + MLP head (all train actors)": "hsemotion_head/mlp"}


def read_scores(path: Path, cond: str) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    return s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/distill"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--teacher", type=Path, default=Path("results/distill_teacher"))
    parser.add_argument("--out", type=Path, default=Path("results/distill_compare"))
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    folds = [f for f in range(N_OUTER) if all((args.runs / f"distill_{a}_f{f}_s0" / "scores.jsonl").exists() for a in ARMS)]
    assert folds, "no fold with both arms"
    status = "three folds pooled (locked protocol)" if folds == list(range(N_OUTER)) else f"folds {folds} only: technical check, not the locked result"
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))

    runs = []
    for a in ARMS:
        for f in folds:
            st = json.loads((args.runs / f"distill_{a}_f{f}_s0" / "state.json").read_text())
            assert st.get("train_done") and (st["arm"], st["fold"], st["seed"]) == (a, f, 0) and st["start"] == f"stage1_s1_noI_f{f}_s0", st
            assert st["loss_weights"] == [0.5, 0.5, 0.0] and abs(st["alpha"] - (0.0 if a == "cont" else 1 / 3)) < 1e-9, st
            runs.append({"arm": a, "fold": f, "optimizer steps": st["opt_step"], "OOM": st["oom"],
                         "train CE AV / V": " / ".join(f"{st['history'][-1].get(f'train_CE_{c}', np.nan):.3f}" for c in ("AV", "V")),
                         "GPU hours logged": round(sum(h.get("session_hours", 0) for h in st["history"]), 1)})
    tables = {"runs": pd.DataFrame(runs)}

    teacher = pd.concat([pd.read_csv(args.teacher / f"teacher_v_f{f}.csv").query("role == 'heldout'") for f in folds]).set_index("uid")[LABELS]
    Q, meta = {}, {}
    for c in ("AV", "V", "A"):
        pr = {a: pd.concat([read_scores(args.runs / f"distill_{a}_f{f}_s0" / "scores.jsonl", c) for f in folds]) for a in ARMS}
        pr["s1_noI (start)"] = pd.concat([read_scores(args.stage1 / f"stage1_s1_noI_f{f}_s0" / "scores.jsonl", c) for f in folds])
        common = sorted(set(df.index[(df[f"{KEY[c]}_n"] > 0) & df.fold.isin(folds)]).intersection(*(set(p.index) for p in pr.values())))
        Q[c] = {a: softmax(p.loc[common].to_numpy(float)) for a, p in pr.items()}
        if c == "V":
            common_t = [u for u in common if u in teacher.index]
            assert len(common_t) == len(common), "teacher reference missing for held-out clips"
            Q[c]["teacher (5 cross-fit mean)"] = teacher.loc[common].to_numpy(float)
            for name, path in REFS.items():
                s = pd.DataFrame([json.loads(l) for l in open(ROOT / "results" / path / "scores.jsonl") if l.strip()]).set_index("uid")
                if set(common) <= set(s.index):
                    Q[c][name] = softmax(s.loc[common, [f"logit_{l}" for l in LABELS]].to_numpy(float))
        d = df.loc[common]
        P = counts(d, KEY[c])
        P = P / P.sum(1, keepdims=True)
        meta[c] = {"d": d, "P": P, "hp": norm_entropy(P), "emo": d.intended.to_numpy(), "fold": d.fold.to_numpy(),
                   "top": P == P.max(1, keepdims=True), "boot": Bootstrap(d.actor.to_numpy(), N_BOOT)}

    def metrics(c, q):
        m = meta[c]
        j, hit, hq = jsd(q, m["P"]), m["top"][np.arange(len(q)), q.argmax(1)], norm_entropy(q)
        return {"JSD": lambda i: j[i].mean(), "partial": lambda i: partial_spearman(hq[i], m["hp"][i], m["emo"][i]),
                "Spearman(H(q), A)": lambda i: spearman(hq[i], m["hp"][i]), "top label among raters' top": lambda i: hit[i].mean(),
                "mean H(q)": lambda i: hq[i].mean()}

    tables["readings"] = pd.DataFrame([{"condition": c, "predictor": n, **{k: meta[c]["boot"](f) for k, f in metrics(c, q).items()},
                                        **{f"JSD fold {f}": round(float(jsd(q, meta[c]["P"])[meta[c]["fold"] == f].mean()), 4) for f in folds},
                                        "human mean H": round(float(meta[c]["hp"].mean()), 3)}
                                       for c in ("AV", "V", "A") for n, q in Q[c].items()])
    stat, prow = {}, []
    for c in ("AV", "V", "A"):
        for new, ref in (("distill", "cont"), ("distill", "s1_noI (start)"), ("cont", "s1_noI (start)")):
            a, b = metrics(c, Q[c][new]), metrics(c, Q[c][ref])
            rec = {"condition": c, "comparison": f"{new} minus {ref}"}
            for k in a:
                samples = np.array([a[k](i) - b[k](i) for i in meta[c]["boot"].samples])
                point = a[k](meta[c]["boot"].all) - b[k](meta[c]["boot"].all)
                lo, hi = np.nanpercentile(samples, [2.5, 97.5])
                rec[k] = fmt(point, samples)
                stat[(c, new, ref, k)] = (point, lo, hi)
            rec.update({f"JSD fold {f}": round(float(jsd(Q[c][new], meta[c]["P"])[meta[c]["fold"] == f].mean()
                                                     - jsd(Q[c][ref], meta[c]["P"])[meta[c]["fold"] == f].mean()), 4) for f in folds})
            prow.append(rec)
    tables["paired"] = pd.DataFrame(prow)

    # ---- the pre-specified reading -------------------------------------------------------------------------------
    _, lj, hj = stat[("V", "distill", "cont", "JSD")]
    _, lp, hp_ = stat[("V", "distill", "cont", "partial")]
    v_better, v_rank = hj < 0, lp > 0
    _, lja, hja = stat[("AV", "distill", "cont", "JSD")]
    _, lpa, hpa = stat[("AV", "distill", "cont", "partial")]
    ni_jsd = "meets the margin" if hja < MARGIN_JSD else "degradation beyond the margin" if lja > MARGIN_JSD else "not shown"
    ni_par = "meets the margin" if lpa > MARGIN_PARTIAL else "degradation beyond the margin" if hpa < MARGIN_PARTIAL else "not shown"
    av_better = hja < 0
    if not v_better:
        overall = "V not better: this configuration did not transfer the teacher's benefit"
    elif av_better and ni_par == "meets the margin":
        overall = "V and AV both better: further evidence that the benefit reaches AV"
    elif ni_jsd == "meets the margin" and ni_par == "meets the margin":
        overall = "V transfer useful, AV preserved within the project's margins (relative to cont)"
    elif "degradation beyond the margin" in (ni_jsd, ni_par):
        overall = "V better but AV degraded beyond a margin: trade-off, not the joint goal"
    else:
        overall = "V transfer shown; AV preservation not shown (CI crosses a margin)"
    tables["reading"] = pd.DataFrame([{
        "JSD V (distill - cont)": "{:.4f} [{:.4f}, {:.4f}]".format(*stat[("V", "distill", "cont", "JSD")]),
        "partial V": "{:.4f} [{:.4f}, {:.4f}]".format(*stat[("V", "distill", "cont", "partial")]),
        "V": ("JSD clearly lower" if v_better else "JSD not clearly lower") + ("; partial clearly higher (ranking)" if v_rank else "; partial not clearly higher"),
        "JSD AV": "{:.4f} [{:.4f}, {:.4f}]".format(*stat[("AV", "distill", "cont", "JSD")]), "AV JSD margin (+0.002)": ni_jsd,
        "partial AV": "{:.4f} [{:.4f}, {:.4f}]".format(*stat[("AV", "distill", "cont", "partial")]), "AV partial margin (-0.02)": ni_par,
        "overall": overall, "status": status}])

    # ---- retained share of the teacher's V benefit (secondary) ---------------------------------------------------
    m = meta["V"]
    jc, jd, jt = (jsd(Q["V"][n], m["P"]) for n in ("cont", "distill", "teacher (5 cross-fit mean)"))
    den = np.array([jc[i].mean() - jt[i].mean() for i in m["boot"].samples])
    num = np.array([jc[i].mean() - jd[i].mean() for i in m["boot"].samples])
    dlo = np.nanpercentile(den, 2.5)
    rec = {"JSD cont - teacher": fmt(jc.mean() - jt.mean(), den), "JSD cont - distill": fmt(jc.mean() - jd.mean(), num)}
    rec["retained share"] = fmt((jc.mean() - jd.mean()) / (jc.mean() - jt.mean()), num / den) if dlo > 0 else \
        "not reported: the teacher's advantage over cont is not clearly above 0"
    tables["retained"] = pd.DataFrame([rec])

    rows = []
    for c in ("V", "AV"):
        mm = meta[c]
        jn, jr = jsd(Q[c]["distill"], mm["P"]), jsd(Q[c]["cont"], mm["P"])
        for e in sorted(set(mm["emo"])):
            sel = np.flatnonzero(mm["emo"] == e)
            be = Bootstrap(mm["d"].actor.to_numpy()[sel], N_BOOT)
            rows.append({"condition": c, "intended emotion": e, "clips": len(sel), "JSD difference (distill - cont)": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean())})
    tables["by_emotion_paired"] = pd.DataFrame(rows)

    lines = ["# V distillation pilot: HSE-only teacher mixed into the V target vs continued training on the votes\n",
             f"Generated by `analysis/distill_compare.py`. {status}. Reading rules in the docstring (fixed before the runs). "
             "Reference not recomputed here: the linear residual readout (results/av_complement_residual) reached JSD V 0.129 "
             "with heads fitted within the held-out actors.\n"]
    for name, t in tables.items():
        t.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(t.astype(object).where(t.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
