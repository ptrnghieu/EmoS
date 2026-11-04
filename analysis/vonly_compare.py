"""Is the weak video-only prediction a limit of the representation, or partly a consequence of joint adaptation?
(Controlled V-only baseline, 02/11/2026.)

Video-only readings of the same held-out actors (3 outer folds over actors, seed 0; each clip read by the model of the
fold that never saw its actor), against the video-only raters' votes p^V:
    base     zero-shot Qwen2.5-Omni-7B (results/cremad_qwen/scores.jsonl, condition V)
    m3       LoRA trained on the AV votes only (V never supervised)
    s1_noI   LoRA trained on AV + V + A votes, V loss weight 0.1, A 0.5 (AV-criterion lambda search)
    v_only   same model, LoRA, input, clips, epochs and optimizer steps as s1_noI; the V task alone, weight 1

Metrics: JSD(q^V, p^V) (lower is better); top label among the raters' top labels; Spearman(H(q^V), H(p^V)); the same
partial given the intended emotion; mean H(q^V). Paired differences v_only minus s1_noI (and minus m3), 95% bootstrap
over actors, and per fold. References for JSD: the simulated human re-runs of cross_condition_diagnostic.py
(Dirichlet(counts + 0.5) and empirical Bayes). The training budget of each arm is reported beside the results.

Reading (to be confirmed before the runs are seen):
    v_only clearly better than s1_noI (JSD lower with CI below 0 and at least a third of the gap to the
    empirical-Bayes re-run closed) -> joint training / task weighting limits V learning: study that first
    similar -> the representation is the limit: compare a dedicated visual emotion model, same supervision and splits
    JSD better but entropy ranking (Spearman, partial) not -> the remaining problem is narrower than recognition

Usage (from the repository root; downloads the CREMA-D rating files):
    python analysis/vonly_compare.py --out results/vonly_compare
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import counts, fit_dirichlet, jsd
from preprocess_cremad import LABELS, clip_table
from rq3_analysis import Bootstrap, softmax
from rq_analysis import markdown, spearman
from visual_probe_analysis import N_OUTER, norm_entropy, outer_folds, partial_spearman

N_BOOT, N_REP = 1000, 20


def read_scores(path: Path, cond: str = "V") -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    return s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/stage1"))
    parser.add_argument("--zero-shot", type=Path, default=Path("results/cremad_qwen/scores.jsonl"))
    parser.add_argument("--out", type=Path, default=Path("results/vonly_compare"))
    parser.add_argument("--extra", nargs="*", default=[], help="name=path of further video-only predictions (scores.jsonl, condition V)")
    parser.add_argument("--pairs", nargs="*", default=[], help="further paired comparisons new:reference (predictor names)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))
    arms, budget = {}, []
    if args.zero_shot.exists():
        arms["base (zero-shot)"] = read_scores(args.zero_shot)
    for arm in ("m3", "s1_noI", "v_only"):
        parts, hours, steps, clips = [], 0.0, 0, 0
        for f in range(N_OUTER):
            run = args.runs / f"stage1_{arm}_f{f}_s0"
            if not (run / "scores.jsonl").exists():
                break
            st = json.loads((run / "state.json").read_text())
            assert st.get("train_done") and st.get("phase") == "outer", run
            parts.append(read_scores(run / "scores.jsonl"))
            hours += sum(h.get("session_hours", 0) for h in st["history"])
            steps += st["opt_step"]
            clips += st["sizes"][0]
        if len(parts) == N_OUTER:
            arms[arm] = pd.concat(parts)
            readings = {"m3": "AV", "s1_noI": "AV + V + A", "v_only": "V"}[arm]
            weight = {"m3": "0 (not trained)", "s1_noI": f"{st['lambdas'][0]:g}", "v_only": f"{st['lambdas'][0]:g}"}[arm]
            budget.append({"arm": arm, "readings trained per clip": readings, "V loss weight": weight,
                           "V-supervised clips (3 folds, 1 epoch)": clips if arm != "m3" else 0, "optimizer steps (3 folds)": steps,
                           "GPU hours logged (last session per fold)": round(hours, 1)})
    for item in args.extra:
        name, path = item.split("=", 1)
        arms[name] = read_scores(Path(path))
    common = sorted(set(df.index[df.v_n > 0]).intersection(*(set(a.index) for a in arms.values())))
    d = df.loc[common]
    P = counts(d, "v")
    P = P / P.sum(1, keepdims=True)
    hp, emo, folds = norm_entropy(P), d.intended.to_numpy(), d.fold.to_numpy()
    top = P == P.max(1, keepdims=True)
    boot = Bootstrap(d.actor.to_numpy(), N_BOOT)
    Q = {name: softmax(a.loc[common].to_numpy(float)) for name, a in arms.items()}

    def metrics(q):
        j, hit, hq = jsd(q, P), top[np.arange(len(q)), q.argmax(1)], norm_entropy(q)
        return {"JSD to V votes": lambda i: j[i].mean(), "top label among raters' top": lambda i: hit[i].mean(),
                "Spearman(H(q), H(p))": lambda i: spearman(hq[i], hp[i]),
                "partial given emotion": lambda i: partial_spearman(hq[i], hp[i], emo[i]),
                "mean H(q)": lambda i: hq[i].mean()}

    rows = [{"predictor": n, **{k: boot(f) for k, f in metrics(q).items()},
             **{f"JSD fold {f}": round(float(jsd(q, P)[folds == f].mean()), 3) for f in range(N_OUTER)}} for n, q in Q.items()]
    C = counts(d, "v")
    alpha, pi = fit_dirichlet(C)
    rng = np.random.default_rng(0)
    for label, extra in (("human re-run, Dirichlet(counts + 0.5)", 0.5), ("human re-run, empirical Bayes", alpha * pi)):
        js = []
        for _ in range(N_REP):
            g = rng.gamma(C + extra)
            js.append(jsd(rng.multinomial(C.sum(1).astype(int), g / g.sum(1, keepdims=True)) / C.sum(1, keepdims=True), P).mean())
        rows.append({"predictor": label, "JSD to V votes": f"{np.mean(js):.3f}", "mean H(q)": f"(human {hp.mean():.3f})"})
    tables = {"v_reading": pd.DataFrame(rows)}
    rows = []
    extra_pairs = [tuple(x.split(":", 1)) for x in args.pairs]
    pairs = [("v_only", "s1_noI"), ("v_only", "m3")] + [(name.split("=")[0], "v_only") for name in args.extra] + extra_pairs
    for new, ref in pairs:
        if new in Q and ref in Q:
            a, b = metrics(Q[new]), metrics(Q[ref])
            rows.append({"comparison": f"{new} minus {ref}", **{k: boot(lambda i, k=k: a[k](i) - b[k](i)) for k in a},
                         **{f"JSD fold {f}": round(float(jsd(Q[new], P)[folds == f].mean() - jsd(Q[ref], P)[folds == f].mean()), 3)
                            for f in range(N_OUTER)}})
    if rows:
        tables["paired"] = pd.DataFrame(rows)
    tables["training_budget"] = pd.DataFrame(budget)
    # paired differences within each intended emotion: JSD and the within-class Spearman of entropies
    rows = []
    for new_, ref in [(name.split("=")[0], "v_only") for name in args.extra] + [("v_only", "s1_noI")] + extra_pairs:
        if new_ not in Q or ref not in Q:
            continue
        jn, jr = jsd(Q[new_], P), jsd(Q[ref], P)
        hn, hr = norm_entropy(Q[new_]), norm_entropy(Q[ref])
        for e in sorted(set(emo)):
            sel = np.flatnonzero(emo == e)
            be = Bootstrap(d.actor.to_numpy()[sel], N_BOOT)
            rows.append({"comparison": f"{new_} minus {ref}", "intended emotion": e, "clips": len(sel),
                         "JSD difference": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean()),
                         "within-class Spearman: first": be(lambda i: spearman(hn[sel][i], hp[sel][i])),
                         "within-class Spearman: second (reference)": be(lambda i: spearman(hr[sel][i], hp[sel][i])),
                         "within-class Spearman difference": be(lambda i: spearman(hn[sel][i], hp[sel][i]) - spearman(hr[sel][i], hp[sel][i]))})
    if rows:
        tables["by_emotion_paired"] = pd.DataFrame(rows)
    tables["by_emotion"] = pd.DataFrame([{"intended emotion": e, "clips": int((emo == e).sum()), "human mean H": round(float(hp[emo == e].mean()), 3),
                                          **{f"{n}: JSD": round(float(jsd(q, P)[emo == e].mean()), 3) for n, q in Q.items()},
                                          **{f"{n}: mean H": round(float(norm_entropy(q)[emo == e].mean()), 3) for n, q in Q.items()}}
                                         for e in sorted(set(emo))])

    # ---- metric diagnostic: s1_noI's A and AV readings kept, its V reading replaced by v_only's ----------------
    # "AV disambiguation" = H(AV) < min(H(V), H(A)) - 0.05 (cross_condition_diagnostic.py). Not a fusion model:
    # it shows whether the better V reading changes the per-emotion over-flagging of that metric.
    if "v_only" in Q and "s1_noI" in Q:
        s1 = {c: pd.concat([read_scores(args.runs / f"stage1_s1_noI_f{f}_s0" / "scores.jsonl", c) for f in range(N_OUTER)]) for c in ("A", "AV")}
        cc = [u for u in common if df.loc[u, "a_n"] > 0 and df.loc[u, "av_n"] > 0 and u in s1["A"].index and u in s1["AV"].index]
        pos = {u: i for i, u in enumerate(common)}
        ix = np.array([pos[u] for u in cc])
        dd = df.loc[cc]
        Ph = {c: counts(dd, k) for c, k in (("V", "v"), ("A", "a"), ("AV", "av"))}
        Ph = {c: v / v.sum(1, keepdims=True) for c, v in Ph.items()}
        H = norm_entropy
        dis = lambda v, a, av: H(av) < np.minimum(H(v), H(a)) - 0.05
        human = dis(Ph["V"], Ph["A"], Ph["AV"])
        qa, qav = softmax(s1["A"].loc[cc].to_numpy(float)), softmax(s1["AV"].loc[cc].to_numpy(float))
        flags = {"s1_noI (own V reading)": dis(Q["s1_noI"][ix], qa, qav), "s1_noI with v_only's V reading": dis(Q["v_only"][ix], qa, qav)}
        Cn = {c: counts(dd, k) for c, k in (("V", "v"), ("A", "a"), ("AV", "av"))}
        rr = np.random.default_rng(0)
        reruns = []
        for _ in range(N_REP):
            rep = {}
            for c, Cc in Cn.items():
                g = rr.gamma(Cc + 0.5)
                rep[c] = rr.multinomial(Cc.sum(1).astype(int), g / g.sum(1, keepdims=True)) / Cc.sum(1, keepdims=True)
            reruns.append(dis(rep["V"], rep["A"], rep["AV"]))
        em = dd.intended.to_numpy()
        rows = []
        for e in ["all"] + sorted(set(em)):
            sel = np.ones(len(cc), bool) if e == "all" else em == e
            rec = {"intended emotion": e, "clips": int(sel.sum()), "human rate": round(float(human[sel].mean()), 3)}
            for name, fl in list(flags.items()) + [("human re-run", None)]:
                if fl is None:
                    rate = np.mean([x[sel].mean() for x in reruns])
                    prec = np.nanmean([human[sel & x].mean() for x in reruns])
                    fpr = np.mean([x[sel & ~human].mean() for x in reruns])
                    rec_ = np.mean([x[sel & human].mean() for x in reruns])
                else:
                    rate, rec_, fpr = fl[sel].mean(), fl[sel & human].mean(), fl[sel & ~human].mean()
                    prec = human[sel & fl].mean() if (sel & fl).any() else np.nan
                rec.update({f"{name}: rate": round(float(rate), 3), f"{name}: recall": round(float(rec_), 3),
                            f"{name}: precision": round(float(prec), 3), f"{name}: FPR": round(float(fpr), 3)})
            rows.append(rec)
        tables["disambiguation_with_v_only_V"] = pd.DataFrame(rows)
        a, b = flags["s1_noI with v_only's V reading"], flags["s1_noI (own V reading)"]
        bb = Bootstrap(dd.actor.to_numpy(), N_BOOT)
        tables["disambiguation_swap_paired"] = pd.DataFrame([{
            "comparison": "v_only's V minus s1_noI's own V (A and AV readings of s1_noI)",
            "rate": bb(lambda i: a[i].mean() - b[i].mean()),
            "recall": bb(lambda i: a[i][human[i]].mean() - b[i][human[i]].mean()),
            "precision": bb(lambda i: human[i][a[i]].mean() - human[i][b[i]].mean()),
            "FPR": bb(lambda i: a[i][~human[i]].mean() - b[i][~human[i]].mean())}])

    lines = ["# Video-only reading: joint training (s1_noI) against the V task alone (v_only)\n",
             f"Generated by `analysis/vonly_compare.py`. {len(common)} clips, {d.actor.nunique()} actors, read by the fold "
             f"model that never saw the actor. CIs: 95% bootstrap over actors ({N_BOOT}).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
