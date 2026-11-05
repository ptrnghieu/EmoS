"""Transfer pilot: frozen HSEmotion features as four extra tokens in Qwen, against continued training with tokens that
carry no clip-specific information. (Design locked 04/11/2026, before any run of the pilot was seen.)

Input: results/transfer/transfer_<arm>_f<k>_s0/ (kaggle/transfer_cremad.ipynb), arms
    hse_tok    the projector receives the clip's HSEmotion features (mean + SD over the A2 face crops, standardised)
    hse_const  the projector receives the zero vector (same prompt, four tokens, architecture and budget)
both continued for one epoch from the s1_noI adapter of fold k (results/stage1/stage1_s1_noI_f<k>_s0), loss
1/2 CE_AV + 1/2 CE_V, last checkpoint, seed 0. hse_const is continued training with extra tokens that carry no
clip-specific information, not continued training with the original prompt.

Metrics per condition c (AV, V, and A as a drift check), on the held-out actors read by the model of their fold:
JSD(q^c, p^c); top label among the raters' top labels; Spearman(H(q^c), A^c); the same partial given the intended
emotion; mean H(q^c). Paired differences with a 95% bootstrap over actors, pooled over the folds present.

Reading, fixed before the runs (hse_tok minus hse_const; "clear" = 95% interval excludes 0 in the favourable direction):
    main metric of the pilot: JSD AV. Pre-specified for the disagreement target: partial Spearman AV.
    JSD V and JSD AV both clearly lower   -> better reproduction of the vote distributions in both conditions
    partial AV clearly higher             -> evidence of better AV disagreement ranking after controlling for emotion
    only JSD V clearly lower              -> the benefit is shown for the visual reading only
    only JSD AV clearly lower             -> benefit in AV, not shown to go through a better V prediction
    intervals contain 0                   -> no clear improvement with the configuration and budget tried
    "Any one metric significant" is not a success rule. Both arms are also read against s1_noI (beating the control
    while staying below the starting point means something else than an absolute gain).
The locked protocol is the three folds pooled; fewer folds are reported as a pipeline check only. A run whose held-out
scores were looked at before changing LR, tokens or epochs is exploratory and is not pooled here.

Checks (hse_tok, held-out actors, AV and V): the features permuted between clips (fixed-seed cyclic permutation)
and the zero input, against the clip's own features. A drop supports use of the match between clip and features;
neither check by itself shows that the features are used in the intended way.

Exploratory round (05/11/2026; --suffix _rms; arms hse_tok_rms / hse_const_rms, fold 2 only; output tokens
RMS-normalised to the text-embedding RMS, nothing else changed). Read hse_tok_rms against hse_const_rms only (the
normalisation also changes how the constant tokens learn), never against the old control, and never pooled with the
locked runs. Decision, fixed before the runs (direction, size and interval read together; one fold need not exclude 0):
    training still unstable                          -> stop the token branch for now
    stable, HSE close to its control on AV JSD and partial AV -> stop; stability fixed, no benefit shown
    only V improves                                  -> keep the V-benefit statement; no further runs hoping for AV
    AV JSD and partial AV both favourable            -> run both arms on folds 0-1 with the configuration unchanged
Fold 2 was chosen because it failed: a recovery there is exploratory evidence only.

References (V only): HSEmotion + head and the feature concatenation (results/hsemotion_head, results/combined_head).

Usage (from the repository root):
    python analysis/transfer_compare.py --runs results/transfer --out results/transfer_compare
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
ARMS = ("hse_tok", "hse_const")
CONDS = ("AV", "V", "A")
KEY = {"AV": "av", "V": "v", "A": "a"}
ROOT = Path(__file__).resolve().parents[1]
REFS = {"HSEmotion + linear head": "hsemotion_head/linear", "HSEmotion + MLP head": "hsemotion_head/mlp",
        "concatenation, linear": "combined_head/linear", "concatenation, MLP": "combined_head/mlp"}


def read_scores(path: Path, cond: str, check: str | None = None) -> pd.DataFrame:
    s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
    s = s[(s.condition == cond) & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
    if check is not None:
        s = s[s.check == check]
    return s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]


def interval(boot: Bootstrap, stat) -> tuple[float, float, float, str]:
    samples = np.array([stat(i) for i in boot.samples])
    point = stat(boot.all)
    lo, hi = np.nanpercentile(samples, [2.5, 97.5])
    return point, lo, hi, fmt(point, samples)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/transfer"))
    parser.add_argument("--stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--out", type=Path, default=Path("results/transfer_compare"))
    parser.add_argument("--suffix", default="", help="_rms: the exploratory RMS-normalised arms (never pooled with the locked runs)")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    folds = [f for f in range(N_OUTER) if all((args.runs / f"transfer_{a}{args.suffix}_f{f}_s0" / "scores.jsonl").exists() for a in ARMS)]
    assert folds, f"no fold with both arms under {args.runs}"
    locked = folds == list(range(N_OUTER)) and not args.suffix
    label = f"exploratory configuration (arms *{args.suffix}), folds {folds}; not pooled with the locked result" if args.suffix else None
    df = clip_table().set_index("uid")
    df["fold"] = df.actor.map(outer_folds(df.reset_index()))

    runs, budget = {}, []
    for a in ARMS:
        for f in folds:
            run = args.runs / f"transfer_{a}{args.suffix}_f{f}_s0"
            st = json.loads((run / "state.json").read_text())
            assert st.get("train_done") and (st["arm"], st["fold"], st["seed"]) == (a + args.suffix, f, 0) and st["start"] == f"stage1_s1_noI_f{f}_s0", run
            assert st["loss_weights"] == [0.5, 0.5, 0.0] and st["n_tok"] == 4, st
            chk = pd.read_csv(run / "start_check.csv") if (run / "start_check.csv").exists() else None
            g = st.get("first_step_grad_norms", {})
            budget.append({"arm": a, "fold": f, "training clips": st["sizes"][0], "optimizer steps": st["opt_step"],
                           "GPU hours logged": round(sum(h.get("session_hours", 0) for h in st["history"]), 1),
                           "final train CE AV / V": " / ".join(f"{st['history'][-1].get(f'train_CE_{c}', np.nan):.3f}" for c in ("AV", "V")),
                           "start check: max |logit diff|": None if chk is None else round(float(chk["max |logit diff| off vs s1_noI"].max()), 4),
                           "first-step grad (LoRA, fc1.weight, fc2)": ", ".join(f"{g.get(k, np.nan):.2e}" for k in ("LoRA", "proj fc1.weight", "proj fc2"))})
    tables = {"runs": pd.DataFrame(budget)}

    def stack(paths, cond, check=None):
        return pd.concat([read_scores(p, cond, check) for p in paths])

    Q, meta = {}, {}
    for c in CONDS:
        preds = {a: stack([args.runs / f"transfer_{a}{args.suffix}_f{f}_s0" / "scores.jsonl" for f in folds], c) for a in ARMS}
        preds["s1_noI (start)"] = stack([args.stage1 / f"stage1_s1_noI_f{f}_s0" / "scores.jsonl" for f in folds], c)
        if c in ("AV", "V"):
            for k in ("permuted", "zero"):
                paths = [args.runs / f"transfer_hse_tok{args.suffix}_f{f}_s0" / "scores_checks.jsonl" for f in folds]
                if all(p.exists() for p in paths):
                    preds[f"hse_tok, {k} features"] = stack(paths, c, k)
        if c == "V":
            for name, path in REFS.items():
                path = ROOT / "results" / path / "scores.jsonl"
                if path.exists():
                    preds[name] = read_scores(path, "V")
        core = [a for a in preds if not a.startswith("HSEmotion") and not a.startswith("concatenation")]
        common = sorted(set(df.index[(df[f"{KEY[c]}_n"] > 0) & df.fold.isin(folds)]).intersection(*(set(preds[a].index) for a in core)))
        refs = [a for a in preds if a not in core and set(common) <= set(preds[a].index)]
        d = df.loc[common]
        P = counts(d, KEY[c])
        P = P / P.sum(1, keepdims=True)
        meta[c] = {"d": d, "P": P, "hp": norm_entropy(P), "emo": d.intended.to_numpy(), "fold": d.fold.to_numpy(),
                   "top": P == P.max(1, keepdims=True), "boot": Bootstrap(d.actor.to_numpy(), N_BOOT)}
        Q[c] = {a: softmax(preds[a].loc[common].to_numpy(float)) for a in core + refs}

    def metrics(c, q):
        m = meta[c]
        j, hit, hq = jsd(q, m["P"]), m["top"][np.arange(len(q)), q.argmax(1)], norm_entropy(q)
        return {f"JSD to {c} votes": lambda i: j[i].mean(), "top label among raters' top": lambda i: hit[i].mean(),
                f"Spearman(H(q), A^{c})": lambda i: spearman(hq[i], m["hp"][i]),
                "partial given emotion": lambda i: partial_spearman(hq[i], m["hp"][i], m["emo"][i]), "mean H(q)": lambda i: hq[i].mean()}

    rows = []
    for c in CONDS:
        m = meta[c]
        for name, q in Q[c].items():
            rows.append({"condition": c, "predictor": name, "clips": len(q), **{k: m["boot"](f) for k, f in metrics(c, q).items()},
                         **{f"JSD fold {f}": round(float(jsd(q, m["P"])[m["fold"] == f].mean()), 3) for f in folds},
                         "human mean H": round(float(m["hp"].mean()), 3)})
    tables["readings"] = pd.DataFrame(rows)

    rows, verdict = [], {}
    pairs = [("hse_tok", "hse_const"), ("hse_tok", "s1_noI (start)"), ("hse_const", "s1_noI (start)"),
             ("hse_tok, permuted features", "hse_tok"), ("hse_tok, zero features", "hse_tok")]
    for c in CONDS:
        for new, ref in pairs:
            if new not in Q[c] or ref not in Q[c]:
                continue
            a, b = metrics(c, Q[c][new]), metrics(c, Q[c][ref])
            rec = {"condition": c, "comparison": f"{new} minus {ref}"}
            for k in a:
                point, lo, hi, text = interval(meta[c]["boot"], lambda i, k=k: a[k](i) - b[k](i))
                rec[k.replace(f" to {c} votes", "").replace(f"A^{c}", "A")] = text
                verdict[(c, new, ref, k.split(" ")[0] if k.startswith("JSD") else k)] = (point, lo, hi)
            rec.update({f"JSD fold {f}": round(float(jsd(Q[c][new], meta[c]["P"])[meta[c]["fold"] == f].mean()
                                                     - jsd(Q[c][ref], meta[c]["P"])[meta[c]["fold"] == f].mean()), 3) for f in folds})
            rows.append(rec)
    tables["paired"] = pd.DataFrame(rows)

    # ---- the pre-specified reading ------------------------------------------------------------------------------
    def clear(c, new, ref, k, lower):
        point, lo, hi = verdict[(c, new, ref, k)]
        return (hi < 0) if lower else (lo > 0)

    rows = []
    for new, ref in (("hse_tok", "hse_const"), ("hse_tok", "s1_noI (start)"), ("hse_const", "s1_noI (start)")):
        jv, jav = clear("V", new, ref, "JSD", True), clear("AV", new, ref, "JSD", True)
        pav = clear("AV", new, ref, "partial given emotion", False)
        if jv and jav:
            dist = "JSD V and AV both clearly lower: better reproduction of the vote distributions in both conditions"
        elif jv:
            dist = "only JSD V clearly lower: the benefit is shown for the visual reading only"
        elif jav:
            dist = "only JSD AV clearly lower: benefit in AV, not shown to go through a better V prediction"
        else:
            worse = [c for c in ("V", "AV") if clear(c, new, ref, "JSD", False)]
            dist = (f"JSD clearly higher in {' and '.join(worse)}: the first predictor is worse there" if worse else
                    "JSD intervals contain 0: no clear improvement with the configuration and budget tried")
        rank = ("partial AV clearly higher: evidence of better AV disagreement ranking after controlling for emotion" if pav
                else "partial AV not clearly higher: no evidence of better AV disagreement ranking")
        rows.append({"comparison": f"{new} minus {ref}", "JSD AV (main)": "{:.4f} [{:.4f}, {:.4f}]".format(*verdict[("AV", new, ref, "JSD")]),
                     "partial AV (pre-specified)": "{:.4f} [{:.4f}, {:.4f}]".format(*verdict[("AV", new, ref, "partial given emotion")]),
                     "JSD V": "{:.4f} [{:.4f}, {:.4f}]".format(*verdict[("V", new, ref, "JSD")]),
                     "distributions": dist, "disagreement ranking": rank,
                     "status": label or ("locked protocol (3 folds)" if locked else f"pipeline check only (folds {folds})")})
    tables["reading"] = pd.DataFrame(rows)

    rows = []
    for c in ("AV", "V"):
        m = meta[c]
        jn, jr = jsd(Q[c]["hse_tok"], m["P"]), jsd(Q[c]["hse_const"], m["P"])
        hn, hr = norm_entropy(Q[c]["hse_tok"]), norm_entropy(Q[c]["hse_const"])
        for e in sorted(set(m["emo"])):
            sel = np.flatnonzero(m["emo"] == e)
            be = Bootstrap(m["d"].actor.to_numpy()[sel], N_BOOT)
            rows.append({"condition": c, "intended emotion": e, "clips": len(sel),
                         "JSD difference (hse_tok minus hse_const)": be(lambda i: jn[sel][i].mean() - jr[sel][i].mean()),
                         "within-class Spearman difference": be(lambda i: spearman(hn[sel][i], m["hp"][sel][i]) - spearman(hr[sel][i], m["hp"][sel][i]))})
    tables["by_emotion_paired"] = pd.DataFrame(rows)

    status = label or ("three folds pooled (locked protocol)" if locked else f"folds {folds} only: pipeline check, not the locked result")
    lines = ["# Transfer pilot: HSEmotion features as extra tokens in Qwen vs continued training with uninformative tokens\n",
             f"Generated by `analysis/transfer_compare.py`. {status}. Reading rules: see the docstring (fixed before the runs). "
             "`hse_const` is continued training with extra tokens that carry no clip-specific information, not continued "
             "training with the original prompt. Both arms have the same architecture, number of tokens and optimisation "
             "budget; they differ in whether the projector receives the clip's features.\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
