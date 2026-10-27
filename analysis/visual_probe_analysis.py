"""Experiment B (+ A as an input ablation): is the video-only raters' disagreement decodable from Qwen2.5-Omni-7B's
visual representations, where, and does a different visual input change it? (Design approved 26/10/2026.)

Inputs: results/visual_probe/vprobe_<model>_<view>[_f<fold>]/ (shard_*.npz, meta.json) from kaggle/probe_visual.ipynb.
Every clip read once under video only. Saved per clip:
    vis            vision encoder output (visual tokens as handed to the language model), mean over tokens  [place 1]
    vid7 ... vid28 residual stream at the video-token positions after layer 7/14/21/28, mean over tokens   [place 2]
    last7 ... last28, norm   residual stream at the answer position after those layers / after the final norm [place 3]
    logits         the six label logits (the model's own prediction, U_model = H(softmax) / log 6)
    grid           (base, A0 only) vision encoder tokens pooled to 4 x 4 cells per temporal patch

Targets (humans): p_V = the video-only raters' vote distribution (about 10 per clip); A^V = H(p_V) / log 6; A^AV the
same for the audio-visual raters.

Probe (fixed before the runs): softmax regression from one saved vector to the six labels, trained on p_V as a soft
target (mean cross-entropy + lambda ||W||^2, features standardised with the training clips' statistics); lambda in
{1, 0.1, 0.01, 1e-3, 1e-4} chosen by the cross-entropy on 15% of the training actors held out, then refitted on all
training actors. U_probe = H(probe prediction) / log 6. The same procedure on the six logits gives `logits
(recalibrated)`: what a new linear read-out of the model's own label scores reaches.
    P1 (base model; main result): the three Stage 1 outer folds over actors; train on two folds, predict the third.
    P2 (every model; the comparison across checkpoints): within each outer fold's actors only (the actors that fold's
       Stage 1 adapter never saw), five folds over actors. Base is run under P2 as well, on the same clips.

Comparisons are always within one checkpoint, on the same clips: U_probe vs U_model of the same model.
    metrics      Spearman with A^V and with A^AV; partial Spearman with A^V given the intended emotion (ranks
                 residualised on emotion indicators, the main check against an emotion prior); AUROC for the top
                 quarter of A^V and of A^AV (thresholds from the other folds' actors); cross-entropy to p_V
    references   emotion prior (mean A^V of the training actors' clips per intended emotion; same folds as the
                 probe), emotion x sentence prior, U_model
    per class    n, Spearman with CI, spread of A^V and U within each intended emotion (descriptive)
    attention    base A0, P1: attention pooling vs mean pooling over the grid tokens, same head and training
                 (weight decay by the held-out actors' cross-entropy, early stopping on them)
    views        base, P1, `vis` and `norm` and U_model: A1/A2/A3 minus A0 on the clips every view has. A view goes on
                 to training only if Spearman(U_probe, A^V) at `norm` improves with the 95% CI above 0, the
                 partial Spearman given emotion improves too, and the gain stays positive when any one emotion is
                 left out (leave-one-emotion-out)
    reliability  of A^V: random split-half of each clip's raters (repeated), the Dirichlet-multinomial check
                 (about 7 raters predict 3, empirical vs simulated) and the DM ceiling (expected Spearman of a
                 perfect predictor with the observed A^V), as for A^AV in cremad_human.py

All CIs: 95% bootstrap over actors (paired where a difference is shown).

Usage (from the repository root; downloads the CREMA-D rating files; the attention probe needs torch):
    python analysis/visual_probe_analysis.py --runs results/visual_probe --out results/visual_probe_analysis
"""

from __future__ import annotations

import argparse
import json
import multiprocessing
import os
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from cremad_human import ceiling, counts, empirical_split
from preprocess_cremad import KNOWN_BAD, LABELS, clip_table
from probe_analysis import fit_softmax, predict
from rq3_analysis import N_BOOT, Bootstrap, softmax
from rq_analysis import auroc, markdown, spearman

K = len(LABELS)
LAMBDAS = [1.0, 1e-1, 1e-2, 1e-3, 1e-4]
N_OUTER, SPLIT_SEED, INNER_SHARE, N_INNER_CV = 3, 0, 0.15, 5
NAME = re.compile(r"vprobe_(?P<model>base|m3|s1_noI)_(?P<view>A[0-3])(_f(?P<fold>\d))?$")
PRIMARY = "norm"
DATA: dict = {}  # filled before the worker pool forks
CACHE: list = [None]


def norm_entropy(p: np.ndarray) -> np.ndarray:
    return -np.sum(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0), 1) / np.log(K)


def outer_folds(df: pd.DataFrame) -> dict[int, int]:
    """The Stage 1 outer folds (kaggle/build_notebooks.py, STAGE1), rebuilt with the same seed."""
    rng = np.random.default_rng(SPLIT_SEED)
    actor_sex = df.groupby("actor")["Sex"].first()
    out = {}
    for _, group in actor_sex.groupby(actor_sex):
        for k, a in enumerate(rng.permutation(group.index.to_numpy())):
            out[int(a)] = k % N_OUTER
    return out


def load_run(path: Path) -> tuple[list[str], dict[str, np.ndarray], dict]:
    uids, parts, grid, grid_len = [], {}, [], []
    for p in sorted(path.glob("shard_*.npz")):
        z = np.load(p, allow_pickle=False)
        uids += z["uid"].tolist()
        for k in z.files:
            if k == "grid":
                grid.append(z[k])
            elif k == "grid_len":
                grid_len.append(z[k])
            elif k != "uid":
                parts.setdefault(k, []).append(z[k])
    arrays = {k: np.concatenate(v) for k, v in parts.items()}
    if grid:
        lens = np.concatenate(grid_len)
        flat = np.concatenate(grid)
        starts = np.concatenate([[0], np.cumsum(lens)[:-1]])
        arrays["grid"] = [flat[s:s + n] for s, n in zip(starts, lens)]
    _, first = np.unique(uids, return_index=True)  # a clip saved twice (resumed run): keep the first copy
    first = np.sort(first)
    arrays = {k: ([v[i] for i in first] if isinstance(v, list) else v[first]) for k, v in arrays.items()}
    meta = json.loads((path / "meta.json").read_text()) if (path / "meta.json").exists() else {}
    return [uids[i] for i in first], arrays, meta


def splits(actors: np.ndarray, folds: np.ndarray, protocol: str) -> list[tuple[np.ndarray, np.ndarray]]:
    """(train rows, test rows) pairs: P1 over the outer folds, P2 within each outer fold over its actors."""
    if protocol == "P1":
        return [(np.flatnonzero(folds != f), np.flatnonzero(folds == f)) for f in range(N_OUTER)]
    out = []
    for f in range(N_OUTER):
        rows = np.flatnonzero(folds == f)
        acts = np.random.default_rng(SPLIT_SEED + 10 + f).permutation(pd.unique(actors[rows]))
        sub = {a: k % N_INNER_CV for k, a in enumerate(acts)}
        s = np.array([sub[a] for a in actors[rows]])
        out += [(rows[s != k], rows[s == k]) for k in range(N_INNER_CV)]
    return out


def inner_holdout(actors: np.ndarray, seed: int) -> np.ndarray:
    acts = np.random.default_rng(seed).permutation(pd.unique(actors))
    held = set(acts[:max(1, round(INNER_SHARE * len(acts)))])
    return np.array([a in held for a in actors])


def probe_oof(X: np.ndarray, Y: np.ndarray, actors: np.ndarray, folds: np.ndarray, protocol: str):
    """Out-of-fold predicted distributions and the chosen lambda per split."""
    P, chosen = np.full((len(X), K), np.nan), []
    for j, (tr, te) in enumerate(splits(actors, folds, protocol)):
        mu, sd = X[tr].mean(0), X[tr].std(0) + 1e-6
        Z = ((X - mu) / sd).astype(np.float32)
        held = inner_holdout(actors[tr], seed=100 + j)
        a, b = tr[~held], tr[held]
        W, best = None, None
        for lam in LAMBDAS:  # large to small, warm-started
            W = fit_softmax(Z[a], Y[a], lam, W)
            Pb = predict(W, Z[b])
            loss = float(-(Y[b] * np.log(np.clip(Pb, 1e-12, None))).sum(1).mean())
            if best is None or loss < best[0]:
                best = (loss, lam, W)
        W = fit_softmax(Z[tr], Y[tr], best[1], best[2])
        P[te] = predict(W, Z[te])
        chosen.append(best[1])
    return P, chosen


def _job(key):
    run, feature, protocol = key
    cache = CACHE[0] / f"{run}__{feature}__{protocol}.npz"
    if cache.exists():  # finished in an earlier run of this script
        z = np.load(cache)
        return key, z["P"], z["lams"].tolist()
    d = DATA[run]
    X = d["arrays"][feature].astype(np.float32)
    P, lams = probe_oof(X, d["Y"], d["actors"], d["folds"], protocol)
    np.savez(cache, P=P, lams=np.array(lams))
    return key, P, lams


def prior_oof(values: np.ndarray, group: np.ndarray, actors: np.ndarray, folds: np.ndarray, protocol: str) -> np.ndarray:
    out = np.full(len(values), np.nan)
    for tr, te in splits(actors, folds, protocol):
        means = pd.Series(values[tr]).groupby(group[tr]).mean()
        out[te] = pd.Series(group[te]).map(means).fillna(values[tr].mean()).to_numpy()
    return out


def attention_oof(tokens: list, Y: np.ndarray, actors: np.ndarray, folds: np.ndarray, pooling: str, seed: int = 0):
    """Grid tokens -> (mean | attention) pooling -> linear -> softmax, trained on p_V. P1 splits."""
    import torch

    torch.manual_seed(seed)
    d = tokens[0].shape[1]
    P = np.full((len(tokens), K), np.nan)

    def batch(rows, mu, sd):
        L = max(len(tokens[i]) for i in rows)
        x = np.zeros((len(rows), L, d), np.float32)
        m = np.zeros((len(rows), L), bool)
        for r, i in enumerate(rows):
            x[r, :len(tokens[i])] = (tokens[i].astype(np.float32) - mu) / sd
            m[r, :len(tokens[i])] = True
        return torch.from_numpy(x), torch.from_numpy(m)

    class Head(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.q = torch.nn.Parameter(torch.zeros(d))  # zeros: starts as mean pooling
            self.out = torch.nn.Linear(d, K)

        def forward(self, x, m):
            s = (x @ self.q) if pooling == "attention" else torch.zeros(m.shape)
            w = torch.softmax(s.masked_fill(~m, -1e9), dim=1)
            return self.out((w.unsqueeze(-1) * x).sum(1))

    def train(rows, val, wd, mu, sd, epochs=40, patience=5):
        head = Head()
        opt = torch.optim.Adam(head.parameters(), lr=1e-3)
        xv, mv = batch(val, mu, sd)
        yv = torch.from_numpy(Y[val]).float()
        best, state, bad = np.inf, None, 0
        rng = np.random.default_rng(seed)
        for _ in range(epochs):
            order = rows[rng.permutation(len(rows))]
            for k in range(0, len(order), 128):
                x, m = batch(order[k:k + 128], mu, sd)
                y = torch.from_numpy(Y[order[k:k + 128]]).float()
                loss = -(y * torch.log_softmax(head(x, m), -1)).sum(-1).mean() + wd * (head.out.weight ** 2).sum()
                opt.zero_grad()
                loss.backward()
                opt.step()
            with torch.no_grad():
                v = float(-(yv * torch.log_softmax(head(xv, mv), -1)).sum(-1).mean())
            if v < best - 1e-4:
                best, state, bad = v, {k: t.clone() for k, t in head.state_dict().items()}, 0
            else:
                bad += 1
                if bad >= patience:
                    break
        head.load_state_dict(state)
        return head, best

    chosen = []
    for j, (tr, te) in enumerate(splits(actors, folds, "P1")):
        allv = np.concatenate([tokens[i] for i in tr]).astype(np.float32)
        mu, sd = allv.mean(0), allv.std(0) + 1e-6
        del allv
        held = inner_holdout(actors[tr], seed=100 + j)
        fits = [(train(tr[~held], tr[held], wd, mu, sd), wd) for wd in (1e-4, 1e-3, 1e-2)]
        (head, _), wd = min(fits, key=lambda f: f[0][1])
        with torch.no_grad():
            for k in range(0, len(te), 256):
                x, m = batch(te[k:k + 256], mu, sd)
                P[te[k:k + 256]] = torch.softmax(head(x, m), -1).numpy()
        chosen.append(wd)
    return P, chosen


def partial_spearman(x: np.ndarray, y: np.ndarray, group: np.ndarray) -> float:
    """Spearman of x and y after removing each group's mean rank (ranks residualised on group indicators)."""
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    codes, g = np.unique(group, return_inverse=True)
    rx = rx - (np.bincount(g, rx) / np.bincount(g))[g]
    ry = ry - (np.bincount(g, ry) / np.bincount(g))[g]
    return float((rx * ry).sum() / np.sqrt((rx ** 2).sum() * (ry ** 2).sum()))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--runs", type=Path, default=Path("results/visual_probe"))
    parser.add_argument("--out", type=Path, default=Path("results/visual_probe_analysis"))
    parser.add_argument("--earlier-zero-shot", type=Path, default=Path("results/cremad_qwen/scores.jsonl"))
    parser.add_argument("--earlier-stage1", type=Path, default=Path("results/stage1"))
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--boot", type=int, default=N_BOOT)
    parser.add_argument("--no-attention", action="store_true")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)

    df = clip_table().set_index("uid")
    fold_of = outer_folds(df.reset_index())
    pV = counts(df, "v") / np.clip(counts(df, "v").sum(1, keepdims=True), 1, None)
    df["A_V"], df["A_AV"] = norm_entropy(pV), norm_entropy(counts(df, "av") / np.clip(counts(df, "av").sum(1, keepdims=True), 1, None))
    df["fold"] = df.actor.map(fold_of)
    pos = {u: i for i, u in enumerate(df.index)}
    tables = {}

    # ---- reliability of A^V --------------------------------------------------------------------------
    V = counts(df[df.v_n > 0], "v")
    rng = np.random.default_rng(0)
    halves = []
    for _ in range(50):
        a, b = [], []
        for c in V.astype(int):
            votes = rng.permutation(np.repeat(np.arange(K), c))
            a.append(np.bincount(votes[:len(votes) // 2], minlength=K))
            b.append(np.bincount(votes[len(votes) // 2:], minlength=K))
        a, b = np.array(a, float), np.array(b, float)
        halves.append(spearman(norm_entropy(a / a.sum(1, keepdims=True)), norm_entropy(b / b.sum(1, keepdims=True))))
    c = ceiling(V)
    tables["reliability_A_V"] = pd.DataFrame([
        {"quantity": "split-half: Spearman between the entropies of two random halves of each clip's raters (50 splits), mean [2.5%, 97.5%]",
         "value": f"{np.mean(halves):.3f} [{np.percentile(halves, 2.5):.3f}, {np.percentile(halves, 97.5):.3f}]"},
        {"quantity": "DM check: about 7 raters predict the other 3, empirical", "value": round(empirical_split(V), 3)},
        {"quantity": "DM check: the same, simulated from the fitted Dirichlet-multinomial (mean)", "value": round(c["split_sim"].mean(), 3)},
        {"quantity": "DM ceiling: Spearman of a perfect predictor with the observed A^V (mean [2.5%, 97.5%]); valid as far as the DM check holds",
         "value": f"{c['perfect'].mean():.3f} [{np.percentile(c['perfect'], 2.5):.3f}, {np.percentile(c['perfect'], 97.5):.3f}]"},
        {"quantity": "clips / raters per clip (median)", "value": f"{len(V)} / {int(np.median(V.sum(1)))}"}])

    # ---- load runs -----------------------------------------------------------------------------------
    raw, checks = {}, []
    for path in sorted(p for p in args.runs.iterdir() if NAME.match(p.name)):
        m = NAME.match(path.name)
        uids, arrays, meta = load_run(path)
        fold = None if m["fold"] is None else int(m["fold"])
        usable = ~df.index.isin(KNOWN_BAD)
        expected = ((df.fold == fold) & usable).sum() if fold is not None else usable.sum()
        raw[path.name] = (m["model"], m["view"], fold, uids, arrays)
        checks.append({"run": path.name, "clips": len(uids), "expected (A0)": int(expected),
                       "grid": "grid" in arrays, "OOM": meta.get("oom_this_run")})
    assert raw, f"no vprobe_* runs under {args.runs}"

    # consistency with the earlier runs' video-only logits
    def earlier(path):
        s = pd.DataFrame([json.loads(l) for l in open(path, encoding="utf-8") if l.strip()])
        s = s[(s.condition == "V") & s.get("error", pd.Series(index=s.index, dtype=object)).isna()]
        return s.drop_duplicates("uid", keep="last").set_index("uid")[[f"logit_{l}" for l in LABELS]]

    for row in checks:
        model, view, fold, uids, arrays = raw[row["run"]]
        ref = None
        if model == "base" and view == "A0" and args.earlier_zero_shot.exists():
            ref = earlier(args.earlier_zero_shot)
        elif model != "base" and (args.earlier_stage1 / f"stage1_{model}_f{fold}_s0" / "scores.jsonl").exists():
            ref = earlier(args.earlier_stage1 / f"stage1_{model}_f{fold}_s0" / "scores.jsonl")
        if ref is not None:
            common = [i for i, u in enumerate(uids) if u in ref.index]
            diff = np.abs(arrays["logits"][common] - ref.loc[[uids[i] for i in common]].to_numpy())
            same = (arrays["logits"][common].argmax(1) == ref.loc[[uids[i] for i in common]].to_numpy().argmax(1)).mean()
            row.update({"clips vs earlier run": len(common), "median |logit diff|": round(float(np.median(diff)), 3),
                        "same top label": round(float(same), 3)})
    tables["run_checks"] = pd.DataFrame(checks)

    # models: base_<view> (all clips) and m3 / s1_noI (three fold runs pooled; each clip read by its fold's adapter)
    models = {}
    for name, (model, view, fold, uids, arrays) in raw.items():
        key = f"{model}_{view}"
        models.setdefault(key, []).append((uids, arrays))
    for key, parts in models.items():
        uids = sum((p[0] for p in parts), [])
        feats = [k for k in parts[0][1] if k not in ("lse", "n_video_tokens", "grid")]
        arrays = {k: np.concatenate([p[1][k] for p in parts]) for k in feats}  # float16 kept (memory); float32 per fit
        if "grid" in parts[0][1]:
            arrays["grid"] = sum((p[1]["grid"] for p in parts), [])
        keep = [i for i, u in enumerate(uids) if u in pos and df.v_n.iloc[pos[u]] > 0]
        uids = [uids[i] for i in keep]
        arrays = {k: ([v[i] for i in keep] if isinstance(v, list) else v[keep]) for k, v in arrays.items()}
        idx = np.array([pos[u] for u in uids])
        DATA[key] = {"uids": uids, "idx": idx, "arrays": arrays, "Y": pV[idx].astype(np.float32),
                     "actors": df.actor.to_numpy()[idx], "folds": df.fold.to_numpy()[idx]}

    CACHE[0] = args.out / "probe_cache"  # out-of-fold predictions per fit, reused when the script is rerun
    CACHE[0].mkdir(exist_ok=True)
    features = lambda key: [k for k in DATA[key]["arrays"] if k != "grid"]
    jobs = []
    for key in DATA:
        protocols = (["P1", "P2"] if key == "base_A0" else ["P1"]) if key.startswith("base") else ["P2"]
        jobs += [(key, f, p) for p in protocols for f in features(key)]
    print(f"{len(jobs)} probe fits on {args.workers} workers", flush=True)
    oof = {}
    with ProcessPoolExecutor(args.workers, mp_context=multiprocessing.get_context("fork")) as pool:  # workers share DATA
        for n_done, (key, P, lams) in enumerate(pool.map(_job, jobs), 1):
            oof[key] = (P, lams)
            print(f"  {n_done}/{len(jobs)} {key}: lambdas {lams}", flush=True)

    # ---- metrics --------------------------------------------------------------------------------------
    def scorers(u, d):
        i0 = d["idx"]
        av, aav = df.A_V.to_numpy()[i0], df.A_AV.to_numpy()[i0]
        emo = df.intended.to_numpy()[i0]
        hi_v, hi_av = np.zeros(len(i0), bool), np.zeros(len(i0), bool)
        for f in range(N_OUTER):  # top quarter, threshold from the other folds' actors
            here, other = d["folds"] == f, df.fold.to_numpy() != f
            hi_v[here] = av[here] >= np.quantile(df.A_V.to_numpy()[other], 0.75)
            hi_av[here] = aav[here] >= np.quantile(df.A_AV.to_numpy()[other], 0.75)
        return {"Spearman(U, A^V)": lambda i: spearman(u[i], av[i]),
                "partial Spearman(U, A^V | emotion)": lambda i: partial_spearman(u[i], av[i], emo[i]),
                "Spearman(U, A^AV)": lambda i: spearman(u[i], aav[i]),
                "AUROC top-25% A^V": lambda i: auroc(u[i], hi_v[i]),
                "AUROC top-25% A^AV": lambda i: auroc(u[i], hi_av[i])}

    def place(f):
        return {"vis": "1 vision encoder", "logits": "output"}.get(f, "2 video tokens" if f.startswith("vid") else "3 answer position")

    rows, paired, per_class, per_sentence, scores_by = [], [], [], [], {}
    for key, d in DATA.items():
        protocols = sorted({p for (k, _, p) in oof if k == key})
        boot = Bootstrap(d["actors"], args.boot)
        i0 = d["idx"]
        native = norm_entropy(softmax(d["arrays"]["logits"].astype(np.float64)))
        for protocol in protocols:
            preds = {"U_model (the model's own prediction)": native,
                     "emotion prior": prior_oof(df.A_V.to_numpy()[i0], df.intended.to_numpy()[i0], d["actors"], d["folds"], protocol),
                     "emotion x sentence prior": prior_oof(df.A_V.to_numpy()[i0], (df.intended + df.sentence_code).to_numpy()[i0],
                                                           d["actors"], d["folds"], protocol)}
            ce = {}
            for f in features(key):
                P, _ = oof[(key, f, protocol)]
                name = "logits (recalibrated)" if f == "logits" else f"probe: {f}"
                preds[name] = norm_entropy(P)
                ce[name] = float(-(d["Y"] * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
            if key == "base_A0" and protocol == "P1" and "grid" in d["arrays"] and not args.no_attention:
                for pooling in ("mean", "attention"):
                    P, wds = attention_oof(d["arrays"]["grid"], d["Y"], d["actors"], d["folds"], pooling)
                    preds[f"grid tokens, {pooling} pooling"] = norm_entropy(P)
                    ce[f"grid tokens, {pooling} pooling"] = float(-(d["Y"] * np.log(np.clip(P, 1e-12, None))).sum(1).mean())
                    print(f"  attention probe ({pooling}): weight decay {wds}", flush=True)
            scores_by[(key, protocol)] = preds
            for name, u in preds.items():
                f = name.replace("probe: ", "")
                rows.append({"model": key, "protocol": protocol, "predictor": name,
                             "place": place(f) if name.startswith("probe") or name.startswith("logits") else "",
                             **{k: boot(s) for k, s in scorers(u, d).items()}, "CE to p_V": round(ce.get(name, np.nan), 4)})
            ref = scorers(native, d)
            prior = scorers(preds["emotion prior"], d)
            for name, u in preds.items():
                if not (name.startswith("probe") or name.startswith("logits") or name.startswith("grid")):
                    continue
                s = scorers(u, d)
                paired.append({"model": key, "protocol": protocol, "predictor": name,
                               "minus U_model: Spearman(U, A^V)": boot(lambda i: s["Spearman(U, A^V)"](i) - ref["Spearman(U, A^V)"](i)),
                               "minus U_model: partial | emotion": boot(lambda i: s["partial Spearman(U, A^V | emotion)"](i) - ref["partial Spearman(U, A^V | emotion)"](i)),
                               "minus U_model: Spearman(U, A^AV)": boot(lambda i: s["Spearman(U, A^AV)"](i) - ref["Spearman(U, A^AV)"](i)),
                               "minus emotion prior: Spearman(U, A^V)": boot(lambda i: s["Spearman(U, A^V)"](i) - prior["Spearman(U, A^V)"](i))})
            primary_run = (key.startswith("base") and protocol == "P1") or (not key.startswith("base"))
            if primary_run and key in ("base_A0", "m3_A0", "s1_noI_A0"):
                av = df.A_V.to_numpy()[i0]
                for col, groups in (("intended", per_class), ("sentence_code", per_sentence)):
                    g = df[col].to_numpy()[i0]
                    for name in ("U_model (the model's own prediction)", f"probe: {PRIMARY}", "probe: vis"):
                        u = preds[name]
                        for level in sorted(pd.unique(g)):
                            sel = np.flatnonzero(g == level)
                            b = Bootstrap(d["actors"][sel], min(args.boot, 500))
                            rec = {"model": key, "protocol": protocol, col: level, "predictor": name, "n": len(sel),
                                   "Spearman(U, A^V)": b(lambda i: spearman(u[sel][i], av[sel][i]))}
                            if col == "intended":
                                rec.update({"A^V mean (SD)": f"{av[sel].mean():.3f} ({av[sel].std():.3f})",
                                            "A^V IQR": f"{np.percentile(av[sel], 25):.3f}-{np.percentile(av[sel], 75):.3f}",
                                            "U mean (SD)": f"{u[sel].mean():.3f} ({u[sel].std():.3f})"})
                            groups.append(rec)
    tables["positions"] = pd.DataFrame(rows)
    tables["paired_vs_own_output"] = pd.DataFrame(paired)
    tables["per_emotion"] = pd.DataFrame(per_class)
    tables["per_sentence"] = pd.DataFrame(per_sentence)

    # ---- views: A1-A3 minus A0 (base, P1), on the clips every view has --------------------------------
    views = sorted(k for k in DATA if k.startswith("base_"))
    if "base_A0" in views and len(views) > 1:
        common = set.intersection(*(set(DATA[k]["uids"]) for k in views))
        sel = {k: np.array([i for i, u in enumerate(DATA[k]["uids"]) if u in common]) for k in views}
        order = {k: np.argsort([DATA[k]["uids"][i] for i in sel[k]]) for k in views}
        rowsel = {k: sel[k][order[k]] for k in views}
        d0 = {kk: (v[rowsel["base_A0"]] if not isinstance(v, list) else None) for kk, v in DATA["base_A0"].items() if kk not in ("uids", "arrays")}
        actors = d0["actors"]
        i0 = d0["idx"]
        av, emo = df.A_V.to_numpy()[i0], df.intended.to_numpy()[i0]
        boot = Bootstrap(actors, args.boot)
        rows = []
        for name in (f"probe: {PRIMARY}", "probe: vis", "U_model (the model's own prediction)"):
            base_u = scores_by[("base_A0", "P1")][name][rowsel["base_A0"]]
            for k in views:
                if k == "base_A0":
                    continue
                u = scores_by[(k, "P1")][name][rowsel[k]]
                rec = {"predictor": name, "view": k.split("_")[1], "clips": len(common),
                       "Spearman(U, A^V): view": boot(lambda i: spearman(u[i], av[i])),
                       "minus A0": boot(lambda i: spearman(u[i], av[i]) - spearman(base_u[i], av[i])),
                       "partial | emotion, minus A0": boot(lambda i: partial_spearman(u[i], av[i], emo[i]) - partial_spearman(base_u[i], av[i], emo[i]))}
                loo = {e: spearman(u[emo != e], av[emo != e]) - spearman(base_u[emo != e], av[emo != e]) for e in sorted(set(emo))}
                rec["leave-one-emotion-out gain, min (emotion left out)"] = f"{min(loo.values()):+.3f} ({min(loo, key=loo.get)})"
                rec["per-emotion gain"] = " ".join(f"{e[:3]} {spearman(u[emo == e], av[emo == e]) - spearman(base_u[emo == e], av[emo == e]):+.3f}"
                                                   for e in sorted(set(emo)))
                rows.append(rec)
        tables["views_vs_A0"] = pd.DataFrame(rows)

    n_clips = {k: len(d["uids"]) for k, d in DATA.items()}
    lines = ["# Experiment B (+ A): the video-only raters' disagreement in Qwen2.5-Omni-7B's visual representations\n",
             f"Generated by `analysis/visual_probe_analysis.py`. Clips per model: {n_clips}. "
             f"CIs: 95% bootstrap over actors ({args.boot} resamples; per emotion / sentence: {min(args.boot, 500)}). "
             "U = normalised entropy of a predicted distribution. Probe and U_model are compared within one model, on "
             "the same clips. P1: three outer folds over actors (base). P2: five folds over actors within each outer "
             "fold (the actors the fold's adapter never saw).\n"]
    for name, table in tables.items():
        table.to_csv(args.out / f"{name}.csv", index=False)
        lines += [f"## {name}\n", markdown(table.astype(object).where(table.notna(), "")), ""]
    (args.out / "report.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
