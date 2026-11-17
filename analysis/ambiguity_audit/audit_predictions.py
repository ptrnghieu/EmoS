"""Descriptive audit of human-distribution error conditional on label agreement.
No model fitting. See protocol_vi.md for scope and interpretation limits.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.special import xlogy
from scipy.stats import spearmanr


def load_rows(path):
    rows = [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]
    if not rows:
        raise ValueError('Empty input')
    classes = rows[0]['classes']
    if len(classes) != 6 or len(set(classes)) != 6:
        raise ValueError('Expected six distinct class names in explicit order')
    seen, truths, actor_folds = set(), {}, {}
    for r in rows:
        for key in ['clip_id', 'actor_id', 'fold', 'model', 'condition']:
            if not isinstance(r[key], str) or not r[key]:
                raise ValueError(f'{key} must be a nonempty string')
        if r['classes'] != classes or r['condition'] not in ['A', 'V', 'AV']:
            raise ValueError('Inconsistent classes or invalid condition')
        c, q = np.asarray(r['counts'], float), np.asarray(r['probs'], float)
        if c.shape != (6,) or q.shape != (6,):
            raise ValueError('Expected six counts/probabilities')
        if not np.isfinite(c).all() or (c < 0).any() or (c != np.floor(c)).any() or c.sum() <= 0:
            raise ValueError('Counts must be finite nonnegative integers, positive total')
        if not np.isfinite(q).all() or (q < 0).any() or not np.isclose(q.sum(), 1, atol=1e-8, rtol=0):
            raise ValueError('Probabilities must be finite, nonnegative, sum to one')
        # Correct only numerical roundoff within the declared tolerance.
        r['probs'] = (q / q.sum()).tolist()
        key = (r['condition'], r['clip_id'])
        identity = (r['actor_id'], r['fold'], tuple(r['counts']))
        if key in truths and truths[key] != identity:
            raise ValueError('Mismatched votes/actor/fold for the same clip-condition')
        truths[key] = identity
        if r['actor_id'] in actor_folds and actor_folds[r['actor_id']] != r['fold']:
            raise ValueError('Actor appears in multiple outer test folds')
        actor_folds[r['actor_id']] = r['fold']
        key = (*key, r['model'])
        if key in seen:
            raise ValueError('Duplicate prediction')
        seen.add(key)
    return rows, classes


def entropy(p):
    return -xlogy(p, p).sum(axis=-1) / np.log(6)


def measures(p, q):
    m = (p + q) / 2
    jsd = (xlogy(p, p).sum(1) - xlogy(p, m).sum(1)
           + xlogy(q, q).sum(1) - xlogy(q, m).sum(1)) / 2
    hp, hq = entropy(p), entropy(q)
    tied = p == p.max(1, keepdims=True)
    ref = np.where(tied, q - p, 0).sum(1) / tied.sum(1)
    return dict(jsd_nats=jsd, entropy_bias=hq-hp, entropy_mae=abs(hq-hp),
                reference_bias=ref, reference_mae=abs(ref))


def unique_correct(p, q):
    human_unique = (p == p.max(1, keepdims=True)).sum(1) == 1
    model_unique = (q == q.max(1, keepdims=True)).sum(1) == 1
    return human_unique & model_unique & (p.argmax(1) == q.argmax(1))


def paired_summary(actors, a, b, n_boot, rng):
    """Cluster bootstrap: sample actors, retain all their selected clips.
    Same draws for baseline/comparator and every metric. Clip-weighted means.
    """
    names = list(a)
    ua, inv = np.unique(actors, return_inverse=True)
    n = len(actors)
    if not n:
        return []
    av, bv = np.stack([a[k] for k in names], 1), np.stack([b[k] for k in names], 1)
    counts = np.bincount(inv)
    sums = np.zeros((len(ua), len(names)))
    np.add.at(sums, inv, bv-av)
    cis = None
    if len(ua) >= 2:
        weights = rng.multinomial(len(ua), np.full(len(ua), 1/len(ua)), size=n_boot)
        samples = (weights @ sums) / (weights @ counts)[:, None]
        cis = np.quantile(samples, [.025, .975], axis=0)
    return [dict(metric=k, n_clips=n, n_actors=len(ua),
                 baseline_mean=float(av[:,j].mean()), comparator_mean=float(bv[:,j].mean()),
                 delta=float((bv[:,j]-av[:,j]).mean()),
                 ci_low=None if cis is None else float(cis[0,j]),
                 ci_high=None if cis is None else float(cis[1,j])) for j,k in enumerate(names)]


def rho(a, b):
    if len(a) < 3 or np.ptp(a) == 0 or np.ptp(b) == 0:
        return None
    return float(spearmanr(a,b).statistic)


def audit(rows, classes, baseline, comparators, condition, n_boot=2000, seed=17):
    selected = [r for r in rows if r['condition'] == condition]
    maps = {name: {r['clip_id']:r for r in selected if r['model']==name}
            for name in [baseline, *comparators]}
    ids = sorted(maps[baseline])
    if not ids:
        raise ValueError('No baseline predictions in this condition')
    for name, mapping in maps.items():
        if set(mapping) != set(ids):
            raise ValueError(f'{name}: clip coverage differs; no silent intersection allowed')
    rr = [maps[baseline][i] for i in ids]
    counts = np.array([r['counts'] for r in rr], float)
    p = counts / counts.sum(1, keepdims=True)
    actors = np.array([r['actor_id'] for r in rr])
    q = {name: np.array([mapping[i]['probs'] for i in ids]) for name,mapping in maps.items()}
    metrics = {name: measures(p,qq) for name,qq in q.items()}
    correct = {name: unique_correct(p,qq) for name,qq in q.items()}
    unique = (p == p.max(1, keepdims=True)).sum(1)==1
    output, strata, ranks, per_clip = [], [], [], []
    rng = np.random.default_rng(seed)
    for name in comparators:
        bc, oc = correct[baseline], correct[name]
        groups = {'all':np.ones(len(ids),bool), 'baseline_correct':bc,
                  'both_correct':bc & oc,
                  'baseline_only_correct':unique & bc & ~oc,
                  'comparator_only_correct':unique & ~bc & oc,
                  'neither_unique_correct':unique & ~bc & ~oc,
                  'human_tied_majority':~unique}
        for group, mask in groups.items():
            strata.append(dict(comparator=name,group=group,n_clips=int(mask.sum()),
                               coverage=float(mask.mean()),n_actors=len(set(actors[mask]))))
            for label, sub in [('ALL',mask), *[(c, mask & unique & (p.argmax(1)==j)) for j,c in enumerate(classes)]]:
                for result in paired_summary(actors[sub], {k:v[sub] for k,v in metrics[baseline].items()},
                                             {k:v[sub] for k,v in metrics[name].items()},n_boot,rng):
                    output.append(dict(comparator=name,group=group,emotion=label,**result))
                ranks.append(dict(comparator=name,group=group,emotion=label,n_clips=int(sub.sum()),
                                  baseline_rho=rho(entropy(p[sub]),entropy(q[baseline][sub])),
                                  comparator_rho=rho(entropy(p[sub]),entropy(q[name][sub]))))
        for j, clip in enumerate(ids):
            per_clip.append(dict(clip_id=clip, actor_id=actors[j],comparator=name,
                human_unique=bool(unique[j]), baseline_correct=bool(bc[j]), comparator_correct=bool(oc[j]),
                **{f'delta_{k}':float(metrics[name][k][j]-metrics[baseline][k][j]) for k in metrics[name]}))
    return dict(paired=output,strata=strata,ranks_descriptive=ranks,per_clip=per_clip,
                metadata=dict(condition=condition,classes=classes,baseline=baseline,comparators=comparators,
                              n_clips=len(ids),n_actors=len(set(actors)),bootstraps=n_boot,seed=seed,
                              ranking_note='Point estimates only; not the project rho_within implementation.',
                              status='Descriptive post-hoc analysis; no causal or equivalence claims.'))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--baseline',default='cont')
    parser.add_argument('--comparators',nargs='+',required=True)
    parser.add_argument('--condition',choices=['A','V','AV'],default='V')
    parser.add_argument('--bootstrap',type=int,default=2000)
    parser.add_argument('--seed',type=int,default=17)
    args=parser.parse_args()
    if args.bootstrap < 100 or args.baseline in args.comparators or len(set(args.comparators))!=len(args.comparators):
        parser.error('Use >=100 bootstrap draws and distinct comparator names excluding baseline')
    rows,classes=load_rows(args.input)
    result=audit(rows,classes,args.baseline,args.comparators,args.condition,args.bootstrap,args.seed)
    result['metadata']['input_sha256']=hashlib.sha256(Path(args.input).read_bytes()).hexdigest()
    out=Path(args.output)
    out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2,allow_nan=False))
    for key in ['paired','strata','ranks_descriptive','per_clip']:
        if result[key]:
            with (out/f'{key}.csv').open('w',newline='') as f:
                writer=csv.DictWriter(f,fieldnames=list(result[key][0]))
                writer.writeheader(); writer.writerows(result[key])
    print(json.dumps(result['metadata'],ensure_ascii=False,indent=2))


if __name__=='__main__':
    main()
