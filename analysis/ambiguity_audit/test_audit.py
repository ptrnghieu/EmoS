import copy
import json
import tempfile
import unittest
from pathlib import Path
import numpy as np
from audit_predictions import audit, load_rows, measures, unique_correct

CLASSES=['Anger','Disgust','Fear','Happy','Neutral','Sad']

def fixture():
    rows=[]
    for i in range(24):
        c=[6+i%4,4-i%4,0,0,0,0]
        for model,q in [('cont',[.99,.01,0,0,0,0]),('reference',np.array(c)/10)]:
            rows.append(dict(clip_id=str(i),actor_id=str(i//3),fold=str((i//3)%3),
                model=model,condition='V',classes=CLASSES,counts=c,probs=list(q)))
    return rows

class TestAudit(unittest.TestCase):
    def test_identity_zero_and_zero_probabilities(self):
        p=np.array([[1.,0,0,0,0,0],[.5,.5,0,0,0,0]])
        for values in measures(p,p).values():
            np.testing.assert_allclose(values,0,atol=1e-14)

    def test_same_correct_label_can_hide_distribution_error(self):
        r=audit(fixture(),CLASSES,'cont',['reference'],'V',100,17)
        point=next(x for x in r['paired'] if x['group']=='both_correct' and x['emotion']=='ALL' and x['metric']=='jsd_nats')
        self.assertEqual(point['n_clips'],24)
        self.assertLess(point['ci_high'],0)
        self.assertAlmostEqual(point['comparator_mean'],0)

    def test_ties_are_not_arbitrarily_called_correct(self):
        p=np.array([[.5,.5,0,0,0,0],[.6,.4,0,0,0,0]])
        q=np.array([[.6,.4,0,0,0,0],[.5,.5,0,0,0,0]])
        self.assertFalse(unique_correct(p,q).any())
        self.assertAlmostEqual(measures(p,q)['reference_bias'][0],0)

    def test_identical_models_have_zero_paired_interval(self):
        rows=fixture()
        for r in rows:
            r['probs']=[.7,.3,0,0,0,0]
        result=audit(rows,CLASSES,'cont',['reference'],'V',100,17)
        for r in result['paired']:
            self.assertEqual(r['delta'],0)
            if r['ci_low'] is not None:
                self.assertEqual(r['ci_low'],0)
                self.assertEqual(r['ci_high'],0)

    def test_coverage_is_not_silently_intersected(self):
        with self.assertRaises(ValueError):
            audit(fixture()[:-1],CLASSES,'cont',['reference'],'V',100,17)

    def test_validation_rejects_duplicates_and_actor_fold_conflicts(self):
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)/'input.jsonl'
            rows=fixture()
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            self.assertEqual(len(load_rows(path)[0]),48)
            path.write_text('\n'.join(json.dumps(r) for r in rows+[rows[0]]))
            with self.assertRaises(ValueError): load_rows(path)
            rows=copy.deepcopy(rows)
            rows[2]['fold']='other'
            path.write_text('\n'.join(json.dumps(r) for r in rows))
            with self.assertRaises(ValueError): load_rows(path)

if __name__=='__main__': unittest.main()
