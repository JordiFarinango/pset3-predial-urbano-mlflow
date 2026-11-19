import unittest,json
import numpy as np
from sklearn.dummy import DummyClassifier
from src.cierre import spaces,business,ROOT,OUT,CACHE
from src.decision import ThresholdClassifier
from sklearn.model_selection import ParameterGrid

class ClosureChecks(unittest.TestCase):
    def test_ninety_distinct_configurations(self):
        for name in ['logistica','mlp','random_forest']:
            configurations=list(ParameterGrid(spaces(name)))
            self.assertEqual(len(configurations),30)
            self.assertEqual(len({repr(x) for x in configurations}),30)
    def test_academic_cost_and_confusion(self):
        result=business(np.array([0,0,1,1]),np.array([0,1,0,1]),3)
        self.assertEqual(result['cost_per_obligation'],1.0)
        self.assertEqual((result['tn'],result['fp'],result['fn'],result['tp']),(1,1,1,1))
    def test_threshold_wrapper_is_portable(self):
        import pickle
        model=DummyClassifier(strategy='prior').fit([[1],[2],[3]],[0,1,1])
        policy=ThresholdClassifier(model,.7);policy.classes_=np.array([0,1])
        restored=pickle.loads(pickle.dumps(policy))
        self.assertEqual(type(restored).__module__,'src.decision')
        np.testing.assert_array_equal(restored.predict([[4],[5]]),[0,0])
    def test_maturity_counts_are_different_cutoffs(self):
        a=json.loads((ROOT/'docs/auditoria_cierre.json').read_text())
        self.assertEqual(a['mature_at_observation_end'],12000)
        self.assertEqual(a['sets']['development']['n']+a['sets']['holdout']['n']+a['sets']['excluded_at_training_cutoff']['n'],12000)
        self.assertFalse(a['dataset_changed'])
    def test_final_decision_is_frozen(self):
        frozen=json.loads((OUT/'selection_frozen.json').read_text());final=json.loads((OUT/'holdout_resultados.json').read_text())
        self.assertFalse(frozen['holdout_evaluated_at_freeze']);self.assertFalse(final['selection_based_on_holdout'])
        self.assertEqual(final['holdout_protocol_evaluations'],1)
        self.assertEqual(frozen['identity'],final['identity'])
