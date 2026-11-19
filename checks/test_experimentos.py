import unittest
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from src.experimentos import seed_everything, holdout_indices, cv_indices, evaluate_cv
class EngineChecks(unittest.TestCase):
    def test_seeds_and_group_holdout(self):
        a=seed_everything(7).random(4); b=seed_everything(7).random(4); np.testing.assert_array_equal(a,b)
        y=np.tile([0,1],12); groups=np.repeat(np.arange(12),2)
        tr,te=holdout_indices(y,strategy='grouped',groups=groups)
        self.assertFalse(set(groups[tr]) & set(groups[te]))
        for tr,va in cv_indices(y,strategy='stratified_grouped',groups=groups):
            self.assertFalse(set(groups[tr]) & set(groups[va]))
    def test_temporal_maturation(self):
        dates=pd.date_range('2020-01-01',periods=20,tz='UTC'); available=dates+pd.Timedelta(days=4)
        tr,te=holdout_indices(np.tile([0,1],10),strategy='temporal',dates=dates,label_available=available,cutoff='2020-01-15')
        self.assertTrue((available[tr]<pd.Timestamp('2020-01-15',tz='UTC')).all())
        folds=list(cv_indices(np.tile([0,1],10),strategy='temporal',dates=dates,label_available=available,temporal_windows=[('2020-01-10','2020-01-15'),('2020-01-15','2020-01-20')]))
        self.assertEqual(len(folds),2)
    def test_scores_and_gap_are_computed(self):
        X=pd.DataFrame({'fixture':np.arange(24,dtype=float)}); y=np.tile([0,1],12)
        result=evaluate_cv(DecisionTreeClassifier(max_depth=1),X,y,splits=cv_indices(y,strategy='stratified'),scoring={'fixture_score':'accuracy'})
        self.assertEqual(len(result.folds),3)
        np.testing.assert_allclose(result.folds['gap_fixture_score'],result.folds['train_fixture_score']-result.folds['validation_fixture_score'])
        self.assertGreaterEqual(result.summary['fit_seconds_mean'],0)
    def test_overlapping_fold_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_cv(DecisionTreeClassifier(),pd.DataFrame({'fixture':[0,1]}),np.array([0,1]),splits=[([0],[0]),([1],[0])],scoring={'fixture_score':'accuracy'})

    def test_decimal_indices_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_cv(DecisionTreeClassifier(),pd.DataFrame({'fixture':[0,1]}),np.array([0,1]),splits=[([0.5],[1]),([1],[0])],scoring={'fixture_score':'accuracy'})
