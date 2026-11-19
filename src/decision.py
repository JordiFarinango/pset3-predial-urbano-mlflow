"""Politica de umbral estable para serializacion entre sesiones."""
import numpy as np
from sklearn.base import ClassifierMixin,BaseEstimator

class ThresholdClassifier(ClassifierMixin,BaseEstimator):
    def __init__(self,estimator,threshold=.5):self.estimator=estimator;self.threshold=threshold
    def predict_proba(self,X):return self.estimator.predict_proba(X)
    def predict(self,X):return (self.predict_proba(X)[:,1]>=self.threshold).astype(int)
