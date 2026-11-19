"""Dos baselines y cinco familias distintas; no contiene un LLM."""
import numpy as np
from scipy.special import expit
from sklearn.base import BaseEstimator,ClassifierMixin
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,HistGradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

class AcademicRuleClassifier(ClassifierMixin,BaseEstimator):
    """Regla academica: importe relativo alto eleva riesgo, estacionalidad lo modula.

    Recibe log(importe) estandarizado en columna 0 y seno mes en columna 3.
    Coeficientes fijos, no regla municipal ni probabilidad calibrada.
    """
    def fit(self,X,y):
        self.classes_=np.array([0,1]);self.n_features_in_=X.shape[1];return self
    def predict_proba(self,X):
        p=expit(.8*np.asarray(X[:,0]).ravel()-.25*np.asarray(X[:,3]).ravel())
        return np.column_stack([1-p,p])
    def predict(self,X):
        return (self.predict_proba(X)[:,1]>=.5).astype(int)


def candidates(seed):
    return [
        ('dummy','baseline_ingenuo',DummyClassifier(strategy='prior')),
        ('regla_academica','baseline_regla',AcademicRuleClassifier()),
        ('logistica','lineal_regularizada',LogisticRegression(C=1,max_iter=500,random_state=seed)),
        ('random_forest','bagging',RandomForestClassifier(n_estimators=120,max_depth=7,min_samples_leaf=20,n_jobs=2,random_state=seed)),
        ('hist_boosting','boosting',HistGradientBoostingClassifier(max_iter=120,max_leaf_nodes=15,l2_regularization=2,early_stopping=False,random_state=seed)),
        ('mlp','red_neuronal',MLPClassifier(hidden_layer_sizes=(32,16),alpha=.1,max_iter=160,batch_size=256,early_stopping=False,tol=1e-3,random_state=seed)),
        ('svm_rbf','kernel_svm',SVC(C=1,kernel='rbf',random_state=seed))]
