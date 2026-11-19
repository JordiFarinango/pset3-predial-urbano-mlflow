"""Preprocesamiento sin fit global y sin seleccion implicita de features/target."""
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from .contratos import FeatureSpec


class FeatureGuard(TransformerMixin, BaseEstimator):
    def __init__(self, spec: FeatureSpec):
        self.spec = spec

    def _check(self, frame):
        self.spec.validate()
        if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
            raise ValueError('Se requiere un DataFrame con columnas unicas.')
        if set(frame.columns) != set(self.spec.features):
            raise ValueError('X debe contener exclusivamente las features aprobadas, sin target/metadatos.')

    def fit(self, X, y=None):
        self._check(X)
        self.feature_names_in_ = np.asarray(self.spec.features, dtype=object)
        self.n_features_in_ = len(self.spec.features)
        return self

    def transform(self, X):
        self._check(X)
        result = X.loc[:, list(self.spec.features)].copy()
        for name in self.spec.numeric_features:
            result[name] = pd.to_numeric(result[name], errors='raise').astype('float64')
            if np.isinf(result[name]).any():
                raise ValueError('Feature numerica con infinito.')
        for name in self.spec.categorical_features:
            # Unificar None/pd.NA/NaN; no mezclar numeros y cadenas en OneHotEncoder.
            result[name] = result[name].map(lambda value: np.nan if pd.isna(value) else str(value)).astype(object)
        return result

    def get_feature_names_out(self, input_features=None):
        return np.asarray(self.spec.features, dtype=object)


def build_preprocessor(spec: FeatureSpec, *, scale_numeric: bool = True,
                       numeric_imputation: str = 'median', sparse_output: bool = True) -> ColumnTransformer:
    spec.validate()
    if numeric_imputation not in ('mean', 'median', 'most_frequent', 'constant'):
        raise ValueError('Estrategia numerica no admitida.')
    numeric_steps = [('impute', SimpleImputer(strategy=numeric_imputation, keep_empty_features=True))]
    if scale_numeric:
        numeric_steps.append(('scale', StandardScaler()))
    categorical = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value='__MISSING__', keep_empty_features=True)),
        ('encode', OneHotEncoder(handle_unknown='ignore', sparse_output=sparse_output)),
    ])
    branches = []
    if spec.numeric_features:
        branches.append(('numeric', Pipeline(numeric_steps), list(spec.numeric_features)))
    if spec.categorical_features:
        branches.append(('categorical', categorical, list(spec.categorical_features)))
    return ColumnTransformer(branches, remainder='drop', verbose_feature_names_out=True)


def build_pipeline(estimator, spec: FeatureSpec, **preprocessing_options) -> Pipeline:
    """Devuelve un estimador SIN entrenar; ajustar exclusivamente en train/CV."""
    return Pipeline([
        ('guard', FeatureGuard(spec)),
        ('preprocess', build_preprocessor(spec, **preprocessing_options)),
        ('model', clone(estimator)),
    ])


def select_xy(frame: pd.DataFrame, spec: FeatureSpec):
    spec.validate()
    if not frame.columns.is_unique or not set((*spec.features, spec.target)).issubset(frame.columns):
        raise ValueError('Faltan columnas aprobadas o existen nombres duplicados.')
    if frame[spec.target].isna().any():
        raise ValueError('El target contiene valores ausentes; resolver elegibilidad antes de entrenar.')
    return frame.loc[:, list(spec.features)].copy(), frame[spec.target].copy()
