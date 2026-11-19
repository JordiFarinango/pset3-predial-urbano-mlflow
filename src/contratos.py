"""Contrato de columnas sin dependencias ML ni acceso a servicios."""
from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureSpec:
    target: str
    numeric_features: tuple[str, ...]
    categorical_features: tuple[str, ...]
    approved_at_prediction: tuple[str, ...]
    excluded_columns: tuple[str, ...] = ()

    @property
    def features(self):
        return self.numeric_features + self.categorical_features

    def validate(self):
        features = self.features
        if not isinstance(self.target, str) or not self.target or not features:
            raise ValueError('Target y features explicitas pendientes de aprobacion.')
        if not all(isinstance(x, str) and x for x in (*features, *self.approved_at_prediction, *self.excluded_columns)):
            raise ValueError('Los nombres de columnas deben ser cadenas no vacias.')
        if len(set(features)) != len(features):
            raise ValueError('Features duplicadas o presentes en ambas ramas.')
        if self.target in features or set(features) & set(self.excluded_columns):
            raise ValueError('Target o columna excluida dentro de las features.')
        if set(features) != set(self.approved_at_prediction):
            raise ValueError('Cada feature requiere aprobacion de disponibilidad al predecir.')

    @classmethod
    def from_dict(cls, config: dict):
        fields = ('numeric_features', 'categorical_features', 'approved_at_prediction', 'excluded_columns')
        if any(not isinstance(config.get(field, []), (list, tuple)) for field in fields):
            raise ValueError('Las listas de columnas deben ser listas explicitas.')
        spec = cls(config.get('target'), tuple(config.get('numeric_features', [])),
                   tuple(config.get('categorical_features', [])),
                   tuple(config.get('approved_at_prediction', [])),
                   tuple(config.get('excluded_columns', [])))
        spec.validate()
        return spec
