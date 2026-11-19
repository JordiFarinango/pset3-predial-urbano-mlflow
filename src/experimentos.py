"""Motor generico. Solo entrena cuando se invocan explicitamente evaluate_cv/tune.

No define columnas, clases, target, umbrales ni metricas de negocio.
"""
from dataclasses import dataclass
import random
from time import perf_counter
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import get_scorer
from sklearn.model_selection import (train_test_split, GroupShuffleSplit, GroupKFold,
    StratifiedKFold, StratifiedGroupKFold, ParameterGrid)
from .datos import dataset_fingerprint
from .tracking import tracked_run, log_results


def seed_everything(seed: int):
    if not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('La semilla debe ser un entero uint32.')
    random.seed(seed)
    np.random.seed(seed)
    return np.random.default_rng(seed)


def seeded_clone(estimator, seed: int):
    model = clone(estimator)
    model.set_params(**{key: seed for key in model.get_params(deep=True) if key.endswith('random_state')})
    return model


def _dates(values, n):
    result = pd.DatetimeIndex(pd.to_datetime(values, utc=True, errors='raise'))
    if len(result) != n or result.isna().any():
        raise ValueError('Fechas ausentes o no alineadas.')
    return result


def holdout_indices(y, *, strategy: str, test_size=.2, seed=42, groups=None,
                    dates=None, label_available=None, cutoff=None):
    """Indices posicionales. Temporal exige maduracion antes del corte."""
    positions = np.arange(len(y))
    if strategy == 'stratified':
        if groups is not None:
            raise ValueError('Usar estrategia grouped para entidades repetidas.')
        train, test = train_test_split(positions, test_size=test_size, stratify=y, random_state=seed)
    elif strategy == 'grouped':
        if groups is None or len(groups) != len(y) or pd.isna(groups).any():
            raise ValueError('Se requieren grupos alineados, sin nulos.')
        train, test = next(GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed).split(positions, y, groups))
    elif strategy == 'temporal':
        if dates is None or label_available is None or cutoff is None:
            raise ValueError('Temporal requiere fechas, disponibilidad de etiqueta y corte explicitos.')
        emitted, available = _dates(dates,len(y)), _dates(label_available,len(y))
        if (available < emitted).any():
            raise ValueError('La etiqueta no puede estar disponible antes de emitir.')
        boundary = pd.to_datetime(cutoff, utc=True)
        train = positions[(emitted < boundary) & (available < boundary)]
        test = positions[emitted >= boundary]
        if groups is not None and set(np.asarray(groups)[train]) & set(np.asarray(groups)[test]):
            raise ValueError('Grupos compartidos: definir protocolo temporal para entidades conocidas/nuevas.')
    else:
        raise ValueError('Estrategia de hold-out desconocida.')
    if not len(train) or not len(test) or set(train) & set(test):
        raise ValueError('Particion vacia o solapada.')
    return np.sort(train), np.sort(test)


def cv_indices(y, *, strategy: str, n_splits=3, seed=42, groups=None, dates=None,
               label_available=None, temporal_windows=None):
    """Temporal usa ventanas calendario [inicio,fin), no un gap de N filas."""
    positions = np.arange(len(y))
    if strategy == 'temporal':
        if dates is None or label_available is None or not temporal_windows:
            raise ValueError('Faltan fechas, maduracion y ventanas temporales explicitas.')
        emitted, available = _dates(dates,len(y)), _dates(label_available,len(y))
        if (available < emitted).any():
            raise ValueError('Disponibilidad temporal invalida.')
        previous_end = None
        for start, end in temporal_windows:
            start, end = pd.to_datetime(start,utc=True), pd.to_datetime(end,utc=True)
            if start >= end or (previous_end is not None and start < previous_end):
                raise ValueError('Ventanas desordenadas o solapadas.')
            previous_end = end
            train = positions[(emitted < start) & (available < start)]
            validation = positions[(emitted >= start) & (emitted < end)]
            if not len(train) or not len(validation):
                raise ValueError('Fold temporal vacio despues de maduracion.')
            if groups is not None and set(np.asarray(groups)[train]) & set(np.asarray(groups)[validation]):
                raise ValueError('Grupos compartidos entre train y validacion temporal.')
            yield train, validation
        return
    if strategy == 'stratified':
        if groups is not None:
            raise ValueError('Usar grouped o stratified_grouped para entidades repetidas.')
        splitter = StratifiedKFold(n_splits=n_splits,shuffle=True,random_state=seed)
    elif strategy in ('grouped','stratified_grouped'):
        if groups is None or len(groups)!=len(y) or pd.isna(groups).any():
            raise ValueError('Grupos requeridos y alineados.')
        splitter = (GroupKFold(n_splits=n_splits) if strategy=='grouped' else
                    StratifiedGroupKFold(n_splits=n_splits,shuffle=True,random_state=seed))
    else:
        raise ValueError('Estrategia CV desconocida.')
    yield from splitter.split(positions, y, groups)


@dataclass
class CVResult:
    folds: pd.DataFrame
    summary: dict


def evaluate_cv(estimator, X, y, *, splits, scoring: dict, seed=42,
                contract=None, provenance=None, tracking=False):
    """Evalua solo DESARROLLO. No recibe ni utiliza hold-out final.

    Los scorers de sklearn son higher-is-better (incluidos neg_*).
    gap = score_train - score_validation. Callables: scorer(estimator,X,y).
    """
    if not isinstance(X,pd.DataFrame) or len(X)!=len(y) or not scoring:
        raise ValueError('X DataFrame, y alineado y scorers explicitos requeridos.')
    if isinstance(y,(pd.Series,pd.DataFrame)) and not X.index.equals(y.index):
        raise ValueError('Indices X/y desalineados.')
    if np.asarray(y).ndim != 1 or pd.isna(np.asarray(y)).any():
        raise ValueError('Se requiere un target unidimensional sin etiquetas ausentes.')
    scorers={name:get_scorer(value) if isinstance(value,str) else value for name,value in scoring.items()}
    y_values=np.asarray(y)
    splits=list(splits)
    if len(splits)<2:
        raise ValueError("Se requieren al menos dos folds.")
    validated=[]
    for train,validation in splits:
        train, validation=np.asarray(train),np.asarray(validation)
        if any(index.ndim != 1 or not np.issubdtype(index.dtype, np.integer) for index in (train,validation)):
            raise ValueError('Los indices deben ser vectores enteros; no se convierten decimales.')
        if (not len(train) or not len(validation) or len(set(train))!=len(train)
            or len(set(validation))!=len(validation) or set(train)&set(validation)
            or min(train.min(),validation.min())<0 or max(train.max(),validation.max())>=len(X)):
            raise ValueError('Indices de fold vacios, repetidos, fuera de rango o solapados.')
        validated.append((train,validation))
    records=[]
    for fold,(train,validation) in enumerate(validated):
        model=seeded_clone(estimator,seed+fold)
        def execute():
            started=perf_counter()
            model.fit(X.iloc[train],y_values[train])
            result={'fold':fold,'train_rows':len(train),'validation_rows':len(validation),
                    'fit_seconds':perf_counter()-started}
            for name,scorer in scorers.items():
                tr=float(scorer(model,X.iloc[train],y_values[train]))
                va=float(scorer(model,X.iloc[validation],y_values[validation]))
                if not np.isfinite([tr,va]).all():
                    raise ValueError('Scorer no finito; el fallo no se oculta.')
                result.update({f'train_{name}':tr,f'validation_{name}':va,f'gap_{name}':tr-va})
            return result
        if tracking:
            with tracked_run(f'fold-{fold}',contract=contract,dataset_metadata=provenance,nested=True):
                import mlflow
                mlflow.set_tags({'pset3.role':'cv_fold','pset3.train_fingerprint':dataset_fingerprint(X.iloc[train]),
                                 'pset3.validation_fingerprint':dataset_fingerprint(X.iloc[validation])})
                record=execute()
                log_results(parameters={'fold':fold,'seed':seed+fold},metrics=record)
        else:
            record=execute()
        records.append(record)
    if len(records)<2:
        raise ValueError('Se requieren al menos dos folds para resumir CV.')
    frame=pd.DataFrame(records)
    summary={}
    for column in frame.columns:
        if column not in ('fold','train_rows','validation_rows'):
            summary[column+'_mean']=float(frame[column].mean())
            summary[column+'_std']=float(frame[column].std(ddof=1))
    return CVResult(frame,summary)


def tune_grid(estimator, X, y, *, parameter_grid, splits, scoring, primary_metric,
              contract, provenance, seed=42, min_configurations=30):
    """Padre tuning > candidato > fold. No reentrena ni elige sobre hold-out."""
    candidates=list(ParameterGrid(parameter_grid))
    if len(candidates)<min_configurations or primary_metric not in scoring:
        raise ValueError('Espacio insuficiente o metrica primaria desconocida.')
    if len({repr(sorted(p.items())) for p in candidates})!=len(candidates):
        raise ValueError('Configuraciones repetidas en el grid.')
    folds=list(splits)
    outcomes=[]
    with tracked_run('tuning',contract=contract,dataset_metadata=provenance):
        for index,parameters in enumerate(candidates):
            with tracked_run(f'candidate-{index}',contract=contract,dataset_metadata=provenance,nested=True) as run:
                result=evaluate_cv(clone(estimator).set_params(**parameters),X,y,splits=folds,
                                   scoring=scoring,seed=seed,contract=contract,provenance=provenance,tracking=True)
                log_results(parameters=parameters,metrics=result.summary)
                outcomes.append({'candidate':index,'run_id':run.info.run_id,'parameters':parameters,**result.summary})
    table=pd.DataFrame(outcomes)
    return table.sort_values('validation_'+primary_metric+'_mean',ascending=False).reset_index(drop=True)
