"""Tracking y Registry por llamadas explicitas; no crea corridas al importar."""
from contextlib import contextmanager
import argparse
import json
import math
import os
from pathlib import Path
from urllib.parse import urlparse


def configure_tracking(*, uri: str | None = None, experiment_name: str | None = None):
    selected = uri or os.environ.get('MLFLOW_TRACKING_URI', 'http://127.0.0.1:5001')
    parsed = urlparse(selected)
    if parsed.scheme != 'http' or parsed.hostname not in ('127.0.0.1', 'localhost') or parsed.username or parsed.password:
        raise ValueError('Esta plantilla solo admite el servidor HTTP local de MLflow.')
    name = experiment_name or os.environ.get('MLFLOW_EXPERIMENT_NAME', 'pset3_sintetico')
    if not name.strip():
        raise ValueError('Se requiere nombre de experimento.')
    import mlflow
    mlflow.set_tracking_uri(selected)
    mlflow.set_registry_uri(selected)
    return mlflow, name


@contextmanager
def tracked_run(name: str, *, contract: dict, dataset_metadata: dict, tracking_uri: str | None = None, nested: bool = False, tags: dict | None = None, experiment_name: str | None = None):
    from .contratos import FeatureSpec
    if contract.get('approved') is not True or not contract.get('dataset_version'):
        raise ValueError('Se requiere un contrato experimental aprobado y versionado.')
    spec = FeatureSpec.from_dict(contract)
    if not dataset_metadata.get('fingerprint') or not dataset_metadata.get('row_count'):
        raise ValueError('No se registra una corrida sin huella y cantidad reales del dataset.')
    if nested:
        import mlflow
        parent = mlflow.active_run()
        if parent is None:
            raise ValueError('Una corrida anidada requiere una corrida padre activa.')
        tracking_uri = tracking_uri or mlflow.get_tracking_uri()
    mlflow, experiment = configure_tracking(uri=tracking_uri, experiment_name=experiment_name)
    if not nested:
        mlflow.set_experiment(experiment)
    run_tags = {
        'pset3.data_classification': 'SYNTHETIC',
        'pset3.dataset_fingerprint': dataset_metadata['fingerprint'],
        'pset3.dataset_version': contract['dataset_version'],
        'pset3.code_version': contract.get('code_version', 'unversioned'),
        **contract.get('run_tags', {}),
    }
    run_tags.update(tags or {})
    with mlflow.start_run(run_name=name, tags=run_tags, nested=nested,
                          experiment_id=parent.info.experiment_id if nested else None) as run:
        mlflow.log_params({
            'dataset_rows': dataset_metadata['row_count'], 'target': spec.target,
            'numeric_features': json.dumps(spec.numeric_features),
            'categorical_features': json.dumps(spec.categorical_features),
        })
        # Solo configuracion permitida; no serializar conexiones, entornos ni filas.
        mlflow.log_dict({key: contract.get(key) for key in (
            'dataset_version', 'target', 'target_rule_version', 'split', 'primary_metric',
            'business_metric', 'random_state', 'numeric_features', 'categorical_features',
            'approved_at_prediction', 'excluded_columns', 'code_version',
        )}, 'experiment_contract.json')
        yield run


def log_results(*, parameters: dict, metrics: dict) -> None:
    """Registrar exclusivamente resultados calculados por el llamador."""
    import mlflow
    if mlflow.active_run() is None:
        raise ValueError('Abrir tracked_run antes de registrar resultados.')
    if any(any(token in key.lower() for token in ('password', 'passphrase', 'secret', 'token', 'private_key'))
           for key in parameters):
        raise ValueError('No se permiten parametros de credenciales.')
    normalized = {name: float(value) for name, value in metrics.items()}
    if not all(math.isfinite(value) for value in normalized.values()):
        raise ValueError('Las metricas deben ser finitas y realmente calculadas.')
    mlflow.log_params(parameters)
    mlflow.log_metrics(normalized)


def register_fitted_model(estimator, *, model_name: str, input_example, trusted_types: tuple[str, ...] = ()):
    """No entrena ni selecciona ganador. Requiere Pipeline previamente ajustado."""
    from sklearn.utils.validation import check_is_fitted
    import mlflow
    import mlflow.sklearn
    if mlflow.active_run() is None:
        raise ValueError('Registrar dentro de una corrida activa con sus metricas.')
    check_is_fitted(estimator)
    from mlflow.models import infer_signature
    signature = infer_signature(input_example, estimator.predict(input_example))
    model = mlflow.sklearn.log_model(sk_model=estimator, name='model',
                                    signature=signature, input_example=input_example, skops_trusted_types=list(trusted_types),
                                    code_paths=[str(Path(__file__).resolve().parent)])
    return mlflow.register_model(model_uri=model.model_uri, name=model_name)


def assign_model_alias(*, model_name: str, version: str, alias: str, uri: str | None = None):
    if alias not in ('champion', 'challenger'):
        raise ValueError('Alias permitido: champion o challenger, tras seleccion explicita.')
    mlflow, _ = configure_tracking(uri=uri)
    mlflow.MlflowClient().set_registered_model_alias(model_name, alias, version)


def export_runs(output: str | Path, *, uri: str | None = None, experiment_name: str | None = None) -> int:
    """Exporta corridas existentes; falla si no hay evidencia en lugar de inventarla."""
    mlflow, name = configure_tracking(uri=uri, experiment_name=experiment_name)
    experiment = mlflow.get_experiment_by_name(name)
    if experiment is None:
        raise ValueError('El experimento no existe; no se crea corridas.csv ficticio.')
    frame = mlflow.search_runs(experiment_ids=[experiment.experiment_id], output_format='pandas')
    if frame.empty:
        raise ValueError('No existen corridas para exportar.')
    dest = Path(output)
    if not dest.parent.is_dir():
        raise ValueError('El directorio de exportacion debe existir.')
    indexed=frame.set_index('run_id')
    def model_name(run_id):
        seen=set()
        while run_id in indexed.index and run_id not in seen:
            seen.add(run_id);row=indexed.loc[run_id]
            value=row.get('tags.pset3.model')
            if isinstance(value,str) and value:return value
            parent=row.get('tags.mlflow.parentRunId')
            if not isinstance(parent,str):return row.get('tags.mlflow.runName')
            run_id=parent
        return None
    frame['model']=frame.run_id.map(model_name)
    frame['phase']=frame.get('tags.pset3.phase')
    frame['dataset_version']=frame.get('tags.pset3.dataset_version')
    frame['registry_name']=frame.get('tags.pset3.registry_name')
    frame['registry_version']=frame.get('tags.pset3.registry_version')
    frame['registry_alias']=frame.get('tags.pset3.registry_alias')
    frame.sort_values('run_id').to_csv(dest, index=False, encoding='utf-8')
    return len(frame)


def main():
    from .configuracion import PROJECT_ROOT, load_local_environment
    parser = argparse.ArgumentParser(description='Exportar evidencia REAL de MLflow; no entrena.')
    parser.add_argument('--export', type=Path, required=True)
    args = parser.parse_args()
    load_local_environment()
    destination = args.export if args.export.is_absolute() else PROJECT_ROOT / args.export
    print(f'Corridas exportadas: {export_runs(destination)}')


if __name__ == '__main__':
    main()
