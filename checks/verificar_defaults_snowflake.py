"""Coherencia offline del estudio nuevo; no red, entrenamiento ni carga de modelos."""
import argparse
import ast
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import re
import statistics

ROOT = Path(__file__).resolve().parents[1]
CANDIDATES = ['dummy', 'regla_academica', 'logistica', 'random_forest', 'hist_boosting', 'mlp', 'svm_rbf']
STUDY = 'defaults-snowflake-20261120-8701a2'
TABLE = '"PSET3_SINTETICO"."EXPERIMENTOS"."OBT_PREDIAL_2B7D72B7735E"'
METRICS = ['roc_auc', 'average_precision', 'balanced_accuracy', 'f1', 'accuracy']


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, label):
    require(math.isfinite(float(actual)) and math.isclose(float(actual), float(expected), rel_tol=0, abs_tol=1e-12), label)


def verify(root=ROOT):
    import pandas as pd
    import sklearn
    from src.estudio_nuevo import default_candidates
    root = Path(root)
    evidence = json.loads((root/'docs/defaults_snowflake_evidencia.json').read_text())
    integration = evidence['publication_integrity']
    records = [json.loads(line) for line in (root/'docs/defaults_snowflake_runs.jsonl').read_text().splitlines()]
    results = pd.read_csv(root/'docs/defaults_snowflake_resultados.csv', keep_default_na=False).to_dict('records')
    generation = json.loads((root/'data/synthetic_v1/manifest.json').read_text())
    parts = json.loads((root/'data/synthetic_v1/partitions.json').read_text())
    contract = json.loads((root/'config/experiment.json').read_text())
    old = [json.loads(line) for line in (root/'docs/mlflow_portable/runs_originales.jsonl').read_text().splitlines()]
    inventory = json.loads((root/'MANIFEST_SHA256.json').read_text())
    require(sklearn.__version__ == evidence['sklearn_version'] == '1.7.2', 'Defaults: usar sklearn 1.7.2 del lock.')
    for name in ['docs/defaults_snowflake_evidencia.json','docs/defaults_snowflake_resultados.csv','docs/defaults_snowflake_runs.jsonl']:
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == inventory['files'][name], 'Archivo cientifico distinto del manifiesto: '+name)
    for name, expected in integration['new_scientific_file_sha256'].items():
        require(hashlib.sha256((root/name).read_bytes()).hexdigest() == expected, 'Evidencia nueva alterada: '+name)
    historical_hash = hashlib.sha256(json.dumps(inventory['preserved_v2_sha256'], sort_keys=True, separators=(',', ':')).encode()).hexdigest()
    require(historical_hash == integration['original_v2_hash_map_sha256'], 'Mapa historico de hashes alterado.')
    ids = [r['info']['run_id'] for r in records]
    require(len(ids) == len(set(ids)) == 42 and all(re.fullmatch('[a-f0-9]{32}', rid) for rid in ids), 'IDs nuevos duplicados, invalidos o incompletos.')
    runs = dict(zip(ids, records))
    require(not set(ids) & {r['info']['run_id'] for r in old}, 'Se mezclaron IDs originales y nuevos.')
    require(Counter(r['info']['status'] for r in records) == {'FINISHED': 36, 'FAILED': 6}, 'Estados distintos de 36 FINISHED / 6 FAILED.')
    require(evidence['experiment']['runs'] == 42 and evidence['experiment']['statuses'] == {'FINISHED':36, 'FAILED':6}, 'Recibo de estados incoherente.')
    require(evidence['experiment']['name'] == 'pset3_new_'+STUDY and evidence['experiment']['tracking_uri'] == 'http://127.0.0.1:5002', 'Experimento o backend incorrecto.')
    require(evidence['study_id'] == STUDY and evidence['source'] == 'SNOWFLAKE', 'Estudio/fuente incorrectos.')
    failed = {r['info']['run_id'] for r in records if r['info']['status'] == 'FAILED'}
    require(failed == {r['run_id'] for r in evidence['failed_runs_preserved']}, 'Faltan fallos conservados.')
    code_versions = {v['code_version'] for v in evidence['code_provenance']}
    fingerprint = generation['hashes']['obt']
    features = contract['numeric_features']+contract['categorical_features']
    require(evidence['features'] == features and len(features) == 8 and fingerprint == parts['fingerprint'], 'Features/snapshot diferentes.')
    for r in records:
        tags = r['tags']; p = r['params']
        require(tags['pset3.source'] == 'SNOWFLAKE' and tags['pset3.study'] == STUDY and tags['pset3.phase'] == 'new_candidates', 'Origen/fase no conciliados.')
        require(tags['pset3.dataset_fingerprint'] == fingerprint and tags['pset3.code_version'] in code_versions, 'Hash de datos/codigo no conciliado.')
        require(tags['git.sha'] == 'UNAVAILABLE_WORKTREE_DIRTY', 'No atribuir un SHA limpio inexistente.')
        require(p['target'] == contract['target'] and json.loads(p['numeric_features']) == contract['numeric_features'] and json.loads(p['categorical_features']) == contract['categorical_features'], 'Contrato de features/target distinto.')
        require(not any('holdout' in key.lower() for key in r['metrics']), 'Hay nuevas metricas hold-out.')
        require(all(math.isfinite(float(v)) for v in r['metrics'].values()), 'Metrica no finita.')
        require(r['info']['end_time'] >= r['info']['start_time'], 'Tiempo de run invalido.')
        if r['parent_run_id']:
            require(r['parent_run_id'] in runs and r['parent_run_id'] != r['info']['run_id'], 'Padre ajeno/ciclico.')
            seen = {r['info']['run_id']}; parent = r['parent_run_id']
            while parent:
                require(parent not in seen, 'Ciclo en la jerarquia de runs.')
                seen.add(parent); parent = runs[parent]['parent_run_id']
    require([row['candidate'] for row in results] == CANDIDATES, 'La tabla no contiene exactamente los siete candidatos.')
    expected_models = set(); checked_defaults = 0
    estimators = {n:m for n,_,m in default_candidates(contract['random_state'], False)}
    for row in results:
        config = runs[row['run_id']]
        require(config['info']['status'] == 'FINISHED' and config['name'] == row['candidate']+'-0', 'Configuracion final no terminada o mal identificada.')
        reused = row.get('reused_cv_config_run_id')
        parent = reused or row['run_id']
        folds = sorted([r for r in records if r['parent_run_id'] == parent and 'fold' in r['params']], key=lambda r:int(r['params']['fold']))
        require(len(folds) == 3 and all(r['info']['status'] == 'FINISHED' for r in folds), 'Folds nuevos incompletos.')
        fold_ids = [r['info']['run_id'] for r in folds]
        require(ast.literal_eval(row['fold_run_ids']) == fold_ids, 'IDs de folds distintos de la tabla.')
        if reused:
            require(runs[reused]['info']['status'] == 'FAILED' and config['tags']['pset3.reused_cv_config_run_id'] == reused, 'Procedencia de CV recuperada perdida.')
            require(ast.literal_eval(row['reused_fold_run_ids']) == fold_ids, 'Folds reutilizados distintos.')
        for i, fold in enumerate(folds):
            require(int(fold['params']['fold']) == i, 'Numero de fold incorrecto.')
            require(fold['metrics']['train_rows'] == len(parts['folds'][i]['train']) and fold['metrics']['validation_rows'] == len(parts['folds'][i]['validation']), 'Tamanos de fold alterados.')
            seed = contract['random_state'] if row['candidate'] == 'regla_academica' else contract['random_state']+i
            require(int(fold['params']['seed']) == seed, 'Semilla real distinta.')
            for metric in METRICS:
                close(fold['metrics']['gap_'+metric], fold['metrics']['train_'+metric]-fold['metrics']['validation_'+metric], 'Brecha de fold incorrecta.')
        for col in ['fit_seconds','train_rows','validation_rows']+[prefix+metric for metric in METRICS for prefix in ['train_','validation_','gap_']]:
            values = [f['metrics'][col] for f in folds]
            for suffix, expected in [('mean', statistics.mean(values)), ('std', statistics.stdev(values))]:
                close(row[col+'_'+suffix], expected, 'Resumen de CV no conciliado.')
                close(config['metrics'][col+'_'+suffix], expected, 'Metrica MLflow/CSV distinta.')
        close(config['metrics']['fit_seconds'], row['refit_fit_seconds'], 'Tiempo de refit distinto.')
        require(config['metrics']['train_rows'] == len(parts['train_indices']) == evidence['development_rows'] == 9510, 'Refit fuera de desarrollo.')
        expected_models.update(fold_ids+[row['run_id']])
        if row['candidate'] in estimators:
            for fitted in [config]+folds:
                expected = type(estimators[row['candidate']])().get_params(deep=False)
                expected['random_state'] = int(fitted['params']['seed'])
                for key, value in expected.items():
                    actual = fitted['params'].get('model__'+key, fitted['params'].get(key))
                    require(actual == str(value), 'No es default estricto: '+row['candidate']+'/'+key)
                checked_defaults += 1
    models = evidence['models']
    require(len(models) == len({m['model_id'] for m in models}) == 28 and {m['source_run_id'] for m in models} == expected_models, 'Los 28 pipelines no corresponden a los 21 folds y siete refits.')
    for model in models:
        require(runs[model['source_run_id']]['info']['status'] == 'FINISHED' and model['status'] == 'READY', 'Pipeline no terminado.')
        require(all(model[k] is True for k in ['signature_present','input_example_present','sklearn_flavor_present','model_payload_present']) and re.fullmatch('[a-f0-9]{64}', model['mlmodel_sha256']), 'Metadatos de modelo incompletos.')
    reads = evidence['snowflake_reads']
    require(len(reads) == len({r['query_id'] for r in reads}) == 3, 'Se requieren tres consultas autenticas documentadas.')
    require({r['query_id'] for r in reads} == {r['query_id'] for r in integration['source_receipts']['snowflake']}, 'Consultas distintas del recibo original.')
    for read in reads:
        require((read['rows'],read['columns'],read['read_limit']) == (12000,20,12001) and read['table'] == TABLE, 'Objeto/tamano/limite incorrectos.')
        require(read['read_query'].startswith('SELECT ') and read['read_query'].endswith(' FROM '+TABLE+' LIMIT %s') and ';' not in read['read_query'], 'SQL documentado no limitado a SELECT autorizado.')
        rec = read['reconciliation']
        require(read['fingerprint'] == rec['training_fingerprint'] == fingerprint and all(rec[k] is True for k in ['keys_unique_and_equal','null_counts_equal','labels_equal_by_id']), 'Reconciliacion Snowflake incompleta.')
    require(evidence['holdout_evaluated_or_used_for_selection'] is False and evidence['champion_challenger_changed'] is False and evidence['new_tuning_executed'] is False, 'Alcance experimental incorrecto.')
    for rel in ['README.md','docs/reproducibilidad.md','docs/defaults_snowflake_evidencia.json','docs/defaults_snowflake_resultados.csv','docs/defaults_snowflake_runs.jsonl']:
        text = (root/rel).read_text()
        require(not re.search(r'(?i)(?:[A-Z]:[\\/](?:Users|Documents and Settings)[\\/]|/(?:home|Users)/)', text), 'Ruta local sensible en evidencia nueva.')
    return {'status':'PASS','mode':'OFFLINE_NEW_DEFAULTS_RECEIPTS','runs':42,'finished':36,'failed_preserved':6,'candidates':7,'folds':21,'development_refits':7,'pipeline_references':28,'default_parameter_sets':checked_defaults,'snowflake_select_receipts':3,'original_run_overlap':0,'holdout_metrics':0,'network_used':False,'training_executed':False,'limits':'Reconciles public receipts and archived scientific metadata. Full model payloads remain in local MLflow volumes; no fresh API, Snowflake query or artifact execution.'}


if __name__ == '__main__':
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(verify(), indent=2))
