"""Modo B opt-in: defaults y tuning NUEVOS, sin hold-out ni Registry historico."""
import argparse
import copy
from datetime import datetime,timezone
import hashlib
import gc
import json
import os
from pathlib import Path
import re
import subprocess
from time import perf_counter
from urllib.parse import urlparse
import warnings

import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import get_scorer

ROOT = Path(__file__).resolve().parents[1]
SCORING = {'roc_auc':'roc_auc', 'average_precision':'average_precision', 'balanced_accuracy':'balanced_accuracy', 'f1':'f1', 'accuracy':'accuracy'}


def default_candidates(seed, baselines=True):
    from .modelos import candidates
    historical = candidates(seed)
    rows = historical[:2] if baselines else []
    # Construct pristine sklearn defaults. Only random_state is overridden.
    for name, family, old in historical[2:]:
        estimator = type(old)()
        estimator.set_params(random_state=seed)
        rows.append((name, family, estimator))
    return rows


def defaults_audit(seed=20261117):
    from .modelos import candidates
    differences = {}
    for name, family, historical in candidates(seed)[2:]:
        defaults = type(historical)().get_params(deep=False)
        differences[name] = {key:{'historical':value, 'sklearn_default':defaults[key], 'reproducibility_exception':key=='random_state'}
                             for key,value in historical.get_params(deep=False).items() if value != defaults[key]}
    return {'sklearn_version':sklearn.__version__, 'differences':differences,
            'new_override_only':['random_state'], 'training_executed':False}


def grid(name):
    spaces = {
        'logistica':{'model__C':[.0001,.0003,.001,.003,.01,.03,.1,.3,.5,1.,2.,3.,10.,30.,100.], 'model__class_weight':[None,'balanced']},
        'mlp':{'model__alpha':[.0001,.001,.01,.1,1.], 'model__hidden_layer_sizes':[(16,),(32,16),(64,)], 'model__learning_rate_init':[.0005,.001]},
        'random_forest':{'model__max_depth':[3,5,7,10,None], 'model__min_samples_leaf':[10,20,40], 'model__max_features':['sqrt',.8]},
        'svm_rbf':{'model__C':[.01,.03,.1,.3,1,3,10,30,100,300], 'model__gamma':['scale',.01,.1]},
        'hist_boosting':{'model__learning_rate':[.02,.05,.1,.2,.3], 'model__max_leaf_nodes':[7,15,31], 'model__l2_regularization':[2.,10.]}}
    return spaces[name]


def code_provenance():
    digest = hashlib.sha256()
    for folder, pattern in [('src','*.py'),('config','*.json')]:
        for path in sorted((ROOT/folder).glob(pattern)):
            digest.update(path.relative_to(ROOT).as_posix().encode()+b'\n'+path.read_bytes())
    try:
        head = subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True,stderr=subprocess.PIPE).strip()
        dirty = bool(subprocess.check_output(['git','status','--porcelain'],cwd=ROOT,text=True,stderr=subprocess.PIPE).strip())
    except subprocess.CalledProcessError:
        head, dirty = 'UNAVAILABLE_NOT_GIT_CHECKOUT', True
    return {'code_version':'sha256:'+digest.hexdigest(), 'git.sha':'UNAVAILABLE_WORKTREE_DIRTY' if dirty else head,
            'git.head_observed':head, 'git.worktree_dirty':str(dirty)}


def load_source(source='local', authorize_snowflake=False, key_password_provider=None):
    from .analisis import load_population
    frozen, manifest = load_population()
    meta = {'fingerprint':manifest['hashes']['obt'], 'row_count':len(frozen),
            'storage':'LOCAL_PARQUET', 'snapshot_cutoff':'2024-01-01', 'observation_end':'2025-12-31',
            'source_path':'data/synthetic_v1/obt.parquet',
            'source_file_sha256':hashlib.sha256((ROOT/'data/synthetic_v1/obt.parquet').read_bytes()).hexdigest()}
    if source == 'local':
        return frozen, meta
    if source != 'snowflake' or not authorize_snowflake:
        raise ValueError('La lectura remota requiere --source snowflake y --authorize-snowflake-read.')
    # Nothing below runs during planning or offline review. Never import a private profile.
    from getpass import getpass
    from .configuracion import load_local_environment
    from .datos import SnowflakeConfig, connect_snowflake
    from .publicar_snowflake import verify_physical_schema, normalized_read, reconcile_frames, reading_sql
    load_local_environment()
    config = SnowflakeConfig.from_env()
    table = 'OBT_PREDIAL_'+manifest['hashes']['obt'][:12].upper()
    if (config.database, config.schema, config.table) != ('PSET3_SINTETICO','EXPERIMENTOS',table):
        raise ValueError('Solo lectura de la OBT sintetica exacta autorizada.')
    original_projects = [ROOT.parent.parent, ROOT.parent.parent.parent/'pset_2']
    if any(config.private_key_path.resolve().is_relative_to(p.resolve()) for p in original_projects):
        raise ValueError('Clave propia fuera de ambos proyectos originales.')
    password=(key_password_provider or getpass)('Passphrase propia: ')
    connection = connect_snowflake(config,private_key_password=password)
    del password
    try:
        physical = verify_physical_schema(connection,config,table,frozen)
        actual, query_id = normalized_read(connection,config.qualified_table,frozen)
        reconciled = reconcile_frames(frozen,actual,kind='obt')
    finally:
        connection.close()
    # Hashes are independent of ordering. Restore frozen key order BEFORE positional folds.
    keys = ['LOTE_ID','OBLIGACION_ID']
    actual = actual.set_index(keys).loc[pd.MultiIndex.from_frame(frozen[keys])].reset_index()[frozen.columns]
    actual = actual.astype(frozen.dtypes.to_dict())
    meta['local_reference_path'] = meta.pop('source_path')
    meta['local_reference_file_sha256'] = meta.pop('source_file_sha256')
    meta.update(storage='SNOWFLAKE', table=config.qualified_table, query_id=query_id,
                read_query=reading_sql(config.qualified_table,frozen), read_limit=len(frozen)+1,
                physical_schema=physical, reconciliation=reconciled,
                read_at_utc=datetime.now(timezone.utc).isoformat())
    return actual, meta


def plan():
    parts = json.loads((ROOT/'data/synthetic_v1/partitions.json').read_text())
    return {'status':'PREPARED_NOT_EXECUTED', 'defaults':defaults_audit(),
            'folds':[{'train':len(f['train']), 'validation':len(f['validation'])} for f in parts['folds']],
            'phases':['candidates: two baselines + five real-default models', 'tuning: top three NEW candidates, 30 configurations each'],
            'planned_fits':{'candidates':28, 'tuning':360},
            'artifact_policy':'Each fitted fold and refitted configuration logs pipeline/signature/input example. Aggregator-only runs do not fit.',
            'holdout_evaluated':False, 'original_registry_modified':False,
            'grid_sizes':{n:len(list(ParameterGrid(grid(n)))) for n,_,_ in default_candidates(20261117,False)},
            'required_new_tracking_uri':'http://127.0.0.1:5002', 'min_available_gib':6,
            'limits':'Prepared code, not integration-tested with training. Preflight is not a hard memory/timeout guarantee. Generative/foundation category still requires separate approval/evidence.'}


def memory_status():
    # Only the tested Windows platform is supported for actual new fits.
    if os.name != 'nt':
        raise ValueError('Ejecucion nueva en otra plataforma pendiente de validar.')
    import ctypes
    class Memory(ctypes.Structure):
        _fields_ = [('length',ctypes.c_ulong),('load',ctypes.c_ulong)]+[(key,ctypes.c_ulonglong) for key in ['total','available','total_page','available_page','total_virtual','available_virtual','extended']]
    memory = Memory();memory.length = ctypes.sizeof(memory)
    if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(memory)):
        raise OSError('No se pudo medir RAM.')
    return {'available_gib':memory.available/1024**3,'memory_load_percent':int(memory.load)}


def memory_preflight():
    state=memory_status()
    if state['available_gib'] < 6:
        raise ValueError('Menos de 6 GiB disponibles: no iniciar entrenamientos nuevos.')
    return state


def execute(study_id, uri, stage='candidates', source='local', authorize_snowflake=False, key_password_provider=None, progress_callback=None, resume_candidates=False):
    if sklearn.__version__ != '1.7.2':
        raise ValueError('Defaults auditados solo con sklearn 1.7.2.')
    if not re.fullmatch('[a-z][a-z0-9_-]{2,50}', study_id):
        raise ValueError('Identificador nuevo no valido.')
    parsed = urlparse(uri)
    if parsed.scheme != 'http' or parsed.hostname not in ('localhost','127.0.0.1') or parsed.port != 5002 or parsed.username or parsed.password or parsed.path not in ('','/') or parsed.query:
        raise ValueError('Usar el backend independiente local en puerto 5002.')
    memory_preflight()
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    from threadpoolctl import threadpool_limits
    from .analisis import derive_features
    from .configuracion import load_experiment_contract
    from .contratos import FeatureSpec
    from .pipeline import build_pipeline,select_xy
    from .experimentos import seeded_clone,seed_everything
    from .tracking import tracked_run,log_results
    client = mlflow.MlflowClient(tracking_uri=uri)
    if client.get_experiment_by_name('pset3_mc_eventos_v1') is not None:
        raise ValueError('Backend historico detectado: no ejecutar.')
    exp = 'pset3_new_'+study_id
    output = ROOT/'estudios'/study_id
    prior_rows=[]
    partial_resume=None
    continuation_suffix="_"+str(1+len(list(output.glob("provenance_continuation*.json")))) if resume_candidates else ""
    if resume_candidates:
        if stage!='candidates' or not (output/'completed.json').is_file() or (output/'candidates.json').exists():
            raise ValueError('Continuacion explicita solo de candidatos parcialmente completados.')
        existing=client.get_experiment_by_name(exp)
        if existing is None:raise ValueError('No existe el experimento parcial.')
        prior_rows=json.loads((output/'completed.json').read_text())
        expected=[n for n,_,_ in default_candidates(20261117)]
        if not prior_rows or [r['candidate'] for r in prior_rows]!=expected[:len(prior_rows)] or len(prior_rows)>=7:
            raise ValueError('El recibo parcial no es un prefijo completo de candidatos.')
        if client.search_runs([existing.experiment_id],filter_string="attributes.status = 'RUNNING'",max_results=1):
            raise ValueError('Hay una corrida activa; no iniciar otra.')
        prior_meta=json.loads((output/'provenance.json').read_text())
        for row in prior_rows:
            old=client.get_run(row['run_id'])
            if old.info.experiment_id!=existing.experiment_id or old.info.status!='FINISHED' or old.data.tags.get('pset3.study')!=study_id or old.data.tags.get('pset3.dataset_fingerprint')!=prior_meta['fingerprint'] or old.data.metrics.get('validation_roc_auc_mean')!=row['validation_roc_auc_mean']:
                raise ValueError('Corrida parcial no conciliada; no continuar.')
        receipt=output/'partial_current.json'
        if receipt.is_file():
            partial=json.loads(receipt.read_text())
            if partial['candidate']==expected[len(prior_rows)] and partial['folds']:
                parent=client.get_run(partial['run_id'])
                if parent.info.experiment_id!=existing.experiment_id or parent.info.status!='FAILED' or parent.data.tags.get('pset3.study')!=study_id or parent.data.tags.get('pset3.dataset_fingerprint')!=prior_meta['fingerprint']:
                    raise ValueError('Recibo de folds ajeno al estudio parcial.')
                folds=client.search_runs([existing.experiment_id],filter_string="tags.`mlflow.parentRunId` = '"+partial['run_id']+"'",max_results=100)
                finished={int(r.data.params['fold']):r for r in folds if r.info.status=='FINISHED' and 'fold' in r.data.params}
                ids=[]
                if not 0<len(partial['folds'])<=3:raise ValueError('Cantidad de folds parcial invalida.')
                for i,metrics in enumerate(partial['folds']):
                    r=finished.get(i)
                    if r is None or any(r.data.metrics.get(k)!=v for k,v in metrics.items()):
                        raise ValueError('Folds parciales no conciliados contra MLflow.')
                    ids.append(r.info.run_id)
                partial_resume={**partial,'fold_run_ids':ids}
    elif stage == 'candidates':
        if output.exists() or client.get_experiment_by_name(exp) is not None:
            raise ValueError('El estudio ya existe; elegir un identificador nuevo.')
    elif not (output/'candidates.json').is_file() or (output/'tuning').exists():
        raise ValueError('Tuning exige candidatos NUEVOS terminados y no se repite.')
    frame, provenance = load_source(source,authorize_snowflake,key_password_provider)
    if progress_callback:progress_callback('SOURCE_VERIFIED',{'rows':len(frame),'columns':len(frame.columns),'provenance':provenance})
    parts = json.loads((ROOT/'data/synthetic_v1/partitions.json').read_text())
    config = copy.deepcopy(load_experiment_contract(ROOT/'config/experiment.json'));spec = FeatureSpec.from_dict(config)
    dev = derive_features(frame).iloc[parts['train_indices']].reset_index(drop=True)
    if dev.OBLIGACION_ID.tolist() != parts['development_keys'] or provenance['fingerprint'] != parts['fingerprint']:
        raise ValueError('Snapshot/folds no conciliados.')
    X,y = select_xy(dev,spec);y = y.astype(int)
    provenance.update(development_rows=len(X), features=list(spec.features),
                      partition_file_sha256=hashlib.sha256((ROOT/'data/synthetic_v1/partitions.json').read_bytes()).hexdigest(),
                      fold_sizes=[{'train':len(f['train']),'validation':len(f['validation'])} for f in parts['folds']])
    del frame,dev  # The training/evaluation routines receive development ONLY.
    version = code_provenance();config['code_version'] = version['code_version']
    seed = config['random_state'];seed_everything(seed)
    if resume_candidates:
        for key in ['storage','fingerprint','development_rows','features','partition_file_sha256','fold_sizes']:
            if prior_meta[key]!=provenance[key]:raise ValueError('Continuacion con fuente/particiones distintas.')
        (output/('provenance_continuation'+continuation_suffix+'.json')).write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    elif stage == 'candidates':
        output.mkdir(parents=True,exist_ok=False)
        (output/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n',encoding='utf-8')
    else:
        prior = json.loads((output/'provenance.json').read_text())
        if prior['storage'] != provenance['storage'] or prior['fingerprint'] != provenance['fingerprint']:
            raise ValueError('Tuning de otra procedencia: crear otro estudio.')
        output = output/'tuning';output.mkdir(exist_ok=False)
    config['run_tags'] = {'pset3.phase':'new_'+stage, 'pset3.study':study_id, 'pset3.source':provenance['storage'],
                          'pset3.family':'aggregate', **{k:v for k,v in version.items() if k.startswith('git.')}}
    (output/('code_provenance_continuation'+continuation_suffix+'.json' if resume_candidates else 'code_provenance.json')).write_text(json.dumps(version,indent=2)+'\n',encoding='utf-8')

    memory_samples=[]
    def fit_log(model, train, valid, label):
        gc.collect()
        state=memory_preflight()
        memory_samples.append({'checkpoint':'before_fit','candidate':name,'completed_configurations':len(rows),**state})
        (output/'memory_checkpoints.json').write_text(json.dumps(memory_samples,indent=2)+'\n',encoding='utf-8')
        if progress_callback:progress_callback('FIT_START',{'candidate':name,'train_rows':len(train),'validation_rows':None if valid is None else len(valid),**state})
        with warnings.catch_warnings(record=True) as captured:
            warnings.simplefilter('always')
            with threadpool_limits(limits=2):
                started = perf_counter();model.fit(X.iloc[train],y.iloc[train]);elapsed = perf_counter()-started
                metrics = {'fit_seconds':elapsed, 'train_rows':len(train)}
                if valid is not None:
                    metrics['validation_rows'] = len(valid)
                for metric, scorer_name in SCORING.items():
                    scorer = get_scorer(scorer_name)
                    metrics['train_'+metric] = float(scorer(model,X.iloc[train],y.iloc[train]))
                    if valid is not None:
                        metrics['validation_'+metric] = float(scorer(model,X.iloc[valid],y.iloc[valid]))
                        metrics['gap_'+metric] = metrics['train_'+metric]-metrics['validation_'+metric]
        log_results(parameters={'seed':int(model.get_params().get('model__random_state',seed))},metrics=metrics)
        example = X.iloc[train].head(3).copy().where(X.iloc[train].head(3).notna(),None)
        mlflow.sklearn.log_model(model,name=label,signature=infer_signature(example,model.predict(example)),input_example=example,
                                serialization_format='cloudpickle',code_paths=[str(ROOT/'src')])
        mlflow.log_dict({'warnings':sorted({str(w.message) for w in captured})},'warnings.json')
        return metrics

    estimators = {n:(family,model) for n,family,model in default_candidates(seed)}
    if stage == 'candidates':
        selected = [n for n in estimators if n not in {r['candidate'] for r in prior_rows}]
    else:
        prior = json.loads((output.parent/'candidates.json').read_text())
        if len(prior) != 7:raise ValueError('Comparacion nueva incompleta.')
        selected = pd.DataFrame(prior).query("candidate not in ['dummy','regla_academica']").sort_values(['validation_roc_auc_mean','validation_roc_auc_std'],ascending=[False,True]).head(3).candidate.tolist()
    rows = list(prior_rows)
    with tracked_run('NEW-'+stage+('-continuation'+continuation_suffix if resume_candidates else ''),contract=config,dataset_metadata=provenance,tracking_uri=uri,experiment_name=exp):
        mlflow.log_dict(provenance,'dataset_provenance.json')
        for name in selected:
            family, estimator = estimators[name]
            config['run_tags']['pset3.family'] = family
            with tracked_run(name,contract=config,dataset_metadata=provenance,nested=True):
                configs = list(ParameterGrid(grid(name))) if stage == 'tuning' else [{}]
                for index,parameters in enumerate(configs):
                    with tracked_run(name+'-'+str(index),contract=config,dataset_metadata=provenance,nested=True) as run:
                        prototype = seeded_clone(build_pipeline(estimator,spec,sparse_output=False).set_params(**parameters),seed)
                        mlflow.log_params({**{'model__'+k:v for k,v in prototype.named_steps['model'].get_params(deep=False).items()},'seed':seed})
                        mlflow.log_dict(provenance,'dataset_provenance.json')
                        reuse=partial_resume if partial_resume and partial_resume['candidate']==name else None
                        scores=list(reuse['folds']) if reuse else []
                        if reuse:
                            mlflow.set_tag('pset3.reused_cv_config_run_id',reuse['run_id'])
                            mlflow.log_dict(reuse,'reused_folds.json')
                        for i, fold in enumerate(parts['folds']):
                            if i<len(scores):continue
                            with tracked_run('fold-'+str(i),contract=config,dataset_metadata=provenance,nested=True):
                                model = seeded_clone(build_pipeline(estimator,spec,sparse_output=False).set_params(**parameters),seed+i)
                                mlflow.log_params({**{'model__'+k:v for k,v in model.named_steps['model'].get_params(deep=False).items()},'fold':i})
                                mlflow.log_dict(provenance,'dataset_provenance.json')
                                from .datos import dataset_fingerprint
                                mlflow.set_tags({'pset3.train_fingerprint':dataset_fingerprint(X.iloc[fold['train']]),
                                                 'pset3.validation_fingerprint':dataset_fingerprint(X.iloc[fold['validation']])})
                                scores.append(fit_log(model,fold['train'],fold['validation'],'pipeline'))
                                mlflow.log_dict({'completed_folds':scores},'partial_cv_results.json')
                                (output/'partial_current.json').write_text(json.dumps({'candidate':name,'run_id':run.info.run_id,'folds':scores},indent=2)+'\n',encoding='utf-8')
                                del model
                                gc.collect()
                        table = pd.DataFrame(scores)
                        summary = {col+'_'+suffix:float(getattr(table[col],suffix)(**({'ddof':1} if suffix=='std' else {}))) for col in table for suffix in ['mean','std']}
                        log_results(parameters={},metrics=summary)
                        # Refit on development only, artifact on each configuration run.
                        final_model = seeded_clone(build_pipeline(estimator,spec,sparse_output=False).set_params(**parameters),seed)
                        refit = fit_log(final_model,list(range(len(X))),None,'development_pipeline')
                        row = {'candidate':name,'config_index':index,'parameters':parameters,'run_id':run.info.run_id,**summary,'refit_fit_seconds':refit['fit_seconds']}
                        if reuse:row.update(reused_cv_config_run_id=reuse['run_id'],reused_fold_run_ids=reuse['fold_run_ids'])
                        rows.append(row)
                        mlflow.log_dict({'folds':scores},'cv_results.json')
                        (output/'completed.json').write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
                        del final_model,prototype
                        gc.collect()
                        if progress_callback:progress_callback('CANDIDATE_COMPLETED',{'candidate':name,'run_id':row['run_id'],'cv_auc':row['validation_roc_auc_mean'],**memory_status()})
    export = mlflow.search_runs([mlflow.get_experiment_by_name(exp).experiment_id],output_format='pandas')
    export.to_csv(output/'runs_new.csv',index=False)
    (output/('candidates.json' if stage=='candidates' else 'tuning.json')).write_text(json.dumps(rows,indent=2)+'\n',encoding='utf-8')
    return {'study':study_id,'stage':stage,'configurations_completed':len(rows),'holdout_used':False,'registry_aliases_changed':False,'output':str(output)}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--execute',action='store_true')
    p.add_argument('--study-id')
    p.add_argument('--tracking-uri',default='http://127.0.0.1:5002')
    p.add_argument('--stage',choices=['candidates','tuning'],default='candidates')
    p.add_argument('--source',choices=['local','snowflake'],default='local')
    p.add_argument('--authorize-snowflake-read',action='store_true')
    p.add_argument('--resume-candidates',action='store_true',help='Continuar explicitamente un prefijo y folds conciliados; no repetir fits terminados.')
    a = p.parse_args()
    if not a.execute:
        print(json.dumps(plan(),indent=2));return
    if not a.study_id:p.error('--execute requiere --study-id')
    print(json.dumps(execute(a.study_id,a.tracking_uri,a.stage,a.source,a.authorize_snowflake_read,resume_candidates=a.resume_candidates),indent=2))


if __name__ == '__main__':
    main()
