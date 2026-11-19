"""Primeros experimentos academicos sobre simulacion local; no evalua hold-out."""
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter
import warnings
import numpy as np
import pandas as pd
from sklearn.metrics import get_scorer,roc_auc_score
from .analisis import ROOT,DATA,load_population,derive_features
from .contratos import FeatureSpec
from .configuracion import load_experiment_contract
from .experimentos import seed_everything,evaluate_cv,seeded_clone
from .pipeline import build_pipeline,select_xy
from .modelos import candidates
from .tracking import tracked_run,log_results,export_runs

EXPERIMENT='pset3_mc_eventos_v1'


def code_hash():
    digest=hashlib.sha256()
    for folder in ['src','config']:
        for path in sorted((ROOT/folder).glob('*.py' if folder=='src' else '*.json')):
            if path.name=='experiment.json':continue
            digest.update(str(path.relative_to(ROOT)).encode());digest.update(path.read_bytes())
    return digest.hexdigest()


def leakage_diagnostics():
    frame,manifest=load_population();config=load_experiment_contract();parts=json.loads((DATA/'partitions.json').read_text())
    dev=derive_features(frame).iloc[parts['train_indices']].reset_index(drop=True)
    spec=FeatureSpec.from_dict(config);X,y=select_xy(dev,spec);y=y.astype(int)
    numeric={}
    for name in spec.numeric_features:
        values=X[name];values=values.fillna(values.median())
        auc=float(roc_auc_score(y,values));numeric[name]=max(auc,1-auc)
    assert max(numeric.values())<.95,'Relacion trivial detectada en feature individual.'
    # Control negativo: etiquetas train permutadas independientemente en cada fold.
    from sklearn.linear_model import LogisticRegression
    rng=np.random.default_rng(config['random_state']);scores=[]
    for fold in parts['folds']:
        tr,va=fold['train'],fold['validation'];model=build_pipeline(LogisticRegression(max_iter=300),spec)
        model.fit(X.iloc[tr],rng.permutation(y.iloc[tr]))
        scores.append(float(roc_auc_score(y.iloc[va],model.predict_proba(X.iloc[va])[:,1])))
    assert abs(float(np.mean(scores))-.5)<.12,'Control negativo no cercano al azar: investigar.'
    result={'univariate_orientation_free_auc':numeric,'permuted_train_label_auc':scores,
       'negative_control_auc_mean':float(np.mean(scores)),'holdout_used':False,
       'excluded_future_columns':[x for x in ['PAGADO_90_CENT','SALDO_90_CENT','N_MOVIMIENTOS','LABEL_AVAILABLE_AT'] if x not in spec.features]}
    (ROOT/'docs/leakage_diagnosticos.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')
    return result


def run_candidates(*, refresh=False):
    import mlflow
    import mlflow.sklearn
    from mlflow.models import infer_signature
    os.environ['MLFLOW_EXPERIMENT_NAME']=EXPERIMENT
    os.environ['MLFLOW_TRACKING_URI']='http://127.0.0.1:5001'
    config=load_experiment_contract();seed=config['random_state'];seed_everything(seed)
    frame,manifest=load_population();parts=json.loads((DATA/'partitions.json').read_text())
    assert parts['fingerprint']==manifest['hashes']['obt']
    development=derive_features(frame).iloc[parts['train_indices']].reset_index(drop=True)
    assert development.OBLIGACION_ID.tolist()==parts['development_keys']
    spec=FeatureSpec.from_dict(config);X,y=select_xy(development,spec);y=y.astype(int)
    splits=[(np.array(f['train']),np.array(f['validation'])) for f in parts['folds']]
    config['code_version']='sha256:'+code_hash()
    provenance={'fingerprint':manifest['hashes']['obt'],'row_count':len(frame),
       'development_rows':len(X),'storage':'LOCAL_PARQUET','source_path':'data/synthetic_v1/obt.parquet',
       'snowflake_status':'NOT_PUBLISHED_USER_SELECTED_LOCAL','snowflake_table':None,
       'observation_end':manifest['config']['observation_end'],'cutoff':'2024-01-01',
       'lote_ids':manifest['lote_ids'],'development_fingerprint':__import__('src.datos',fromlist=['dataset_fingerprint']).dataset_fingerprint(development)}
    scoring={'roc_auc':'roc_auc','average_precision':'average_precision','balanced_accuracy':'balanced_accuracy','f1':'f1','accuracy':'accuracy'}
    if not refresh and (ROOT/'docs/resultados_candidatos.csv').exists():
        cached=pd.read_csv(ROOT/'docs/resultados_candidatos.csv');client=mlflow.MlflowClient()
        if len(cached)==7 and all(client.get_run(r).info.status=='FINISHED' and client.get_run(r).data.tags.get('pset3.dataset_fingerprint')==manifest['hashes']['obt'] for r in cached.run_id):
            print('Reutilizando siete corridas reales verificadas del mismo dataset; sin nuevos entrenamientos.')
            return cached
    summaries=[]; failures=[]
    result_path=ROOT/'docs/resultados_candidatos.csv'
    for name,family,estimator in candidates(seed):
        run_id=None
        try:
            config['run_tags']={'pset3.phase':'baseline_candidates','pset3.family':family,'git.sha':'UNAVAILABLE_PSET3_NOT_VERSIONED','pset3.source':'LOCAL_PARQUET','pset3.cutoff':'2024-01-01'}
            with tracked_run(name,contract=config,dataset_metadata=provenance,experiment_name=EXPERIMENT,
                   tags={'pset3.phase':'baseline_candidates','pset3.family':family,'git.sha':'UNAVAILABLE_PSET3_NOT_VERSIONED',
                         'pset3.source':'LOCAL_PARQUET','pset3.snowflake_status':provenance['snowflake_status'],'pset3.cutoff':'2024-01-01'}) as run:
                run_id=run.info.run_id
                with warnings.catch_warnings(record=True) as captured:
                    warnings.simplefilter('always')
                    pipeline=build_pipeline(estimator,spec,sparse_output=False)
                    result=evaluate_cv(pipeline,X,y,splits=splits,scoring=scoring,seed=seed,contract=config,provenance=provenance,tracking=True)
                    # Refit SOLO sobre desarrollo; no accede a las etiquetas hold-out.
                    fitted=seeded_clone(pipeline,seed);start=perf_counter();fitted.fit(X,y);refit_seconds=perf_counter()-start
                    train_scores={f'refit_train_{k}':float(get_scorer(v)(fitted,X,y)) for k,v in scoring.items()}
                parameters={f'model__{k}':v for k,v in estimator.get_params(deep=False).items()}
                parameters.update({'seed':seed,'family':family,'source':'LOCAL_PARQUET','cutoff':'2024-01-01','observation_end':manifest['config']['observation_end']})
                metrics={**result.summary,**train_scores,'refit_seconds':refit_seconds}
                log_results(parameters=parameters,metrics=metrics)
                result.folds.to_csv(ROOT/f'docs/folds_{name}.csv',index=False)
                mlflow.log_artifact(str(ROOT/f'docs/folds_{name}.csv'),'cv')
                mlflow.log_dict(provenance,'dataset_provenance.json')
                mlflow.log_dict(manifest,'generation_manifest.json')
                mlflow.log_dict({'development_keys':parts['development_keys'],'folds':parts['folds'],'holdout_not_evaluated':True},'partitions.json')
                warning_messages=sorted({f'{type(w.message).__name__}: {w.message}' for w in captured})
                mlflow.log_dict({'warnings':warning_messages},'warnings.json')
                # Artefacto sklearn, sin Registrar/seleccionar champion. Pickle solo modelo
                # propio de esta corrida: no cargar artefactos externos no confiables.
                example=X.head(3).copy()
                example=example.where(example.notna(),None)
                model=mlflow.sklearn.log_model(fitted,name='pipeline',signature=infer_signature(example,fitted.predict(example)),
                    input_example=example,serialization_format='cloudpickle',code_paths=[str(ROOT/'src')],
                    pip_requirements=['scikit-learn==1.7.2','numpy==2.3.5','pandas==3.0.1','scipy==1.16.3','cloudpickle==3.1.2'])
                row={'candidate':name,'family':family,'run_id':run_id,'status':'FINISHED','warnings':len(warning_messages),
                     'model_uri':model.model_uri,**metrics}
                summaries.append(row)
        except Exception as error:
            # Historial conserva la corrida fallida; no se fabrican metricas.
            failure={'candidate':name,'run_id':run_id,'error_type':type(error).__name__,'error':str(error)}
            failures.append(failure);print('FALLO REAL:',failure)
        pd.DataFrame(summaries).to_csv(result_path,index=False)
        (ROOT/'docs/errores_candidatos.json').write_text(json.dumps(failures,indent=2)+'\n',encoding='utf-8')
    count=export_runs(ROOT/'docs/corridas.csv',experiment_name=EXPERIMENT)
    report={'experiment':EXPERIMENT,'candidates_successful':len(summaries),'failures':failures,'exported_runs':count,
       'holdout_evaluated':False,'git_sha':'UNAVAILABLE_PSET3_NOT_VERSIONED','code_sha256':config['code_version'].removeprefix('sha256:'),
       'snowflake_status':'NOT_PUBLISHED_USER_SELECTED_LOCAL','llm_status':'NOT_RUN_NO_LOCAL_WEIGHTS_OR_PROVIDER_CONFIGURED'}
    (ROOT/'docs/experimentos_resumen.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return pd.DataFrame(summaries)
