"""Comparacion fundacional local, presupuesto limitado y sin datos externos."""
import os
from pathlib import Path
from time import perf_counter
import json
import numpy as np
import pandas as pd
from .cierre import context,dump,OUT
from .ejecutar_estudio import EXPERIMENT
from .tracking import tracked_run,log_results,configure_tracking


def compare_foundation():
    result_path=Path(__file__).resolve().parents[1]/'docs/fundacional_resultado.json'
    if result_path.exists():return json.loads(result_path.read_text())
    root=Path(__file__).resolve().parents[1]
    os.environ.update({'TABPFN_NO_BROWSER':'1','TABPFN_DISABLE_TELEMETRY':'1','HF_HUB_DISABLE_IMPLICIT_TOKEN':'1',
      'HF_HOME':str(root/'data/hf_cache'),'TABPFN_MODEL_CACHE_DIR':str(root/'data/tabpfn_models'),'TABPFN_ALLOW_CPU_LARGE_DATASET':'true'})
    frame,m,p,c,spec,X,y,identity,meta=context();configure_tracking(experiment_name=EXPERIMENT)
    c['run_tags']={'pset3.phase':'foundation_candidate','pset3.family':'tabular_foundation','pset3.source':'LOCAL_PARQUET','git.sha':'UNAVAILABLE_PSET3_NOT_VERSIONED'}
    run_id=None;start=perf_counter()
    try:
        with tracked_run('tabpfn_v2',contract=c,dataset_metadata=meta,experiment_name=EXPERIMENT) as run:
            run_id=run.info.run_id
            import torch
            torch.set_num_threads(2)
            from tabpfn import TabPFNClassifier
            from tabpfn.constants import ModelVersion
            from .pipeline import build_pipeline
            from .experimentos import evaluate_cv
            model=TabPFNClassifier.create_default_for_version(ModelVersion.V2,n_estimators=1,device='cpu',
                fit_mode='fit_with_cache',memory_saving_mode=True,random_state=c['random_state'],n_preprocessing_jobs=1)
            log_results(parameters={'tabpfn_version':'10.2.0','weights_version':'V2','device':'cpu','n_estimators':1,'seed':c['random_state'],
                'gpu_detected':'GTX1650_4GiB','weight_license':'Prior Labs v2 Apache2+attribution','cpu_threads':2},metrics={})
            from .experimentos import evaluate_cv
            cv=evaluate_cv(build_pipeline(model,spec,sparse_output=False),X,y,
                splits=[(np.array(f['train']),np.array(f['validation'])) for f in p['folds']],
                scoring={'roc_auc':'roc_auc','average_precision':'average_precision','balanced_accuracy':'balanced_accuracy','f1':'f1','accuracy':'accuracy'},
                seed=c['random_state'],contract=c,provenance=meta,tracking=True)
            log_results(parameters={},metrics=cv.summary)
            cv.folds.to_csv(OUT/'foundation_folds.csv',index=False)
            import mlflow
            mlflow.log_artifact(str(OUT/'foundation_folds.csv'))
            report={'status':'SUCCESS','result':{'candidate':'tabpfn_v2','family':'tabular_foundation','run_id':run_id,**cv.summary},'seconds':perf_counter()-start,'holdout_used':False}
    except Exception as error:
        report={'status':'BLOCKED','run_id':run_id,'error_type':type(error).__name__,'error':str(error),
            'seconds':perf_counter()-start,'holdout_used':False,'installed':'tabpfn10.2.0 torch2.15.0','hardware':'19GiB RAM; GTX1650 4GiB; CPU used',
            'note':'No license gate bypass; no automatic login; no municipal data. Failure kept in MLflow.'}
    dump(result_path,report);return report

if __name__=='__main__':print(json.dumps(compare_foundation(),indent=2))
