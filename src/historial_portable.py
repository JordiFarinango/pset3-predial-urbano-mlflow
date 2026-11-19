"""Archivo de metadatos originales. Nunca importa/recrea corridas."""
import argparse,csv,hashlib,json,math,re
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'docs/mlflow_portable'


def verify():
    metadata=json.loads((OUT/'manifest.json').read_text(encoding='utf-8'))
    assert hashlib.sha256((ROOT/'docs/corridas.csv').read_bytes()).hexdigest()==metadata['original_csv_sha256']
    payload=(OUT/'runs_originales.jsonl').read_bytes();assert hashlib.sha256(payload).hexdigest()==metadata['jsonl_sha256']
    rows=[json.loads(x) for x in payload.decode('utf-8').splitlines()];index={r['info']['run_id']:r for r in rows}
    with (ROOT/'docs/corridas.csv').open(encoding='utf-8',newline='') as stream:csv_rows=list(csv.DictReader(stream))
    assert len(rows)==len(index)==len(csv_rows)==396
    assert set(index)=={r['run_id'] for r in csv_rows}
    compared=0
    for row in csv_rows:
        record=index[row['run_id']];assert record['info']['status']==row['status']
        for name,value in record['metrics'].items():
            expected=row.get('metrics.'+name,'')
            assert expected!='' and math.isclose(float(expected),value,rel_tol=0,abs_tol=1e-12),(row['run_id'],name)
            compared+=1
        parent=record['tags'].get('mlflow.parentRunId')
        if parent:assert parent in index
    return {'status':'VERIFIED_ORIGINAL_METADATA_ARCHIVE','run_ids':len(index),'metrics_compared_to_original_csv':compared,
        'numeric_tolerance':1e-12,'jsonl_sha256':metadata['jsonl_sha256'],'creates_or_imports_runs':False,'requires_server_to_verify':False}


def export_originals():
    import mlflow
    client=mlflow.MlflowClient(tracking_uri='http://127.0.0.1:5001')
    experiment=client.get_experiment_by_name('pset3_mc_eventos_v1')
    if experiment is None:raise ValueError('Experimento original no disponible.')
    runs=client.search_runs([experiment.experiment_id],max_results=1000);assert len(runs)==396
    sensitive=re.compile(r'password|passphrase|private_key|api_key|authorization|credential|secret|token',re.I)
    records=[];removed=0
    for run in runs:
        params={}
        for key,value in run.data.params.items():
            if sensitive.search(key):removed+=1;continue
            if re.search(r'https?://[^/@\s]+:[^/@\s]+@',value):raise ValueError('URL con autenticacion no permitida.')
            params[key]=value
        tags={key:value for key,value in run.data.tags.items() if (key.startswith('pset3.') or key in ['git.sha','mlflow.runName','mlflow.parentRunId']) and not sensitive.search(key)}
        records.append({'info':{'run_id':run.info.run_id,'experiment_id':run.info.experiment_id,'status':run.info.status,
          'start_time_ms_utc':run.info.start_time,'end_time_ms_utc':run.info.end_time},'params':params,'metrics':run.data.metrics,'tags':tags})
    OUT.mkdir(exist_ok=True)
    payload=('\n'.join(json.dumps(r,sort_keys=True,ensure_ascii=False,allow_nan=False) for r in sorted(records,key=lambda r:r['info']['run_id']))+'\n').encode('utf-8')
    (OUT/'runs_originales.jsonl').write_bytes(payload)
    code_versions=sorted({r['tags'].get('pset3.code_version') for r in records if r['tags'].get('pset3.code_version')})
    metadata={'mode':'READ_ONLY_EXPORT_OF_ORIGINAL_RUN_METADATA','source_tracking_uri':'http://127.0.0.1:5001','experiment_name':experiment.name,
      'experiment_id':experiment.experiment_id,'runs':len(records),'jsonl_sha256':hashlib.sha256(payload).hexdigest(),
      'original_csv_sha256':hashlib.sha256((ROOT/'docs/corridas.csv').read_bytes()).hexdigest(),'code_versions_as_recorded':code_versions,
      'sensitive_params_excluded':removed,'not_a_backend_backup':True,'not_recreated_runs':True,'artifacts_not_included_here':True,
      'notes':'Original server, IDs, statuses, times and metrics retained. Only scientific params/tags exported; no user profiles or credentials. CSV remains untouched.'}
    (OUT/'manifest.json').write_bytes((json.dumps(metadata,indent=2)+'\n').encode('utf-8'))
    result=verify();(OUT/'verificacion.json').write_bytes((json.dumps(result,indent=2)+'\n').encode('utf-8'));return result

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--export',action='store_true');args=parser.parse_args()
    print(json.dumps(export_originals() if args.export else verify(),indent=2))
