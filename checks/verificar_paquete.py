"""Auditoria offline de entrega; no fit, ingestas, servidor o credenciales."""
import json,hashlib,argparse
from pathlib import Path
import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import roc_auc_score
from src.analisis import ROOT,DATA,load_population,derive_features
from src.configuracion import load_experiment_contract
from src.contratos import FeatureSpec
from src.pipeline import select_xy


def audit():
    checksums=ROOT/'MANIFEST_SHA256.json';verified=0
    if checksums.exists():
        inventory=json.loads(checksums.read_text(encoding='utf-8'))
        for relative,expected in inventory['files'].items():
            relative_path=Path(relative)
            if relative_path.is_absolute() or '..' in relative_path.parts:raise ValueError('Ruta fuera del paquete.')
            actual=ROOT/relative_path
            if not actual.is_file() or hashlib.sha256(actual.read_bytes()).hexdigest()!=expected:raise ValueError('Archivo no coincide con manifiesto: '+relative)
            verified+=1
    frame,manifest=load_population();parts=json.loads((DATA/'partitions.json').read_text())
    exported=json.loads((ROOT/'models/manifest.json').read_text());assert manifest['hashes']['obt']==exported['population_hash']==parts['fingerprint']
    config=load_experiment_contract();spec=FeatureSpec.from_dict(config)
    hold=derive_features(frame).iloc[parts['holdout_indices']].reset_index(drop=True)
    X,y=select_xy(hold,spec);roles={}
    for role,item in exported['roles'].items():
        path=ROOT/item['file'];assert hashlib.sha256(path.read_bytes()).hexdigest()==item['sha256']
        # Solo modelos propios incluidos en esta entrega, nunca pickle externo.
        model=joblib.load(path);score=model.predict_proba(X)[:,1];prediction=model.predict(X)
        reference=pd.read_parquet(ROOT/item['reference_predictions'])
        np.testing.assert_array_equal(prediction,reference.prediction.to_numpy())
        np.testing.assert_allclose(score,reference.score.to_numpy(),rtol=0,atol=0)
        auc=float(roc_auc_score(y.astype(int),score));assert auc==item['expected_roc_auc']
        roles[role]={'exact_scores_and_classes':True,'roc_auc':auc,'source_run_id':item['source_run_id']}
    runs=pd.read_csv(ROOT/'docs/corridas.csv');tuning=pd.read_csv(ROOT/'docs/cierre/tuning_configuraciones.csv')
    assert len(runs)==396 and len(tuning)==90 and tuning.config_key.nunique()==90
    assert tuning.groupby('candidate').size().eq(30).all()
    for _,row in tuning.iterrows():
        matched=runs.loc[runs.run_id==row.run_id];assert len(matched)==1 and matched.iloc[0].status=='FINISHED'
        folds=runs.loc[runs['tags.mlflow.parentRunId']==row.run_id];assert len(folds)==3 and (folds.status=='FINISHED').all()
    from src.historial_portable import verify
    portable=verify()
    markdown=[*ROOT.glob('*.md'),*(ROOT/'docs').rglob('*.md')]
    for path in markdown:
        payload=path.read_bytes().replace(b'\r\n',b'\n')
        if b'\r' in payload or any(x<32 and x not in (9,10) for x in payload):raise ValueError('Caracter de control en Markdown: '+str(path.relative_to(ROOT)))
    from checks.verificar_publicacion import AUTHORIZED_PDF,validate_academic_pdf
    assert {name for name in inventory['files'] if Path(name).suffix.lower()=='.pdf'}=={AUTHORIZED_PDF}
    pdf_info=validate_academic_pdf()
    return {'mode':'OFFLINE_FROZEN_EVIDENCE_VERIFICATION','files_hash_verified':verified,'population_hash':manifest['hashes']['obt'],
       'rows':len(frame),'events':manifest['quality']['events'],'roles':roles,'academic_runs':len(runs),'tuning_configurations':len(tuning),
       'original_metadata_archive':portable,'markdown_control_character_scan':'PASS','pdf_status':'AUTHORIZED_AND_VERIFIED','academic_pdf':pdf_info,'training_executed':False,'threshold_reoptimized':False,'snowflake_connected':False,'mlflow_server_required':False,
       'note':'Original run IDs in CSV are historical. This verifies portable model artifacts and stored results, not availability of original tracking volumes.'}

if __name__=='__main__':print(json.dumps(audit(),indent=2))
