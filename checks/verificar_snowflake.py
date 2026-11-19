"""Conciliacion offline de evidencia Snowflake; no abre conexiones."""
import json,re
from uuid import UUID
from pathlib import Path
from src.datos import dataset_fingerprint
from src.publicar_snowflake import logical_fingerprint,sql_type
ROOT=Path(__file__).resolve().parents[1]
def verify(frames,evidence=None):
    e=evidence if evidence is not None else json.loads((ROOT/'docs/snowflake_evidencia.json').read_text(encoding='utf-8'))
    assert e['classification']=='SYNTHETIC' and e['status']=='READBACK_VERIFIED'
    assert (e['database'],e['schema'])==('PSET3_SINTETICO','EXPERIMENTOS')
    assert e['experimental_training_source']=='LOCAL_PARQUET' and e['publication_changes_historical_training_lineage'] is False
    stages={'snowflake_publicacion.json':'PUBLISHED_AND_READBACK_VERIFIED','snowflake_relectura.json':'READBACK_VERIFIED'}
    assert set(e['source_receipts'])==set(stages)
    for name,status in stages.items():
        assert set(e['source_receipts'][name])=={'status','sha256'}
        assert e['source_receipts'][name]['status']==status
        assert re.fullmatch('[0-9a-f]{64}',e['source_receipts'][name]['sha256'])
    suffix=e['source_obt_fingerprint'][:12].upper()
    expected={'obligaciones':('OBLIGACIONES',12000,15),'movimientos':('MOVIMIENTOS',18988,6),'obt':('OBT_PREDIAL',12000,20)}
    assert set(e['objects'])==set(expected)
    assert dataset_fingerprint(frames['obt'])==e['source_obt_fingerprint']
    for name,(prefix,n,c) in expected.items():
        obj=e['objects'][name];frame=frames[name];rec=obj['reconciliation']
        assert obj['table']=='.'.join('"'+x+'"' for x in ['PSET3_SINTETICO','EXPERIMENTOS',prefix+'_'+suffix])
        assert frame.shape==(n,c) and obj['rows']==rec['rows']==n and obj['columns']==rec['columns']==c and obj['read_limit']==n+1
        assert obj['fingerprint']==rec['training_fingerprint']==dataset_fingerprint(frame)
        assert rec['logical_fingerprint_version']=='pset3-logical-v1' and rec['logical_fingerprint']==logical_fingerprint(frame)
        assert rec['keys_unique_and_equal'] is True and rec['null_counts_equal'] is True
        assert set(obj['query_ids'])==set(stages)
        for query in obj['query_ids'].values():UUID(query)
        assert [x[0] for x in obj['physical_schema']]==list(frame.columns)
        for col,typ,precision,scale in obj['physical_schema']:
            kind=sql_type(frame[col]);assert typ==('TEXT' if kind=='VARCHAR' else kind.split('(')[0])
            if kind.startswith('NUMBER('):assert [precision,scale]==list(map(int,kind[7:-1].split(',')))
        if name=='obt':
            assert rec['labels_equal_by_id'] is True
            assert rec['classes']=={str(k):int(v) for k,v in frame.TARGET_NO_PAGADA_90D.value_counts().items()}
            assert rec['cohort_classes']=={str(y):{str(k):int(v) for k,v in g.TARGET_NO_PAGADA_90D.value_counts().items()} for y,g in frame.groupby('ANIO_FISCAL')}
    return {'status':'PASS','receipts_status':'READBACK_VERIFIED','objects_verified':3,'remote_connection_opened':False,'historical_training_source':'LOCAL_PARQUET'}
