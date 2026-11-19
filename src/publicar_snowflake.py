"""Publicacion opt-in de snapshots ficticios, sin reemplazar tablas existentes.

Preparado, no ejecutado en esta entrega. Nunca usa dbt/Spark/SQL Server.
"""
import argparse
from getpass import getpass
import json
import numpy as np
import pandas as pd
from .analisis import DATA,ROOT,load_population
from .datos import SnowflakeConfig,identifier,connect_snowflake,dataset_fingerprint
from .configuracion import load_local_environment


def publication_plan(config,manifest):
    config.validate()
    if identifier(config.database)!='PSET3_SINTETICO' or identifier(config.schema)!='EXPERIMENTOS':
        raise ValueError('Escritura solo en PSET3_SINTETICO.EXPERIMENTOS; ambas bases PSet2 estan bloqueadas.')
    suffix=manifest['hashes']['obt'][:12].upper()
    return {key:prefix+'_'+suffix for key,prefix in [('obligaciones','OBLIGACIONES'),('movimientos','MOVIMIENTOS'),('obt','OBT_PREDIAL')]}


def sql_type(series):
    if series.name in ('IMPORTE_PRINCIPAL','TOTAL_OBLIGACION_DECLARADO','AVALUO_SIMULADO','AREA_M2'):return 'NUMBER(16,2)'
    if pd.api.types.is_datetime64_any_dtype(series):return 'TIMESTAMP_TZ'
    if pd.api.types.is_integer_dtype(series):return 'NUMBER(38,0)'
    if pd.api.types.is_float_dtype(series):return 'FLOAT'
    return 'VARCHAR'


def normalized_read(connection,qualified,original):
    with connection.cursor() as cursor:
        cursor.execute('SELECT '+','.join('"'+identifier(c)+'"' for c in original.columns)+' FROM '+qualified+' LIMIT %s',(len(original)+1,))
        result=pd.DataFrame(cursor.fetchall(),columns=original.columns)
        query_id=cursor.sfqid
    for name in original.columns:
        if pd.api.types.is_datetime64_any_dtype(original[name]):
            result[name]=pd.to_datetime(result[name],utc=True).astype(original[name].dtype)
        else:
            result[name]=result[name].astype(original[name].dtype)
    return result,query_id


def publish(config,connection):
    # load_population verifica huella ANTES de crear o escribir objetos.
    obt,manifest=load_population();names=publication_plan(config,manifest)
    frames={name:pd.read_parquet(DATA/f'{name}.parquet') for name in names}
    for name,frame in frames.items():
        if dataset_fingerprint(frame)!=manifest['hashes'][name]:raise ValueError('Snapshot local alterado: no publicar.')
    tables={name:'.'.join('"'+identifier(x)+'"' for x in [config.database,config.schema,table]) for name,table in names.items()}
    with connection.cursor() as cursor:
        # Solo metadatos de la base PSET3 autorizada. Si existe cualquier tabla del
        # snapshot, aborta antes de DDL/DML; no borra ni intenta completar a ciegas.
        cursor.execute(f'SELECT TABLE_NAME FROM "{identifier(config.database)}".INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=%s',(identifier(config.schema),))
        existing={r[0] for r in cursor.fetchall()}
        if existing & set(names.values()):raise ValueError('Snapshot ya existente o parcial: revisar; no se sobreescribe.')
        cursor.execute(f'CREATE SCHEMA IF NOT EXISTS "{identifier(config.database)}"."{identifier(config.schema)}"')
    from snowflake.connector.pandas_tools import write_pandas
    receipts={}
    for name,frame in frames.items():
        with connection.cursor() as cursor:
            columns=','.join('"'+identifier(col)+'" '+sql_type(frame[col]) for col in frame.columns)
            cursor.execute('CREATE TABLE '+tables[name]+' ('+columns+')') # sin OR REPLACE
        success,chunks,rows,_=write_pandas(connection,frame,table_name=names[name],database=identifier(config.database),
            schema=identifier(config.schema),auto_create_table=False,overwrite=False,use_logical_type=True)
        if not success or rows!=len(frame):raise ValueError('Carga incompleta: conservar evidencia y revisar snapshot parcial.')
        physical=verify_physical_schema(connection,config,names[name],frame)
        actual,query_id=normalized_read(connection,tables[name],frame)
        reconciliation=reconcile_frames(frame,actual,kind=name)
        receipts[name]={'table':tables[name],'rows':rows,'query_id':query_id,'fingerprint':dataset_fingerprint(actual),'physical_schema':physical,'reconciliation':reconciliation,'read_query':reading_sql(tables[name],frame),'read_limit':len(frame)+1}
    connection.commit()
    receipt={'status':'PUBLISHED_AND_READBACK_VERIFIED','objects':receipts,'source_obt_fingerprint':manifest['hashes']['obt']}
    (ROOT/'docs/snowflake_publicacion.json').write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf-8')
    return receipt



def verify_physical_schema(connection,config,table,original):
    with connection.cursor() as cursor:
        cursor.execute(f'SELECT COLUMN_NAME,DATA_TYPE,NUMERIC_PRECISION,NUMERIC_SCALE FROM "{identifier(config.database)}".INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION',(identifier(config.schema),identifier(table)))
        records=cursor.fetchall()
    actual={row[0]:row for row in records}
    if set(actual)!=set(original.columns):raise ValueError('Columnas Snowflake distintas del snapshot.')
    for name in original:
        kind=sql_type(original[name]);expected='TEXT' if kind=='VARCHAR' else kind.split('(')[0]
        if actual[name][1]!=expected:raise ValueError('Tipo fisico incompatible: '+name)
        if kind.startswith('NUMBER('):
            precision,scale=map(int,kind[7:-1].split(','))
            if tuple(actual[name][2:])!=(precision,scale):raise ValueError('Precision/escala incompatibles: '+name)
    return [list(row) for row in records]


def verify_existing(config,connection):
    _,manifest=load_population();names=publication_plan(config,manifest);receipts={}
    for key,table in names.items():
        original=pd.read_parquet(DATA/f'{key}.parquet')
        if dataset_fingerprint(original)!=manifest['hashes'][key]:raise ValueError('Archivo local alterado.')
        qualified='.'.join('"'+identifier(x)+'"' for x in (config.database,config.schema,table))
        physical=verify_physical_schema(connection,config,table,original)
        actual,query_id=normalized_read(connection,qualified,original)
        reconciliation=reconcile_frames(original,actual,kind=key)
        receipts[key]={'table':qualified,'rows':len(actual),'fingerprint':dataset_fingerprint(actual),'query_id':query_id,'physical_schema':physical,'reconciliation':reconciliation,'read_query':reading_sql(qualified,original),'read_limit':len(original)+1}
    report={'status':'READBACK_VERIFIED','objects':receipts,'source_obt_fingerprint':manifest['hashes']['obt']}
    (ROOT/'docs/snowflake_relectura.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');return report



def reading_sql(qualified,original):
    return 'SELECT '+','.join('"'+identifier(c)+'"' for c in original.columns)+' FROM '+qualified+' LIMIT %s'


def logical_fingerprint(frame):
    """Hash semantico por contrato, independiente del orden y del dtype fisico.

    Dinero de dos decimales, centavos enteros y timestamps UTC; el fingerprint
    original de entrenamiento tambien debe coincidir tras restaurar dtypes.
    """
    from decimal import Decimal
    import hashlib
    columns=sorted(frame.columns);rows=[]
    for values in frame[columns].itertuples(index=False,name=None):
        record=[]
        for name,value in zip(columns,values):
            if pd.isna(value):record.append(['null']);continue
            kind=sql_type(frame[name])
            if kind=='NUMBER(16,2)':
                amount=Decimal(str(value));fixed=amount.quantize(Decimal('.01'))
                if fixed!=amount:raise ValueError('Dinero fuera de escala de dos decimales: '+name)
                record.append(['money2',format(fixed,'f')])
            elif kind=='NUMBER(38,0)':
                if Decimal(str(value))!=Decimal(int(value)):raise ValueError('Entero no integral: '+name)
                record.append(['int',str(int(value))])
            elif kind=='TIMESTAMP_TZ':record.append(['utc',pd.Timestamp(value).tz_convert('UTC').isoformat()])
            elif kind=='FLOAT':record.append(['float',float(value).hex()])
            else:record.append(['text',str(value)])
        rows.append(json.dumps(record,separators=(',',':'),ensure_ascii=False))
    payload='pset3-logical-v1\n'+json.dumps(columns)+'\n'+'\n'.join(sorted(rows))
    return hashlib.sha256(payload.encode('utf-8')).hexdigest()


def reconcile_frames(original,actual,*,kind):
    keys=['LOTE_ID','MOVIMIENTO_ID'] if kind=='movimientos' else ['LOTE_ID','OBLIGACION_ID']
    if len(actual)!=len(original) or list(actual.columns)!=list(original.columns):raise ValueError('Filas/columnas distintas.')
    if actual[keys].isna().any().any() or actual.duplicated(keys).any():raise ValueError('IDs nulos/duplicados.')
    if set(map(tuple,actual[keys].itertuples(index=False,name=None)))!=set(map(tuple,original[keys].itertuples(index=False,name=None))):raise ValueError('Claves diferentes.')
    if dataset_fingerprint(actual)!=dataset_fingerprint(original):raise ValueError('Snapshot canonico no concilia tras normalizar tipos.')
    logical=logical_fingerprint(original)
    if logical_fingerprint(actual)!=logical:raise ValueError('Hash logico no concilia.')
    result={'rows':len(actual),'columns':len(actual.columns),'keys_unique_and_equal':True,'training_fingerprint':dataset_fingerprint(original),
            'logical_fingerprint':logical,'logical_fingerprint_version':'pset3-logical-v1','null_counts_equal':original.isna().sum().to_dict()==actual.isna().sum().to_dict()}
    if 'TARGET_NO_PAGADA_90D' in actual:
        ordered_original=original.sort_values(keys).reset_index(drop=True);ordered_actual=actual.sort_values(keys).reset_index(drop=True)
        if not ordered_original.TARGET_NO_PAGADA_90D.equals(ordered_actual.TARGET_NO_PAGADA_90D):raise ValueError('Etiquetas diferentes por obligacion.')
        result['labels_equal_by_id']=True;result['classes']={str(k):int(v) for k,v in actual.TARGET_NO_PAGADA_90D.value_counts(dropna=False).items()}
        result['cohort_classes']={str(year):{str(k):int(v) for k,v in group.TARGET_NO_PAGADA_90D.value_counts().items()} for year,group in actual.groupby('ANIO_FISCAL')}
    return result


def main():
    parser=argparse.ArgumentParser();group=parser.add_mutually_exclusive_group();group.add_argument('--execute',action='store_true');group.add_argument('--verify',action='store_true');args=parser.parse_args()
    load_local_environment();config=SnowflakeConfig.from_env()
    _,manifest=load_population();print(json.dumps({'database':config.database,'schema':config.schema,'tables':publication_plan(config,manifest)},indent=2))
    if not (args.execute or args.verify):return
    connection=connect_snowflake(config,private_key_password=getpass('Passphrase propia: '))
    try:print(json.dumps(verify_existing(config,connection) if args.verify else publish(config,connection),indent=2))
    finally:connection.close()
if __name__=='__main__':main()
