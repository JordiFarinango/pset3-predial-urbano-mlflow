"""Control previo a publicar; no conexiones, entrenamientos ni cambios en Git."""
import argparse,hashlib,json,re,subprocess
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
AUTHORIZED_PDF='docs/PSet3_memo_Farinango_Andrango.pdf'
RETIRED_PATHS={'docs/mlflow_registry.jpg', 'docs/mlflow_comparacion.jpg', 'docs/PSet3_memo_estilo_PSet2_Farinango_Andrango.pdf', 'docs/mlflow_comparacion_procedencia.json'}
def digest(data):return hashlib.sha256(data).hexdigest()
def validate_academic_pdf():
    from pypdf import PdfReader
    path=ROOT/AUTHORIZED_PDF
    reader=PdfReader(path,strict=True)
    if reader.is_encrypted or len(reader.pages)!=6:raise ValueError('El PDF definitivo debe ser legible, no cifrado y tener seis paginas.')
    text=[]
    for page in reader.pages:
        page.get_contents()
        text.append(page.extract_text() or '')
    # The authorized binary is also checked against its exact manifest SHA.
    patterns=[r'-----BEGIN (?:[A-Z ]* )?PRIVATE KEY-----',r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',r'\bsk-[A-Za-z0-9_-]{24,}',r'\bAKIA[A-Z0-9]{16}\b',r'https?://[^/@\s]+:[^/@\s]+@']
    if any(re.search(pattern,'\n'.join(text)) for pattern in patterns):raise ValueError('Posible secreto en PDF; revisar sin imprimir contenido.')
    return {'file':AUTHORIZED_PDF,'pages':len(reader.pages),'sha256':digest(path.read_bytes()),'structural_validation':'PASS','extracted_text_secret_scan':'PASS'}


def audit(staged=False):
    m=json.loads((ROOT/'MANIFEST_SHA256.json').read_text(encoding='utf-8'))
    old={'files':m['preserved_v2_sha256']}
    if RETIRED_PATHS.intersection(m['files']):raise ValueError('Archivo retirado de la publicacion.')
    pdf_paths={name for name in m['files'] if Path(name).suffix.lower()=='.pdf'}
    if pdf_paths!={AUTHORIZED_PDF}:raise ValueError('Solo se permite el PDF academico expresamente autorizado.')
    pdf_info=validate_academic_pdf()
    required={'docs/corridas.csv','docs/cierre/tuning_configuraciones.csv','docs/cierre/selection_frozen.json','docs/cierre/holdout_resultados.json','models/manifest.json','models/champion.joblib','models/challenger.joblib','data/synthetic_v1/manifest.json','data/synthetic_v1/partitions.json','data/synthetic_v1/obligaciones.parquet','data/synthetic_v1/movimientos.parquet','data/synthetic_v1/obt.parquet'}
    required.update({'notebooks/01_eda.ipynb','notebooks/02_features.ipynb','notebooks/03_baseline_candidatos.ipynb','notebooks/04_tuning.ipynb','notebooks/05_seleccion.ipynb'})
    if not required.issubset(m['files']):raise ValueError('Falta evidencia fundamental.')
    allowed=set(m['files'])|{'MANIFEST_SHA256.json'}
    actual={p.relative_to(ROOT).as_posix() for p in ROOT.rglob('*') if p.is_file() and '.git' not in p.relative_to(ROOT).parts and '.venv' not in p.relative_to(ROOT).parts and '__pycache__' not in p.relative_to(ROOT).parts and '._revision_privada' not in p.relative_to(ROOT).parts}
    # Ignored local runtime files may exist later, but cannot enter this publishing snapshot/index.
    for rel,expected in m['files'].items():
        p=ROOT/rel
        if p.is_symlink() or not p.resolve().is_relative_to(ROOT.resolve()):raise ValueError('Enlace/ruta externa: '+rel)
        if digest(p.read_bytes())!=expected:raise ValueError('SHA distinto: '+rel)
        if Path(rel).parts[0] in {'entrega','tmp'}:raise ValueError('Directorio privado de ejecucion: '+rel)
        name=p.name.lower();parts={x.lower() for x in p.relative_to(ROOT).parts}
        if (name.startswith('.env') and rel!='.env.example') or p.suffix.lower() in {'.key','.pem','.p8','.pfx','.p12','.zip'} or parts & {'.git','.venv','._revision_privada','tmp','cierre_cache','hf_cache','mlruns','.aws','.snowflake','.snowsql'}:raise ValueError('Archivo no publicable: '+rel)
        if p.suffix.lower() in {'.py','.ps1','.json','.jsonl','.md','.txt','.csv','.yml','.yaml','.ipynb'} or rel=='.env.example':
            s=p.read_text(encoding='utf-8-sig')
            patterns=[r'-----BEGIN (?:[A-Z ]* )?PRIVATE KEY-----',r'\b(?:ghp_|github_pat_)[A-Za-z0-9_]{20,}',r'\bsk-[A-Za-z0-9_-]{24,}',r'\bAKIA[A-Z0-9]{16}\b',r'https?://[^/@\s]+:[^/@\s]+@']
            if any(re.search(pattern,s) for pattern in patterns):raise ValueError('Posible secreto; revisar sin imprimir contenido: '+rel)
    for line in (ROOT/'.env.example').read_text(encoding='utf-8').splitlines():
        if not line or line.startswith('#') or '=' not in line:continue
        k,v=line.split('=',1)
        if k in {'SNOWFLAKE_ACCOUNT','SNOWFLAKE_USER','SNOWFLAKE_ROLE','SNOWFLAKE_WAREHOUSE','SNOWFLAKE_PRIVATE_KEY_PATH'} and v.strip():raise ValueError('Configuracion personal en ejemplo.')
    # All original data, models, experiments, notebooks and contracts must remain exact v2 bytes.
    immutable=list(old['files'])
    for rel in immutable:
        if digest((ROOT/rel).read_bytes())!=old['files'][rel]:raise ValueError('Evidencia original modificada: '+rel)
    import pandas as pd
    from src.datos import dataset_fingerprint
    frames={name:pd.read_parquet(ROOT/f'data/synthetic_v1/{name}.parquet') for name in ['obligaciones','movimientos','obt']}
    generation=json.loads((ROOT/'data/synthetic_v1/manifest.json').read_text(encoding='utf-8'))
    for name,frame in frames.items():
        if dataset_fingerprint(frame)!=generation['hashes'][name]:raise ValueError('Huella de datos distinta: '+name)
    for name in ['obligaciones','obt']:
        if set(frames[name].DATA_CLASSIFICATION)!={'SYNTHETIC'} or set(frames[name].GENERATOR_VERSION)!={'mc-eventos-v1'}:raise ValueError('Clasificacion/version distinta.')
    keys=set(zip(frames['obligaciones'].LOTE_ID,frames['obligaciones'].OBLIGACION_ID))
    if not set(zip(frames['movimientos'].LOTE_ID,frames['movimientos'].OBLIGACION_ID)).issubset(keys):raise ValueError('Eventos ajenos a obligaciones sinteticas.')
    if len(frames['obligaciones'])!=12000 or len(frames['movimientos'])!=18988 or len(frames['obt'])!=12000:raise ValueError('Conteos distintos.')
    from checks.verificar_snowflake import verify
    snowflake_evidence=verify(frames)
    from checks.verificar_defaults_snowflake import verify as verify_defaults
    defaults_evidence=verify_defaults(ROOT)
    staged_count=None
    if staged:
        def git(*args):return subprocess.check_output(['git',*args],cwd=ROOT)
        paths=set(git('ls-files','-z').decode('utf-8').strip('\0').split('\0'))
        if paths!=allowed:raise ValueError('Index fuera del manifiesto/archivos faltantes.')
        for rel in paths:
            if digest(git('show',':'+rel))!=digest((ROOT/rel).read_bytes()):raise ValueError('Index distinto de bytes auditados: '+rel)
        staged_count=len(paths)
    extras=actual-allowed
    return {'status':'PASS','files_verified':len(m['files']),'original_v2_immutable_files':len(immutable),'synthetic_obligations':12000,'synthetic_events':18988,'synthetic_obt_rows':12000,'seed':generation['config']['seed'],'population_hash':generation['hashes']['obt'],'secret_marker_scan':'PASS','extra_local_files_outside_manifest':len(extras),'staged_files_verified':staged_count,'academic_pdf':pdf_info,'snowflake_evidence':snowflake_evidence,'new_defaults_evidence':defaults_evidence,'no_database_connections':True,'no_training':True,'note':'Procedencia sintetica sustentada en generador, contratos, clasificacion, relaciones y hashes de v2; escaneo de secretos no garantiza ausencia de toda clase de secreto.'}
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--staged',action='store_true');a=p.parse_args();print(json.dumps(audit(a.staged),indent=2))
