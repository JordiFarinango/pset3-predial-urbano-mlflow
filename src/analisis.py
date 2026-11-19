"""EDA de desarrollo y features de emision, sin acceso al hold-out final."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from .sinteticos import ROOT,TARGET

DATA=ROOT/'data/synthetic_v1'
EDA=ROOT/'docs/eda';EDA.mkdir(exist_ok=True)


def load_population():
    manifest=json.loads((DATA/'manifest.json').read_text(encoding='utf-8'))
    frame=pd.read_parquet(DATA/'obt.parquet')
    from .datos import dataset_fingerprint
    if dataset_fingerprint(frame)!=manifest['hashes']['obt']:
        raise ValueError('Huella de poblacion diferente del manifiesto.')
    return frame,manifest


def development():
    frame,manifest=load_population()
    # Corte fijado por calendario antes de inspeccionar resultados.
    cutoff=pd.Timestamp('2024-01-01',tz='UTC')
    return frame.loc[(frame.FECHA_EMISION_DECLARADA<cutoff)&(frame.LABEL_AVAILABLE_AT<cutoff)].copy(),manifest


def run_eda():
    dev,manifest=development()
    numeric=['TOTAL_OBLIGACION_DECLARADO','AVALUO_SIMULADO','AREA_M2','PLAZO_DIAS_SIMULADO']
    description=dev[numeric].describe().T
    description['skew']=dev[numeric].skew()
    counts=dev[TARGET].value_counts().sort_index()
    profile=pd.DataFrame({'dtype':dev.dtypes.astype(str),'missing':dev.isna().sum(),'completeness':dev.notna().mean()})
    checks=[]
    for name in numeric:
        values=dev[name];q1,q3=values.quantile([.25,.75]);iqr=q3-q1
        z=(values-values.mean())/values.std(ddof=0)
        checks.append({'feature':name,'iqr_lower':float(q1-1.5*iqr),'iqr_upper':float(q3+1.5*iqr),
                       'iqr_outliers':int(((values<q1-1.5*iqr)|(values>q3+1.5*iqr)).sum()),
                       'abs_z_gt_3':int((z.abs()>3).sum())})
    outliers=pd.DataFrame(checks)
    correlation=dev[numeric].corr(method='spearman')
    associations={k:float(dev[k].corr(dev[TARGET].astype(float),method='spearman')) for k in numeric}
    findings=[
      {'evidence':f"Asimetria del importe: {description.loc['TOTAL_OBLIGACION_DECLARADO','skew']:.3f}; atipicos IQR: {int(outliers.iloc[0].iqr_outliers)}.",
       'consequence':'Evaluar log1p(importe) como feature de emision; conservar cola sin eliminar extremos por target.'},
      {'evidence':f"Area ausente en {int(dev.AREA_M2.isna().sum())} filas de desarrollo; canal ausente en {int(dev.CANAL_NOTIFICACION.isna().sum())}.",
       'consequence':'Imputar exclusivamente dentro de cada fold y admitir categoria faltante; no imputar globalmente.'},
      {'evidence':f"Fraccion target=1: {dev[TARGET].mean():.4f}; correlacion Spearman avaluo/importe: {correlation.loc['AVALUO_SIMULADO','TOTAL_OBLIGACION_DECLARADO']:.4f}.",
       'consequence':'Usar ROC-AUC primaria y AP/F1/balanced accuracy secundarias; excluir avaluo redundante de esta primera comparacion.'}
    ]
    for name,df in [('descriptivos',description),('completitud',profile),('outliers',outliers),('spearman',correlation)]:
        df.to_csv(EDA/f'{name}.csv')
    fig,axs=plt.subplots(1,2,figsize=(11,4));counts.plot.bar(ax=axs[0],color=['#2563eb','#f97316']);axs[0].set_title('Clases - solo desarrollo');axs[0].set_xlabel('Target no pagada 90d')
    axs[1].hist(dev.TOTAL_OBLIGACION_DECLARADO,bins=60,color='#2563eb');axs[1].set_title('Importe de emision');fig.tight_layout();fig.savefig(EDA/'distribuciones.png',dpi=140);plt.close(fig)
    fig,axs=plt.subplots(1,2,figsize=(11,4));axs[0].boxplot(dev.TOTAL_OBLIGACION_DECLARADO.dropna());axs[0].set_title('Boxplot importe (IQR)')
    axs[1].hist(((dev.TOTAL_OBLIGACION_DECLARADO-dev.TOTAL_OBLIGACION_DECLARADO.mean())/dev.TOTAL_OBLIGACION_DECLARADO.std(ddof=0)),bins=60);axs[1].axvline(3,color='red');axs[1].set_title('Z-score (diagnostico, sin filtrar)');fig.tight_layout();fig.savefig(EDA/'outliers.png',dpi=140);plt.close(fig)
    fig,ax=plt.subplots(figsize=(7,5));im=ax.imshow(correlation,vmin=-1,vmax=1,cmap='RdBu_r');ax.set_xticks(range(len(numeric)),numeric,rotation=60,ha='right');ax.set_yticks(range(len(numeric)),numeric);fig.colorbar(im,ax=ax);fig.tight_layout();fig.savefig(EDA/'correlaciones.png',dpi=140);plt.close(fig)
    report={'development_rows':len(dev),'target_counts':{str(k):int(v) for k,v in counts.items()},'positive_rate':float(dev[TARGET].mean()),
            'spearman_feature_target':associations,'findings':findings,'holdout_used':False}
    (EDA/'hallazgos.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    return report


def derive_features(frame):
    # Transformaciones deterministas por fila: no ajustan estadisticos globales.
    out=frame.copy()
    out['LOG_IMPORTE_EMISION']=np.log1p(out.TOTAL_OBLIGACION_DECLARADO)
    month=out.FECHA_EMISION_DECLARADA.dt.month
    out['MES_EMISION_SIN']=np.sin(2*np.pi*month/12)
    out['MES_EMISION_COS']=np.cos(2*np.pi*month/12)
    out['LOG_AREA_M2']=np.log1p(out.AREA_M2)
    return out


def prepare_features():
    if not (EDA/'hallazgos.json').exists(): raise ValueError('Ejecutar EDA de desarrollo primero.')
    frame,manifest=load_population();frame=derive_features(frame)
    from .experimentos import holdout_indices,cv_indices
    train,test=holdout_indices(frame[TARGET],strategy='temporal',dates=frame.FECHA_EMISION_DECLARADA,
           label_available=frame.LABEL_AVAILABLE_AT,cutoff='2024-01-01')
    # No etiquetas de Q4-2023 inmaduras a 1-ene-2024 en entrenamiento final.
    subset=frame.iloc[train]
    windows=[(f'{y}-01-01',f'{y+1}-01-01') for y in [2021,2022,2023]]
    folds=list(cv_indices(subset[TARGET],strategy='temporal',dates=subset.FECHA_EMISION_DECLARADA,
             label_available=subset.LABEL_AVAILABLE_AT,temporal_windows=windows))
    numeric=['LOG_IMPORTE_EMISION','LOG_AREA_M2','PLAZO_DIAS_SIMULADO','MES_EMISION_SIN','MES_EMISION_COS']
    categorical=['ZONA_SIMULADA','USO_PREDIO','CANAL_NOTIFICACION']
    excluded=[x for x in frame.columns if x not in numeric+categorical+[TARGET]]
    config={'approved':True,'approval_scope':'User-authorized first synthetic experimental version; academic assumptions',
        'dataset_version':manifest['config']['version'],'code_version':'sha256:computed-at-run',
        'target':TARGET,'target_rule_version':'net-cents-day90-inclusive-v1',
        'lote_ids':manifest['lote_ids'],'numeric_features':numeric,'categorical_features':categorical,
        'approved_at_prediction':numeric+categorical,'excluded_columns':excluded,
        'split':{'strategy':'temporal','cutoff':'2024-01-01','windows':windows,'label_maturation_required':True},
        'primary_metric':'roc_auc','business_metric':'none_academic_simulation','random_state':20261117}
    (ROOT/'config/experiment.json').write_text(json.dumps(config,indent=2)+'\n',encoding='utf-8')
    # Folds posicionales se refieren EXACTAMENTE a development_keys en este archivo.
    payload={'train_indices':train.tolist(),'holdout_indices':test.tolist(),
         'development_keys':subset.OBLIGACION_ID.tolist(),'holdout_keys':frame.iloc[test].OBLIGACION_ID.tolist(),
         'folds':[{'train':a.tolist(),'validation':b.tolist()} for a,b in folds],
         'fingerprint':manifest['hashes']['obt'],'seed':config['random_state']}
    (DATA/'partitions.json').write_text(json.dumps(payload)+'\n',encoding='utf-8')
    summary={'features':numeric+categorical,'train_rows':len(train),'holdout_rows':len(test),
        'embargo_rows':len(frame)-len(train)-len(test),'folds':[{'train':len(a),'validation':len(b)} for a,b in folds],
        'holdout_evaluated':False,'cutoff':'2024-01-01','final_observation_end':manifest['config']['observation_end']}
    (ROOT/'docs/features_particiones.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8')
    return summary
