"""Cierre reproducible: tuning recuperable y evaluacion final congelada."""
import hashlib,json,os,time,warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import roc_auc_score,average_precision_score,balanced_accuracy_score,f1_score,accuracy_score,precision_score,recall_score,confusion_matrix,roc_curve
from sklearn.model_selection import ParameterGrid
from threadpoolctl import threadpool_limits
from .analisis import ROOT,DATA,load_population,derive_features
from .configuracion import load_experiment_contract
from .contratos import FeatureSpec
from .pipeline import select_xy,build_pipeline
from .experimentos import seeded_clone
from .modelos import candidates
from .ejecutar_estudio import EXPERIMENT,code_hash
from .tracking import tracked_run,log_results,configure_tracking,export_runs

OUT=ROOT/'docs/cierre';OUT.mkdir(exist_ok=True)
CACHE=DATA/'cierre_cache';CACHE.mkdir(exist_ok=True)
REGISTRY='pset3-predial-urbano-modelo'


def dump(path,value):
    path=Path(path);tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2,default=str)+'\n',encoding='utf-8');tmp.replace(path)


def context():
    frame,manifest=load_population();part=json.loads((DATA/'partitions.json').read_text())
    if part['fingerprint']!=manifest['hashes']['obt']:raise ValueError('Particiones de otra poblacion.')
    contract=load_experiment_contract();spec=FeatureSpec.from_dict(contract)
    dev=derive_features(frame).iloc[part['train_indices']].reset_index(drop=True)
    if dev.OBLIGACION_ID.tolist()!=part['development_keys']:raise ValueError('Orden de desarrollo alterado.')
    X,y=select_xy(dev,spec);y=y.astype(int)
    identity=hashlib.sha256(json.dumps({'dataset':manifest['hashes']['obt'],'parts':part,'features':spec.features,'seed':contract['random_state']},sort_keys=True).encode()).hexdigest()
    contract['code_version']='sha256:'+code_hash()
    meta={'fingerprint':manifest['hashes']['obt'],'row_count':len(frame),'development_rows':len(X),'storage':'LOCAL_PARQUET','snowflake_status':'NOT_PUBLISHED_NO_CONFIGURATION'}
    return frame,manifest,part,contract,spec,X,y,identity,meta


def metric_values(y,score,pred):
    return {'roc_auc':float(roc_auc_score(y,score)),'average_precision':float(average_precision_score(y,score)),
      'balanced_accuracy':float(balanced_accuracy_score(y,pred)),'f1':float(f1_score(y,pred,zero_division=0)),
      'accuracy':float(accuracy_score(y,pred))}


def spaces(name):
    if name=='logistica':return {'model__C':[.0001,.0003,.001,.003,.01,.03,.1,.3,.5,1.,2.,3.,10.,30.,100.], 'model__class_weight':[None,'balanced']}
    if name=='mlp':return {'model__alpha':[.0001,.001,.01,.1,1.], 'model__hidden_layer_sizes':[(16,),(32,16),(64,)],'model__learning_rate_init':[.0005,.001]}
    if name=='random_forest':return {'model__max_depth':[3,5,7,10,None],'model__min_samples_leaf':[10,20,40],'model__max_features':['sqrt',.8]}
    # Espacios equivalentes si el candidato fundacional ejecutado cambia el ranking.
    if name=='svm_rbf':return {'model__C':[.01,.03,.1,.3,1,3,10,30,100,300],'model__gamma':['scale',.01,.1]}
    if name=='hist_boosting':return {'model__learning_rate':[.02,.05,.1,.2,.3],'model__max_leaf_nodes':[7,15,31],'model__l2_regularization':[2.,10.]}
    raise ValueError('Definir un espacio defendible para '+name+' antes de tunear.')


def tune_all():
    import mlflow
    frame,manifest,part,c,spec,X,y,identity,meta=context()
    configure_tracking(experiment_name=EXPERIMENT);os.environ['MLFLOW_EXPERIMENT_NAME']=EXPERIMENT
    baseline=pd.read_csv(ROOT/'docs/resultados_candidatos.csv')
    ranking=baseline.loc[~baseline.candidate.isin(['dummy','regla_academica'])].copy()
    foundation=ROOT/'docs/fundacional_resultado.json'
    if foundation.exists():
        f=json.loads(foundation.read_text())
        if f['status']=='SUCCESS':ranking=pd.concat([ranking,pd.DataFrame([f['result']])],ignore_index=True)
    ranking=ranking.sort_values(['validation_roc_auc_mean','validation_roc_auc_std'],ascending=[False,True])
    selected=ranking.head(3).candidate.tolist()
    if 'tabpfn_v2' in selected:raise ValueError('TabPFN entro en top3: definir tuning de ensamblado/temperatura antes de continuar.')
    plan_path=OUT/'tuning_plan.json'
    plan={'identity':identity,'models':selected,'primary_metric':'roc_auc','n_folds':3,'method':'exhaustive bounded grid (30 distinct configurations/model)',
         'spaces':{name:spaces(name) for name in selected},'n_configurations':90,'seed':c['random_state'],'holdout_used':False}
    if plan_path.exists():
        prior=json.loads(plan_path.read_text())
        if prior['identity']!=identity or prior['models']!=selected:raise ValueError('Plan congelado incompatible.')
    else:dump(plan_path,plan)
    estimators={name:(family,model) for name,family,model in candidates(c['random_state'])}
    client=mlflow.MlflowClient();all_rows=[]
    complete=OUT/'tuning_resumen.json'
    if complete.exists():
        cached=json.loads(complete.read_text());table_path=OUT/'tuning_configuraciones.csv'
        if cached['identity']==identity and cached['completed_configurations']>=90 and table_path.exists():
            table=pd.read_csv(table_path)
            if len(table)==90 and all(client.get_run(r).info.status=='FINISHED' for r in table.run_id):
                print('90 configuraciones terminadas verificadas: recuperadas sin reentrenar.')
                return pd.DataFrame(cached['best'])
    for name in selected:
        family,base=estimators[name]
        configs=list(ParameterGrid(spaces(name)));assert len(configs)>=30
        parent_file=CACHE/f'{name}_parent.json'
        c['run_tags']={'pset3.phase':'tuning','pset3.family':family,'pset3.model':name,'pset3.study':identity,'git.sha':'UNAVAILABLE_PSET3_NOT_VERSIONED','pset3.source':'LOCAL_PARQUET'}
        if parent_file.exists():
            parent_id=json.loads(parent_file.read_text())['run_id'];parent=mlflow.start_run(run_id=parent_id)
        else:
            mlflow.set_experiment(EXPERIMENT);parent=mlflow.start_run(run_name='tuning-'+name,tags={**c['run_tags'],'pset3.role':'tuning_parent','pset3.dataset_fingerprint':meta['fingerprint']})
            dump(parent_file,{'run_id':parent.info.run_id,'identity':identity})
        try:
            mlflow.log_dict(plan,'tuning_plan.json');mlflow.log_dict(meta,'dataset_provenance.json')
            for index,parameters in enumerate(configs):
                key=hashlib.sha256(json.dumps({'study':identity,'model':name,'params':parameters},sort_keys=True).encode()).hexdigest()
                record_file=CACHE/(key+'.json');oof_file=CACHE/(key+'_oof.parquet')
                if record_file.exists() and oof_file.exists():
                    row=json.loads(record_file.read_text())
                    if client.get_run(row['run_id']).info.status=='FINISHED':all_rows.append(row);continue
                with tracked_run(f'{name}-config-{index:02d}',contract=c,dataset_metadata=meta,nested=True,
                     tags={'pset3.role':'tuning_configuration','pset3.config_key':key}) as run:
                    rows=[];oof=[];warning_text=[]
                    for fold_index,fold in enumerate(part['folds']):
                        tr,va=np.array(fold['train']),np.array(fold['validation'])
                        with tracked_run('fold-'+str(fold_index),contract=c,dataset_metadata=meta,nested=True,tags={'pset3.role':'tuning_fold'}) as frun:
                            model=seeded_clone(build_pipeline(base,spec,sparse_output=False).set_params(**parameters),c['random_state']+fold_index)
                            with warnings.catch_warnings(record=True) as caught,threadpool_limits(limits=2):
                                warnings.simplefilter('always');start=time.perf_counter();model.fit(X.iloc[tr],y.iloc[tr]);seconds=time.perf_counter()-start
                                st=model.predict_proba(X.iloc[tr])[:,1];sv=model.predict_proba(X.iloc[va])[:,1]
                            warning_text+=sorted({str(w.message) for w in caught})
                            mt=metric_values(y.iloc[tr],st,st>=.5);mv=metric_values(y.iloc[va],sv,sv>=.5)
                            record={'fold':fold_index,'fit_seconds':seconds,**{'train_'+k:v for k,v in mt.items()},**{'validation_'+k:v for k,v in mv.items()},**{'gap_'+k:mt[k]-mv[k] for k in mt}}
                            log_results(parameters={'seed':c['random_state']+fold_index,'train_rows':len(tr),'validation_rows':len(va)},metrics=record)
                            rows.append(record);oof.extend({'position':int(pos),'fold':fold_index,'y':int(y.iloc[pos]),'score':float(s)} for pos,s in zip(va,sv))
                    df=pd.DataFrame(rows);summary={col+'_'+stat:float(getattr(df[col],stat)(**({'ddof':1} if stat=='std' else {}))) for col in df if col!='fold' for stat in ['mean','std']}
                    log_results(parameters={**parameters,'seed':c['random_state']},metrics=summary)
                    pd.DataFrame(oof).to_parquet(oof_file,index=False)
                    mlflow.log_artifact(str(oof_file),'oof');mlflow.log_dict({'folds':rows,'warnings':sorted(set(warning_text))},'cv_results.json')
                    row={'candidate':name,'family':family,'config_index':index,'config_key':key,'parameters':parameters,'run_id':run.info.run_id,'parent_run_id':parent.info.run_id,'warnings':len(set(warning_text)),**summary}
                    mlflow.log_dict(row,'configuration_result.json')
                dump(record_file,row);all_rows.append(row)
                pd.DataFrame(all_rows).to_csv(OUT/'tuning_configuraciones.csv',index=False)
                print(f'TUNING {name} {index+1}/30 AUC={row["validation_roc_auc_mean"]:.5f}',flush=True)
            mlflow.end_run(status='FINISHED')
        except BaseException:
            mlflow.end_run(status='FAILED');raise
    table=pd.DataFrame(all_rows);table.to_csv(OUT/'tuning_configuraciones.csv',index=False)
    best=table.sort_values(['validation_roc_auc_mean','validation_roc_auc_std'],ascending=[False,True]).groupby('candidate',sort=False).head(1)
    comparisons=[]
    for _,row in best.iterrows():
        original=baseline.loc[baseline.candidate==row.candidate].iloc[0]
        comparisons.append({'candidate':row.candidate,'default_auc':float(original.validation_roc_auc_mean),'default_sd':float(original.validation_roc_auc_std),
          'tuned_auc':row.validation_roc_auc_mean,'tuned_sd':row.validation_roc_auc_std,'delta':row.validation_roc_auc_mean-float(original.validation_roc_auc_mean),
          'improvement_conclusive':False,'run_id':row.run_id,'config_key':row.config_key,'parameters':row.parameters,'gap_roc_auc_mean':row.gap_roc_auc_mean})
    dump(OUT/'tuning_resumen.json',{'identity':identity,'completed_configurations':len(table),'per_model':table.groupby('candidate').size().to_dict(),'best':comparisons,'holdout_used':False})
    export_runs(ROOT/'docs/corridas.csv',experiment_name=EXPERIMENT)
    return pd.DataFrame(comparisons)


from .decision import ThresholdClassifier


def business(y,pred,ratio):
    tn,fp,fn,tp=confusion_matrix(y,pred,labels=[0,1]).ravel()
    return {'cost_per_obligation':float((fp+ratio*fn)/len(y)),'tn':int(tn),'fp':int(fp),'fn':int(fn),'tp':int(tp),
            'precision':float(precision_score(y,pred,zero_division=0)),'recall':float(recall_score(y,pred,zero_division=0)),
            'flagged_fraction':float(np.mean(pred))}


def freeze_selection():
    frame,m,p,c,spec,X,y,identity,meta=context();path=OUT/'selection_frozen.json'
    if path.exists():
        frozen=json.loads(path.read_text())
        if frozen['identity']!=identity:raise ValueError('Evaluacion congelada de otro dataset/split.')
        return frozen
    tuning=json.loads((OUT/'tuning_resumen.json').read_text());assert tuning['completed_configurations']>=90
    best=sorted(tuning['best'],key=lambda r:(-r['tuned_auc'],r['tuned_sd']))[:2]
    ratios=[1,2,3,5,10];roles={}
    for role,row in zip(['champion','challenger'],best):
        oof=pd.read_parquet(CACHE/(row['config_key']+'_oof.parquet'));assert not oof.position.duplicated().any()
        thresholds={}
        for ratio in ratios:
            losses=[(business(oof.y,oof.score>=t,ratio)['cost_per_obligation'],abs(t-.5),-t,t) for t in np.linspace(.05,.95,91)]
            optimum=min(losses);thresholds[str(ratio)]={'threshold':float(optimum[-1]),'development_oof_cost':float(optimum[0])}
        roles[role]={**row,'threshold':thresholds['3']['threshold'],'cost_sensitivity':thresholds,'oof_rows':len(oof)}
    frozen={'identity':identity,'registry':REGISTRY,'selection_metric':'development CV ROC-AUC; sd tie-break only',
        'classification_policy':'OOF-only minimize FP+3*FN; thresholds fixed before holdout','cost_units':'academic relative units, not municipal money',
        'cost_ratios':ratios,'roles':roles,'holdout_evaluated_at_freeze':False,'population_hash':m['hashes']['obt'],
        'inference_timing_policy':'single batch 2000 rows; wall time; no threshold or model selection using holdout'}
    dump(path,frozen);return frozen


def evaluate_final():
    import mlflow,mlflow.sklearn,joblib
    from mlflow.models import infer_signature
    frozen=freeze_selection();result_path=OUT/'holdout_resultados.json'
    if result_path.exists():return json.loads(result_path.read_text())
    frame,m,p,c,spec,X,y,identity,meta=context();configure_tracking(experiment_name=EXPERIMENT)
    estimators={name:(family,model) for name,family,model in candidates(c['random_state'])}
    # Los dos pipelines se entrenan y persisten ANTES de obtener X/y hold-out.
    fit_receipts={}
    for role,row in frozen['roles'].items():
        path=CACHE/(role+'_final.joblib');receipt=CACHE/(role+'_fit.json')
        if path.exists() and receipt.exists():fit_receipts[role]=json.loads(receipt.read_text());continue
        base=estimators[row['candidate']][1];model=seeded_clone(build_pipeline(base,spec,sparse_output=False).set_params(**row['parameters']),c['random_state'])
        with threadpool_limits(limits=2):start=time.perf_counter();model.fit(X,y);seconds=time.perf_counter()-start
        wrapped=ThresholdClassifier(model,row['threshold']);wrapped.classes_=np.array([0,1]);wrapped.n_features_in_=len(spec.features)
        joblib.dump(wrapped,path);fit_receipts[role]={'fit_seconds':seconds};dump(receipt,fit_receipts[role])
    # Acceso final solo despues de seleccionar, congelar umbrales y entrenar ambos.
    hold=derive_features(frame).iloc[p['holdout_indices']].reset_index(drop=True)
    Xh,yh=select_xy(hold,spec);yh=yh.astype(int);assert hold.OBLIGACION_ID.tolist()==p['holdout_keys']
    outcomes={};predictions={}
    for role,row in frozen['roles'].items():
        scored=CACHE/(role+'_holdout.parquet')
        model=joblib.load(CACHE/(role+'_final.joblib'))
        timing_file=CACHE/(role+'_inference.json')
        if scored.exists():scores=pd.read_parquet(scored);timing=json.loads(timing_file.read_text())
        else:
            with threadpool_limits(limits=2):start=time.perf_counter();score=model.predict_proba(Xh)[:,1];elapsed=time.perf_counter()-start
            scores=pd.DataFrame({'OBLIGACION_ID':hold.OBLIGACION_ID,'y':yh,'score':score,'prediction':(score>=row['threshold']).astype(int)})
            scores.to_parquet(scored,index=False);timing={'inference_seconds':elapsed,'batch_rows':len(Xh)};dump(timing_file,timing)
        predictions[role]=scores
        result={**metric_values(scores.y,scores.score,scores.prediction),**business(scores.y,scores.prediction,3),**fit_receipts[role],**timing}
        result['inference_ms_per_row']=result['inference_seconds']/len(Xh)*1000
        sensitivity={ratio:business(scores.y,scores.score>=policy['threshold'],int(ratio))|policy for ratio,policy in row['cost_sensitivity'].items()}
        c['run_tags']={'pset3.phase':'final_selection','pset3.family':estimators[row['candidate']][0],'pset3.role':role,'pset3.source':'LOCAL_PARQUET','git.sha':'UNAVAILABLE_PSET3_NOT_VERSIONED'}
        run_checkpoint=CACHE/(role+'_registered.json')
        if run_checkpoint.exists():registered=json.loads(run_checkpoint.read_text())
        else:
            with tracked_run('final-'+role+'-'+row['candidate'],contract=c,dataset_metadata=meta,experiment_name=EXPERIMENT) as run:
                log_results(parameters={**row['parameters'],'threshold':row['threshold'],'cost_fn_to_fp':3,'seed':c['random_state']},metrics={**{'holdout_'+k:v for k,v in result.items() if isinstance(v,(int,float))},'cv_auc_mean':row['tuned_auc'],'cv_auc_std':row['tuned_sd']})
                mlflow.log_dict(frozen,'selection_frozen.json');mlflow.log_dict(sensitivity,'business_sensitivity.json');mlflow.log_artifact(str(scored),'final_predictions')
                example=X.head(3).copy().where(X.head(3).notna(),None)
                logged=mlflow.sklearn.log_model(model,name='pipeline',input_example=example,signature=infer_signature(example,model.predict(example)),
                    serialization_format='cloudpickle',code_paths=[str(ROOT/'src')],pip_requirements=['scikit-learn==1.7.2','numpy==2.3.5','pandas==3.0.1','scipy==1.16.3','cloudpickle==3.1.2'])
                version=mlflow.register_model(logged.model_uri,REGISTRY)
                registered={'run_id':run.info.run_id,'model_uri':logged.model_uri,'version':str(version.version)}
            dump(run_checkpoint,registered)
        mlflow.MlflowClient().set_registered_model_alias(REGISTRY,role,registered['version'])
        outcomes[role]={'candidate':row['candidate'],'threshold':row['threshold'],'cv_auc_mean':row['tuned_auc'],'cv_auc_sd':row['tuned_sd'],
            'cv_gap_auc':row['gap_roc_auc_mean'],'metrics':result,'sensitivity':sensitivity,**registered}
    rng=np.random.default_rng(c['random_state']);deltas=[]
    for _ in range(1000):
        ix=rng.integers(0,len(yh),len(yh));yy=yh.iloc[ix]
        if yy.nunique()<2:continue
        deltas.append(roc_auc_score(yy,predictions['champion'].score.iloc[ix])-roc_auc_score(yy,predictions['challenger'].score.iloc[ix]))
    report={'identity':identity,'registry':REGISTRY,'roles':outcomes,'population_hash':m['hashes']['obt'],'holdout_rows':len(yh),
       'holdout_protocol_evaluations':1,'selection_based_on_holdout':False,'auc_difference_bootstrap_95':np.quantile(deltas,[.025,.975]).tolist(),
       'auc_difference_bootstrap_note':'Paired holdout bootstrap uncertainty only, no model/threshold reselection; 1000 resamples.',
       'dummy_all_positive_cost':business(yh,np.ones(len(yh),dtype=int),3)['cost_per_obligation'],'snowflake_status':'NOT_PUBLISHED_NO_CONFIGURATION'}
    dump(result_path,report);export_runs(ROOT/'docs/corridas.csv',experiment_name=EXPERIMENT);return report


def reproduce_champion():
    """Sesion limpia: cargar por alias y verificar metricas sin fit u optimizacion."""
    import mlflow,mlflow.sklearn
    configure_tracking(experiment_name=EXPERIMENT)
    report=json.loads((OUT/'holdout_resultados.json').read_text())
    frame,m,p,c,spec,X,y,identity,meta=context();hold=derive_features(frame).iloc[p['holdout_indices']]
    Xh,yh=select_xy(hold,spec)
    loaded=mlflow.sklearn.load_model(f'models:/{REGISTRY}@champion')
    score=loaded.predict_proba(Xh)[:,1];pred=loaded.predict(Xh)
    auc=float(roc_auc_score(yh.astype(int),score));expected=report['roles']['champion']['metrics']['roc_auc']
    assert auc==expected,(auc,expected)
    reference=pd.read_parquet(CACHE/'champion_holdout.parquet')
    np.testing.assert_array_equal(pred,reference.prediction.to_numpy())
    result={'registry_uri':f'models:/{REGISTRY}@champion','auc':auc,'expected_auc':expected,'exact_auc_match':True,'exact_predictions_match':True,'refit':False,'threshold_optimized_again':False}
    dump(OUT/'champion_reproduccion.json',result);return result

if __name__=='__main__':
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument('stage',choices=['tuning','selection','reproduce']);stage=parser.parse_args().stage
    if stage=='tuning':print(tune_all().to_string(index=False))
    elif stage=='selection':print(json.dumps(evaluate_final(),indent=2))
    else:print(json.dumps(reproduce_champion(),indent=2))
