"""Simulacion Monte Carlo de eventos, completamente ficticia y sin conexiones."""
import hashlib
import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL
import numpy as np
import pandas as pd
from scipy.special import expit
from .datos import dataset_fingerprint

ROOT = Path(__file__).resolve().parents[1]
TARGET = 'TARGET_NO_PAGADA_90D'


def configuration():
    return json.loads((ROOT/'config/generacion_v1.json').read_text(encoding='utf-8'))


def label_from_events(obligations, events, observation_end, horizon_days=90):
    """UTC dias calendario: saldo al cierre inclusivo del dia emision+90.

    Reversiones firmadas reducen pago neto. Eventos administrativos no cancelan
    deuda en esta convencion academica. Sin horizonte maduro => etiqueta NA.
    """
    out=obligations.copy()
    cutoff=out.FECHA_EMISION_DECLARADA+pd.Timedelta(days=horizon_days)
    selected=events.merge(out[['OBLIGACION_ID','LOTE_ID','FECHA_EMISION_DECLARADA']],on=['OBLIGACION_ID','LOTE_ID'],validate='many_to_one')
    eligible=(selected.FECHA_EVENTO<=selected.FECHA_EMISION_DECLARADA+pd.Timedelta(days=horizon_days)) & (selected.FECHA_EVENTO<=pd.Timestamp(observation_end,tz='UTC'))
    net=selected.loc[eligible].groupby(['LOTE_ID','OBLIGACION_ID']).IMPORTE_CENT.sum()
    out['PAGADO_90_CENT']=[int(net.get((l,o),0)) for l,o in zip(out.LOTE_ID,out.OBLIGACION_ID)]
    out['SALDO_90_CENT']=out.TOTAL_CENT-out.PAGADO_90_CENT
    out['LABEL_AVAILABLE_AT']=cutoff+pd.Timedelta(days=1) # disponible al dia siguiente del cierre
    mature=out.LABEL_AVAILABLE_AT<=pd.Timestamp(observation_end,tz='UTC')
    out[TARGET]=pd.array(np.where(mature,(out.SALDO_90_CENT>0).astype(int),np.nan),dtype='Int64')
    count=events.groupby(['LOTE_ID','OBLIGACION_ID']).size()
    out['N_MOVIMIENTOS']=[int(count.get((l,o),0)) for l,o in zip(out.LOTE_ID,out.OBLIGACION_ID)]
    return out


def generate(config=None):
    c=config or configuration(); a=c['assumptions']
    if not 100<=c['per_year']<=10000 or len(c['years'])>10 or c['horizon_days']!=90:
        raise ValueError('Configuracion fuera del presupuesto/contrato v1.')
    seeds=np.random.SeedSequence(c['seed']).spawn(len(c['years']))
    cfg_hash=hashlib.sha256(json.dumps(c,sort_keys=True).encode()).hexdigest()
    lote=str(uuid5(NAMESPACE_URL,'pset3:'+cfg_hash))
    obligations=[]; events=[]; event_number=0
    for cohort,(year,seed) in enumerate(zip(c['years'],seeds)):
        feature_seed,event_seed=np.random.SeedSequence(seed.entropy,spawn_key=seed.spawn_key).spawn(2)
        rng=np.random.default_rng(feature_seed); ev=np.random.default_rng(event_seed)
        for i in range(c['per_year']):
            oid=hashlib.sha256(f"{lote}:{year}:{i}".encode()).hexdigest()
            issued=pd.Timestamp(f'{year}-01-01',tz='UTC')+pd.Timedelta(days=int(rng.integers(0,365)))
            zone=int(rng.choice(3,p=[.40,.38,.22])); use=str(rng.choice(['RESIDENCIAL','COMERCIAL','MIXTO'],p=[.70,.20,.10]))
            area=float(np.clip(rng.lognormal(a['area_log_mean']+.12*(use=='COMERCIAL'),a['area_log_sd']),20,4500))
            value=area*a['value_per_m2'][zone]*rng.lognormal(0,a['value_log_sd'])*(1+.035*cohort)
            principal=max(1,int(round(value*a['academic_rate']*100)))
            total=principal+int(round(a['fixed_charge']*100))
            channel=str(rng.choice(['DIGITAL','PAPEL','PRESENCIAL'],p=[.30+.035*cohort,.45-.025*cohort,.25-.01*cohort]))
            period=int(rng.choice([30,45,60],p=[.5,.3,.2]))
            # Heterogeneidad oculta y efectos moderados: ningun umbral genera el target.
            latent=ev.normal(0,a['frailty_sd'])
            eta=(.30*(channel=='DIGITAL')-.22*(use=='COMERCIAL')-.24*np.log1p(total/20000)
                 +.15*zone+.35*np.sin(issued.month/12*2*np.pi)+.15*np.sin(cohort)+latent)
            obligations.append({'LOTE_ID':lote,'OBLIGACION_ID':oid,'ANIO_FISCAL':year,
                'FECHA_EMISION_DECLARADA':issued,'TOTAL_CENT':total,'IMPORTE_PRINCIPAL':principal/100,
                'TOTAL_OBLIGACION_DECLARADO':total/100,'AVALUO_SIMULADO':round(value,2),
                'AREA_M2':np.nan if rng.random()<a['area_missing_probability'] else round(area,2),
                'ZONA_SIMULADA':f'Z{zone+1}','USO_PREDIO':use,
                'CANAL_NOTIFICACION':None if rng.random()<a['channel_missing_probability'] else channel,
                'PLAZO_DIAS_SIMULADO':period,'DATA_CLASSIFICATION':'SYNTHETIC','GENERATOR_VERSION':c['version']})
            def add(day,amount,kind):
                nonlocal event_number
                date=issued+pd.Timedelta(days=int(day))
                if date>pd.Timestamp(c['observation_end'],tz='UTC'): return
                event_number+=1
                events.append({'LOTE_ID':lote,'OBLIGACION_ID':oid,
                    'MOVIMIENTO_ID':hashlib.sha256(f'{lote}:event:{event_number}'.encode()).hexdigest(),
                    'FECHA_EVENTO':date,'IMPORTE_CENT':int(amount),'TIPO_EVENTO':kind})
            balance=total; day=0
            # Masa sin oportunidades y proceso de renovacion Weibull; no se sortea etiqueta.
            no_contact=ev.random()<expit(-1.25-.45*eta)
            if not no_contact:
                for _ in range(a['max_opportunities']):
                    day+=max(1,int(np.ceil(ev.weibull(a['weibull_shape'])*a['arrival_scale_days']*np.exp(-.40*eta))))
                    if day>c['trajectory_days']: break
                    full=ev.random()<expit(np.log(a['full_payment_probability']/(1-a['full_payment_probability']))+.35*eta+.002*day)
                    payment=balance if full else max(1,int(round(balance*ev.beta(2,3))))
                    payment=min(payment,balance); add(day,payment,'PAGO'); balance-=payment
                    if ev.random()<a['reversal_probability']:
                        day+=int(ev.integers(1,8))
                        if day<=c['trajectory_days']:
                            add(day,-payment,'REVERSO'); balance+=payment
                    if balance==0: break
            if ev.random()<a['administrative_event_probability']:
                add(int(ev.integers(1,c['trajectory_days']+1)),0,'ADMINISTRATIVO')
    raw=pd.DataFrame(obligations)
    movements=pd.DataFrame(events).sort_values(['LOTE_ID','OBLIGACION_ID','FECHA_EVENTO','MOVIMIENTO_ID']).reset_index(drop=True)
    obt=label_from_events(raw,movements,c['observation_end'],c['horizon_days'])
    return raw,movements,obt


def validate(raw,events,obt,config):
    assert len(raw)==len(config['years'])*config['per_year']==len(obt)
    assert not raw.duplicated(['LOTE_ID','OBLIGACION_ID']).any()
    assert not events.MOVIMIENTO_ID.duplicated().any()
    assert not raw[['LOTE_ID','OBLIGACION_ID','FECHA_EMISION_DECLARADA','TOTAL_CENT']].isna().any().any()
    joined=events.merge(raw[['LOTE_ID','OBLIGACION_ID','FECHA_EMISION_DECLARADA','TOTAL_CENT']],on=['LOTE_ID','OBLIGACION_ID'],how='left',validate='many_to_one')
    assert not joined.TOTAL_CENT.isna().any()
    assert (joined.FECHA_EVENTO>joined.FECHA_EMISION_DECLARADA).all()
    assert (joined.FECHA_EVENTO<=pd.Timestamp(config['observation_end'],tz='UTC')).all()
    assert (events.loc[events.TIPO_EVENTO=='PAGO','IMPORTE_CENT']>0).all()
    assert (events.loc[events.TIPO_EVENTO=='REVERSO','IMPORTE_CENT']<0).all()
    # mismo dia pago/reverso: ordena movimientos positivos antes de negativos.
    joined=joined.sort_values(['OBLIGACION_ID','FECHA_EVENTO','IMPORTE_CENT'],ascending=[True,True,False])
    cumulative=joined.groupby('OBLIGACION_ID').IMPORTE_CENT.cumsum()
    assert ((cumulative>=0)&(cumulative<=joined.TOTAL_CENT)).all()
    assert (obt.SALDO_90_CENT>=0).all()
    assert set(obt[TARGET].dropna().unique())=={0,1}
    cohort=obt.groupby('ANIO_FISCAL')[TARGET].agg(['size','mean'])
    assert cohort['mean'].between(.10,.90).all(), 'Simulacion degenerada; revisar parametros antes de modelar.'
    assert not obt[TARGET].isna().any(), 'Poblacion principal requiere horizonte maduro.'
    report={'obligations':len(raw),'events':len(events),'obt_rows':len(obt),
        'class_counts':{str(k):int(v) for k,v in obt[TARGET].value_counts().items()},
        'cohorts':json.loads(cohort.to_json(orient='index')),
        'missing':{k:int(v) for k,v in raw.isna().sum().items()},
        'event_types':{k:int(v) for k,v in events.TIPO_EVENTO.value_counts().items()},
        'obligations_without_events':int((obt.N_MOVIMIENTOS==0).sum()),
        'target_from_events':True,'integrity_passed':True}
    return report


def save_population():
    config=configuration();raw,events,obt=generate(config);quality=validate(raw,events,obt,config)
    # Segunda generacion completa: reproducibilidad verificada, no solo semilla declarada.
    again=generate(config)
    for a,b in zip((raw,events,obt),again):
        assert dataset_fingerprint(a)==dataset_fingerprint(b)
    destination=ROOT/'data'/'synthetic_v1';destination.mkdir(parents=True,exist_ok=True)
    hashes={}
    for name,frame in [('obligaciones',raw),('movimientos',events),('obt',obt)]:
        frame.to_parquet(destination/f'{name}.parquet',index=False)
        restored=pd.read_parquet(destination/f'{name}.parquet')
        assert dataset_fingerprint(frame)==dataset_fingerprint(restored)
        hashes[name]=dataset_fingerprint(frame)
    manifest={'config':config,'lote_ids':sorted(raw.LOTE_ID.unique()),'hashes':hashes,
        'fingerprint_version':'pset3-fingerprint-v1','reproducibility_verified':True,
        'storage':'LOCAL_PARQUET','snowflake_status':'NOT_PUBLISHED_NO_CONFIGURATION',
        'quality':quality}
    (destination/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    (ROOT/'docs/generacion_validacion.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding='utf-8')
    return manifest
