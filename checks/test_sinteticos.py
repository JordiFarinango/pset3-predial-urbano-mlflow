import unittest
import pandas as pd
from src.sinteticos import label_from_events,TARGET

class LabelChecks(unittest.TestCase):
    def test_day90_net_reversals_and_censoring(self):
        emitted=pd.Timestamp('2024-01-01',tz='UTC')
        ids=['full90','late91','partial','none','reversed','immature']
        raw=pd.DataFrame({'LOTE_ID':['fixture']*6,'OBLIGACION_ID':ids,'TOTAL_CENT':[100]*6,
           'FECHA_EMISION_DECLARADA':[emitted]*5+[pd.Timestamp('2025-12-20',tz='UTC')]})
        events=pd.DataFrame({'LOTE_ID':['fixture']*5,'OBLIGACION_ID':['full90','late91','partial','reversed','reversed'],
           'FECHA_EVENTO':[emitted+pd.Timedelta(days=d) for d in [90,91,20,10,30]],'IMPORTE_CENT':[100,100,50,100,-100]})
        result=label_from_events(raw,events,'2025-12-31').set_index('OBLIGACION_ID')
        self.assertEqual(result.loc['full90',TARGET],0)
        for key in ['late91','partial','none','reversed']:self.assertEqual(result.loc[key,TARGET],1)
        self.assertTrue(pd.isna(result.loc['immature',TARGET]))
    def test_features_do_not_use_future(self):
        from src.analisis import derive_features
        frame=pd.DataFrame({'TOTAL_OBLIGACION_DECLARADO':[10.],'AREA_M2':[100.],
           'FECHA_EMISION_DECLARADA':[pd.Timestamp('2020-01-01',tz='UTC')],'SALDO_90_CENT':[1]})
        columns=['LOG_IMPORTE_EMISION','LOG_AREA_M2','MES_EMISION_SIN','MES_EMISION_COS']
        original=derive_features(frame)[columns]
        frame['SALDO_90_CENT']=99999
        pd.testing.assert_frame_equal(original,derive_features(frame)[columns])

class PublicationChecks(unittest.TestCase):
    def test_pset2_writes_are_blocked(self):
        from src.publicar_snowflake import publication_plan
        from checks.test_infraestructura import configuration
        from dataclasses import replace
        manifest={'hashes':{'obt':'a'*64}}
        with self.assertRaises(ValueError):publication_plan(configuration(),manifest)
        own=replace(configuration(),database='PSET3_SINTETICO',allowed_databases=('PSET3_SINTETICO',),schema='EXPERIMENTOS')
        self.assertEqual(publication_plan(own,manifest)['obt'],'OBT_PREDIAL_AAAAAAAAAAAA')
