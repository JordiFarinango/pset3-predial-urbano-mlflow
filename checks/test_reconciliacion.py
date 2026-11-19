import unittest
import pandas as pd
from src.publicar_snowflake import reconcile_frames,logical_fingerprint

class ReconciliationChecks(unittest.TestCase):
    def frame(self):
        return pd.DataFrame({'LOTE_ID':['synthetic']*2,'OBLIGACION_ID':['a','b'],'ANIO_FISCAL':[2024,2024],
          'TOTAL_CENT':[1234,5678],'TOTAL_OBLIGACION_DECLARADO':[12.34,56.78],'TARGET_NO_PAGADA_90D':pd.Series([0,1],dtype='Int64')})
    def test_reordered_rows_reconcile_without_changing_label(self):
        data=self.frame();r=reconcile_frames(data,data.iloc[::-1],kind='obt')
        self.assertTrue(r['labels_equal_by_id']);self.assertEqual(r['classes'],{'1':1,'0':1})
    def test_changed_target_and_duplicates_fail(self):
        data=self.frame();changed=data.copy();changed.loc[0,'TARGET_NO_PAGADA_90D']=1
        with self.assertRaises(ValueError):reconcile_frames(data,changed,kind='obt')
        with self.assertRaises(ValueError):reconcile_frames(data,pd.concat([data.iloc[:1]]*2),kind='obt')
    def test_money_extra_precision_is_rejected(self):
        data=self.frame();data.loc[0,'TOTAL_OBLIGACION_DECLARADO']=12.345
        with self.assertRaises(ValueError):logical_fingerprint(data)

    def test_publication_rejects_other_pset3_databases(self):
        from dataclasses import replace
        from checks.test_infraestructura import configuration
        from src.publicar_snowflake import publication_plan
        cfg=replace(configuration(),database='PSET3_OTRA',allowed_databases=('PSET3_OTRA',),schema='EXPERIMENTOS')
        with self.assertRaises(ValueError):publication_plan(cfg,{'hashes':{'obt':'a'*64}})
