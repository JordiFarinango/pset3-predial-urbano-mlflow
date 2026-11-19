"""Comprobaciones locales; sin red, secretos, datasets o ajuste de modelos.

Los pocos valores en memoria son fixtures unitarios, no datos de experimentos.
"""
from dataclasses import replace
from decimal import Decimal
import ast
import json
import importlib.util
from pathlib import Path
import unittest

import numpy as np
import pandas as pd

from src.configuracion import PROJECT_ROOT, load_experiment_contract
from src.datos import (SnowflakeConfig, dataset_fingerprint, identifier, read_obt,
                       validate_schema)
from src.contratos import FeatureSpec
HAS_SKLEARN = importlib.util.find_spec('sklearn') is not None
if HAS_SKLEARN:
    from src.pipeline import FeatureGuard, build_pipeline, build_preprocessor

LOTE = '11111111-2222-4333-8444-555555555555'
OTHER = '99999999-2222-4333-8444-555555555555'


def configuration(database='PSET3_UNIT_FIXTURE'):
    # Ruta deliberadamente inexistente y externa. Nunca se abre una conexion.
    external = Path(PROJECT_ROOT.anchor) / 'pset3-unit-fixture' / 'not-a-key.p8'
    return SnowflakeConfig('fixture-account', 'fixture-user', 'FIXTURE_ROLE', 'FIXTURE_WH',
                           database, 'OBT', 'FIXTURE_TABLE', external,
                           (database,), 'SYNTHETIC')


SCHEMA = {'columns': {
    'LOTE_ID': {'data_type':'TEXT', 'character_maximum_length':36},
    'OBLIGACION_ID': {'data_type':'TEXT', 'character_maximum_length':64},
    'AMOUNT': {'data_type':'NUMBER', 'numeric_precision':16, 'numeric_scale':2},
}}


class CursorDouble:
    def __init__(self, connection):
        self.connection = connection
        self.sfqid = 'unit-fixture-no-snowflake'
        self.description = []
        self.rows = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return False

    def execute(self, query, parameters):
        self.connection.queries.append((query, parameters))
        if 'INFORMATION_SCHEMA.COLUMNS' in query:
            fields = ('COLUMN_NAME','DATA_TYPE','NUMERIC_PRECISION','NUMERIC_SCALE',
                      'CHARACTER_MAXIMUM_LENGTH','DATETIME_PRECISION','IS_NULLABLE')
            self.description = [(field,) for field in fields]
            self.rows = [
                ('LOTE_ID','TEXT',None,None,36,None,'NO'),
                ('OBLIGACION_ID','TEXT',None,None,64,None,'NO'),
                ('AMOUNT','NUMBER',16,2,None,None,'YES'),
            ]
        else:
            self.rows = list(self.connection.records)

    def fetchall(self):
        return list(self.rows)

    def fetchmany(self, count):
        result, self.rows = self.rows[:count], self.rows[count:]
        return result


class ConnectionDouble:
    def __init__(self, records):
        self.records = records
        self.queries = []

    def cursor(self):
        return CursorDouble(self)


class FingerprintChecks(unittest.TestCase):
    def test_order_does_not_change_fingerprint(self):
        frame = pd.DataFrame({'amount':[Decimal('1.00'),Decimal('2.30')], 'code':['x',None]})
        self.assertEqual(dataset_fingerprint(frame), dataset_fingerprint(frame.iloc[::-1][['code','amount']]))

    def test_value_type_and_multiplicity_are_detected(self):
        frame = pd.DataFrame({'a':[1,2]})
        original = dataset_fingerprint(frame)
        self.assertNotEqual(original, dataset_fingerprint(pd.DataFrame({'a':[1,3]})))
        self.assertNotEqual(original, dataset_fingerprint(frame.astype(str)))
        self.assertNotEqual(original, dataset_fingerprint(pd.concat([frame,frame.iloc[:1]])))

    def test_decimal_scale_and_invalid_values(self):
        self.assertEqual(dataset_fingerprint(pd.DataFrame({'a':[Decimal('1.00')]})),
                         dataset_fingerprint(pd.DataFrame({'a':[Decimal('1.0')]})))
        with self.assertRaises(ValueError):
            dataset_fingerprint(pd.DataFrame({'a':[float('inf')]}))


class DataGuardChecks(unittest.TestCase):
    def test_municipal_and_unacknowledged_destinations_are_rejected(self):
        with self.assertRaises(ValueError):
            configuration('PSET2_CAYAMBE').validate()
        with self.assertRaises(ValueError):
            replace(configuration(), classification='').validate()
        with self.assertRaises(ValueError):
            replace(configuration(), allowed_databases=()).validate()
        with self.assertRaises(ValueError):
            replace(configuration(), private_key_path=PROJECT_ROOT/'not-a-key.p8').validate()

    def test_identifiers_cannot_contain_sql(self):
        self.assertEqual(identifier(' obt '), 'OBT')
        with self.assertRaises(ValueError):
            identifier('OBT; DROP TABLE X')

    def test_schema_rejects_wrong_precision_and_extra_columns(self):
        row = {'column_name':'AMOUNT','data_type':'NUMBER','numeric_precision':16,'numeric_scale':2}
        expected = {'columns':{'AMOUNT':SCHEMA['columns']['AMOUNT']}}
        validate_schema([row], expected)
        with self.assertRaises(ValueError):
            validate_schema([{**row, 'numeric_scale':3}], expected)
        with self.assertRaises(ValueError):
            validate_schema([row, {**row,'column_name':'OTHER'}], expected)

    def test_read_uses_explicit_projection_and_parameters(self):
        connection = ConnectionDouble([(LOTE,'a'*64,Decimal('1.25'))])
        snapshot = read_obt(connection, configuration(), lote_ids=(LOTE,), expected_schema=SCHEMA)
        self.assertEqual(snapshot.metadata['row_count'], 1)
        self.assertEqual(snapshot.frame.iloc[0]['AMOUNT'], Decimal('1.25'))
        query, params = connection.queries[-1]
        self.assertNotIn('SELECT *', query)
        self.assertNotIn(LOTE, query)
        self.assertEqual(params, (LOTE,100001))

    def test_read_rejects_truncation_duplicates_and_missing_lotes(self):
        records = [(LOTE,'a'*64,Decimal('1')), (LOTE,'b'*64,Decimal('2'))]
        with self.assertRaises(ValueError):
            read_obt(ConnectionDouble(records), configuration(), lote_ids=(LOTE,), expected_schema=SCHEMA, max_rows=1)
        with self.assertRaises(ValueError):
            read_obt(ConnectionDouble(records[:1]*2), configuration(), lote_ids=(LOTE,), expected_schema=SCHEMA)
        with self.assertRaises(ValueError):
            read_obt(ConnectionDouble(records), configuration(), lote_ids=(LOTE,OTHER), expected_schema=SCHEMA)


@unittest.skipUnless(HAS_SKLEARN, 'scikit-learn no instalado; requiere el entorno propio del PSet #3.')
class PreprocessingChecks(unittest.TestCase):
    def setUp(self):
        # Nombres de fixtures: no son features escogidas para el PSet #3.
        self.spec = FeatureSpec('fixture_label', ('fixture_numeric',), ('fixture_category',),
                                ('fixture_numeric','fixture_category'))

    def test_explicit_approval_and_target_exclusion(self):
        with self.assertRaises(ValueError):
            replace(self.spec, approved_at_prediction=()).validate()
        with self.assertRaises(ValueError):
            replace(self.spec, target='fixture_numeric').validate()
        with self.assertRaises(ValueError):
            replace(self.spec, excluded_columns=('fixture_numeric',)).validate()

    def test_train_statistics_and_unseen_category(self):
        train = pd.DataFrame({'fixture_numeric':[1.,np.nan,3.], 'fixture_category':['a',None,'a']})
        holdout = pd.DataFrame({'fixture_numeric':[1000.,np.nan], 'fixture_category':['unseen',None]})
        guard = FeatureGuard(self.spec)
        normalized = guard.fit_transform(train)
        preprocessor = build_preprocessor(self.spec, sparse_output=False)
        preprocessor.fit(normalized)  # Solo preprocesamiento de tres fixtures, no estimador ML.
        numeric = preprocessor.named_transformers_['numeric']
        before = numeric.named_steps['impute'].statistics_.copy()
        transformed = preprocessor.transform(guard.transform(holdout))
        np.testing.assert_array_equal(before, [2.])
        np.testing.assert_array_equal(before, numeric.named_steps['impute'].statistics_)
        self.assertTrue(np.isfinite(transformed).all())
        with self.assertRaises(ValueError):
            guard.transform(train.assign(fixture_label=0))

    def test_all_missing_columns_remain_and_model_is_unfitted(self):
        from sklearn.dummy import DummyClassifier
        frame = pd.DataFrame({'fixture_numeric':[np.nan,np.nan], 'fixture_category':[None,None]})
        normalized = FeatureGuard(self.spec).fit_transform(frame)
        result = build_preprocessor(self.spec, sparse_output=False).fit_transform(normalized)
        self.assertEqual(result.shape[0],2)
        self.assertGreaterEqual(result.shape[1],2)
        pipeline = build_pipeline(DummyClassifier(), self.spec)
        self.assertFalse(hasattr(pipeline.named_steps['model'], 'classes_'))


class ScaffoldChecks(unittest.TestCase):
    def test_feature_contract_can_be_validated_without_sklearn(self):
        with self.assertRaises(ValueError):
            FeatureSpec('target', ('target',), (), ('target',)).validate()
        with self.assertRaises(ValueError):
            FeatureSpec('label', ('field',), (), ()).validate()
        FeatureSpec('label', ('field',), (), ('field',)).validate()

    def test_example_contract_cannot_be_approved_implicitly(self):
        with self.assertRaises(ValueError):
            load_experiment_contract(PROJECT_ROOT/'config'/'experiment.example.json')

    def test_notebooks_code_parses_and_outputs_have_no_errors(self):
        paths = sorted((PROJECT_ROOT/'notebooks').glob('*.ipynb'))
        self.assertEqual(len(paths),5)
        for path in paths:
            notebook = json.loads(path.read_text(encoding='utf-8'))
            self.assertEqual(notebook['nbformat'],4)
            for cell in notebook['cells']:
                if cell['cell_type']=='code':
                    self.assertFalse(any(output.get('output_type')=='error' for output in cell['outputs']))
                    if cell['execution_count'] is None:
                        self.assertEqual(cell['outputs'],[])
                    ast.parse(''.join(cell['source']), filename=path.name)

    def test_physical_contract_has_90_columns_and_no_target(self):
        schema = json.loads((PROJECT_ROOT/'config'/'obt_schema_pset2.json').read_text(encoding='utf-8'))
        self.assertEqual(len(schema['columns']),90)
        self.assertNotIn('TARGET', schema['columns'])


if __name__ == '__main__':
    unittest.main()
