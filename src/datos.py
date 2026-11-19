"""Lectura Snowflake explicita, acotada y exclusiva de destinos sinteticos aprobados.

No hay conexiones, consultas ni lectura de credenciales al importar este modulo.
"""
from dataclasses import dataclass
from datetime import date, datetime, timezone
from decimal import Decimal
import hashlib
import json
import math
from numbers import Integral, Real
import os
from pathlib import Path
import re
from typing import Mapping
from uuid import UUID

import pandas as pd
import numpy as np

IDENTIFIER = re.compile(r'^[A-Z_][A-Z0-9_]{0,254}$')


def identifier(value: str) -> str:
    normalized = value.strip().upper()
    if not IDENTIFIER.fullmatch(normalized):
        raise ValueError('Identificador Snowflake no permitido.')
    return normalized


@dataclass(frozen=True, repr=False)
class SnowflakeConfig:
    account: str
    user: str
    role: str
    warehouse: str
    database: str
    schema: str
    table: str
    private_key_path: Path
    allowed_databases: tuple[str, ...]
    classification: str

    def validate(self) -> None:
        if self.classification != 'SYNTHETIC':
            raise ValueError('Solo se admite clasificacion explicita SYNTHETIC.')
        if not self.account or not self.user:
            raise ValueError('Falta configurar cuenta o usuario propios.')
        for value in (self.role, self.warehouse, self.database, self.schema, self.table):
            identifier(value)
        allowed = {identifier(value) for value in self.allowed_databases}
        database = identifier(self.database)
        if database == 'PSET2_CAYAMBE' or database not in allowed:
            raise ValueError('Base municipal o destino fuera de la lista sintetica aprobada.')
        if not (database.startswith('PSET3_') or database == 'PSET2_DEMO_SINTETICO'):
            raise ValueError('El destino debe ser una base PSET3_* o PSET2_DEMO_SINTETICO.')
        key = self.private_key_path
        project = Path(__file__).resolve().parents[1]
        municipal_project = project.parent / 'pset_2'
        if not key.is_absolute() or key.resolve().is_relative_to(project) or key.resolve().is_relative_to(municipal_project):
            raise ValueError('La clave propia debe estar fuera de pset_3 y pset_2.')

    @property
    def qualified_table(self) -> str:
        self.validate()
        return '.'.join(f'"{identifier(x)}"' for x in (self.database, self.schema, self.table))

    @classmethod
    def from_env(cls, environment: Mapping[str, str] | None = None):
        env = os.environ if environment is None else environment
        required = ('ACCOUNT', 'USER', 'ROLE', 'WAREHOUSE', 'DATABASE', 'PRIVATE_KEY_PATH')
        if any(not env.get('SNOWFLAKE_' + key, '').strip() for key in required):
            raise ValueError('Configuracion Snowflake incompleta; no se intentara conectar.')
        config = cls(
            account=env['SNOWFLAKE_ACCOUNT'].strip(), user=env['SNOWFLAKE_USER'].strip(),
            role=env['SNOWFLAKE_ROLE'].strip(), warehouse=env['SNOWFLAKE_WAREHOUSE'].strip(),
            database=env['SNOWFLAKE_DATABASE'].strip(), schema=env.get('SNOWFLAKE_SCHEMA', 'OBT'),
            table=env.get('SNOWFLAKE_TABLE', 'OBT_URBANO_OBLIGACIONES'),
            private_key_path=Path(env['SNOWFLAKE_PRIVATE_KEY_PATH']).expanduser(),
            allowed_databases=tuple(x.strip() for x in env.get('PSET3_ALLOWED_DATABASES', '').split(',') if x.strip()),
            classification=env.get('PSET3_DATA_CLASSIFICATION', ''),
        )
        config.validate()
        return config


def connect_snowflake(config: SnowflakeConfig, *, private_key_password: str):
    """Abre una conexion solo al ser invocada. No escribe ni modifica objetos."""
    config.validate()
    if not config.private_key_path.is_file() or not private_key_password:
        raise ValueError('Falta una clave cifrada propia o su passphrase de sesion.')
    import snowflake.connector
    return snowflake.connector.connect(
        account=config.account, user=config.user, role=config.role,
        warehouse=config.warehouse, database=config.database, schema=config.schema,
        authenticator='SNOWFLAKE_JWT', private_key_file=str(config.private_key_path),
        private_key_file_pwd=private_key_password, autocommit=False,
        login_timeout=30, network_timeout=60,
        session_parameters={'QUERY_TAG': 'pset3_synthetic_read', 'STATEMENT_TIMEOUT_IN_SECONDS': 120},
    )


def load_schema(path: Path | str) -> dict:
    """Contrato fisico local; no define features ni etiqueta."""
    return json.loads(Path(path).read_text(encoding='utf-8'))


def dataset_metadata(connection, config: SnowflakeConfig) -> list[dict]:
    """Consulta INFORMATION_SCHEMA explicitamente; la cuenta debe ser sintetica."""
    config.validate()
    query = (
        'SELECT COLUMN_NAME,DATA_TYPE,NUMERIC_PRECISION,NUMERIC_SCALE,'
        'CHARACTER_MAXIMUM_LENGTH,DATETIME_PRECISION,IS_NULLABLE '
        f'FROM "{identifier(config.database)}".INFORMATION_SCHEMA.COLUMNS '
        'WHERE TABLE_SCHEMA=%s AND TABLE_NAME=%s ORDER BY ORDINAL_POSITION'
    )
    with connection.cursor() as cursor:
        cursor.execute(query, (identifier(config.schema), identifier(config.table)))
        columns = [x[0].lower() for x in cursor.description]
        records = [dict(zip(columns, row)) for row in cursor.fetchall()]
    if not records:
        raise ValueError('Tabla inexistente, sin permisos o sin columnas visibles.')
    return records


def validate_schema(actual: list[dict], expected: dict, *, allow_extra: bool = False) -> None:
    indexed = {row['column_name'].upper(): row for row in actual}
    wanted = {identifier(name): spec for name, spec in expected['columns'].items()}
    if len(indexed) != len(actual) or not wanted:
        raise ValueError('Metadatos duplicados o contrato de columnas vacio.')
    if not set(wanted).issubset(indexed) or (not allow_extra and set(indexed) != set(wanted)):
        raise ValueError('Las columnas visibles no coinciden con el contrato aprobado.')
    for name, specification in wanted.items():
        row = indexed[name]
        for field, value in specification.items():
            if field not in ('data_type', 'numeric_precision', 'numeric_scale',
                             'character_maximum_length', 'datetime_precision', 'is_nullable'):
                raise ValueError('Propiedad desconocida en el contrato fisico.')
            if row.get(field) != value:
                raise ValueError(f'Tipo o propiedad incompatible en columna {name}.')


def _canonical(value):
    if value is None or value is pd.NA or value is pd.NaT:
        return ['null']
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise ValueError('Decimal no finito en el dataset.')
        text = format(value, 'f')
        if '.' in text:
            text = text.rstrip('0').rstrip('.')
        return ['decimal', '0' if value == 0 else text]
    if isinstance(value, (bool, np.bool_)):
        return ['bool', bool(value)]
    if isinstance(value, Integral):
        return ['integer', str(int(value))]
    if isinstance(value, Real):
        numeric = float(value)
        if math.isnan(numeric):
            return ['null']
        if not math.isfinite(numeric):
            raise ValueError('Numero no finito en el dataset.')
        return ['float64', numeric.hex()]
    if isinstance(value, (datetime, pd.Timestamp)):
        stamp = pd.Timestamp(value)
        if stamp.tzinfo is not None:
            stamp = stamp.tz_convert('UTC')
        return ['timestamp', stamp.isoformat()]
    if isinstance(value, date):
        return ['date', value.isoformat()]
    if isinstance(value, str):
        return ['text', value]
    raise TypeError(f'Tipo no admitido para huella: {type(value).__name__}')


def dataset_fingerprint(frame: pd.DataFrame) -> str:
    """SHA-256 v1: contenido + nombres/tipos, independiente del orden de filas/columnas.

    Conserva multiplicidad de filas. Un cambio de tipos cambia la huella.
    Las columnas operativas se excluyen SOLO si el llamador declara esa proyeccion.
    """
    if not frame.columns.is_unique or not all(isinstance(x, str) for x in frame.columns):
        raise ValueError('La huella requiere nombres de columna unicos y textuales.')
    columns = sorted(frame.columns)
    dump = lambda value: json.dumps(value, ensure_ascii=False, separators=(',', ':'), allow_nan=False)
    header = ['pset3-fingerprint-v1', [(name, str(frame[name].dtype)) for name in columns]]
    digest = hashlib.sha256(dump(header).encode('utf-8'))
    rows = sorted(dump([_canonical(v) for v in row]) for row in frame[columns].itertuples(index=False, name=None))
    for row in rows:
        digest.update(b'\n')
        digest.update(row.encode('utf-8'))
    return digest.hexdigest()


@dataclass
class DatasetSnapshot:
    frame: pd.DataFrame
    metadata: dict


def read_obt(connection, config: SnowflakeConfig, *, lote_ids: tuple[str, ...],
             expected_schema: dict, max_rows: int = 100_000) -> DatasetSnapshot:
    """SELECT acotado por lotes explicitos. Nunca ultimo lote automatico ni SELECT *."""
    config.validate()
    if not lote_ids or len(set(lote_ids)) != len(lote_ids) or len(lote_ids) > 1000:
        raise ValueError('Se requieren de 1 a 1000 UUID de lote distintos y explicitos.')
    for lote in lote_ids:
        if str(UUID(lote)) != lote:
            raise ValueError('UUID de lote no canonico.')
    if not 1 <= max_rows <= 1_000_000:
        raise ValueError('Limite de memoria fuera del contrato local.')
    physical = dataset_metadata(connection, config)
    validate_schema(physical, expected_schema)
    names = [identifier(x) for x in expected_schema['columns']]
    keys = ['LOTE_ID', 'OBLIGACION_ID']
    if not set(keys).issubset(names):
        raise ValueError('La lectura OBT requiere las claves LOTE_ID y OBLIGACION_ID.')
    projection = ','.join('"' + name + '"' for name in names)
    placeholders = ','.join(['%s'] * len(lote_ids))
    query = (f'SELECT {projection} FROM {config.qualified_table} '
             f'WHERE "LOTE_ID" IN ({placeholders}) '
             'ORDER BY "LOTE_ID","OBLIGACION_ID" LIMIT %s')
    with connection.cursor() as cursor:
        cursor.execute(query, (*lote_ids, max_rows + 1))
        records = []
        while batch := cursor.fetchmany(1000):
            records.extend(batch)
            if len(records) > max_rows:
                raise ValueError('Dataset excede max_rows; no se devuelve una muestra truncada.')
        query_id = getattr(cursor, 'sfqid', None)
    frame = pd.DataFrame.from_records(records, columns=names)
    if frame.empty or frame[keys].isna().any().any() or frame.duplicated(keys).any():
        raise ValueError('OBT vacia o claves nulas/duplicadas.')
    if set(frame['LOTE_ID']) != set(lote_ids):
        raise ValueError('Faltan lotes solicitados; no se acepta una lectura parcial.')
    metadata = {
        'table': config.qualified_table, 'row_count': len(frame), 'columns': physical,
        'lote_ids': sorted(lote_ids), 'query_id': query_id,
        'read_at_utc': datetime.now(timezone.utc).isoformat(),
        'fingerprint': dataset_fingerprint(frame), 'fingerprint_version': 'pset3-fingerprint-v1',
    }
    return DatasetSnapshot(frame, metadata)
