"""Rutas y contrato experimental explicito, independiente del generador pendiente."""
from pathlib import Path
import json
import os

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def load_local_environment(path: Path | None = None) -> None:
    """Solo por llamada explicita; nunca busca .env en carpetas del PSet #2."""
    from dotenv import load_dotenv
    load_dotenv(path or PROJECT_ROOT / '.env', override=False)


def load_experiment_contract(path: Path | str | None = None) -> dict:
    selected = Path(path or os.environ.get('PSET3_EXPERIMENT_CONFIG', 'config/experiment.json'))
    if not selected.is_absolute():
        selected = PROJECT_ROOT / selected
    if not selected.is_file():
        raise ValueError('Falta config/experiment.json aprobado; consultar experiment.example.json.')
    config = json.loads(selected.read_text(encoding='utf-8'))
    if config.get('approved') is not True:
        raise ValueError('El contrato experimental no esta aprobado.')
    required = ('dataset_version', 'target', 'target_rule_version', 'lote_ids', 'split', 'primary_metric', 'business_metric')
    if any(not config.get(name) for name in required):
        raise ValueError('Faltan dataset, target, lotes, particion o metricas por aprobar.')
    from .contratos import FeatureSpec
    FeatureSpec.from_dict(config)
    # La validez de la etiqueta y disponibilidad temporal requieren revision de dominio.
    return config
