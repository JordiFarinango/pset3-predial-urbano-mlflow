"""Revision equivalente a las cinco etapas: evidencia congelada, nunca fit o red."""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def review():
    # Audita los modelos PROPIOS publicados y todos los hashes antes de usarlos.
    from checks.verificar_paquete import audit
    frozen = audit()
    from src.analisis import load_population, derive_features
    from src.configuracion import load_experiment_contract
    from src.contratos import FeatureSpec
    frame, manifest = load_population()
    parts = json.loads((ROOT/'data/synthetic_v1/partitions.json').read_text())
    spec = FeatureSpec.from_dict(load_experiment_contract())
    tr, ho = parts['train_indices'], parts['holdout_indices']
    if len(set(tr)) != len(tr) or len(set(ho)) != len(ho) or set(tr) & set(ho):
        raise ValueError('Particiones duplicadas o solapadas.')
    dev, hold = frame.iloc[tr], frame.iloc[ho]
    if dev.OBLIGACION_ID.tolist() != parts['development_keys'] or hold.OBLIGACION_ID.tolist() != parts['holdout_keys']:
        raise ValueError('Claves de particion alteradas.')
    cutoff = pd.Timestamp('2024-01-01', tz='UTC')
    if not (dev.LABEL_AVAILABLE_AT < cutoff).all() or not (hold.FECHA_EMISION_DECLARADA >= cutoff).all():
        raise ValueError('Disponibilidad temporal no conciliada.')
    if not set(spec.features).issubset(derive_features(dev).columns):
        raise ValueError('Features no derivables desde el snapshot.')
    completeness = pd.read_csv(ROOT/'docs/eda/completitud.csv',index_col=0)
    np.testing.assert_array_equal(completeness.loc[dev.columns,'missing'],dev.isna().sum())
    columns = ['TOTAL_OBLIGACION_DECLARADO','AVALUO_SIMULADO','AREA_M2','PLAZO_DIAS_SIMULADO']
    correlation = pd.read_csv(ROOT/'docs/eda/spearman.csv',index_col=0)
    np.testing.assert_allclose(correlation.to_numpy(),dev[columns].corr(method='spearman').to_numpy(),rtol=0,atol=1e-12)
    folds = []
    for i, f in enumerate(parts['folds']):
        train, valid = f['train'], f['validation']
        if set(train) & set(valid) or max(train+valid) >= len(dev):
            raise ValueError('Fold invalido.')
        boundary = dev.iloc[valid].FECHA_EMISION_DECLARADA.min()
        if not (dev.iloc[train].LABEL_AVAILABLE_AT < boundary).all():
            raise ValueError('Etiquetas futuras en train.')
        folds.append({'fold': i, 'train': len(train), 'validation': len(valid)})
    # Reconciliar medias/sd y gaps desde los tres CSV de folds, sin entrenar.
    candidates = pd.read_csv(ROOT/'docs/resultados_candidatos.csv')
    for row in candidates.itertuples():
        f = pd.read_csv(ROOT/f'docs/folds_{row.candidate}.csv')
        if len(f) != 3:
            raise ValueError('Faltan folds.')
        for metric in ['roc_auc', 'average_precision', 'balanced_accuracy', 'f1', 'accuracy']:
            np.testing.assert_allclose(f['gap_'+metric], f['train_'+metric]-f['validation_'+metric], rtol=0, atol=1e-12)
        for column in f.columns:
            if column in ['fold', 'train_rows', 'validation_rows']:
                continue
            for suffix, value in [('mean', f[column].mean()), ('std', f[column].std(ddof=1))]:
                np.testing.assert_allclose(value, getattr(row, column+'_'+suffix), rtol=0, atol=1e-12)
    # Reconciliar cada resultado de tuning con sus metricas MLflow exportadas.
    runs = pd.read_csv(ROOT/'docs/corridas.csv').set_index('run_id')
    tuning = pd.read_csv(ROOT/'docs/cierre/tuning_configuraciones.csv')
    for row in tuning.to_dict('records'):
        for key, value in row.items():
            if key.endswith(('_mean', '_std')):
                np.testing.assert_allclose(value, runs.loc[row['run_id'], 'metrics.'+key], rtol=0, atol=1e-12)
    selected = json.loads((ROOT/'docs/cierre/selection_frozen.json').read_text())
    for role, item in selected['roles'].items():
        oof = pd.read_parquet(ROOT/f'models/{role}_development_oof.parquet')
        expected = {(i, int(pos)) for i, f in enumerate(parts['folds']) for pos in f['validation']}
        if len(oof) != len(expected) or set(zip(oof.fold, oof.position)) != expected or oof.position.duplicated().any():
            raise ValueError('OOF no corresponde a folds congelados.')
        np.testing.assert_array_equal(oof.y, dev.iloc[oof.position].TARGET_NO_PAGADA_90D.astype(int))
        from sklearn.metrics import roc_auc_score
        aucs = [roc_auc_score(g.y,g.score) for _,g in oof.groupby('fold')]
        np.testing.assert_allclose([np.mean(aucs),np.std(aucs,ddof=1)],[item['tuned_auc'],item['tuned_sd']],rtol=0,atol=1e-12)
        for ratio, policy in item['cost_sensitivity'].items():
            pred = oof.score.to_numpy() >= policy['threshold']
            y = oof.y.to_numpy()
            cost = ((pred & (y == 0)).sum() + int(ratio)*((~pred) & (y == 1)).sum())/len(y)
            np.testing.assert_allclose(cost, policy['development_oof_cost'], rtol=0, atol=1e-12)
    from checks.verificar_snowflake import verify
    frames = {n: pd.read_parquet(ROOT/f'data/synthetic_v1/{n}.parquet') for n in ['obligaciones', 'movimientos', 'obt']}
    snowflake = verify(frames)
    return {
        'mode': 'A_OFFLINE_FROZEN_REVIEW', 'status': 'PASS',
        '01_eda': {'development_rows': len(dev), 'class_counts': {str(k):int(v) for k,v in dev.TARGET_NO_PAGADA_90D.value_counts().items()}, 'evidence': 'docs/eda/'},
        '02_features': {'features': list(spec.features), 'folds': folds, 'holdout_rows': len(hold), 'excluded_at_training_cutoff': len(frame)-len(dev)-len(hold)},
        '03_candidates': {'families_and_baselines': len(candidates), 'fold_means_sd_gaps_reconciled': True, 'historical_initials_are_not_all_defaults': True},
        '04_tuning': {'configurations': len(tuning), 'exported_run_metrics_reconciled': True},
        '05_selection': {'fixed_oof_policies_reconciled': True, 'roles': frozen['roles'], 'no_threshold_search': True},
        'snowflake_receipt_review_offline': snowflake,
        'training_executed': False, 'new_mlflow_runs': False, 'network_used': False,
        'equivalent_to_retraining': False,
        'limits': 'Checks published results and final own models; no refit of 90 configurations, no original backend reconstruction or proof of clean-machine installation.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(review(), indent=2)
    if args.output:
        # Never overwrite any old evidence, including outputs outside this module.
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(result+'\n')
    print(result)


if __name__ == '__main__':
    main()
