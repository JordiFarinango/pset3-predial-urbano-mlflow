# PSet #3 - Predial urbano y MLflow

Proyecto academico de Ciencia de Datos, USFQ. Objetivo: clasificar al emitir una
obligacion ficticia si seguira sin pagarse completamente al dia 90. Clase 1 =
saldo neto positivo; clase 0 = saldo cero.

## Dataset y metodo

Monte Carlo por eventos, semilla 20261117, version mc-eventos-v1: 12.000
obligaciones, 18.988 eventos y OBT analitica de 20 columnas. Las formulas son
supuestos academicos; no son reglas tributarias municipales ni se calibraron con
contribuyentes reales. Se conservan los movimientos individuales y los snapshots
sinteticos congelados. [Metodo, target y controles](docs/metodologia_sintetica.md).

La fuente de los 396 experimentos originales fue LOCAL_PARQUET. El mismo snapshot
sintetico ya fue publicado y releido fisicamente en Snowflake: READBACK_VERIFIED
en PSET3_SINTETICO.EXPERIMENTOS. Esta publicacion posterior no cambia el origen
historico de los entrenamientos. [Evidencia verificable](docs/snowflake_evidencia.json).
No se ejecuto
Kestra/dbt/Spark para esta OBT. No se necesitan VPN, SQL Server ni usuario municipal.
La instancia de pruebas del proyecto original contiene datos reales: no se
incluyen registros, extracciones, respaldos ni credenciales. HMAC es
seudonimizacion, no anonimato irreversible.

## Estructura

- notebooks/: cinco etapas y sus salidas autenticas.
- src/: generador, lectura, features, pipelines, tracking y experimentacion.
- config/: contratos y parametros; data/synthetic_v1/: snapshots y particiones.
- models/: champion/challenger propios, manifiesto y predicciones congeladas.
- docs/eda/, docs/folds_*.csv: analisis y validacion de candidatos.
- docs/cierre/: tuning, decision, resultados y limites del estudio.
- docs/corridas.csv y docs/mlflow_portable/: evidencia de 396 corridas originales.
- checks/: controles de publicacion/reproduccion y pruebas unitarias.

## Instalacion y verificacion offline

Desde la raiz del clon, Windows y Python 3.12.14 (otras plataformas no validadas):

```powershell
py -3.12 -m venv .venv
.\.venv\Scripts\python.exe -X utf8 -m pip install -r requirements-lock-windows-py312.txt
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_publicacion
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_paquete
```

Dependencias directas en requirements.txt; lock del entorno usado en el estudio.
La instalacion completa en otro ordenador sigue sin demostrarse. El primer
control revisa hashes/procedencia/exclusiones. El segundo carga solo modelos
propios y compara scores, clases y AUC congelados sin fit, seleccion ni tuning.
No requiere MLflow, Snowflake o regenerar datos. No cargar pickle/joblib externos.
MANIFEST_SHA256.json identifica la seleccion publica actual, no experimentos nuevos.
Memoria academica: [PSet #3 - Farinango y Andrango](docs/PSet3_memo_Farinango_Andrango.pdf).
Los controles verifican su hash, estructura y limite de seis paginas, ademas de datos y modelos.

## MLflow y notebooks

Docker Desktop con contenedores Linux y Compose v2. Servicio local:

```powershell
docker compose --env-file .env.example up -d mlflow
Invoke-WebRequest http://127.0.0.1:5001/health
```

MLflow 3.18.0 en http://127.0.0.1:5001; backend SQLite, Registry y artefactos en
volumenes persistentes. Solo loopback. No ejecutar otra copia del Compose en
paralelo: conserva nombre pset3-ml, puerto y volumenes. No usar down -v.
Un servidor nuevo no contiene las corridas ni los aliases historicos.

Registrar el kernel del entorno y abrir Jupyter (no se ejecutaron estos comandos
durante la limpieza):

```powershell
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name pset3 --display-name "Python (PSet 3)"
.\.venv\Scripts\python.exe -m jupyterlab
```

Seleccionar Python (PSet 3) y revisar en orden:

1. 01_eda.ipynb: EDA de desarrollo.
2. 02_features.ipynb: features de emision, guardas y protocolo temporal.
3. 03_baseline_candidatos.ipynb: dos baselines, cinco familias e intento TabPFN.
4. 04_tuning.ipynb: 30 configuraciones por familia, 90 en total.
5. 05_seleccion.ipynb: decision congelada, hold-out y Registry.

Las salidas autenticas permiten revisar el trabajo sin Run All. La reejecucion
03-05 requiere backend/artefactos/checkpoints correspondientes al estudio original;
el clon no incluye volumenes ni caches privados. Una reejecucion sin esos recursos
puede fallar o iniciar calculos: para auditar esta entrega usar los controles
offline. Cambiar datos/configuracion exige un estudio nuevo y trazabilidad propia.

## Resultados y modelos

Desarrollo 9.510 filas; hold-out temporal 2024 de 2.000; 490 excluidas del train
por disponibilidad de etiqueta, maduras al cierre final. CV expansiva con tres
folds. Imputacion, escalado y codificacion se ajustan dentro de cada train.

| Modelo | ROC-AUC CV ajustada +/- sd | ROC-AUC hold-out |
|---|---:|---:|
| Champion: logistica | 0.570509 +/- 0.002199 | 0.5459667435 |
| Challenger: MLP | 0.569234 +/- 0.005586 | 0.5485606352 |

Registro original: `models:/pset3-predial-urbano-modelo@champion` y `@challenger`.
Modelos propios exportados en models/. Parametros, folds, scores y decision:
[configuraciones](docs/cierre/tuning_configuraciones.csv),
[seleccion congelada](docs/cierre/selection_frozen.json),
[resultados finales](docs/cierre/holdout_resultados.json).
La memoria definitiva incluye tres capturas del servidor MLflow real aportadas
por el autor: comparacion de corridas y metricas, y Model Registry (pagina 3).

396 corridas academicas: 394 FINISHED y 2 FAILED; smoke tecnico excluido. El
CSV/JSONL conserva IDs, metricas y hashes de codigo originales, sin recrear runs.
Las corridas preceden al repositorio: su git.sha sigue siendo
UNAVAILABLE_PSET3_NOT_VERSIONED. El commit de publicacion identifica esta entrega,
no retroactivamente sus entrenamientos. Los enlaces historicos necesitan el
backend original o respaldo completo de sus volumenes.

## Limitaciones comprobadas

- Snowflake: [publicacion y relectura verificadas](docs/snowflake.md). La cuenta
  academica original no se comparte; otro evaluador configura su propia cuenta.
  Los entrenamientos originales siguen siendo LOCAL_PARQUET.
- La memoria definitiva incluye la publicacion y relectura Snowflake (pagina 6);
  los detalles verificables se conservan en docs/snowflake_evidencia.json.
- TabPFN completo solo un fold por presion de RAM, sin media CV comparable:
  [resultado](docs/fundacional_resultado.json) y [atribucion](docs/cierre/TabPFN_atribucion.md).
  SVM no sustituye un modelo fundacional.
- LLM: [justificacion y plantilla](docs/cierre/llm_justificacion.md); no hubo prompting.
- AUC final cercana a 0,55. Los umbrales OOF con costo academico FP+3FN marcan
  todos los casos; no hay mejora operacional sobre marcar todo ni superioridad
  concluyente entre finalistas. No se reelige el ganador con hold-out.
- Una poblacion principal, mecanismos simplificados y sin validacion municipal.

El historial de publicacion se reconstruye con un nuevo commit raiz que conserva
los resultados originales y publica solo la memoria definitiva. Los hashes de
codigo y run IDs historicos no se reescriben ni se les atribuye ese commit.
Sustituir main retira las versiones anteriores de su historial alcanzable; no
garantiza borrar copias descargadas, forks, enlaces por SHA o caches de GitHub.
