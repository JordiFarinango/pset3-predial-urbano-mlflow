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

## Estudio complementario — Defaults reales desde Snowflake (20/11/2026)

Dos líneas base y cinco familias convencionales, con las mismas ocho features y
los tres folds temporales congelados. scikit-learn 1.7.2 con defaults reales;
única excepción: random_state. No es una comparación fundacional completa.

| Modelo | AUC train | AUC CV ± sd | Brecha | Fit medio/fold (s) | Run ID terminado |
|---|---:|---:|---:|---:|---|
| dummy | 0.500000 | 0.500000 ± 0.000000 | 0.000000 | 0.0274 | `b8c6da92780f4eec90f1e364b5be29bd` |
| regla_academica | 0.522189 | 0.518377 ± 0.026910 | 0.003812 | 0.0252 | `56fd455bdefa49a1b772395c256ce1f8` |
| logistica | 0.575070 | 0.570464 ± 0.002261 | 0.004606 | 0.0382 | `271f9454b72e49b89b84543e040f28c5` |
| random_forest | 1.000000 | 0.518676 ± 0.007376 | 0.481324 | 0.6101 | `2d0c24b22bba4a3ea677749add3ed7e5` |
| hist_boosting | 0.883806 | 0.542359 ± 0.020856 | 0.341447 | 0.2546 | `9190a6019db04a4e867b6938cd686542` |
| mlp | 0.712385 | 0.534452 ± 0.013804 | 0.177933 | 2.6454 | `7439cfc2c7024f0b9c1e312718a7a263` |
| svm_rbf | 0.657288 | 0.554149 ± 0.015608 | 0.103139 | 1.3952 | `b5b1ce43c67049c5a14b17ec7987bf8b` |

Fuente real: `PSET3_SINTETICO.EXPERIMENTOS.OBT_PREDIAL_2B7D72B7735E`,
12.000 filas y 20 columnas. Tres SELECT auténticos conciliaron esquema, claves,
etiquetas y huellas con el Parquet congelado; se entrenó con el DataFrame remoto.
Query IDs:

- `01c887ee-030a-0263-0013-16ae0007b022`.
- `01c887f5-030a-0263-0013-16ae0007b02a`.
- `01c887fb-010a-0265-0013-16ae00079052`.

[MLflow 5002](http://127.0.0.1:5002/#/experiments/1) es independiente del servidor
histórico 5001: 42 runs nuevos (36 FINISHED, seis FAILED conservados), 28 ajustes
únicos y 28 pipelines con firma/ejemplo. Los artefactos completos siguen en sus
volúmenes locales; Git conserva recibos filtrados, no el backend ni los payloads.
En otro equipo ese enlace localhost no contiene estas corridas automáticamente.

El fallo inicial por categorical_features duplicado se corrigió con el prefijo
model__; el bloqueo posterior por RAM conservó los folds terminados de boosting,
reutilizados sin repetirlos. MLP avisó de no convergencia a200 iteraciones.
La sd es muestral entre tres folds, no un intervalo de confianza; el tiempo fit
excluye scoring, serialización y logging. No hay nuevos scores de hold-out.

Este estudio NO reemplaza las396 corridas originales LOCAL_PARQUET, las90
configuraciones históricas ni la selección/aliases champion y challenger.
Tuning de los mejores del estudio NUEVO, fundacional/LLM y validación en otra
cuenta siguen pendientes. Los hashes de código ejecutado se conservan; no se
atribuye a estas corridas un commit limpio inexistente.

[Métricas completas](docs/defaults_snowflake_resultados.csv) ·
[42 runs exportados](docs/defaults_snowflake_runs.jsonl) ·
[Consultas, modelos y hashes](docs/defaults_snowflake_evidencia.json) ·
Los hashes de los recibos originales se conservan en el mismo JSON de evidencia.

Controles offline, sin servidor, Snowflake ni entrenamiento:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_publicacion
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_defaults_snowflake
.\.venv\Scripts\python.exe -B -X utf8 -m src.revision_offline
```

El control específico reconcilia IDs/estados, defaults, métricas y folds contra
los recibos, y el modo A conserva su revisión de las cinco etapas originales.
El PDF definitivo conserva su contenido; esta comparación se documenta en el
README y en los tres archivos científicos enlazados.

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
.\.venv\Scripts\python.exe -B -X utf8 -m src.revision_offline
```

Dependencias directas en requirements.txt; lock del entorno usado en el estudio.
La instalacion completa en otro ordenador sigue sin demostrarse. El modo A
revisa las cinco etapas desde evidencias publicadas y carga solo modelos propios
para comparar scores, clases y AUC congelados, sin fit, seleccion ni tuning.
No requiere MLflow, Snowflake o regenerar datos. No cargar pickle/joblib externos.
MANIFEST_SHA256.json identifica los archivos públicos actuales, incluidos los
recibos nuevos; no sustituye el linaje ni el hash del código ejecutado.
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

El historial actual parte de un commit raíz preparado en una tarea anterior,
con los resultados originales y solo la memoria definitiva. Esta integración
prepara un commit normal adicional; no reconstruye ni reescribe ese historial.
Los hashes de código y run IDs ejecutados conservan su procedencia. La retirada
anterior de versiones del historial alcanzable no garantiza borrar copias,
forks, enlaces por SHA o caches de GitHub.

## Revisión offline y límites de reproducción

[Procedimiento de revisión y ejecución nueva](docs/reproducibilidad.md). El modo A
revisa las cinco etapas originales sin backend privado ni entrenamiento. No
certifica Restart+Run All en una instalación limpia. Los iniciales históricos
no eran defaults estrictos en cuatro modelos: sus campos default_auc y sus396
corridas conservan sus valores y linaje. No se reparan retrospectivamente.

El recibo nuevo documenta28 pipelines con firma/ejemplo; sus payloads completos
son locales. La cobertura histórica de campos y artefactos no es uniforme entre
sus396 runs. No se atribuye un Git SHA inexistente a los entrenamientos.
SOTA/TabPFN, prompting LLM y tuning complementario siguen pendientes.
