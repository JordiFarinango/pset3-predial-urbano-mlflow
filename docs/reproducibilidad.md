# Revisión offline y reproducción del estudio complementario

Usar Python3.12 y el entorno del lock del README. No se necesitan credenciales
para revisar la entrega. Los siguientes controles no crean runs ni entrenan:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_publicacion
.\.venv\Scripts\python.exe -B -X utf8 -m checks.verificar_defaults_snowflake
.\.venv\Scripts\python.exe -B -X utf8 -m src.revision_offline
```

Publicación verifica hashes, procedencia sintética, exclusiones y el único PDF
autorizado, y llama al control del complemento. Este último cruza42 IDs y sus
estados36 FINISHED/6 FAILED, siete candidatos,21 folds, siete refits, parámetros
default y28 referencias de modelos. Recalcula medias, desviaciones y brechas
usando la exportación científica; comprueba las tres consultas y los hashes de
procedencia en [el recibo](defaults_snowflake_evidencia.json). No consulta APIs
ni descarga payloads. CSV y JSONL mantienen exactamente los bytes exportados.

## A. Revisión de las cinco etapas originales

El modo A contrasta EDA, features/particiones, candidatos,90 configuraciones de
tuning y selección congelada. Carga solo modelos propios verificados por SHA y
reproduce scores/clases de referencia, sin fit, selección o búsqueda de umbral.
Comprueba OOF, maduración de etiquetas y políticas de costo previamente fijadas.
No modifica las cinco libretas ni sus salidas. Se ejecutó con Pipeline.fit,
generación y socket.connect bloqueados y sin llamadas a esas operaciones.

No equivale al Restart+Run All literal:03-05 necesitan backend/checkpoints/aliases
originales y02 incluye ajustes de control negativo. Un servidor vacío no restaura
las396 corridas ni sus artefactos. No se ha probado instalación limpia en otra
máquina. No cargar pickle externos; los joblib publicados son los propios.

## B. Nueva ejecución separada, solo tras autorización

El estudio del20/11/2026 usó realmente Snowflake. Sus396 predecesores siguen
siendo LOCAL_PARQUET. Mantiene ocho features,9.510 filas de desarrollo y los tres
folds congelados. No usa hold-out para comparar, seleccionar o ajustar.
Los cinco estimadores usan defaults de sklearn1.7.2 salvo random_state. Los
iniciales históricos tenían configuraciones propias en cuatro modelos; no se
reescriben sus campos default_auc. El plan permite inspeccionar las diferencias:

```powershell
.\.venv\Scripts\python.exe -B -X utf8 -m src.estudio_nuevo
```

Para repetir con credenciales y recursos propios: configurar .env según
[Snowflake](snowflake.md), clave cifrada externa y permisos SELECT/USAGE. La
passphrase se introduce localmente mediante getpass; no se versiona. Mantener
el objeto sintético exacto y el snapshot. No se validó otra cuenta.

El Compose siguiente usa proyecto/volúmenes nuevos, puerto5002 y loopback;
no iniciar ni alterar el Compose histórico5001. Los payloads de28 modelos
permanecen en los volúmenes locales pset3-experimentos_mlflow_backend y
pset3-experimentos_mlflow_artifacts. Clonar Git no los copia ni recrea los IDs.
No usar down -v. Después de autorizar OTRA ejecución, con al menos6GiB libres:

```powershell
docker compose --env-file .env.example -f docker-compose.experimentos.yml up -d mlflow
Invoke-WebRequest http://127.0.0.1:5002/health
.\.venv\Scripts\python.exe -B -X utf8 -m src.estudio_nuevo --execute --study-id snowflake-replica-01 --tracking-uri http://127.0.0.1:5002 --stage candidates --source snowflake --authorize-snowflake-read
```

El identificador debe ser nuevo. El ejecutor rechaza5001 y backends históricos.
La lectura valida esquema, claves, etiquetas y huellas de12.000 filas/20 columnas;
ordena las filas remotas por claves congeladas antes de aplicar los folds. Solo
consulta la OBT autorizada y INFORMATION_SCHEMA mediante SELECT; no publica ni
altera tablas. No usa Kestra/dbt/Spark ni representa la OBT90 del PSet2.

Se crean nuevos IDs en pset3_new_<study-id> y salidas ignoradas en estudios/.
Cada fold y refit registra parámetros efectivos, semilla, contrato/procedencia,
train/CV/sd/brecha/tiempo, warnings y pipeline con firma/ejemplo. Los padres son
agregadores. Semillas de fold20261117/18/19; refit20261117. La regla académica es
determinista, sin RNG; su seed contextual es20261117. SVC usa decision_function
con probability=False. El refit usa desarrollo solamente.

--resume-candidates permite continuación explícita de un estudio parcial:
verifica fuente/folds, runs terminados y ausencia de runs activos; reutiliza
folds FINISHED sin repetirlos y conserva los padres FAILED. No es retry
automático. La corrida inicial falló por categorical_features duplicado entre
contrato y estimador; se corrigió el logging nuevo con model__. La RAM bloqueó
posteriormente el refit de boosting; se recuperaron sus tres folds. Ambos grupos
de fallos permanecen documentados. MLP llegó a200 iteraciones sin converger en
sus cuatro ajustes; no se aumentó max_iter para alterar los defaults.

Compose limita MLflow a768MiB/una CPU; deshabilita el ejecutor opcional de jobs,
conservando tracking, Registry y protecciones HTTP. Fit/scoring usan dos hilos y
preflight6GiB antes de cada ajuste. El supervisor externo usado en la ejecución
original no forma parte del CLI: el preflight no garantiza un límite duro de RSS.
El tiempo fit excluye scoring, serialización y logging; sd entre folds no es IC.

Con cambios sin commit se registra UNAVAILABLE_WORKTREE_DIRTY, HEAD observado y
hash real de código, sin atribuir retrospectivamente un commit de publicación.
Los modelos originales, los396 runs y las90 configuraciones no se modifican.
No hay tuning de los mejores del estudio nuevo, nueva selección/aliases ni
nuevos scores de hold-out. La función de tuning preparada no acredita resultados
nuevos. SVM no sustituye la familia fundacional y prompting LLM sigue pendiente.
