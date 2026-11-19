# Snowflake: publicacion y relectura verificadas

El snapshot sintetico congelado ya esta publicado y verificado fisicamente en
PSET3_SINTETICO.EXPERIMENTOS. Se completaron realmente:

```powershell
python -m src.publicar_snowflake --execute
python -m src.publicar_snowflake --verify
```

Estado de publicacion PUBLISHED_AND_READBACK_VERIFIED; relectura posterior
READBACK_VERIFIED. [Evidencia publica](snowflake_evidencia.json).

| Tabla en PSET3_SINTETICO.EXPERIMENTOS | Filas | Columnas |
|---|---:|---:|
| OBLIGACIONES_2B7D72B7735E | 12.000 | 15 |
| MOVIMIENTOS_2B7D72B7735E | 18.988 | 6 |
| OBT_PREDIAL_2B7D72B7735E | 12.000 | 20 |

Ambos recibos originales concuerdan en tablas, conteos, esquema fisico y
reconciliacion. Coinciden las huellas del snapshot de entrenamiento y los hashes
logicos pset3-logical-v1, claves, nulos y etiquetas por obligacion. La OBT conserva
6.865 clase 1 y 5.135 clase 0, tambien conciliadas por cohorte. Lecturas acotadas a
N+1 con proyeccion explicita; no se publican registros individuales en el recibo.

El JSON resume campos permitidos de los recibos originales snowflake_publicacion.json
y snowflake_relectura.json conservados en pset_3/docs/ fuera de este repositorio.
Incluye sus SHA-256, estados, query IDs de ambas etapas, esquema, nombres de tablas
y hashes. No incluye cuenta, usuario, rol, warehouse, claves o secretos. Los query
IDs permiten al responsable autorizado contrastar el historial de su propia cuenta;
no otorgan acceso publico a Snowflake. Los recibos no contienen timestamp de
comprobacion: no se infiere uno del UUID ni se certifica disponibilidad permanente.

La comprobacion de esta incorporacion fue offline: se contrastaron ambos recibos
con los Parquet congelados. No se repitieron cargas o consultas remotas.

## Linaje experimental

Las 396 corridas originales utilizaron LOCAL_PARQUET. La publicacion posterior
valida transporte, persistencia y reconciliacion, sin cambiar retroactivamente
su procedencia ni sus metricas. Esta OBT de 20 columnas se construyo con pandas
desde obligaciones/eventos Monte Carlo; no recorrio Kestra, dbt o Spark ni equivale
a la OBT fisica de 90 columnas del PSet2. La memoria definitiva incluye este hito en su pagina 6.
Los JSON historicos de generacion/experimentos conservan los estados que registraron
entonces; el presente recibo documenta la publicacion posterior.

## Reproducir con una cuenta academica propia

1. Crear .env local desde .env.example, configurando cuenta, usuario, rol,
   warehouse y ruta de clave cifrada propios. No usar secretos municipales.
   La clave debe estar fuera de este repositorio y de los proyectos originales;
   la passphrase se solicita mediante getpass y no se versiona.
2. Mantener PSET3_SINTETICO.EXPERIMENTOS y clasificacion SYNTHETIC. El administrador
   prepara base/esquema con el [SQL auxiliar](preparar_destino_pset3.sql) y concede
   USAGE en base/esquema/warehouse y CREATE TABLE en EXPERIMENTOS.
3. Revisar primero el plan sin conectar: `python -m src.publicar_snowflake`.
4. En destino nuevo autorizado usar --execute. Si el snapshot ya existe, usar
   --verify: el publicador no sobrescribe ni reemplaza tablas y aborta ante estado
   parcial. DDL puede haberse confirmado; revisar manualmente antes de reintentar.

La configuracion en otra cuenta y su ejecucion no se han demostrado. Los recibos
locales generados por esas ejecuciones no deben reemplazar la evidencia original
sin una revision y trazabilidad nuevas. No se requiere SQL Server o VPN municipal.
