# Metodo, contrato y limites de la simulacion v1

## Metodo realmente implementado

Monte Carlo de eventos de renovacion, implementado en src/sinteticos.py.
No se ajusto ningun parametro con datos municipales. Configuracion completa en
config/generacion_v1.json; todos los numeros son supuestos academicos.

Una tabla de diez obligaciones deterministas sirve para demostrar ingenieria,
pero no para estimar variacion fuera de muestra. Remuestrear esa tabla repetiria
su soporte; entrenar GAN/copulas sin una poblacion autorizada tampoco aportaria
una distribucion empirica defendible. La simulacion mecanistica permite conservar
movimientos individuales, explicitar dependencias y auditar la etiqueta. Su
robustez se refiere a reproducibilidad, coherencia y controles; no a fidelidad
municipal demostrada. No es un modelo tributario legal ni un gemelo digital.

1. Se generan seis cohortes anuales 2019-2024, 2.000 obligaciones cada una.
   Zona (tres categorias) y uso (residencial/comercial/mixto) son ficticios.
   Area lognormal, con limites de simulacion 20-4.500 m2; valor depende de area,
   zona, ruido lognormal y deriva nominal del 3,5% por cohorte. Importe principal
   = redondeo a centavos de valor * 0,0012, mas cargo fijo 12. Son formulas
   academicas; no se atribuyen al municipio. El plazo 30/45/60 es una covariable
   simulada sin efecto directo sobre intereses, sanciones o vencimientos legales.
2. Canal de notificacion cambia de mezcla por cohorte. Fechas uniformes en 365
   dias por ano: en bisiestos el ultimo dia queda fuera por convencion v1.
   Se mantienen fluctuaciones por mes y cohorte, y ruido individual no observado.
   La entidad ficticia de cada obligacion es independiente; no se modelan
   repeticiones longitudinales de contribuyentes o predios.
3. Se generan oportunidades de pago con esperas Weibull de forma 1,25 y escala
   base 58 dias, modificada moderadamente por variables de emision y frailty
   normal sd=0,95. Existe una masa probabilistica sin oportunidades. En cada
   oportunidad se paga el saldo completo o una fraccion Beta(2,3); no se sortea
   directamente la clase final. Maximo diez oportunidades en 365 dias.
4. Puede haber reverso exacto del pago previo (probabilidad 0,025), antes de una
   nueva oportunidad, y evento administrativo de importe cero (0,015). El reverso
   resta pago neto y reabre saldo; no produce importes negativos acumulados ni
   sobrepago. Administrativos no extinguen deuda en esta convencion. No se simula
   anulacion legal, remision, interes, descuento o multa municipal.
5. Faltantes MCAR de area (0,035) y canal (0,02) ejercitan preprocesamiento.
   No se inyectan IDs rotos, fechas imposibles ni etiquetas arbitrarias. Estos
   casos de calidad son coherentes y quedan diferenciados de eventos futuros.

NumPy SeedSequence deriva streams independientes por cohorte y por componente
features/eventos. La semilla principal es 20261117. El lote UUID5 depende de toda
la configuracion; IDs SHA-256 derivan exclusivamente de etiquetas ficticias.
Dos generaciones completas y readback Parquet dieron iguales huellas SHA-256
incluyendo tipos. No hay HMAC municipal ni acceso a registros reales.
[NumPy documenta el uso de SeedSequence](https://numpy.org/doc/stable/reference/random/parallel.html).

## Etiqueta y convenciones

TARGET_NO_PAGADA_90D = 1 cuando total en centavos - suma neta de movimientos
hasta el dia 90 es estrictamente positivo; 0 si el saldo es cero. Pagos del dia
90 cuentan, los del 91 no. Todas las fechas simuladas son dias calendario a
00:00 UTC, sin horas intradia: no se interpreta una politica municipal de cierre.
Etiqueta disponible al dia siguiente del cierre (emision+91). Sin horizonte
completo => NA, nunca clase 1 por ausencia de observacion. Poblacion principal
completamente madura al 31-dic-2025. Tests conocidos cubren pago dia 90, pago dia
91, parcial, sin eventos, reverso y censura. Dinero autoritativo en enteros de
centavos; se conservan eventos individuales para reauditar cualquier corte.

## Tamano, calidad y limitaciones estadisticas

12.000 filas permiten reservar 2.000 de 2024 y obtener mas de 1.500 por ventana
CV sin llegar al limite de lectura local de 100.000. Con 2.000 observaciones,
el error estandar binomial maximo de una proporcion seria sqrt(.25/2000)=1,12
puntos porcentuales bajo independencia. No es una garantia del error de AUC ni
un intervalo de desempeno municipal. Tres folds dan una primera medida de
variacion, no una prueba de significancia ni intervalos independientes.

Se ejecutan 7 pipelines x (3 folds + 1 refit desarrollo), con RF limitado a 120
arboles y dos jobs, MLP pequeno y SVM sobre menos de 10.000 filas. El presupuesto
posterior de 3 familias x 30 configuraciones x 3 folds implica al menos 270
ajustes: medir tiempos antes de ampliar poblacion. No se aumentaron muestras
para mejorar metricas ni se reescogio semilla tras observar resultados.

Controles: cardinalidad, claves, FKs, fechas, signos, acumulados, saldos, clases
por cohorte (guardas predefinidas 10%-90%), faltantes, ausencia de duplicados,
huellas reproducibles y readback. Reporte exacto: generacion_validacion.json.
Control de trivialidad en desarrollo: AUC univariada numerica orientada <0,95;
control negativo con etiquetas train permutadas y auditoria de whitelist de
features. No prueban ausencia de toda fuga posible. La señal predictiva es
moderada/debil y ruidosa; no se fuerza una AUC minima ni se ocultan malos modelos.

## Features, EDA y validacion

EDA usa 9.510 filas elegibles de desarrollo. No usa el hold-out para elegir
variables, transformar o ajustar hiperparametros. Describe tipos, completitud,
IQR/boxplots, Z-score y Spearman (cola no normal y relacion monotona); incluye
tres hallazgos cuantificados y consecuencias. Los graficos y tablas proceden
realmente de la simulacion, no son ejemplos ilustrativos.

Features: log1p(importe), log1p(area), plazo, seno/coseno de mes, zona, uso y canal.
Las cuatro transformaciones son deterministas por fila de emision. Avalúo y
principal se excluyen por redundancia con importe; ano/cohorte, IDs, lote,
metadatos, saldos, pagos, movimientos, disponibilidad de etiqueta y target no
entran en X. La lista explicita es comprobada por FeatureGuard. Imputacion,
escalado y codificacion se ajustan dentro de cada train/fold. Este protocolo
sigue la recomendacion de [Pipeline para prevenir fuga por preprocesamiento](https://scikit-learn.org/stable/common_pitfalls.html).

Corte final 1-ene-2024: 9.510 desarrollo, 2.000 hold-out, 490 excluidas por
etiqueta no disponible antes del corte. CV expansiva valida 2021, 2022 y 2023,
con entrenamiento etiquetado antes del inicio de cada ventana. La tercera
validacion usa 1.510 filas elegibles de 2023, no todas las emisiones de ese ano.
No existe particion aleatoria adicional. Claves, indices y folds quedan en
partitions.json junto con la huella. La comparacion temporal tiene
[dependencias distintas de K-fold aleatoria](https://scikit-learn.org/stable/modules/cross_validation.html).
El hold-out se consulto despues de congelar la seleccion y los umbrales OOF. Las metricas finales se conservan en cierre/holdout_resultados.json; no se reutiliza para ajustar decisiones.


## Experimentos, linaje y limites

Dos baselines y cinco familias convencionales comparten el protocolo temporal.
Se completaron 30 configuraciones de logistica, 30 de MLP y 30 de RF, con tres
folds por configuracion. Los resultados completos estan en cierre/ y corridas.csv.
El champion es logistica por CV; el challenger es MLP. Su mayor AUC de hold-out
no se uso para reconsiderar el ganador. FP+3FN es un costo relativo academico:
ambos umbrales finales marcan todos los casos, sin mejora operacional demostrada.

La fuente realmente utilizada fue Parquet local. Esta OBT de 20 columnas se
construyo con pandas desde obligaciones/eventos; no recorrio Kestra, dbt o Spark
ni es la OBT fisica de 90 columnas del PSet2. Se conservan convenciones analiticas
y grano obligacion+lote. No se incluye ni necesita infraestructura municipal.
El mismo snapshot se publico y releyo posteriormente en [Snowflake](snowflake.md),
con estado READBACK_VERIFIED; las corridas originales conservan origen LOCAL_PARQUET.

TabPFN completo un fold y se detuvo por presion de memoria; no tiene media CV
comparable. Ver fundacional_resultado.json y [atribucion](cierre/TabPFN_atribucion.md).
LLM: [justificacion y plantilla](cierre/llm_justificacion.md), sin prompting ejecutado.

Las 396 corridas preceden al repositorio; conservan UNAVAILABLE_PSET3_NOT_VERSIONED
y sus hashes reales. Un commit futuro identifica la publicacion, no retroactivamente
los entrenamientos. CSV y metadatos originales no son un backend MLflow completo.

La simulacion no esta calibrada con registros reales, supone entidades independientes,
faltantes MCAR y mecanismos simplificados; usa una sola poblacion principal.
Reproducibilidad no implica fidelidad municipal ni eficacia sobre contribuyentes.
