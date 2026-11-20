# Tratamiento del apartado LLM mediante prompting

**Estado: justificacion tecnica, no experimento LLM ejecutado.** Si la rubrica
exige necesariamente una ejecucion de prompting, ese requisito sigue pendiente.
Un modelo fundacional tabular y un LLM no son equivalentes; SVM tampoco lo es.

El objetivo depende de ocho covariables tabulares de emision, algunas con codigos
ficticios arbitrarios, sin texto libre ni semantica municipal que recuperar.
El generador tiene frailty no observable y eventos futuros estocasticos: un LLM
no puede conocerlos por razonamiento linguistico. Un prompt zero-shot impondria
priors de lenguaje que no estan calibrados con estas reglas sinteticas. No hay
un valor predictivo demostrable de reformular numeros como texto.

Few-shot requeriria elegir ejemplos SOLO del train de cada fold (3513, 5487,
7514 filas disponibles), sin saldo/pagos ni consultas externas municipales,
controlar contexto, formato de numeros, faltantes, orden, balance de ejemplos y
version/hash de pesos. No se puede comparar una evaluacion de pocas filas con
los 5510 casos CV de los demas candidatos como si fuera el mismo protocolo.
Se necesitarian scores comparables para ROC-AUC, no solo una etiqueta o una
probabilidad textual no calibrada. Hay que registrar salidas invalidas y latencia.

Protocolo propuesto (NO ejecutado): inferencia local, temperatura cero y seed
cuando aplique; mismos folds; ejemplos seleccionados con semilla solo del train;
plantilla versionada y campo score de clase 1 validado. No optimizar el prompt
con hold-out. No usar un servicio de pago sin cuenta/presupuesto definidos.
En esta tarea no se configuro un proveedor/pesos LLM ni se envio dataset a una
API externa. Se priorizo la comparacion tabular reproducible y se preservo el
requisito como justificacion, sin inventar outputs, tiempos o exactitud.

Plantilla concreta de referencia:

```text
Sistema: Clasifica una obligacion COMPLETAMENTE FICTICIA al emitir. Clase 1:
no pagada completamente al dia 90; clase 0: pagada completamente. No tienes
movimientos futuros ni datos municipales. Usa solo los campos de emision.
Ejemplos: {ejemplos_extraidos_solo_del_train_del_fold}
Caso: {importe_emision, area_m2_o_null, plazo_simulado, mes_emision,
       zona_simulada, uso_predio, canal_notificacion_o_null}
Devuelve exclusivamente JSON {"class": 0|1, "score_class_1": numero_0_a_1}.
```

La plantilla ilustra el diseno, no resultados. No constituye una razon para
reemplazar la etiqueta de eventos por lo que responda un LLM. El impedimento de
TabPFN se documenta separadamente: instalacion/pesos funcionaron, comparacion
completa limitada por recursos, no falta de autorizacion para datos ficticios.

La justificacion responde a la opcion de discutir adecuacion permitida por el
enunciado. No prueba que un LLM no pueda funcionar ni su inferioridad empirica;
la frailty limita tambien modelos tabulares. No se ejecuto prompting.

## Alcance contrastado con el enunciado oficial

La pagina4 permite explicar la exclusion si el problema no se puede plantear
mediante prompting. La plantilla anterior SI formula un caso tabular: falta de
texto libre o costo no prueban imposibilidad. Esta discusion no acredita por si
sola esa condicion literal; falta aceptacion docente o un experimento prompting
comparable autorizado. No se presenta como requisito cerrado ni como prueba de
inferioridad de LLM. Este requisito permanece pendiente.
