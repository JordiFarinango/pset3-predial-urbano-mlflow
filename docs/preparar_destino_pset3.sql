-- PROPUESTA MANUAL, NO EJECUTADA. Administrador de una cuenta propia.
-- No reutilizar roles/claves municipales. No concede privilegios globales.
CREATE DATABASE IF NOT EXISTS PSET3_SINTETICO;
CREATE SCHEMA IF NOT EXISTS PSET3_SINTETICO.EXPERIMENTOS;
-- Administrador: crear/asignar usuario propio, clave publica, rol propio y warehouse.
-- El rol de publicacion requiere USAGE en base/esquema/warehouse,
-- CREATE TABLE en EXPERIMENTOS y propiedad de los objetos que cree.
-- CREATE SCHEMA solo si el esquema no existe (es preferible prepararlo aqui).
-- Completar GRANTs con identificadores aprobados; no hay cuentas personales por defecto.
-- No se usa DROP, TRUNCATE, DELETE, OR REPLACE ni se referencia una base PSet2.
