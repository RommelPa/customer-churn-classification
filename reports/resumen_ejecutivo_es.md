# Resumen Ejecutivo — Clasificación de Churn de Clientes

## 1. Objetivo

Este proyecto construye y evalúa modelos de clasificación para predecir abandono de clientes.

El objetivo no es solo clasificar clientes, sino también evaluar el riesgo de churn con métricas relevantes para negocio, analizar umbrales de decisión y generar recomendaciones de retención.

## 2. Contexto de negocio

El churn ocurre cuando un cliente deja de usar el servicio de una empresa.

En negocios basados en suscripción, predecir churn permite priorizar campañas de retención, reducir pérdida de ingresos y mejorar el valor de vida del cliente.

En este problema, la accuracy no es suficiente. Un modelo puede tener accuracy alta prediciendo siempre la clase mayoritaria, pero fallar completamente en detectar clientes que abandonan.

El problema de negocio requiere balancear:

- capturar la mayor cantidad posible de clientes que abandonan,
- evitar demasiados falsos positivos,
- ajustar el umbral a la capacidad real del equipo de retención.

## 3. Alcance del dataset

El proyecto usa el dataset Telco Customer Churn.

El dataset contiene:

| Dataset | Filas | Columnas |
|---|---:|---:|
| Datos crudos | 7,043 | 21 |
| Datos limpios | 7,043 | 22 |
| Partición de entrenamiento | 5,634 | 22 |
| Partición de validación | 1,409 | 22 |

La variable objetivo es `Churn`.

La variable binaria de modelado es `ChurnLabel`, donde:

| Etiqueta | Significado |
|---:|---|
| 0 | No churn |
| 1 | Churn |

## 4. Distribución del target

La distribución muestra un desbalance moderado:

| Clase | Proporción |
|---|---:|
| No churn | 73.46% |
| Churn | 26.54% |

Un modelo ingenuo que siempre predice “No churn” alcanzaría 73.46% de accuracy, pero detectaría cero clientes churn.

Por esta razón, la evaluación se enfoca en precision, recall, F1-score, ROC-AUC, Average Precision, matrices de confusión y análisis de umbrales.

## 5. Hallazgos de calidad de datos

El dataset no tiene filas duplicadas ni IDs de cliente duplicados.

El principal problema de calidad de datos es `TotalCharges`, que aparece como texto y contiene 11 valores en blanco.

Todos los registros con `TotalCharges` en blanco tienen `tenure = 0`, lo que indica clientes recién registrados sin cargos acumulados.

La regla de limpieza usada fue:

```text
TotalCharges en blanco con tenure = 0 → TotalCharges = 0
```

Esto preserva el significado de negocio del dato, en lugar de imputar con media o mediana.

## 6. Señales iniciales de negocio

La auditoría mostró patrones claros:

| Segmento | Tasa de churn |
|---|---:|
| Contrato month-to-month | 42.71% |
| Contrato two-year | 2.83% |
| Servicio de internet Fiber optic | 41.89% |
| Pago con Electronic check | 45.29% |
| Sin OnlineSecurity | 41.77% |
| Sin TechSupport | 41.64% |
| Paperless billing activado | 33.57% |

Estos patrones sugieren que el churn se asocia con tipo de contrato, tipo de servicio, servicios de soporte/seguridad y método de pago.

## 7. Modelos comparados

Se evaluaron los siguientes modelos:

| Modelo | Propósito |
|---|---|
| Baseline de clase mayoritaria | Punto mínimo de comparación |
| Logistic Regression | Clasificador lineal interpretable |
| Decision Tree | Clasificador no lineal simple |
| Random Forest | Modelo de ensamble |
| Gradient Boosting | Modelo de ensamble boosting |

## 8. Resultados en validación con umbral 0.50

Con el umbral estándar de 0.50, Gradient Boosting obtuvo el mejor desempeño de ranking:

| Modelo | Accuracy | Precision | Recall | F1 | ROC-AUC | Average Precision |
|---|---:|---:|---:|---:|---:|---:|
| Gradient Boosting | 0.8048 | 0.6689 | 0.5241 | 0.5877 | 0.8442 | 0.6631 |
| Random Forest | 0.7615 | 0.5354 | 0.7674 | 0.6308 | 0.8433 | 0.6507 |
| Logistic Regression | 0.7381 | 0.5043 | 0.7834 | 0.6136 | 0.8416 | 0.6327 |
| Decision Tree | 0.7452 | 0.5131 | 0.7861 | 0.6209 | 0.8332 | 0.6230 |
| Baseline | 0.7346 | 0.0000 | 0.0000 | 0.0000 | 0.5000 | 0.2654 |

La accuracy del baseline es engañosa porque no detecta ningún cliente churn.

## 9. Resultados de validación cruzada

La validación cruzada confirmó que Gradient Boosting es el modelo más fuerte para ordenar riesgo de churn.

| Modelo | Accuracy promedio | ROC-AUC promedio | Average Precision promedio |
|---|---:|---:|---:|
| Gradient Boosting | 0.8046 | 0.8471 | 0.6616 |
| Random Forest | 0.7657 | 0.8465 | 0.6576 |
| Logistic Regression | 0.7456 | 0.8450 | 0.6555 |
| Decision Tree | 0.7292 | 0.8306 | 0.6218 |
| Baseline | 0.7346 | 0.5000 | 0.2654 |

Gradient Boosting fue seleccionado como modelo final porque obtuvo el mejor ROC-AUC y Average Precision en validación cruzada.

## 10. Análisis de umbral

El umbral estándar de 0.50 no es ideal para retención.

| Estrategia | Umbral | Precision | Recall | Clientes marcados | Churners capturados | Churners perdidos |
|---|---:|---:|---:|---:|---:|---:|
| Default | 0.50 | 0.6689 | 0.5241 | 293 | 196 | 178 |
| Balanced F1 | 0.24 | 0.5143 | 0.8182 | 595 | 306 | 68 |
| Retention Recall | 0.29 | 0.5306 | 0.7647 | 539 | 286 | 88 |
| Efficient Precision | 0.55 | 0.7046 | 0.4465 | 237 | 167 | 207 |

El umbral recomendado para este proyecto es **0.24**, porque ofrece el mejor balance entre precision y recall y reduce de forma importante los churners perdidos.

Comparado con el umbral 0.50, el umbral 0.24:

- captura 110 churners adicionales,
- reduce los churners perdidos de 178 a 68,
- aumenta los clientes marcados de 293 a 595,
- aumenta los falsos positivos de 97 a 289.

Esto es un trade-off de negocio, no solo una decisión técnica.

## 11. Decisión final del modelo

El modelo final es:

```text
Gradient Boosting Classifier
```

El umbral recomendado es:

```text
0.24
```

Esta configuración es adecuada cuando el negocio quiere priorizar detección de churn y puede gestionar una lista mayor de clientes para campañas de retención.

Si el equipo de retención tiene capacidad limitada, puede usarse un umbral más alto como 0.55 para priorizar menos clientes con mayor probabilidad de churn.

## 12. Interpretación del modelo

La importancia de variables de Gradient Boosting indica que las variables más predictivas incluyen:

- contrato month-to-month,
- tenure,
- TotalCharges,
- servicio de internet Fiber optic,
- MonthlyCharges,
- ausencia de OnlineSecurity,
- ausencia de TechSupport,
- pago con Electronic check.

Los coeficientes de Logistic Regression sugieren asociaciones positivas con churn en:

- servicio Fiber optic,
- contrato month-to-month,
- mayor TotalCharges,
- servicios de streaming,
- pago con Electronic check,
- ausencia de OnlineSecurity,
- ausencia de TechSupport.

Las asociaciones negativas incluyen:

- mayor tenure,
- contrato two-year,
- servicio DSL,
- no tener servicio de internet,
- no usar paperless billing,
- tener dependents.

Estos hallazgos deben interpretarse como asociaciones, no como efectos causales.

## 13. Recomendaciones de negocio

1. Usar el modelo para priorizar campañas de retención, no como sistema automático de decisión.
2. Usar el umbral 0.24 cuando el negocio quiera maximizar la detección de churn.
3. Usar un umbral más alto cuando la capacidad del equipo de retención sea limitada.
4. Enfocar campañas en clientes month-to-month, con bajo tenure y patrones de servicio de alto riesgo.
5. Investigar por qué los clientes con Fiber optic tienen mayor churn.
6. Mejorar ofertas de soporte y seguridad, porque la ausencia de OnlineSecurity y TechSupport se asocia fuertemente con churn.
7. Medir costo de campaña, tasa de retención y valor de vida del cliente para refinar el umbral.

## 14. Limitaciones

- El dataset es estático y no incluye comportamiento temporal del cliente.
- El modelo no incluye tickets de soporte, quejas, satisfacción del cliente, precios de competidores ni historial de campañas de retención.
- La selección del umbral debe ajustarse con costos reales de negocio y capacidad del equipo.
- La interpretación de variables es asociativa, no causal.
- El modelo estima probabilidad de churn, pero no explica por sí solo por qué un cliente específico abandonará.

## 15. Próximos pasos

- Agregar optimización de umbral sensible a costos usando customer lifetime value y costo de campaña.
- Comparar desempeño con probabilidades calibradas.
- Agregar explicabilidad con SHAP en una versión futura.
- Construir un dashboard para equipos de retención.
- Desplegar una API de scoring de churn en un proyecto posterior enfocado en producción.
