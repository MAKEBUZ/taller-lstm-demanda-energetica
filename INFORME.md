# Taller: arquitecturas LSTM para demanda energética

Autor: Diego Alejandro Ocampo Madroñero.

## 1. Objetivo

Predecir la demanda eléctrica de la hora siguiente a partir de una ventana de observaciones históricas multivariadas. Se implementan tres arquitecturas (LSTM base, LSTM profunda y LSTM con atención) y se comparan ventanas de 12, 24 y 48 horas.

## 2. Datos y limpieza

El archivo original contiene 17 550 filas entre 2025-01-01 y 2026-12-31. Se retiraron 30 duplicados exactos, quedando 17 520 horas únicas sin huecos temporales. Había 80 etiquetas de `periodo_dia` con espacios o variaciones de mayúsculas; se normalizaron con `strip` y minúsculas. Se detectaron faltantes en temperatura (70), humedad (70), viento (71), precio (70) y objetivo (1, correspondiente a la última hora).

La limpieza marca como inválidas mediciones imposibles: demanda no positiva, humedad fuera de 0–100 %, viento, radiación o precipitación negativos, y periodo del día desconocido. Además se aplican límites conservadores de cuatro rangos intercuartílicos a demanda, temperatura, viento y precio; sus cuartiles se estiman **solo con el tramo de entrenamiento**. Los precios negativos no se clasifican automáticamente como imposibles, porque pueden existir en un mercado eléctrico. Ningún valor anómalo se reemplaza o inventa. Una ventana se descarta si contiene al menos una entrada inválida o un objetivo ausente.

Los archivos `resultados/dataset_limpio_auditado.csv` y `resultados/auditoria.json` permiten auditar cada regla. La limpieza no elimina filas del eje temporal; las marca para impedir que al crear ventanas se unan horas que antes no eran consecutivas.

## 3. Entradas, objetivo y normalización

La etiqueta de la fila de tiempo *t*, `demanda_objetivo`, representa la demanda de *t + 1 hora*. No se incluye entre las entradas. Las entradas contienen demanda observada, temperatura, humedad, viento, radiación, precipitación, precio, festivo y variables de calendario. Hora, día de la semana y mes se codifican con seno y coseno; `periodo_dia` se codifica en cuatro indicadores. Las variables de calendario se derivan de `timestamp` para evitar contradicciones en columnas redundantes del CSV.

Las medias y desviaciones estándar de las entradas se estiman únicamente sobre entradas válidas de entrenamiento. La escala del objetivo se estima únicamente sobre etiquetas de entrenamiento. Los parámetros quedan en `resultados/normalizacion.json`; los valores transformados, en `resultados/dataset_normalizado.csv`. Se aplica el mismo escalado en validación, prueba y app web, sin reajustarlo.

## 4. División temporal y ventanas

Se divide según la hora del **objetivo**, no la última hora de entrada:

| Partición | Hora objetivo |
|---|---|
| Entrenamiento | Hasta 2026-03-31 23:00 |
| Validación | 2026-04-01 a 2026-06-30 |
| Prueba | Desde 2026-07-01 |

Cada secuencia contiene 12, 24 o 48 horas consecutivas y predice la siguiente. Una secuencia de validación o prueba puede consultar historia previa al comienzo de su partición, como sucedería en uso real. No se utiliza `train_test_split` aleatorio ni se mezcla una etiqueta futura dentro de entradas de entrenamiento.

Para comparar los nueve modelos de forma justa, la puntuación de todos se calcula sobre las **mismas 531 horas objetivo de validación y 1 854 de prueba**, que son las disponibles también para la ventana de 48 horas. Las secuencias de entrenamiento válidas son 8 677, 6 798 y 4 147 para ventanas de 12, 24 y 48 horas, respectivamente.

## 5. Arquitecturas y entrenamiento

| Modelo | Arquitectura | Justificación |
|---|---|---|
| Base | LSTM(32) → Dense(1) | Referencia simple con memoria temporal. |
| Profunda | LSTM(48) → Dropout(0,2) → LSTM(24) → Dense(1) | Mayor capacidad para patrones temporales; Dropout reduce dependencia de unidades concretas. |
| Propuesta | LSTM(32, secuencia) → atención temporal → promedio global → Dense(16) → Dense(1) | Permite ponderar relaciones entre horas observadas dentro de la ventana. |

Se usa Adam con tasa 0,001, pérdida MSE, lotes de 256, máximo 15 épocas y Early Stopping con paciencia de tres épocas sobre pérdida de validación, restaurando los mejores pesos. Se fija la semilla 42. Las nueve combinaciones se entrenan con el mismo protocolo. El modelo destinado a la app se elige mediante **MAE de validación**; prueba queda reservada para la evaluación final. Los pesos se guardan en formato NumPy y se cargan en la misma arquitectura definida en el código.

## 6. Resultados

La tabla completa se genera en `resultados/tabla_resultados.csv`. Los errores se calculan en MW, tras revertir la escala del objetivo. Cada fila usa las mismas horas de evaluación.

| Modelo | Ventana | MAE | MSE | RMSE | MAPE % | R² | Épocas |
|---|---:|---:|---:|---:|---:|---:|---:|
| Base | 12 | 25,34 | 1053,63 | 32,46 | 4,28 | 0,883 | 10 |
| Base | 24 | 23,94 | 974,02 | 31,21 | 3,97 | 0,892 | 9 |
| **Base** | **48** | **22,83** | **867,22** | **29,45** | **3,81** | **0,904** | **15** |
| Profunda | 12 | 23,17 | 864,60 | 29,40 | 3,85 | 0,904 | 15 |
| Profunda | 24 | 42,82 | 2679,39 | 51,76 | 6,88 | 0,702 | 6 |
| Profunda | 48 | 22,83 | 844,32 | 29,06 | 3,80 | 0,906 | 15 |
| Atención | 12 | 25,39 | 1054,13 | 32,47 | 4,31 | 0,883 | 9 |
| Atención | 24 | 38,90 | 2392,69 | 48,92 | 6,30 | 0,734 | 8 |
| Atención | 48 | 52,90 | 4206,09 | 64,85 | 8,89 | 0,533 | 11 |

La referencia de **persistencia** (predecir que la demanda siguiente será igual a la observada ahora) obtuvo MAE 37,59 MW, RMSE 49,52 MW y R² 0,728 en las mismas horas. La LSTM base de 48 horas reduce su MAE aproximadamente **39,3 %**. Su MAE de validación fue 22,86 MW; la diferencia de 0,03 MW respecto a prueba sugiere buena generalización en este corte temporal. La LSTM profunda de 48 horas tiene MAE de prueba prácticamente igual, pero validación ligeramente peor (23,08 MW) y mayor complejidad; por eso se conserva la base.

Las cuatro gráficas exigidas son `curvas_loss.png`, `real_vs_predicha.png`, `error_prediccion.png` y `comparacion_modelos.png`. La propuesta con atención empeoró al aumentar la ventana; su inclusión es un resultado experimental, no una mejora garantizada.

## 7. Preguntas de análisis

1. **¿Por qué LSTM?** Porque recibe secuencias ordenadas y puede conservar información de horas anteriores, útil para ciclos diarios, persistencia y efectos meteorológicos. Su utilidad concreta se juzga con las métricas frente a alternativas simples.
2. **¿Qué efecto tiene la ventana?** En la arquitectura base, el MAE de prueba bajó de 25,34 MW (12 h) a 23,94 MW (24 h) y 22,83 MW (48 h). Sin embargo, la ventana mayor deja menos ejemplos de entrenamiento válidos (4 147 frente a 8 677) y exige más historial para desplegar. En otras arquitecturas, ampliar la ventana no garantizó mejora.
3. **¿Hay sobreajuste?** En `curvas_loss.png` hay inestabilidad y separación entre entrenamiento y validación, especialmente en la LSTM profunda de 24 h y la de atención de 24–48 h; sus resultados de prueba son peores. En la LSTM base de 48 h ambas curvas descienden juntas y los MAE de validación y prueba (22,86 y 22,83 MW) son cercanos. No hay señal fuerte de sobreajuste en el modelo recomendado, aunque un solo corte temporal no descarta degradación futura.
4. **¿Qué es Dropout?** Desactiva aleatoriamente una fracción de activaciones durante entrenamiento. En la LSTM profunda se usa 0,2 para reducir coadaptación y probar si mejora la generalización.
5. **¿Qué es Early Stopping?** Detiene el entrenamiento cuando la pérdida de validación deja de mejorar durante tres épocas; se restauran los mejores pesos para evitar conservar un estado posterior con peor validación.
6. **¿Más neuronas implican mejores resultados?** No. La LSTM profunda de 24 h obtuvo MAE 42,82 MW, frente a 23,94 MW de la base de 24 h. Con 48 h ambas resultaron cercanas, sin ventaja suficiente para justificar la mayor complejidad de la profunda.
7. **¿Qué modelo recomendar para producción?** La LSTM base con ventana de 48 h: menor MAE de validación (22,86 MW), MAE de prueba 22,83 MW, curva estable y arquitectura más sencilla que la profunda. Se registra en `resultados/mejor_modelo.json`. Antes de un uso operativo haría monitoreo de errores y reentrenamiento periódico.
8. **¿Qué variables influyen más?** Al permutarlas en validación, los mayores aumentos de MAE fueron `fin_semana` (+15,59 MW), `demanda_mw` (+12,58 MW), `radiacion_wm2` (+10,38 MW) y los indicadores de periodo del día (+7–8 MW). La tabla íntegra está en `resultados/importancia_permutacion_validacion.csv`. La permutación mide dependencia predictiva, no causalidad; variables de calendario correlacionadas comparten información.
9. **¿Qué limitaciones hay?** Datos sintéticos o específicos del periodo, anomalías y faltantes, incertidumbre fuera del rango de entrenamiento, ausencia de eventos no modelados y degradación ante cambios estructurales. Las predicciones requieren monitoreo y actualización antes de uso operativo.

## 8. Despliegue

`app.py` permite cargar un CSV o editar una tabla con las últimas horas observadas, valida continuidad y rangos físicos, aplica el escalado guardado y muestra la demanda predicha para la hora siguiente. La interfaz no solicita `demanda_objetivo` ni valores futuros. Se inicia con `streamlit run app.py` después del entrenamiento.

La app se probó localmente: inició sin excepciones y produjo una predicción finita al pulsar el botón. Para disponer de un enlace accesible desde otro computador, el repositorio está preparado para Streamlit Community Cloud; las instrucciones se encuentran en `README.md`. La publicación externa requiere cargar el proyecto en una cuenta de GitHub y crear la app en la cuenta de Streamlit del estudiante.
