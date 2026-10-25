# Evaluación y validación de modelos

Proyecto académico para comparar modelos de clasificación de sentimiento en el conjunto Twitter US Airline Sentiment. Se comparan un clasificador base, Regresión Logística y Random Forest con TF-IDF; el F1 macro es la métrica principal por el desequilibrio entre clases.

## Datos

El análisis espera un archivo local `Tweets.csv` con las columnas `text` y `airline_sentiment`. El archivo entregado contiene 14,640 registros. No se incluye en el control de versiones. La colección se conoce como Twitter US Airline Sentiment y se distribuye en Kaggle/CrowdFlower; véase la ficha del conjunto en https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment.

La preparación elimina filas sin texto o etiqueta, textos vacíos, etiquetas fuera de las tres clases esperadas y grupos de textos idénticos con etiquetas contradictorias. Los textos idénticos con una sola etiqueta se reducen a una observación. Las cantidades excluidas se guardan en `resultados/resumen.json`.

## Estructura

```text
src/evaluacion_modelos.py      Evaluación, validación y exportación de resultados
src/prueba_ab.py               Simulación A/B sintética y analizador de CSV real
tests/test_project.py          Pruebas automáticas
notebooks/evaluacion_validacion_modelos.ipynb
resultados/                    CSV, JSON y gráficos exportados
reporte/Reporte_Tecnico_Evaluacion_Validacion.docx
plantillas/                    Esquema vacío para una futura prueba A/B real
Tweets.csv                     Archivo local del conjunto, no versionado
```

## Instalación local

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Activa Jupyter con:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab
```

## Ejecución

Desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m src.evaluacion_modelos --data Tweets.csv --output resultados
.\.venv\Scripts\python.exe -m src.prueba_ab --output resultados
.\.venv\Scripts\python.exe -m src.generar_reporte
.\.venv\Scripts\python.exe -m pytest -q
```

El primer comando crea métricas, validación cruzada, reportes por clase, matrices de confusión, curvas ROC y análisis de umbrales. El segundo crea una **SIMULACIÓN CON DATOS SINTÉTICOS** reproducible. El analizador real se ejecuta indicando `--real-csv archivo.csv`; espera `grupo`, `caso_id`, `negativo_real` y `negativo_detectado`.

La plantilla vacía está en `plantillas/plantilla_ab_real.csv`. Cuando haya observaciones reales, complétala y ejecuta:

```powershell
.\.venv\Scripts\python.exe -m src.prueba_ab --output resultados --real-csv plantillas\plantilla_ab_real.csv
```

## Google Colab

1. Sube el proyecto como ZIP a Colab y extrae su contenido, o abre el notebook después de publicar el repositorio.
2. Si Colab no encuentra los archivos fuente, carga `evaluacion_modelos.py`, `prueba_ab.py` y `generar_reporte.py` desde `src/` cuando el notebook lo solicite.
3. Sube `Tweets.csv` cuando se solicite.
4. Ejecuta las celdas en orden. El notebook instala las dependencias, inspecciona el dataset, llama a los módulos del proyecto y muestra las tablas, gráficas y el reporte.

No se requieren rutas locales exclusivas de esta computadora.

## Ejecución local del notebook

Desde la carpeta raíz del proyecto, abre el notebook con:

```powershell
.\.venv\Scripts\python.exe -m jupyter lab notebooks\evaluacion_validacion_modelos.ipynb
```

`Tweets.csv` debe estar en la carpeta raíz. El notebook ya contiene una ejecución local guardada con salidas visibles. Para repetirla, elige **Restart Kernel and Run All Cells**. Las celdas vuelven a generar los archivos en `resultados/` y el documento en `reporte/`.

El notebook presenta cada etapa y reutiliza las funciones de `src/`, donde se mantiene la implementación principal para evitar duplicar la lógica de evaluación.

## Repositorio en GitHub

El proyecto se publica en [Papila9/actividad6-evaluacion-validacion](https://github.com/Papila9/actividad6-evaluacion-validacion). El repositorio contiene código, notebook, reporte y resultados derivados. `Tweets.csv` y `.venv` quedan excluidos por `.gitignore`.

Para obtener el proyecto:

```powershell
git clone https://github.com/Papila9/actividad6-evaluacion-validacion.git
cd actividad6-evaluacion-validacion
```

Coloca el archivo autorizado `Tweets.csv` en la carpeta raíz para volver a ejecutar la evaluación. El repositorio no distribuye el dataset.

## Resultados e interpretación

La partición usa `StratifiedGroupKFold` con semilla 42, agrupando por texto para que textos iguales no compartan entrenamiento y prueba. TF-IDF vive dentro de cada pipeline. El modelo se selecciona por F1 macro medio en validación cruzada sobre entrenamiento y se evalúa una vez en el holdout reservado. El umbral negativo recomendado se selecciona maximizando F1 binario sobre predicciones out-of-fold de entrenamiento; el holdout se usa para informar su rendimiento final, no para escogerlo.

La tabla principal se encuentra en `resultados/metricas_modelos.csv`; la tabla de cinco folds está en `resultados/validacion_cruzada.csv`. `resultados/resumen.json` reúne métricas y características de las particiones. Las gráficas y matrices se guardan en PNG y CSV.

La simulación A/B usa supuestos fijados en el código y no representa clientes ni operación real. No se proporcionó un CSV operacional, así que el análisis de A/B real queda pendiente. La función para CSV real solo estima la comparación; una prueba operacional requiere asignación adecuada, grupos comparables, etiquetado independiente y protocolo previo. No se calculan costos ni ROI por falta de datos de negocio.

## Referencias

- Kaggle/CrowdFlower, Twitter US Airline Sentiment: https://www.kaggle.com/datasets/crowdflower/twitter-airline-sentiment
- Scikit-learn, validación cruzada y StratifiedGroupKFold: https://scikit-learn.org/stable/modules/cross_validation.html
- McNemar, Q. (1947). Note on the sampling error of the difference between correlated proportions or percentages. Psychometrika, 12(2), 153–157. https://doi.org/10.1007/BF02295996
- Dietterich, T. G. (1998). Approximate Statistical Tests for Comparing Supervised Classification Learning Algorithms. Neural Computation, 10(7), 1895–1923. https://doi.org/10.1162/089976698300017197
