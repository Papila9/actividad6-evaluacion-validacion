# Plantilla para analizar una prueba A/B real

Completa el archivo `plantilla_ab_real.csv` con registros de una prueba operacional, una fila por caso y un identificador no sensible.

- `grupo`: `A` para control o `B` para tratamiento.
- `caso_id`: identificador único por caso.
- `negativo_real`: `1` si la etiqueta de referencia independiente es negativa; `0` en otro caso.
- `negativo_detectado`: `1` si el proceso evaluado marcó el comentario como negativo; `0` en otro caso.
- `tiempo_revision_min`: columna opcional con el tiempo de revisión en minutos.

No llenes la plantilla con datos sintéticos si se usará para analizar una prueba real. Se requieren ambos grupos y casos negativos reales en cada grupo. El analizador omitirá la prueba de significancia si alguna celda de detección/no detección tiene menos de cinco observaciones.

Ejecuta el análisis desde la carpeta del proyecto:

```powershell
.\.venv\Scripts\python.exe -m src.prueba_ab --output resultados --real-csv plantillas\plantilla_ab_real.csv
```
