from __future__ import annotations

import base64
import html
import json
from pathlib import Path
from typing import Any

import pandas as pd


def _image(path: Path, alt: str) -> str:
    if not path.exists():
        return f'<div class="missing">No se encontró la figura: {html.escape(path.name)}</div>'
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return f'<img src="data:image/png;base64,{encoded}" alt="{html.escape(alt)}">'


def _table(frame: pd.DataFrame, digits: int = 3) -> str:
    headers = "".join(f"<th>{html.escape(str(column))}</th>" for column in frame.columns)
    rows = []
    for _, record in frame.iterrows():
        cells = []
        for value in record:
            if isinstance(value, (float,)):
                value = f"{value:.{digits}f}"
            cells.append(f"<td>{html.escape(str(value))}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return f"<div class='table-wrap'><table><thead><tr>{headers}</tr></thead><tbody>{''.join(rows)}</tbody></table></div>"


def build_dashboard(results_dir: str | Path = "resultados",
                    output: str | Path | None = None) -> Path:
    results = Path(results_dir)
    destination = Path(output) if output else results / "tablero_resultados.html"
    destination.parent.mkdir(parents=True, exist_ok=True)

    metrics = pd.read_csv(results / "metricas_modelos.csv").sort_values("f1_macro", ascending=False)
    cv_rows: list[dict[str, Any]] = json.loads((results / "validacion_cruzada.json").read_text(encoding="utf-8"))
    cv = pd.DataFrame(cv_rows)
    thresholds = pd.read_csv(results / "analisis_umbral.csv")
    summary = json.loads((results / "resumen.json").read_text(encoding="utf-8"))
    impact = json.loads((results / "impacto_ilustrativo_1000.json").read_text(encoding="utf-8"))
    ab = json.loads((results / "simulacion_ab_sintetica.json").read_text(encoding="utf-8"))
    stats = summary["statistics"]
    selected = summary["best_by_macro_f1"]
    selected_row = metrics.loc[metrics.model == selected].iloc[0]
    selected_cv = cv.loc[cv.model == selected].iloc[0]

    cv_bars = []
    for record in cv_rows:
        label = html.escape(record["model"])
        value = float(record["f1_macro_mean"])
        cv_bars.append(
            f"<div class='bar-row'><span>{label}</span><div class='track'><div class='fill' style='width:{value * 100:.1f}%'></div></div>"
            f"<strong>{value:.3f}</strong><small>± {record['f1_macro_std']:.3f}</small></div>"
        )

    model_table = metrics.rename(columns={
        "model": "Modelo", "accuracy": "Exactitud", "precision_macro": "Precisión macro",
        "recall_macro": "Recall macro", "f1_macro": "F1 macro", "f1_weighted": "F1 ponderado"
    })
    cv_table = cv.rename(columns={
        "model": "Modelo", "f1_macro_mean": "F1 macro medio", "f1_macro_std": "Desv. estándar",
        "f1_macro_min": "Mínimo", "f1_macro_max": "Máximo", "accuracy_mean": "Exactitud media"
    })[["Modelo", "F1 macro medio", "Desv. estándar", "Mínimo", "Máximo", "Exactitud media"]]
    threshold_table = thresholds.rename(columns={
        "threshold": "Umbral", "precision": "Precisión", "recall": "Recall", "f1": "F1",
        "false_positives": "Falsos positivos", "false_negatives": "Falsos negativos"
    })
    ab_groups = pd.DataFrame(ab["groups"])[[
        "grupo", "casos_total", "negativos_reales", "negativos_detectados", "recall", "alertas_falsas", "precision_alertas"
    ]].rename(columns={
        "grupo": "Grupo", "casos_total": "Casos totales", "negativos_reales": "Negativos reales",
        "negativos_detectados": "Negativos detectados", "recall": "Recall", "alertas_falsas": "Alertas falsas",
        "precision_alertas": "Precisión de alertas"
    })

    body = f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Tablero de evaluación de modelos</title>
<style>
:root{{--navy:#15243a;--blue:#3c73b9;--teal:#2b8a82;--bg:#f3f6fa;--card:#fff;--text:#243246;--muted:#62738a;--line:#dce4ee;--amber:#9a6814}}
*{{box-sizing:border-box}}body{{margin:0;background:var(--bg);color:var(--text);font:15px/1.5 Segoe UI,Arial,sans-serif}}
header{{background:linear-gradient(115deg,#14253c,#24476f);color:#fff;padding:30px max(24px,calc((100vw - 1240px)/2))}}
header h1{{margin:0 0 6px;font-size:28px}}header p{{margin:0;color:#d8e5f5}}main{{max-width:1240px;margin:24px auto;padding:0 20px}}
.banner{{background:#fff7e6;border-left:5px solid #e1a333;padding:12px 16px;margin-bottom:18px;color:#614819}}
.grid{{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-bottom:16px}}
.card,.panel{{background:var(--card);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 14px #22334a0a}}
.card{{padding:16px}}.card .label{{font-size:12px;color:var(--muted);text-transform:uppercase;letter-spacing:.06em}}.card .value{{font-size:27px;font-weight:700;color:var(--navy);margin-top:5px}}.card .note{{font-size:12px;color:var(--muted)}}
.panel{{padding:18px;margin:14px 0}}h2{{font-size:19px;margin:0 0 12px;color:var(--navy)}}h3{{font-size:15px;margin:0 0 8px;color:var(--navy)}}
.two{{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}}.figure{{min-width:0}}img{{display:block;max-width:100%;height:auto;margin:8px auto;border-radius:7px}}.figure img{{width:100%}}
.table-wrap{{overflow-x:auto}}table{{border-collapse:collapse;width:100%;font-size:13px}}th,td{{padding:8px 10px;border-bottom:1px solid var(--line);text-align:left;white-space:nowrap}}th{{background:#edf3f9;color:#334a65}}tbody tr:nth-child(even){{background:#fafbfd}}
.bar-row{{display:grid;grid-template-columns:165px 1fr 48px 78px;gap:10px;align-items:center;margin:11px 0;font-size:13px}}.track{{height:13px;background:#e7edf4;border-radius:99px;overflow:hidden}}.fill{{height:100%;background:linear-gradient(90deg,var(--blue),var(--teal));border-radius:99px}}.bar-row small{{color:var(--muted)}}
.selection{{padding:14px 16px;border-radius:10px;background:#eaf4f2;border:1px solid #c8e5de}}.badge{{display:inline-block;padding:4px 9px;border-radius:99px;background:#f6e8c9;color:var(--amber);font-size:12px;font-weight:700}}
.foot{{color:var(--muted);font-size:12px;padding:16px 0 32px}}.missing{{padding:20px;color:#8b3e32;background:#fff1ee;border-radius:8px}}
@media(max-width:800px){{.grid{{grid-template-columns:repeat(2,minmax(0,1fr))}}.two{{grid-template-columns:1fr}}.bar-row{{grid-template-columns:125px 1fr 42px}}.bar-row small{{display:none}}}}
@media(max-width:480px){{.grid{{grid-template-columns:1fr 1fr;gap:8px}}.card{{padding:12px}}.card .value{{font-size:22px}}main{{padding:0 12px}}}}
</style>
</head>
<body>
<header><h1>Evaluación y validación de modelos</h1><p>Clasificación de sentimiento · Twitter US Airline Sentiment · semilla {summary['diagnostics']['seed']}</p></header>
<main>
<div class="banner"><strong>Lectura de resultados:</strong> el modelo se elige por F1 macro medio de validación cruzada en entrenamiento. El holdout queda reservado para la evaluación final. La sección A/B está marcada como sintética.</div>
<div class="grid">
 <div class="card"><div class="label">Modelo seleccionado</div><div class="value" style="font-size:21px">{html.escape(selected)}</div><div class="note">Mayor F1 macro medio en validación</div></div>
 <div class="card"><div class="label">F1 macro en holdout</div><div class="value">{selected_row['f1_macro']:.3f}</div><div class="note">{selected_cv['f1_macro_mean']:.3f} medio en 5 folds</div></div>
 <div class="card"><div class="label">Exactitud en holdout</div><div class="value">{selected_row['accuracy']:.3f}</div><div class="note">{summary['diagnostics']['partition']['test_rows']:,} casos de prueba</div></div>
 <div class="card"><div class="label">AUC macro OvR</div><div class="value">{stats['auc_ovr_macro_logistic']:.3f}</div><div class="note">Regresión logística</div></div>
</div>
<section class="panel"><h2>Selección del modelo</h2><div class="selection"><strong>{html.escape(selected)}</strong> se seleccionó porque obtuvo la mejor media de F1 macro en validación cruzada ({selected_cv['f1_macro_mean']:.3f} ± {selected_cv['f1_macro_std']:.3f}). En el holdout alcanzó F1 macro {selected_row['f1_macro']:.3f}. La selección no usa el holdout.</div></section>
<section class="panel"><h2>Comparación de modelos en el holdout</h2>{_table(model_table)}</section>
<section class="panel"><h2>Validación cruzada agrupada · cinco folds</h2><div class="two"><div>{''.join(cv_bars)}</div><div>{_table(cv_table)}</div></div></section>
<section class="panel"><h2>Matrices de confusión</h2><div class="two"><div class="figure"><h3>Regresión Logística</h3>{_image(results / 'matriz_confusion_logistica.png','Matriz de confusión de Regresión Logística')}</div><div class="figure"><h3>Random Forest</h3>{_image(results / 'matriz_confusion_random_forest.png','Matriz de confusión de Random Forest')}</div></div></section>
<section class="panel"><h2>Curvas ROC multiclase · One-vs-Rest</h2><p>AUC macro: Regresión Logística {stats['auc_ovr_macro_logistic']:.3f}; Random Forest {stats['auc_ovr_macro_random_forest']:.3f}.</p>{_image(results / 'curvas_roc.png','Comparación de curvas ROC OvR')}</section>
<section class="panel"><h2>Umbral para comentarios negativos</h2><p>Umbral recomendado: <strong>{stats['recommended_negative_threshold']:.2f}</strong>, seleccionado por F1 binario sobre probabilidades out-of-fold del entrenamiento (F1 de validación {stats['validation_f1_at_recommended_threshold']:.3f}). En el holdout: precisión {stats['test_metrics_at_recommended_threshold']['precision']:.3f}, recall {stats['test_metrics_at_recommended_threshold']['recall']:.3f}, {stats['test_metrics_at_recommended_threshold']['false_positives']} falsos positivos y {stats['test_metrics_at_recommended_threshold']['false_negatives']} falsos negativos.</p><div class="two"><div>{_table(threshold_table)}</div><div class="figure">{_image(results / 'analisis_umbral.png','Métricas por umbral de detección')}</div></div></section>
<section class="panel"><h2>Impacto ilustrativo · 1,000 comentarios</h2><p>Proyección escalada con prevalencia, recall y tasa de falsos positivos observados en el holdout; no son resultados operativos ni costos monetarios.</p><div class="grid"><div class="card"><div class="label">Negativos detectados</div><div class="value">{impact['projected_negative_detected']}</div></div><div class="card"><div class="label">Negativos omitidos</div><div class="value">{impact['projected_negative_missed']}</div></div><div class="card"><div class="label">Alertas falsas</div><div class="value">{impact['projected_false_alerts']}</div></div><div class="card"><div class="label">Alertas para revisión</div><div class="value">{impact['projected_alerts_for_review']}</div></div></div></section>
<section class="panel"><h2><span class="badge">SIMULACIÓN CON DATOS SINTÉTICOS</span> A/B</h2><p>Supuestos fijados: recall A={ab['assumptions']['baseline_recall']:.2f}, recall B={ab['assumptions']['assisted_recall']:.2f}, α={ab['assumptions']['alpha']:.2f}, potencia objetivo={ab['assumptions']['target_power']:.0%}, semilla={ab['assumptions']['seed']}. Tamaño estimado: {ab['estimated_n_negative_per_group']} casos negativos por grupo.</p>{_table(ab_groups)}<p>Diferencia observada de recall B−A: {ab['test']['difference_b_minus_a']:.3f}; p={ab['test']['p_value']:.3f}; IC95% [{ab['test']['confidence_interval_95'][0]:.3f}, {ab['test']['confidence_interval_95'][1]:.3f}]. Esta simulación ilustra el procedimiento; no demuestra una mejora real.</p>{_image(results / 'simulacion_ab_sintetica.png','Resultados de simulación A/B sintética')}</section>
<div class="foot">Fuente: ejecución local de los scripts del proyecto. Datos académicos: {summary['diagnostics']['records_before']:,} filas; después de limpieza: {summary['diagnostics']['records_after']:,}. El dataset original no se distribuye con este tablero.</div>
</main></body></html>"""
    destination.write_text(body, encoding="utf-8")
    return destination


if __name__ == "__main__":
    print(build_dashboard())
