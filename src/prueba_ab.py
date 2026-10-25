from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

SEED = 42
REQUIRED = {"grupo", "caso_id", "negativo_real", "negativo_detectado"}


def sample_size_two_proportions(p_control: float = 0.65, p_treatment: float = 0.75,
                                alpha: float = 0.05, power: float = 0.80) -> int:
    if not 0 < p_control < 1 or not 0 < p_treatment < 1 or p_control == p_treatment:
        raise ValueError("Las proporciones deben estar entre 0 y 1 y ser diferentes")
    if not 0 < alpha < 1 or not 0 < power < 1:
        raise ValueError("alpha y power deben estar entre 0 y 1")
    pbar = (p_control + p_treatment) / 2
    z_alpha = norm.ppf(1 - alpha / 2)
    z_power = norm.ppf(power)
    n = ((z_alpha * np.sqrt(2 * pbar * (1 - pbar)) +
          z_power * np.sqrt(p_control * (1 - p_control) + p_treatment * (1 - p_treatment))) /
         (p_treatment - p_control)) ** 2
    return int(np.ceil(n))


def _proportion_test(success_a: int, n_a: int, success_b: int, n_b: int) -> dict:
    p_a, p_b = success_a / n_a, success_b / n_b
    pooled = (success_a + success_b) / (n_a + n_b)
    se = np.sqrt(pooled * (1 - pooled) * (1 / n_a + 1 / n_b))
    z = (p_b - p_a) / se if se else 0.0
    p_value = float(2 * norm.sf(abs(z)))
    diff = p_b - p_a
    se_unpooled = np.sqrt(p_a * (1 - p_a) / n_a + p_b * (1 - p_b) / n_b)
    ci = [float(diff - norm.ppf(.975) * se_unpooled), float(diff + norm.ppf(.975) * se_unpooled)]
    return {"recall_a": p_a, "recall_b": p_b, "difference_b_minus_a": diff,
            "z_statistic": float(z), "p_value": p_value, "confidence_interval_95": ci}


def simulate_ab(seed: int = SEED, n_per_group: int | None = None) -> tuple[pd.DataFrame, dict]:
    assumptions = {"baseline_recall": 0.65, "assisted_recall": 0.75, "negative_prevalence": 0.30,
                   "alpha": 0.05, "target_power": 0.80, "seed": seed}
    n_required = sample_size_two_proportions(assumptions["baseline_recall"], assumptions["assisted_recall"])
    n_per_group = n_per_group or n_required
    if n_per_group < 2:
        raise ValueError("Se requieren al menos dos casos negativos por grupo")
    rng = np.random.default_rng(seed)
    rows = []
    for group, recall in (("A", assumptions["baseline_recall"]), ("B", assumptions["assisted_recall"])):
        n_negative = n_per_group
        detected = rng.binomial(1, recall, size=n_negative)
        rows.extend({"grupo": group, "caso_id": f"{group}-N-{i}", "negativo_real": 1,
                     "negativo_detectado": int(value)} for i, value in enumerate(detected))
        n_nonnegative = int(round(n_per_group * (1 / assumptions["negative_prevalence"] - 1)))
        false_alerts = rng.binomial(1, 0.18 if group == "A" else 0.22, size=n_nonnegative)
        rows.extend({"grupo": group, "caso_id": f"{group}-P-{i}", "negativo_real": 0,
                     "negativo_detectado": int(value)} for i, value in enumerate(false_alerts))
    df = pd.DataFrame(rows)
    results = []
    for group in ("A", "B"):
        part = df[df.grupo == group]
        positives = part[part.negativo_real == 1]
        results.append({"grupo": group, "casos_total": len(part), "negativos_reales": len(positives),
                        "negativos_detectados": int(positives.negativo_detectado.sum()),
                        "recall": float(positives.negativo_detectado.mean()),
                        "alertas_falsas": int(((part.negativo_real == 0) & (part.negativo_detectado == 1)).sum()),
                        "precision_alertas": float(((part.negativo_real == 1) & (part.negativo_detectado == 1)).sum() /
                                                    max(1, part.negativo_detectado.sum()))})
    inferential = _proportion_test(results[0]["negativos_detectados"], results[0]["negativos_reales"],
                                   results[1]["negativos_detectados"], results[1]["negativos_reales"])
    return pd.DataFrame(results), {"label": "SIMULACIÓN CON DATOS SINTÉTICOS", "assumptions": assumptions,
                                   "estimated_n_negative_per_group": n_required, "simulated_n_negative_per_group": n_per_group,
                                   "groups": results, "test": inferential}


def analyze_real_csv(path: str | Path) -> tuple[pd.DataFrame, dict]:
    df = pd.read_csv(path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise ValueError(f"Faltan columnas: {sorted(missing)}")
    optional_time = "tiempo_revision_min" in df.columns
    columns = ["grupo", "caso_id", "negativo_real", "negativo_detectado"]
    if optional_time:
        columns.append("tiempo_revision_min")
    df = df[columns].copy()
    if df.isna().any().any():
        raise ValueError("El archivo contiene valores faltantes en las columnas requeridas")
    if df["grupo"].isin(["A", "B"]).all() is False or set(df.grupo.unique()) != {"A", "B"}:
        raise ValueError("grupo debe contener ambos grupos A y B, sin otros valores")
    if df["caso_id"].duplicated().any():
        raise ValueError("caso_id debe ser único")
    if (df["grupo"].value_counts().reindex(["A", "B"], fill_value=0) < 2).any():
        raise ValueError("Se requieren al menos dos casos en cada grupo")
    for column in ("negativo_real", "negativo_detectado"):
        if not df[column].isin([0, 1]).all():
            raise ValueError(f"{column} solo puede contener 0 o 1")
    if optional_time:
        df["tiempo_revision_min"] = pd.to_numeric(df["tiempo_revision_min"], errors="coerce")
        if df["tiempo_revision_min"].isna().any() or (df["tiempo_revision_min"] < 0).any():
            raise ValueError("tiempo_revision_min debe ser numérico y no negativo")
    if len(df) < 4:
        raise ValueError("Se requieren al menos dos registros por grupo")
    summary = []
    for group in ("A", "B"):
        part = df[df.grupo == group]
        n_pos = int((part.negativo_real == 1).sum())
        if n_pos == 0:
            raise ValueError(f"El grupo {group} no contiene casos negativos reales")
        tp = int(((part.negativo_real == 1) & (part.negativo_detectado == 1)).sum())
        fp = int(((part.negativo_real == 0) & (part.negativo_detectado == 1)).sum())
        result = {"grupo": group, "casos": len(part), "negativos_reales": n_pos,
                  "negativos_detectados": tp, "recall": tp / n_pos, "precision_alertas": tp / (tp + fp) if tp + fp else 0.0,
                  "alertas_falsas": fp}
        if optional_time:
            result["tiempo_revision_min_promedio"] = float(part["tiempo_revision_min"].mean())
        summary.append(result)
    insufficient = []
    for row in summary:
        misses = row["negativos_reales"] - row["negativos_detectados"]
        if row["negativos_detectados"] < 5 or misses < 5:
            insufficient.append(row["grupo"])
    if insufficient:
        inference = {"significance_test_performed": False, "p_value": None, "alpha": 0.05,
                     "reason": "Frecuencias de detecciones o no detecciones negativas menores que 5 en el/los grupos: " + ", ".join(insufficient)}
    else:
        inference = _proportion_test(summary[0]["negativos_detectados"], summary[0]["negativos_reales"],
                                     summary[1]["negativos_detectados"], summary[1]["negativos_reales"])
        inference["significance_test_performed"] = True
    inference["real_experiment_conclusion_allowed"] = bool(inference["significance_test_performed"])
    inference["caveat"] = "Interpretar solo con diseño, asignación y etiquetado preespecificados."
    return pd.DataFrame(summary), inference


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="resultados")
    parser.add_argument("--real-csv")
    args = parser.parse_args()
    out = Path(args.output); out.mkdir(parents=True, exist_ok=True)
    table, result = simulate_ab()
    table.to_csv(out / "simulacion_ab_sintetica.csv", index=False)
    (out / "simulacion_ab_sintetica.json").write_text(pd.Series(result).to_json(force_ascii=False, indent=2), encoding="utf-8")
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(table.grupo, table.recall, color=["#5879a8", "#5b9a75"])
    ax.set(ylim=(0, 1), ylabel="Recall", title="SIMULACIÓN CON DATOS SINTÉTICOS")
    fig.tight_layout(); fig.savefig(out / "simulacion_ab_sintetica.png", dpi=160); plt.close(fig)
    if args.real_csv:
        real, test = analyze_real_csv(args.real_csv)
        real.to_csv(out / "analisis_ab_real.csv", index=False)
        (out / "analisis_ab_real.json").write_text(pd.Series(test).to_json(force_ascii=False, indent=2), encoding="utf-8")
    print(result["test"])


if __name__ == "__main__":
    main()
