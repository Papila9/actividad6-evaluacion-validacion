from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binomtest, chi2
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, precision_score, recall_score, roc_auc_score,
                             roc_curve, auc)
from sklearn.model_selection import StratifiedGroupKFold, cross_val_predict, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import label_binarize

SEED = 42
LABELS = ["negative", "neutral", "positive"]
REQUIRED = {"text", "airline_sentiment"}


def load_and_clean(path: str | Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    raw = pd.read_csv(path)
    df = raw.copy()
    missing_cols = REQUIRED - set(df.columns)
    if missing_cols:
        raise ValueError(f"Faltan columnas requeridas: {sorted(missing_cols)}")
    initial = len(df)
    null_text = int(df["text"].isna().sum())
    null_label = int(df["airline_sentiment"].isna().sum())
    df = df.dropna(subset=["text", "airline_sentiment"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["airline_sentiment"] = df["airline_sentiment"].astype(str).str.strip().str.lower()
    empty_text = int(df["text"].eq("").sum())
    df = df[df["text"].ne("") & df["airline_sentiment"].isin(LABELS)].copy()
    label_counts = df.groupby("text")["airline_sentiment"].nunique()
    conflict_texts = set(label_counts[label_counts > 1].index)
    conflict_rows = int(df["text"].isin(conflict_texts).sum())
    nonconflicting = df[~df["text"].isin(conflict_texts)].copy()
    duplicate_text_rows_removed = int(nonconflicting["text"].duplicated().sum())
    df = nonconflicting.drop_duplicates(subset=["text"])
    info = {
        "columns": list(raw.columns),
        "data_types": {str(column): str(dtype) for column, dtype in raw.dtypes.items()},
        "missing_by_column": {str(column): int(count) for column, count in raw.isna().sum().items()},
        "records_before": initial,
        "missing_text": null_text,
        "missing_label": null_label,
        "empty_text": empty_text,
        "invalid_label_rows": int((raw["airline_sentiment"].notna() &
                                    ~raw["airline_sentiment"].astype(str).str.strip().str.lower().isin(LABELS)).sum()),
        "duplicate_rows": int(raw.duplicated().sum()),
        "duplicate_text_rows_before_cleaning": int(raw["text"].duplicated().sum()),
        "conflicting_text_groups": len(conflict_texts),
        "conflicting_rows_excluded": conflict_rows,
        "records_after": len(df),
        "duplicate_text_rows_removed": duplicate_text_rows_removed,
        "class_distribution_before": {str(k): int(v) for k, v in raw["airline_sentiment"].value_counts(dropna=False).items()},
        "class_distribution_after": {str(k): int(v) for k, v in df["airline_sentiment"].value_counts().items()},
        "seed": SEED,
    }
    return df.reset_index(drop=True), info


def make_models() -> dict[str, Pipeline]:
    return {
        "DummyClassifier": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000)),
                                      ("model", DummyClassifier(strategy="most_frequent"))]),
        "Regresión logística": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=40000, sublinear_tf=True)),
                                          ("model", LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED, C=2.0))]),
        "Random Forest": Pipeline([("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=30000, sublinear_tf=True)),
                                    ("model", RandomForestClassifier(n_estimators=120, max_depth=45, min_samples_leaf=2,
                                                                      class_weight="balanced_subsample", n_jobs=-1, random_state=SEED))]),
    }


def grouped_holdout(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    train, test = next(splitter.split(df["text"], df["airline_sentiment"], groups=df["text"]))
    if set(df.iloc[train]["text"]) & set(df.iloc[test]["text"]):
        raise AssertionError("Fuga de textos idénticos entre particiones")
    return train, test


def threshold_table(y_true: np.ndarray, negative_prob: np.ndarray) -> pd.DataFrame:
    truth = (y_true == "negative").astype(int)
    rows = []
    for threshold in (0.30, 0.40, 0.50, 0.60, 0.70):
        pred = (negative_prob >= threshold).astype(int)
        tn, fp, fn, tp = confusion_matrix(truth, pred, labels=[0, 1]).ravel()
        rows.append({"threshold": threshold, "precision": precision_score(truth, pred, zero_division=0),
                     "recall": recall_score(truth, pred, zero_division=0), "f1": f1_score(truth, pred, zero_division=0),
                     "false_positives": int(fp), "false_negatives": int(fn)})
    return pd.DataFrame(rows)


def mcnemar(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> dict[str, Any]:
    correct_a, correct_b = pred_a == y_true, pred_b == y_true
    both = int(np.sum(correct_a & correct_b))
    a_only = int(np.sum(correct_a & ~correct_b))
    b_only = int(np.sum(~correct_a & correct_b))
    neither = int(np.sum(~correct_a & ~correct_b))
    discordant = a_only + b_only
    p = float(binomtest(min(a_only, b_only), discordant, 0.5, alternative="two-sided").pvalue) if discordant else 1.0
    stat = float((abs(a_only - b_only) - 1) ** 2 / discordant) if discordant else 0.0
    return {"contingency_correct_both": both, "contingency_logistic_only": a_only,
            "contingency_forest_only": b_only, "contingency_both_wrong": neither,
            "mcnemar_chi2_cc": stat, "mcnemar_exact_p": p, "alpha": 0.05}


def run_evaluation(data_path: str | Path, output_dir: str | Path) -> dict[str, Any]:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    df, diagnostics = load_and_clean(data_path)
    train_idx, test_idx = grouped_holdout(df)
    train, test = df.iloc[train_idx], df.iloc[test_idx]
    X_train, y_train = train["text"], train["airline_sentiment"].to_numpy()
    X_test, y_test = test["text"], test["airline_sentiment"].to_numpy()
    groups_train = train["text"].to_numpy()
    diagnostics["partition"] = {"train_rows": len(train), "test_rows": len(test), "seed": SEED,
                                 "train_class_distribution": train["airline_sentiment"].value_counts().to_dict(),
                                 "test_class_distribution": test["airline_sentiment"].value_counts().to_dict(),
                                 "shared_texts": len(set(X_train) & set(X_test))}
    metrics, cv_rows, predictions = [], [], {}
    models = make_models()
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    scoring = {"f1_macro": "f1_macro", "accuracy": "accuracy"}
    for name, model in models.items():
        scores = cross_validate(model, X_train, y_train, groups=groups_train, cv=cv, scoring=scoring,
                                n_jobs=1, return_train_score=False)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions[name] = pred
        row = {"model": name, "accuracy": accuracy_score(y_test, pred),
               "precision_macro": precision_score(y_test, pred, average="macro", zero_division=0),
               "recall_macro": recall_score(y_test, pred, average="macro", zero_division=0),
               "f1_macro": f1_score(y_test, pred, average="macro", zero_division=0),
               "f1_weighted": f1_score(y_test, pred, average="weighted", zero_division=0)}
        metrics.append(row)
        cv_rows.append({"model": name, "f1_macro_mean": float(scores["test_f1_macro"].mean()),
                        "f1_macro_std": float(scores["test_f1_macro"].std(ddof=1)),
                        "f1_macro_min": float(scores["test_f1_macro"].min()),
                        "f1_macro_max": float(scores["test_f1_macro"].max()),
                        "accuracy_mean": float(scores["test_accuracy"].mean()),
                        "fold_f1_macro": scores["test_f1_macro"].tolist()})
        if name in ("Regresión logística", "Random Forest"):
            pd.DataFrame(classification_report(y_test, pred, labels=LABELS, output_dict=True, zero_division=0)).T.to_csv(out / f"reporte_clases_{'logistica' if name.startswith('Regresión') else 'random_forest'}.csv")
            cm = confusion_matrix(y_test, pred, labels=LABELS)
            pd.DataFrame(cm, index=LABELS, columns=LABELS).to_csv(out / f"matriz_confusion_{'logistica' if name.startswith('Regresión') else 'random_forest'}.csv")
            fig, ax = plt.subplots(figsize=(6, 5))
            ax.imshow(cm, cmap="Blues")
            ax.set(xticks=range(3), xticklabels=LABELS, yticks=range(3), yticklabels=LABELS,
                   xlabel="Predicción", ylabel="Etiqueta real", title=f"Matriz de confusión: {name}")
            for (i, j), value in np.ndenumerate(cm): ax.text(j, i, str(value), ha="center", va="center")
            fig.tight_layout(); fig.savefig(out / f"matriz_confusion_{'logistica' if name.startswith('Regresión') else 'random_forest'}.png", dpi=160); plt.close(fig)
    metrics_df = pd.DataFrame(metrics).sort_values("f1_macro", ascending=False)
    cv_df = pd.DataFrame(cv_rows)
    metrics_df.to_csv(out / "metricas_modelos.csv", index=False)
    cv_df.to_csv(out / "validacion_cruzada.csv", index=False)
    winner = max(cv_rows, key=lambda row: row["f1_macro_mean"])["model"]
    roc_models = {"Regresión logística": models["Regresión logística"], "Random Forest": models["Random Forest"]}
    roc_results = {}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, (roc_name, roc_model) in zip(axes, roc_models.items()):
        roc_probs = roc_model.predict_proba(X_test)
        classes = list(roc_model.named_steps["model"].classes_)
        y_bin = label_binarize(y_test, classes=classes)
        roc_results[roc_name] = float(roc_auc_score(y_test, roc_probs, multi_class="ovr", average="macro", labels=classes))
        for i, label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], roc_probs[:, i])
            ax.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title=f"OvR: {roc_name}")
        ax.legend(fontsize=8)
        roc_file = "logistica" if roc_name.startswith("Regresión") else "random_forest"
        fig_single, ax_single = plt.subplots(figsize=(7, 6))
        for i, label in enumerate(classes):
            fpr, tpr, _ = roc_curve(y_bin[:, i], roc_probs[:, i])
            ax_single.plot(fpr, tpr, label=f"{label} (AUC={auc(fpr, tpr):.3f})")
        ax_single.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax_single.set(xlabel="Tasa de falsos positivos", ylabel="Tasa de verdaderos positivos", title=f"Curvas ROC OvR: {roc_name}")
        ax_single.legend(); fig_single.tight_layout()
        fig_single.savefig(out / f"curvas_roc_{roc_file}.png", dpi=160); plt.close(fig_single)
    fig.tight_layout(); fig.savefig(out / "curvas_roc.png", dpi=160); plt.close(fig)
    lr = models["Regresión logística"]
    probs = lr.predict_proba(X_test)
    classes = list(lr.named_steps["model"].classes_)
    roc_auc = roc_results["Regresión logística"]
    neg_idx = classes.index("negative")
    cv_lr = make_models()["Regresión logística"]
    validation_probs = cross_val_predict(cv_lr, X_train, y_train, groups=groups_train, cv=cv,
                                         method="predict_proba", n_jobs=1)
    validation_classes = sorted(np.unique(y_train))
    validation_thresholds = threshold_table(
        y_train, validation_probs[:, validation_classes.index("negative")]
    )
    validation_thresholds.to_csv(out / "validacion_umbral.csv", index=False)
    selected_validation = validation_thresholds.sort_values(
        ["f1", "recall", "precision"], ascending=False
    ).iloc[0]
    recommended_threshold = float(selected_validation["threshold"])
    thresholds = threshold_table(y_test, probs[:, neg_idx])
    thresholds.to_csv(out / "analisis_umbral.csv", index=False)
    fig, ax = plt.subplots(figsize=(7, 5))
    for col in ("precision", "recall", "f1"): ax.plot(thresholds["threshold"], thresholds[col], marker="o", label=col)
    ax.set(xlabel="Umbral de probabilidad negativa", ylabel="Métrica", ylim=(0, 1), title="Métricas por umbral")
    ax.legend(); fig.tight_layout(); fig.savefig(out / "analisis_umbral.png", dpi=160); plt.close(fig)
    stats = mcnemar(y_test, predictions["Regresión logística"], predictions["Random Forest"])
    stats["auc_ovr_macro_logistic"] = roc_auc
    stats["auc_ovr_macro_random_forest"] = roc_results["Random Forest"]
    stats["roc_results"] = roc_results
    test_row = thresholds.loc[thresholds["threshold"] == recommended_threshold].iloc[0]
    test_at_selected_threshold = {
        key: int(value) if key in ("false_positives", "false_negatives") else float(value)
        for key, value in test_row.items()
    }
    stats["selected_model_by_cv_f1_macro"] = str(winner)
    stats["selection_note"] = "El modelo se selecciona por F1 macro medio de validación cruzada en entrenamiento; el holdout se reserva para evaluación final."
    stats["recommended_negative_threshold"] = recommended_threshold
    stats["threshold_selection_metric"] = "F1 binario máximo sobre probabilidades out-of-fold del entrenamiento"
    stats["validation_f1_at_recommended_threshold"] = float(selected_validation["f1"])
    stats["test_metrics_at_recommended_threshold"] = test_at_selected_threshold
    metrics_df.to_json(out / "metricas_modelos.json", orient="records", force_ascii=False, indent=2)
    cv_df.to_json(out / "validacion_cruzada.json", orient="records", force_ascii=False, indent=2)
    reference = test_at_selected_threshold
    n_negative = int(np.sum(y_test == "negative"))
    n_nonnegative = len(y_test) - n_negative
    observed_fpr = float(reference["false_positives"] / n_nonnegative)
    projected_negative = int(round(1000 * n_negative / len(y_test)))
    projected_nonnegative = 1000 - projected_negative
    impact = {
        "volume": 1000,
        "threshold": recommended_threshold,
        "basis": "Observed holdout prevalence, recall and false-positive rate; illustrative projection only.",
        "holdout_negative_prevalence": n_negative / len(y_test),
        "observed_recall": float(reference["recall"]),
        "observed_false_positive_rate": observed_fpr,
        "projected_negative_detected": int(round(projected_negative * reference["recall"])),
        "projected_negative_missed": projected_negative - int(round(projected_negative * reference["recall"])),
        "projected_false_alerts": int(round(projected_nonnegative * observed_fpr)),
    }
    impact["projected_alerts_for_review"] = impact["projected_negative_detected"] + impact["projected_false_alerts"]
    pd.DataFrame([{"measure": key, "value": value, "interpretation": impact.get("basis", "Projection")}
                  for key, value in impact.items()]).to_csv(out / "impacto_ilustrativo_1000.csv", index=False)
    (out / "impacto_ilustrativo_1000.json").write_text(json.dumps(impact, ensure_ascii=False, indent=2), encoding="utf-8")
    summary = {"diagnostics": diagnostics, "metrics": metrics, "cross_validation": cv_rows, "statistics": stats,
               "thresholds": thresholds.to_dict(orient="records"),
               "validation_thresholds": validation_thresholds.to_dict(orient="records"),
               "business_impact": impact,
               "best_by_macro_f1": str(winner)}
    (out / "resumen.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="Tweets.csv")
    parser.add_argument("--output", default="resultados")
    args = parser.parse_args()
    result = run_evaluation(args.data, args.output)
    print(json.dumps(result["statistics"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
