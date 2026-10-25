import numpy as np
import pandas as pd
import pytest

from src.evaluacion_modelos import grouped_holdout, load_and_clean, make_models, mcnemar, run_evaluation, threshold_table
from src.prueba_ab import analyze_real_csv, sample_size_two_proportions, simulate_ab


def test_cleaning_and_group_split(tmp_path):
    path = tmp_path / "tiny.csv"
    texts = [f"t{i}" for i in range(15)]
    labels = ["negative"] * 5 + ["neutral"] * 5 + ["positive"] * 5
    pd.DataFrame({"text": texts, "airline_sentiment": labels}).to_csv(path, index=False)
    frame, details = load_and_clean(path)
    train, test = grouped_holdout(frame)
    assert not (set(frame.iloc[train].text) & set(frame.iloc[test].text))
    assert details["records_after"] == len(frame)
    assert details["columns"] == ["text", "airline_sentiment"]
    assert details["missing_by_column"] == {"text": 0, "airline_sentiment": 0}


def test_required_columns_are_validated(tmp_path):
    path = tmp_path / "missing.csv"
    pd.DataFrame({"text": ["comment"]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="Faltan columnas requeridas"):
        load_and_clean(path)


def test_conflicting_labels_excluded(tmp_path):
    path = tmp_path / "tiny.csv"
    pd.DataFrame({"text": ["x", "x", "y", "z"], "airline_sentiment": ["negative", "positive", "neutral", "negative"]}).to_csv(path, index=False)
    frame, details = load_and_clean(path)
    assert "x" not in set(frame.text)
    assert details["conflicting_text_groups"] == 1


def test_pipeline_contains_tfidf():
    assert all("tfidf" in model.named_steps for model in make_models().values())


def test_threshold_indicators():
    table = threshold_table(np.array(["negative", "neutral", "negative"]), np.array([.8, .1, .2]))
    row = table.iloc[2]
    assert row.false_positives == 0
    assert row.false_negatives == 1
    assert 0 <= row.f1 <= 1


def test_metrics_and_mcnemar():
    result = mcnemar(np.array(["a", "b"]), np.array(["a", "a"]), np.array(["b", "b"]))
    assert result["contingency_logistic_only"] == 1
    assert result["contingency_forest_only"] == 1
    assert 0 <= result["mcnemar_exact_p"] <= 1


def test_ab_simulation_reproducible_and_power():
    first, result1 = simulate_ab(seed=123, n_per_group=100)
    second, result2 = simulate_ab(seed=123, n_per_group=100)
    pd.testing.assert_frame_equal(first, second)
    assert result1["test"] == result2["test"]
    assert sample_size_two_proportions() > 0


def test_real_ab_validator(tmp_path):
    path = tmp_path / "ab.csv"
    rows = []
    for group in ("A", "B"):
        for i in range(12):
            rows.append({"grupo": group, "caso_id": f"{group}{i}", "negativo_real": 1,
                         "negativo_detectado": int(i < (6 if group == "A" else 7)),
                         "tiempo_revision_min": 1.0 + i / 10})
        for i in range(6):
            rows.append({"grupo": group, "caso_id": f"{group}p{i}", "negativo_real": 0,
                         "negativo_detectado": 0, "tiempo_revision_min": 2.0})
    pd.DataFrame(rows).to_csv(path, index=False)
    table, result = analyze_real_csv(path)
    assert len(table) == 2
    assert "p_value" in result
    assert result["significance_test_performed"] is True
    assert "tiempo_revision_min_promedio" in table.columns


def test_real_ab_suppresses_significance_when_sample_is_small(tmp_path):
    path = tmp_path / "small.csv"
    pd.DataFrame({"grupo": ["A", "A", "B", "B"], "caso_id": ["1", "2", "3", "4"],
                  "negativo_real": [1, 0, 1, 0], "negativo_detectado": [1, 0, 1, 1]}).to_csv(path, index=False)
    _, result = analyze_real_csv(path)
    assert result["significance_test_performed"] is False
    assert result["p_value"] is None


def test_real_ab_rejects_missing_classes_and_values(tmp_path):
    path = tmp_path / "bad.csv"
    pd.DataFrame({"grupo": ["A", "A", "B", "B"], "caso_id": ["1", "2", "3", "4"],
                  "negativo_real": [1, 0, 0, 0], "negativo_detectado": [1, 0, 0, 0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="no contiene casos negativos"):
        analyze_real_csv(path)
    pd.DataFrame({"grupo": ["A", "A", "B", "B"], "caso_id": ["1", "2", "3", "4"],
                  "negativo_real": [1, 0, 1, 0], "negativo_detectado": [1, 2, 1, 0]}).to_csv(path, index=False)
    with pytest.raises(ValueError, match="solo puede contener"):
        analyze_real_csv(path)


def test_artifacts_export(tmp_path):
    table, _ = simulate_ab(seed=5, n_per_group=30)
    path = tmp_path / "synthetic.csv"
    table.to_csv(path, index=False)
    assert path.exists() and path.stat().st_size > 0


def test_evaluation_exports_results(tmp_path):
    path = tmp_path / "tweets.csv"
    rows = []
    for label, word in (("negative", "awful"), ("neutral", "okay"), ("positive", "great")):
        for index in range(10):
            rows.append({"text": f"{word} airline service token{index}", "airline_sentiment": label})
    pd.DataFrame(rows).to_csv(path, index=False)
    out = tmp_path / "out"
    summary = run_evaluation(path, out)
    for name in ("metricas_modelos.csv", "validacion_cruzada.csv", "validacion_umbral.csv", "resumen.json", "curvas_roc.png",
                 "matriz_confusion_logistica.csv", "matriz_confusion_random_forest.csv", "analisis_umbral.csv"):
        assert (out / name).exists()
    metric_frame = pd.read_csv(out / "metricas_modelos.csv")
    metric_columns = ["accuracy", "precision_macro", "recall_macro", "f1_macro", "f1_weighted"]
    assert metric_frame[metric_columns].apply(lambda column: column.between(0, 1).all()).all()
    assert pd.read_csv(out / "matriz_confusion_logistica.csv", index_col=0).shape == (3, 3)
    assert pd.read_csv(out / "matriz_confusion_random_forest.csv", index_col=0).shape == (3, 3)
    cv_frame = pd.read_csv(out / "validacion_cruzada.csv")
    assert summary["best_by_macro_f1"] == cv_frame.loc[cv_frame.f1_macro_mean.idxmax(), "model"]
    assert summary["statistics"]["recommended_negative_threshold"] in [0.30, 0.40, 0.50, 0.60, 0.70]
