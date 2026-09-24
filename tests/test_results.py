"""Deliverables integrity: every recorded result must be present and
internally consistent with its artifacts.

These tests pass on a fresh clone using only the committed artifacts in
results/. They are the executable half of the project's claims: the docs
state the numbers, this suite proves the numbers are what the repo says
they are, and that per-fold evidence backs each one.
"""

import glob
import json
from pathlib import Path

import numpy as np

RESULTS = Path("results")

# (event, task) -> expected mean and tolerance, from docs/TASKS.md
EXPECTED_EVENTS = {
    ("champion_baseline", 6): (0.8365, 0.002),   # original untuned stack
    ("champion_baseline", 0): (0.8370, 0.002),   # tuned stack (finish run)
    ("retest", 6): (0.8388, 0.002),              # +cat retest was the last one
    ("gate_tuned", 7): (0.8370, 0.002),
    ("gate_ticket", 9): (0.8346, 0.002),
    ("gate_iterimp", 11): (0.8347, 0.002),
    ("gate_hygiene", 13): (0.8382, 0.002),
    ("posthoc_joint", 6): (0.8395, 0.002),
    ("nested_cv", 14): (0.8406, 0.005),
    ("final_cv", None): (0.8370, 0.002),
}

KAGGLE_SCORES = ["0.76555", "0.74641", "0.75358", "0.76076", "0.75837"]


def _load_events():
    events = []
    with open(RESULTS / "results.jsonl") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def test_expected_gate_events_present_and_consistent():
    events = _load_events()
    by_key = {}
    for e in events:
        by_key[(e.get("event"), e.get("task"))] = e  # last occurrence wins
    for (event, task), (expected, tol) in EXPECTED_EVENTS.items():
        key = (event, task)
        assert key in by_key, f"missing event {key} in results.jsonl"
        mean = by_key[key]["mean"]
        assert abs(mean - expected) <= tol, (
            f"{key}: recorded {mean:.4f}, expected {expected} +/- {tol}")


def test_champion_json_matches_spec():
    champ = json.load(open(RESULTS / "champion.json"))
    assert champ["spec"]["members"] == ["rf", "hgb", "lr"]
    assert champ["spec"]["ticket"] is False
    assert champ["tuned"]["hgb"]["max_iter"] == 1165
    assert champ["tuned"]["rf"]["n_estimators"] == 661
    assert abs(champ["cv_mean"] - 0.8370) < 0.002


def test_kaggle_submissions_recorded():
    text = (RESULTS / "kaggle_submissions.txt").read_text()
    for score in KAGGLE_SCORES:
        assert score in text, f"submission score {score} not recorded"


def test_benchmark_artifacts_valid():
    files = sorted(glob.glob(str(RESULTS / "benchmark_*.json")))
    assert files, "no benchmark artifact; run scripts/benchmark.py"
    for path in files:
        report = json.load(open(path))
        host = report["hostname"]
        assert report["environment"]["python"].startswith("3.12")
        champ = report["battery"]["champion_5fold_cv"]
        assert 0.80 < champ["mean"] < 0.87, f"{host}: champion mean {champ['mean']}"
        assert len(champ["fold_scores"]) == 5
        for member in ["tuned_hgb_screen", "tuned_rf_screen"]:
            assert 0.80 < report["battery"][member]["mean"] < 0.87
            assert report["battery"][member]["wall_seconds"] > 0


def test_per_fold_arrays_match_recorded_means():
    """Every committed scores_*.npy must reproduce its results.jsonl mean."""
    events = _load_events()
    by_key = {}
    for e in events:
        if "mean" in e:
            by_key[(e.get("event"), e.get("task"))] = e["mean"]

    npys = sorted(glob.glob(str(RESULTS / "scores_*.npy")))
    assert npys, "no per-fold score arrays committed"
    for path in npys:
        stem = Path(path).stem  # scores_<event>_t<task>
        event_name, _, task_str = stem.rpartition("_t")
        event_name = event_name[len("scores_"):]
        key = (event_name, int(task_str))
        assert key in by_key, f"{stem}: no matching results.jsonl event"
        arr_mean = float(np.load(path).mean())
        assert abs(arr_mean - by_key[key]) < 1e-9, (
            f"{stem}: array mean {arr_mean:.12f} != recorded {by_key[key]:.12f}")


def test_submission_csv_is_valid():
    import pandas as pd
    sub = pd.read_csv("submission.csv")
    assert list(sub.columns) == ["PassengerId", "Survived"]
    assert len(sub) == 418
    assert set(sub["Survived"].unique()) <= {0, 1}
    test = pd.read_csv("data/test.csv")
    assert sub["PassengerId"].tolist() == test["PassengerId"].tolist()
