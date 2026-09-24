"""Model construction and end to end smoke tests."""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, cross_val_score

from src.features import add_base_features, load_data
from src.model import CHAMPION_TUNED, HGB_BASE, build_stack, columns, make_member, make_pre


def test_make_member_constructs_every_name():
    for name in ["rf", "hgb", "lr", "xgb", "cat", "mlp", "svc", "lgbm"]:
        est = make_member(name, seed=0, tuned={"hgb": {}, "rf": {}})
        assert est is not None


def test_champion_tuned_params_override():
    rf = make_member("rf", seed=0, tuned=CHAMPION_TUNED)
    assert rf.n_estimators == 661
    hgb = make_member("hgb", seed=0, tuned=CHAMPION_TUNED)
    assert hgb.max_iter == 1165


def test_preprocessor_output_shape():
    train, _ = load_data()
    num, cat = columns()
    pre = make_pre(num, cat)
    out = pre.fit_transform(train[num + cat], train["Survived"])
    # 7 numeric + Sex(2) + Embarked(3) + Title(5) + Deck(9) = 26
    assert out.shape[1] == 26


def test_champion_fits_and_predicts():
    train, test = load_data()
    num, cat = columns()
    model = build_stack(tuned=CHAMPION_TUNED)
    X = train[num + cat].head(300)
    y = train["Survived"].head(300)
    model.fit(X, y)
    proba = model.predict_proba(test[num + cat].head(50))
    assert proba.shape == (50, 2)
    np.testing.assert_allclose(proba.sum(axis=1), 1.0, atol=1e-6)
    preds = model.predict(test[num + cat].head(50))
    assert set(preds) <= {0, 1}


def test_hgb_reference_score_is_stable():
    """Broad regression guard against environment or version drift.

    The untuned HGB at 3-fold CV on the champion features should sit in a
    wide but bounded band. Not a golden value: BLAS differences across
    machines move the third decimal.
    """
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.pipeline import Pipeline

    train, _ = load_data()
    num, cat = columns()
    model = Pipeline([
        ("preprocess", make_pre(num, cat)),
        ("clf", HistGradientBoostingClassifier(random_state=42, **HGB_BASE)),
    ])
    X, y = train[num + cat], train["Survived"]
    scores = cross_val_score(model, X, y, cv=StratifiedKFold(3, shuffle=True, random_state=0))
    assert 0.70 < scores.mean() < 0.90
