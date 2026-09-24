"""Experiment: greedy stack composition (task 6 step 2).

Repeat testing tool: tries adding each candidate to the champion stack with
paired gates. pipeline/run_all.py embeds this logic; this standalone copy
exists so you can re-run composition without the full pipeline. Run:
    OMP_NUM_THREADS=8 python experiments/t6_stack.py
"""

import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import warnings

warnings.filterwarnings("ignore")

from src.evaluate import cv_scores, paired_test
from src.features import BASE_CAT, BASE_NUM, load_data
from src.model import build_stack

train, _ = load_data()
y = train["Survived"]
X = train[BASE_NUM + BASE_CAT]

champ_members = ["rf", "hgb", "lr"]
candidates = ["xgb", "cat", "mlp", "svc", "lgbm"]


def stack(members):
    spec = {"members": members, "avg_seeds": {}, "ticket": False, "farepp": False,
            "imputer": "median", "target_enc": False, "hygiene": False}
    return build_stack(spec=spec)


s_champ = cv_scores(stack(champ_members), X, y, mode="half")
print(f"Champion (25-fold): {s_champ.mean():.4f}", flush=True)

for name in candidates:
    trial = champ_members + [name]
    s_trial = cv_scores(stack(trial), X, y, mode="half")
    diff, t, p = paired_test(s_trial, s_champ)
    verdict = "PROMISING (retest at 50 folds)" if (diff > 0.001 and p < 0.30) else "reject"
    print(f"  +{name}: {s_trial.mean():.4f} diff={diff:+.4f} p={p:.3f} -> {verdict}", flush=True)
