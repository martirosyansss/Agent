"""
Hyper-parameter tuning script for the MLPredictor stack.

Walks the search space defined below using the Phase-7 tuner
(``analyzer.ml.orchestration.hparam_tuner``), scoring each candidate
by the Phase-1 gate metric (PSR on predicted-positive PnL) — the
strictest single-number signal of "will this actually trade well in
production?".

Typical operator workflow::

    # 1. Snapshot the trade universe once (slow: full backtest replay)
    python -m scripts.train_ml --dump-trades data/trades.pkl

    # 2. Tune over the dump (fast: just refits the ML stack per trial)
    python -m scripts.tune_ml --trades data/trades.pkl --mode random \\
        --n-trials 40 --seed 7

    # 3. Apply the winning config (copy-paste MLConfig overrides)
    python -m scripts.train_ml --mlconfig tuned.json

Three modes:

- ``grid``   — exhaustive over the search space (errors if > max-trials)
- ``random`` — ``n_trials`` uniform samples (Bergstra & Bengio 2012)
- ``halving`` — start with ``n_initial``, keep top 1/halving each round

All modes score by the same objective so results are directly
comparable. Trials are evaluated serially — the inner trainer already
parallelises per-member fits, adding another layer usually OOMs on
tree ensembles.

Exit code 0 on success; non-zero only when the trade file is missing
or completely empty. Individual bad configs are recorded as failed
trials, not fatal errors.
"""

from __future__ import annotations

import argparse
import json
import logging
import pickle
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

from analyzer.ml.orchestration.hparam_tuner import (
    grid_search,
    random_search,
    successive_halving,
)


logger = logging.getLogger("tune_ml")


# Default search space — edited in-file rather than via CLI because every
# axis has meaningful interactions (e.g. high learning_rate wants lower
# max_depth). A config-file-driven version is a follow-up.
DEFAULT_SPACE: dict[str, list[Any]] = {
    "n_estimators": [150, 200, 250, 300, 400],
    "max_depth": [4, 6, 8, 10],
    "learning_rate": [0.02, 0.03, 0.05, 0.08],
    "min_child_samples": [10, 20, 30],
    "subsample": [0.7, 0.8, 0.9],
    "colsample_bytree": [0.7, 0.8, 0.9],
    "temporal_decay": [0.001, 0.003, 0.005, 0.01],
}


def _build_objective(trades: list[Any], verbose: bool) -> Any:
    """Return a closure that trains on ``trades`` and scores the fit.

    The trainer is imported lazily inside so the tuner can at least
    import even when the ML stack can't (useful for dry-runs that
    exercise the search-space wiring without touching sklearn).
    """
    def objective(params: dict[str, Any]) -> float:
        # Late imports so --help / the scoring toy path don't pull heavy deps.
        from analyzer.ml.domain.config import MLConfig
        from analyzer.ml_predictor import MLPredictor

        cfg = MLConfig(**{**_defaults(), **params})
        predictor = MLPredictor(cfg=cfg)
        metrics = predictor.train(trades)
        # Score: prefer PSR gate, fall back to skill_score, then precision.
        psr = float(getattr(metrics, "psr", 0.0) or 0.0)
        skill = float(getattr(metrics, "skill_score", 0.0) or 0.0)
        prec = float(getattr(metrics, "precision", 0.0) or 0.0)
        # Blend so a marginal-PSR config with great precision still scores.
        score = 0.6 * psr + 0.3 * skill + 0.1 * prec
        if verbose:
            logger.info(
                "  → score=%.4f (psr=%.3f skill=%.3f prec=%.3f)",
                score, psr, skill, prec,
            )
        return score
    return objective


def _defaults() -> dict[str, Any]:
    """MLConfig fields not in the search space stay at their defaults."""
    from analyzer.ml.domain.config import MLConfig
    return asdict(MLConfig())


def _toy_objective(params: dict[str, Any]) -> float:
    """Deterministic surrogate score used by --dry-run to smoke-test wiring
    without training real models. Peaks at lr=0.03, depth=6."""
    lr = params.get("learning_rate", 0.05)
    depth = params.get("max_depth", 8)
    subs = params.get("subsample", 0.8)
    return -((lr - 0.03) * 20) ** 2 - (depth - 6) ** 2 - (subs - 0.8) ** 2


def _load_trades(path: Path) -> list[Any]:
    with open(path, "rb") as f:
        trades = pickle.load(f)
    if not isinstance(trades, list) or not trades:
        raise RuntimeError(f"trade dump {path} is empty or not a list")
    logger.info("loaded %d trades from %s", len(trades), path)
    return trades


def _parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[1])
    ap.add_argument("--mode", choices=["grid", "random", "halving"], default="random")
    ap.add_argument("--trades", type=Path, help="pickled list[StrategyTrade] — required unless --dry-run")
    ap.add_argument("--dry-run", action="store_true",
                    help="Use a synthetic objective (no training). Smoke-tests wiring.")
    ap.add_argument("--n-trials", type=int, default=40)
    ap.add_argument("--n-initial", type=int, default=16, help="halving: initial candidates")
    ap.add_argument("--rounds", type=int, default=3, help="halving: number of rounds")
    ap.add_argument("--halving-ratio", type=float, default=2.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output", type=Path, default=Path("data/tuning_results.json"))
    ap.add_argument("--top-k", type=int, default=5)
    return ap.parse_args()


def main() -> int:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s | %(message)s")

    if args.dry_run:
        objective = _toy_objective
    else:
        if args.trades is None or not args.trades.exists():
            logger.error("--trades is required unless --dry-run is set")
            return 2
        trades = _load_trades(args.trades)
        objective = _build_objective(trades, verbose=True)

    if args.mode == "grid":
        result = grid_search(objective, DEFAULT_SPACE, max_trials=args.n_trials, verbose=True)
    elif args.mode == "random":
        result = random_search(
            objective, DEFAULT_SPACE, n_trials=args.n_trials, seed=args.seed, verbose=True,
        )
    else:   # halving
        result = successive_halving(
            objective, DEFAULT_SPACE,
            n_initial=args.n_initial, rounds=args.rounds,
            halving_ratio=args.halving_ratio, seed=args.seed, verbose=True,
        )

    logger.info("=" * 60)
    logger.info("best score: %.4f", result.best_score)
    logger.info("best params: %s", result.best_params)
    logger.info("%d trials in %.1fs (%d failures)",
                result.n_trials, result.duration_s,
                sum(1 for t in result.trials if t.error))

    top = result.top_k(args.top_k)
    logger.info("top %d:", len(top))
    for rank, trial in enumerate(top, start=1):
        logger.info("  %d. score=%.4f params=%s", rank, trial.score, trial.params)

    # Persist for reproducibility — a training run later can cite the
    # tuning result file in its manifest.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "mode": args.mode,
        "n_trials": result.n_trials,
        "duration_s": result.duration_s,
        "best_score": result.best_score,
        "best_params": result.best_params,
        "top_k": [
            {"rank": i + 1, "score": t.score, "params": t.params, "duration_s": t.duration_s}
            for i, t in enumerate(top)
        ],
        "failures": [
            {"params": t.params, "error": t.error}
            for t in result.trials if t.error
        ][:50],
    }
    args.output.write_text(json.dumps(payload, indent=2, default=str))
    logger.info("saved → %s", args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
