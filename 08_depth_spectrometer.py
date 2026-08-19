from __future__ import annotations

import argparse
import itertools
import json
import math
from dataclasses import dataclass, asdict

import numpy as np


@dataclass(frozen=True)
class DepthSpectrometerConfig:
    rho: float = 0.88
    thetas: tuple[float, ...] = (0.25, 0.28, 0.31, 0.34, 0.37, 0.40)
    sigma: float = 0.06
    max_depth: int = 40
    observations: int = 3


def port_means(cfg: DepthSpectrometerConfig) -> np.ndarray:
    """Mean scalar port trace for damped rotations.

    Each hidden machine is

        x[n+1] = rho R(theta) x[n],  x[0] = (1,0)
        y[n]   = first coordinate of x[n]

    so y[n] = rho**n cos(n theta).

    Every candidate has the same infinite-depth fixed point x*=0.  Only its
    transient trajectory identifies theta.
    """
    t = np.arange(1, cfg.max_depth + 1, dtype=np.float64)
    return np.asarray(
        [cfg.rho**t * np.cos(theta * t) for theta in cfg.thetas],
        dtype=np.float64,
    )


def min_pairwise_information(
    means: np.ndarray,
    depths: tuple[int, ...] | list[int],
    sigma: float,
) -> float:
    """Worst pairwise squared separation in noise units for selected depths."""
    cols = np.asarray(depths, dtype=np.int64) - 1
    v = means[:, cols]
    return float(
        min(
            np.sum((v[i] - v[j]) ** 2) / (sigma * sigma)
            for i, j in itertools.combinations(range(v.shape[0]), 2)
        )
    )


def choose_depths(cfg: DepthSpectrometerConfig, means: np.ndarray | None = None) -> tuple[int, ...]:
    """Exhaustively choose depths maximizing the worst pairwise separation.

    The search is tiny for the demo: C(40,3)=9880 sets.  This is an active
    *observation-depth* choice, not training and not an architecture change.
    """
    means = port_means(cfg) if means is None else means
    candidates = itertools.combinations(range(1, cfg.max_depth + 1), cfg.observations)
    return max(candidates, key=lambda ds: min_pairwise_information(means, ds, cfg.sigma))


def classify_accuracy(
    cfg: DepthSpectrometerConfig,
    depths: tuple[int, ...] | list[int],
    trials: int = 50_000,
    seed: int = 806,
    means: np.ndarray | None = None,
) -> float:
    """Nearest-template identification accuracy under iid Gaussian port noise."""
    means = port_means(cfg) if means is None else means
    rng = np.random.default_rng(seed)
    cols = np.asarray(depths, dtype=np.int64) - 1
    truth = rng.integers(0, len(cfg.thetas), size=trials)
    observed = means[truth][:, cols] + rng.normal(0.0, cfg.sigma, size=(trials, len(cols)))
    errors = np.sum((observed[:, None, :] - means[None, :, cols]) ** 2, axis=2)
    pred = np.argmin(errors, axis=1)
    return float(np.mean(pred == truth))


def fisher_by_depth(cfg: DepthSpectrometerConfig) -> np.ndarray:
    """Average scalar Fisher information for theta at each observation depth."""
    t = np.arange(1, cfg.max_depth + 1, dtype=np.float64)
    deriv = np.asarray(
        [-t * cfg.rho**t * np.sin(theta * t) for theta in cfg.thetas],
        dtype=np.float64,
    )
    return np.mean((deriv / cfg.sigma) ** 2, axis=0)


def random_depth_baseline(
    cfg: DepthSpectrometerConfig,
    sets: int = 200,
    trials_per_set: int = 3_000,
    seed: int = 809,
    means: np.ndarray | None = None,
) -> dict:
    means = port_means(cfg) if means is None else means
    rng = np.random.default_rng(seed)
    acc = []
    chosen = []
    for i in range(sets):
        ds = tuple(
            sorted(
                int(x)
                for x in rng.choice(
                    np.arange(1, cfg.max_depth + 1),
                    size=cfg.observations,
                    replace=False,
                )
            )
        )
        chosen.append(ds)
        acc.append(
            classify_accuracy(
                cfg,
                ds,
                trials=trials_per_set,
                seed=1000 + i,
                means=means,
            )
        )
    a = np.asarray(acc, dtype=np.float64)
    return {
        "sets": int(sets),
        "mean_accuracy": float(np.mean(a)),
        "median_accuracy": float(np.median(a)),
        "p10_accuracy": float(np.percentile(a, 10)),
        "p90_accuracy": float(np.percentile(a, 90)),
        "best_accuracy": float(np.max(a)),
        "best_depths": list(chosen[int(np.argmax(a))]),
    }


def demo(cfg: DepthSpectrometerConfig | None = None) -> dict:
    cfg = DepthSpectrometerConfig() if cfg is None else cfg
    means = port_means(cfg)
    selected = choose_depths(cfg, means)
    final_depths = tuple([cfg.max_depth] * cfg.observations)
    early_depths = tuple(range(1, cfg.observations + 1))
    fisher = fisher_by_depth(cfg)
    peak_fisher_depth = int(np.argmax(fisher) + 1)

    payload = {
        "name": "SOL-08 Depth Spectrometer",
        "config": asdict(cfg),
        "system": "x[n+1]=rho*R(theta)*x[n], y[n]=x0[n]; all candidates converge to x*=0",
        "selected_depths": list(selected),
        "selected_min_pairwise_information": min_pairwise_information(means, selected, cfg.sigma),
        "selected_accuracy": classify_accuracy(cfg, selected, seed=806, means=means),
        "final_depth_repeated": list(final_depths),
        "final_depth_accuracy": classify_accuracy(cfg, final_depths, seed=807, means=means),
        "first_depths": list(early_depths),
        "first_depths_accuracy": classify_accuracy(cfg, early_depths, seed=808, means=means),
        "random_depths": random_depth_baseline(cfg, means=means),
        "rho_power_max_depth": float(cfg.rho ** cfg.max_depth),
        "average_fisher_peak_depth": peak_fisher_depth,
        "continuous_envelope_peak": float(-1.0 / math.log(cfg.rho)),
        "interpretation": {
            "positive": "Different contracting machines can share the same fixed point while remaining identifiable in a finite transient computation-depth window.",
            "negative": "This is classical transient system identification in a computation-depth coordinate; it is not a claim that transient probing is new.",
            "use": "Treat iteration number as a probe variable. Choose the depths that make hidden mechanisms distinguishable instead of automatically reading only the converged state.",
        },
    }
    return payload


def main() -> None:
    p = argparse.ArgumentParser(description="Probe a convergent iterative machine across computation depth")
    p.add_argument("--json", action="store_true")
    args = p.parse_args()
    result = demo()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return
    print("SOL-08 — DEPTH SPECTROMETER")
    print("all six hidden machines converge to the same fixed point: x*=0")
    print("selected observation depths:", result["selected_depths"])
    print(f"selected-depth accuracy: {100*result['selected_accuracy']:.2f}%")
    print(f"repeat final depth accuracy: {100*result['final_depth_accuracy']:.2f}%")
    print(f"random 3-depth mean: {100*result['random_depths']['mean_accuracy']:.2f}%")
    print("average Fisher-information peak depth:", result["average_fisher_peak_depth"])
    print("continuous t*rho^t envelope peak:", f"{result['continuous_envelope_peak']:.2f}")


if __name__ == "__main__":
    main()
