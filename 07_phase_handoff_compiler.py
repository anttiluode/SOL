"""SOL-07: Phase Handoff Compiler — optimize the handoff, not one stage alone.

A four-phase state is temporarily merged into two phase wells and then returned
to four-well locking. The locally obvious coarse wells sit halfway between the
source phases, but those same midpoints are exactly the separatrices of the next
fine stage: stage one looks maximally comfortable while the composition is a coin
flip.

The compiler chooses the coarse-well offset that maximizes the *minimum margin
across both stages*. For this C4 -> C2 -> C4 merge, it is alpha = pi/8.

This is a reduced phase/quantizer model, not a hardware claim. The design rule
applies more broadly to cascaded analog quantizers, oscillator states, relocking
stages, and any pipeline where one stage's legal output is the next stage's input.
"""
from __future__ import annotations

import argparse
import numpy as np


def wrap(phi: np.ndarray | float) -> np.ndarray:
    return (np.asarray(phi) + np.pi) % (2 * np.pi) - np.pi


def angular_distance(a: float, b: float) -> float:
    return float(abs(wrap(a - b)))


def nearest_well(phi: np.ndarray, wells: np.ndarray) -> np.ndarray:
    d = np.abs(wrap(phi[..., None] - wells[None, ...]))
    return np.argmin(d, axis=-1)


def composition_margins(alpha: float) -> tuple[float, float, float]:
    """Return capture margin, restore margin, and their minimum.

    Fine source phases 0 and pi/2 are supposed to merge into coarse well alpha.
    A two-well stage has basin half-width pi/2. Restoring C4 gives the desired
    target 0 a basin half-width pi/4. The opposite pair is symmetric.
    """
    sources = (0.0, np.pi / 2)
    capture = min(np.pi / 2 - angular_distance(s, alpha) for s in sources)
    restore = np.pi / 4 - angular_distance(alpha, 0.0)
    return float(capture), float(restore), float(min(capture, restore))


def compile_alpha(samples: int = 2001) -> tuple[float, float]:
    grid = np.linspace(0.0, np.pi / 4, samples)
    scores = np.array([composition_margins(a)[2] for a in grid])
    i = int(np.argmax(scores))
    return float(grid[i]), float(scores[i])


def simulate(alpha: float, sigma: float = 0.08, trials: int = 80000, seed: int = 3) -> float:
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 4, trials)
    phi = src * (np.pi / 2)

    # Stage 1: C4 -> C2. Add perturbation before capture into the coarse wells.
    coarse = np.array([alpha, alpha + np.pi])
    ci = nearest_well(phi + rng.normal(0.0, sigma, trials), coarse)

    # Stage 2: restore C4. Add another perturbation at the inter-stage handoff.
    coarse_state = coarse[ci] + rng.normal(0.0, sigma, trials)
    fine = np.arange(4) * (np.pi / 2)
    out = nearest_well(coarse_state, fine)

    desired = np.where(src < 2, 0, 2)
    return float(np.mean(out == desired))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma", type=float, default=0.08, help="phase noise std in radians")
    ap.add_argument("--trials", type=int, default=80000)
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    compiled, score = compile_alpha()
    candidates = [np.pi / 16, compiled, 3 * np.pi / 16, np.pi / 4]

    print("=" * 78)
    print("SOL-07  PHASE HANDOFF COMPILER — compile the composition, not one stage")
    print("=" * 78)
    print("task: C4 phases {0,pi/2} -> 0 and {pi,3pi/2} -> pi, then restore C4")
    print(f"noise sigma              : {args.sigma:.4f} rad")
    print(f"Monte Carlo trials       : {args.trials}")
    print(f"compiled alpha           : {compiled / np.pi:.6f} * pi")
    print(f"expected optimum         : 0.125000 * pi")
    print(f"worst handoff margin     : {score / np.pi:.6f} * pi")

    print("\n alpha/pi | capture margin/pi | restore margin/pi | end-to-end accuracy")
    print("----------+-------------------+-------------------+--------------------")
    rows = []
    for a in candidates:
        cm, rm, mm = composition_margins(float(a))
        acc = simulate(float(a), args.sigma, args.trials, args.seed)
        rows.append((float(a), cm, rm, mm, acc))
        print(f" {a/np.pi:8.4f} | {cm/np.pi:17.4f} | {rm/np.pi:17.4f} | {100*acc:17.3f}%")

    midpoint = rows[-1]
    compiled_row = min(rows, key=lambda r: abs(r[0] - compiled))
    ok = (
        abs(compiled - np.pi / 8) < 1e-3
        and compiled_row[4] > 0.98
        and 0.45 < midpoint[4] < 0.55
        and midpoint[2] < 1e-10
    )
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("\nThe midpoint pi/4 is locally attractive because it maximizes the first-stage")
    print("capture margin. But it has zero restore margin: it hands the next stage a")
    print("state exactly on a fine-state boundary. pi/8 sacrifices first-stage comfort")
    print("to maximize the worst margin of the whole composition.")


if __name__ == "__main__":
    main()
