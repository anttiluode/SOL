"""SOL-05: Zombie Memory Audit — test whether a claimed reset really forgot.

A recurrent system can make two histories output-identical at the reset event while
leaving a hidden distinction in state. If later dynamics are invertible/lossless,
a common future suffix can rotate that hidden distinction back into the readout.

This demo searches for such an exposing suffix. It then compares the fake reset
with a true singular "pinch" that actually destroys the distinction.

The utility is generic: pass token-labelled state matrices and a readout vector to
``search_exposing_suffix``. It is a small adversarial reset/unlearning audit,
not a claim about any particular neural architecture.
"""
from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass
from typing import Mapping
import numpy as np


@dataclass
class ExposureResult:
    found: bool
    word: tuple[str, ...]
    output_gap: float
    hidden_gap: float
    searched: int


def rotation(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, -s], [s, c]], dtype=float)


def search_exposing_suffix(
    h_a: np.ndarray,
    h_b: np.ndarray,
    operators: Mapping[str, np.ndarray],
    readout: np.ndarray,
    *,
    max_depth: int = 6,
    threshold: float = 0.25,
) -> ExposureResult:
    """Breadth-first search for one common suffix that exposes hidden history.

    The same operator word is applied to both histories. A reset is behaviorally
    suspicious when the current output gap is tiny but some common future word
    makes the output gap large again.
    """
    h_a = np.asarray(h_a, dtype=float)
    h_b = np.asarray(h_b, dtype=float)
    w = np.asarray(readout, dtype=float)
    if h_a.shape != h_b.shape or w.shape != h_a.shape:
        raise ValueError("states and readout must have the same 1-D shape")

    d0 = h_a - h_b
    best_gap = float(abs(w @ d0))
    best_word: tuple[str, ...] = ()
    hidden_gap = float(np.linalg.norm(d0))
    q = deque([(d0, ())])
    searched = 0

    while q:
        d, word = q.popleft()
        searched += 1
        gap = float(abs(w @ d))
        if gap > best_gap:
            best_gap, best_word = gap, word
        if word and gap >= threshold:
            return ExposureResult(True, word, gap, hidden_gap, searched)
        if len(word) >= max_depth:
            continue
        for name, op in operators.items():
            op = np.asarray(op, dtype=float)
            if op.shape != (d.size, d.size):
                raise ValueError(f"operator {name!r} has wrong shape")
            q.append((op @ d, word + (name,)))

    return ExposureResult(False, best_word, best_gap, hidden_gap, searched)


def demo(max_depth: int = 5, threshold: float = 0.5) -> tuple[ExposureResult, ExposureResult]:
    # Two different histories before the claimed reset.
    h_a = np.array([1.0, 0.0])
    h_b = np.array([-1.0, 0.0])
    readout = np.array([1.0, 0.0])

    # "Fake reset": a lossless quarter-turn moves the distinction into the
    # readout nullspace. Output says forgotten; hidden distance is unchanged.
    fake_reset = rotation(np.pi / 2)
    f_a, f_b = fake_reset @ h_a, fake_reset @ h_b

    # Future token dynamics. The auditor does not give the two histories
    # different inputs; it searches one common suffix.
    ops = {
        "hold": rotation(0.0),
        "quarter": rotation(-np.pi / 2),
        "eighth": rotation(np.pi / 4),
    }
    fake = search_exposing_suffix(
        f_a, f_b, ops, readout, max_depth=max_depth, threshold=threshold
    )

    # "True reset": singular projection kills the hidden coordinate that carries
    # the distinction. Once both states coincide, no common linear suffix can
    # resurrect history that is no longer there.
    pinch = np.array([[1.0, 0.0], [0.0, 0.0]])
    t_a, t_b = pinch @ f_a, pinch @ f_b
    true = search_exposing_suffix(
        t_a, t_b, ops, readout, max_depth=max_depth, threshold=threshold
    )
    return fake, true


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--threshold", type=float, default=0.5)
    args = ap.parse_args()

    fake, true = demo(args.depth, args.threshold)

    print("=" * 76)
    print("SOL-05  ZOMBIE MEMORY AUDIT — can a common suffix resurrect a reset history?")
    print("=" * 76)
    print("\nFAKE / LOSSLESS RESET")
    print(f"hidden history gap after reset : {fake.hidden_gap:.6f}")
    print("visible output gap at reset     : ~0")
    print(f"exposing suffix found           : {fake.found}")
    print(f"shortest exposing suffix        : {list(fake.word)}")
    print(f"output gap after suffix         : {fake.output_gap:.6f}")
    print(f"states searched                 : {fake.searched}")

    print("\nTRUE / DESTRUCTIVE RESET")
    print(f"hidden history gap after reset : {true.hidden_gap:.6f}")
    print(f"exposing suffix found           : {true.found}")
    print(f"largest output gap searched     : {true.output_gap:.6e}")
    print(f"states searched                 : {true.searched}")

    ok = fake.found and not true.found and fake.hidden_gap > 1.0 and true.hidden_gap < 1e-10
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("\nInterpretation: equality at the output is not evidence that a recurrent")
    print("state has forgotten. Search a common future suffix. If old history can")
    print("reappear, the reset hid information; it did not destroy it.")


if __name__ == "__main__":
    main()
