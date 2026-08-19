"""SOL-04: Arrow Monitor — a tiny directional anomaly detector for multichannel streams.

Reads the skew half of a lagged covariance A_tau=(C_tau-C_tau.T)/2.  This is the
GeometricNeuron V9 "arrow" stripped of biology and used as an engineering
instrument: detect when the ordering/circulation of a process changes.
"""
from __future__ import annotations
import argparse
import csv
import numpy as np


def lag_skew(X: np.ndarray, tau: int = 1) -> np.ndarray:
    X = np.asarray(X, dtype=float)
    X = X - X.mean(axis=0, keepdims=True)
    C = X[tau:].T @ X[:-tau] / max(1, len(X) - tau)
    return 0.5 * (C - C.T)


def arrow_features(X: np.ndarray, tau: int = 1) -> dict[str, float]:
    A = lag_skew(X, tau)
    eig = np.linalg.eigvals(A)
    skew_energy = float(np.sum(np.abs(eig.imag)))
    # Orientation of the dominant 2-D rotation plane. Sign is basis-convention
    # dependent, so for monitoring we compare A against a learned reference.
    return {"skew_energy": skew_energy, "fro": float(np.linalg.norm(A, "fro"))}


def cosine_matrix(A: np.ndarray, B: np.ndarray) -> float:
    return float(np.sum(A * B) / (np.linalg.norm(A) * np.linalg.norm(B) + 1e-12))


def simulate(n=6, steps=6000, flip_at=3000, coupling=0.27, seed=0):
    rng = np.random.default_rng(seed)
    X = np.zeros((steps, n))
    X[0] = rng.standard_normal(n)
    for t in range(1, steps):
        forward = t < flip_at
        src = np.roll(X[t - 1], 1 if forward else -1)
        X[t] = 0.72 * X[t - 1] + coupling * src + 0.18 * rng.standard_normal(n)
    return X


def read_csv(path: str) -> np.ndarray:
    with open(path, newline="") as f:
        rows = list(csv.reader(f))
    vals = []
    for row in rows:
        try:
            vals.append([float(x) for x in row])
        except ValueError:
            continue  # allow one header row
    X = np.asarray(vals, dtype=float)
    if X.ndim != 2 or X.shape[1] < 2:
        raise ValueError("CSV needs at least two numeric columns")
    return X


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", type=str, default=None, help="optional multichannel numeric CSV")
    ap.add_argument("--tau", type=int, default=1)
    ap.add_argument("--window", type=int, default=800)
    args = ap.parse_args()

    if args.csv:
        X = read_csv(args.csv)
        split = len(X) // 2
        label = args.csv
    else:
        X = simulate()
        split = len(X) // 2
        label = "simulated directed process with a hidden direction reversal"

    A0 = lag_skew(X[:split], args.tau)
    A1 = lag_skew(X[split:], args.tau)
    Ar = lag_skew(X[:split][::-1], args.tau)
    cos_change = cosine_matrix(A0, A1)
    cos_reverse = cosine_matrix(A0, Ar)

    print("=" * 74)
    print("SOL-04  ARROW MONITOR — directional fingerprint of a multichannel stream")
    print("=" * 74)
    print(f"source              : {label}")
    print(f"channels / samples  : {X.shape[1]} / {X.shape[0]}")
    print(f"lag tau             : {args.tau}")
    print(f"reference skew norm : {np.linalg.norm(A0):.5f}")
    print(f"second-half cosine  : {cos_change:+.4f}  (+1 same arrow, -1 reversed arrow)")
    print(f"literal time-reverse: {cos_reverse:+.4f}")

    # Sliding monitor against first-window reference.
    w = min(args.window, max(20, len(X) // 4))
    Aref = lag_skew(X[:w], args.tau)
    print("\nsliding arrow similarity:")
    stride = max(w // 2, 1)
    for start in range(0, len(X) - w + 1, stride):
        A = lag_skew(X[start:start + w], args.tau)
        c = cosine_matrix(Aref, A)
        flag = "  <-- DIRECTIONAL CHANGE" if c < 0 else ""
        print(f"  samples {start:5d}:{start+w:5d}  cosine={c:+.3f}{flag}")

    print("\nUse case: motor/process/audio/EEG-style multichannel monitoring where RMS or")
    print("spectrum may stay similar while the *temporal ordering between channels* changes.")
    print("This is an anomaly fingerprint, not a causality proof.")


if __name__ == "__main__":
    main()
