"""Shared local-scatter and diagnosis primitives for SOL demos.

The core object is deliberately small: a ring of local reciprocal 2-port
scatterers.  The recurrence comes from KYY; exact forward sensitivities and
identifiability-minded diagnosis come from TransientWaveCompiler.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence
import numpy as np


def ring_edges(n: int) -> list[tuple[int, int]]:
    if n < 4:
        raise ValueError("n must be >= 4")
    return [(i, (i + 1) % n) for i in range(n)]


def phase_order(n: int) -> list[int]:
    """Checkerboard-ish order used by KYY: even edges, odd edges, wrap edge last."""
    edges = ring_edges(n)
    wrap = n - 1
    even = [e for e, (i, j) in enumerate(edges) if e != wrap and min(i, j) % 2 == 0]
    odd = [e for e, (i, j) in enumerate(edges) if e != wrap and min(i, j) % 2 == 1]
    return even + odd + [wrap]


def scatter_pair(a: float, b: float, theta: float) -> tuple[float, float]:
    c, s = np.cos(theta), np.sin(theta)
    return c * a + s * b, s * a - c * b


def scatter_pair_derivative(a: float, b: float, theta: float) -> tuple[float, float]:
    """d/dtheta of the symmetric orthogonal two-port scatter cell."""
    c, s = np.cos(theta), np.sin(theta)
    return -s * a + c * b, c * a + s * b


def scatter_sweep(
    h: np.ndarray,
    theta: np.ndarray,
    J: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """One local reciprocal sweep, optionally propagating exact dh/dtheta.

    h: (N,)
    theta: (E,), one persistent parameter per physical edge
    J: (N,E), sensitivity of incoming h to all edge parameters
    """
    h = np.asarray(h, dtype=float).copy()
    n = h.size
    edges = ring_edges(n)
    if theta.shape != (len(edges),):
        raise ValueError(f"theta must have shape ({len(edges)},)")
    if J is not None:
        J = np.asarray(J, dtype=float).copy()
        if J.shape != (n, len(edges)):
            raise ValueError("J has wrong shape")

    for e in phase_order(n):
        i, j = edges[e]
        a, b = float(h[i]), float(h[j])
        c, s = np.cos(theta[e]), np.sin(theta[e])

        if J is not None:
            Ji = J[i].copy()
            Jj = J[j].copy()
            J[i] = c * Ji + s * Jj
            J[j] = s * Ji - c * Jj
            da, db = scatter_pair_derivative(a, b, theta[e])
            J[i, e] += da
            J[j, e] += db

        h[i] = c * a + s * b
        h[j] = s * a - c * b
    return h, J


def make_pulse_probe(n: int, node: int, steps: int = 5, amplitude: float = 1.0) -> np.ndarray:
    if not 0 <= node < n:
        raise ValueError("node out of range")
    u = np.zeros((steps, n), dtype=float)
    u[0, node] = amplitude
    return u


def make_bipolar_probe(n: int, a: int, b: int, steps: int = 5) -> np.ndarray:
    u = np.zeros((steps, n), dtype=float)
    u[0, a] = 1.0
    u[0, b] = -1.0
    return u


def run_probe(
    theta: np.ndarray,
    injections: np.ndarray,
    sweeps: int = 2,
    ports: Sequence[int] | None = None,
    exact_jacobian: bool = True,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Run a transient probe and read final port values plus exact Jacobian."""
    injections = np.asarray(injections, dtype=float)
    if injections.ndim != 2 or injections.shape[1] != theta.size:
        raise ValueError("injections must have shape (steps, state_dim) and state_dim=edge_count for ring")
    n = theta.size
    h = np.zeros(n, dtype=float)
    J = np.zeros((n, theta.size), dtype=float) if exact_jacobian else None
    for u in injections:
        h = h + u
        for _ in range(sweeps):
            h, J = scatter_sweep(h, theta, J)
    if ports is None:
        ports = tuple(range(n))
    ports = np.asarray(ports, dtype=int)
    y = h[ports]
    Jy = J[ports] if J is not None else None
    return y, Jy


def stack_experiment(
    theta: np.ndarray,
    probes: Iterable[np.ndarray],
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
) -> tuple[np.ndarray, np.ndarray]:
    ys, Js = [], []
    for p in probes:
        y, J = run_probe(theta, p, sweeps=sweeps, ports=ports, exact_jacobian=True)
        ys.append(y)
        Js.append(J)
    return np.concatenate(ys), np.vstack(Js)


def measure_experiment(
    theta: np.ndarray,
    probes: Iterable[np.ndarray],
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
    noise_std: float = 0.0,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    y, _ = stack_experiment(theta, probes, sweeps=sweeps, ports=ports)
    if noise_std > 0:
        rng = np.random.default_rng() if rng is None else rng
        y = y + noise_std * rng.standard_normal(y.shape)
    return y


def ridge_solve(J: np.ndarray, dy: np.ndarray, lam: float = 1e-8) -> np.ndarray:
    JTJ = J.T @ J
    return np.linalg.solve(JTJ + lam * np.eye(JTJ.shape[0]), J.T @ dy)


@dataclass
class Diagnosis:
    delta_hat: np.ndarray
    ranking: np.ndarray
    residual_norm: float
    jacobian_rank: int
    condition_number: float


def diagnose_linearized(
    theta_nominal: np.ndarray,
    probes: Sequence[np.ndarray],
    measured: np.ndarray,
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
    lam: float = 1e-7,
) -> Diagnosis:
    y0, J = stack_experiment(theta_nominal, probes, sweeps=sweeps, ports=ports)
    dy = measured - y0
    d = ridge_solve(J, dy, lam=lam)
    resid = dy - J @ d
    s = np.linalg.svd(J, compute_uv=False)
    rank = int(np.linalg.matrix_rank(J, tol=1e-9))
    cond = float(s[0] / max(s[-1], 1e-12)) if s.size else float("inf")
    return Diagnosis(
        delta_hat=d,
        ranking=np.argsort(-np.abs(d)),
        residual_norm=float(np.linalg.norm(resid)),
        jacobian_rank=rank,
        condition_number=cond,
    )


def gauss_newton_fit(
    theta0: np.ndarray,
    probes: Sequence[np.ndarray],
    measured: np.ndarray,
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
    iterations: int = 5,
    lam: float = 1e-6,
) -> np.ndarray:
    """Small exact-Jacobian refit for larger drifts than first-order diagnosis."""
    theta = np.asarray(theta0, dtype=float).copy()
    for _ in range(iterations):
        y, J = stack_experiment(theta, probes, sweeps=sweeps, ports=ports)
        step = ridge_solve(J, measured - y, lam=lam)
        theta += step
        if np.linalg.norm(step) < 1e-10:
            break
    return theta


def information_score(J: np.ndarray, lam: float = 1e-6) -> float:
    """D-optimal-ish score: log det(J^T J + lam I). Higher is more informative."""
    G = J.T @ J + lam * np.eye(J.shape[1])
    sign, logdet = np.linalg.slogdet(G)
    return float(logdet) if sign > 0 else -np.inf


def greedy_probe_nodes(
    theta: np.ndarray,
    n_select: int,
    steps: int = 5,
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
) -> tuple[list[int], list[float]]:
    n = theta.size
    selected: list[int] = []
    scores: list[float] = []
    for _ in range(n_select):
        best_node, best_score = None, -np.inf
        for node in range(n):
            if node in selected:
                continue
            probes = [make_pulse_probe(n, j, steps=steps) for j in selected + [node]]
            _, J = stack_experiment(theta, probes, sweeps=sweeps, ports=ports)
            score = information_score(J)
            if score > best_score:
                best_node, best_score = node, score
        assert best_node is not None
        selected.append(best_node)
        scores.append(best_score)
    return selected, scores


def finite_difference_jacobian(
    theta: np.ndarray,
    probe: np.ndarray,
    sweeps: int = 2,
    ports: Sequence[int] = (0, 4),
    eps: float = 1e-6,
) -> np.ndarray:
    y0, _ = run_probe(theta, probe, sweeps=sweeps, ports=ports, exact_jacobian=False)
    J = np.zeros((len(ports), theta.size))
    for e in range(theta.size):
        tt = theta.copy(); tt[e] += eps
        y1, _ = run_probe(tt, probe, sweeps=sweeps, ports=ports, exact_jacobian=False)
        J[:, e] = (y1 - y0) / eps
    return J
