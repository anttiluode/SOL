"""SOL-06: Behavior / Gauge Doctor — repair relations, then anchor the port.

Two local reciprocal reflection cells generate a tiny noncommutative machine.
Each cell H(theta) is an involution. If their angle difference is 2*pi/3,
(H(a) H(b))^3 = I: the Coxeter/S3 relation closes exactly.

Crucial point: that relation determines only the *relative* angle. A common
rotation of both cells is an invisible gauge direction. Relation-only repair
therefore restores the algebra while leaving the external port shifted.

The demo automatically chooses one scalar probe/port anchor that kills the gauge,
then repairs both the behavioral relation and observable interface. This is the
KYY relation+port idea turned into a self-calibration instrument.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
import numpy as np


def H(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[c, s], [s, -c]], dtype=float)


def dH(theta: float) -> np.ndarray:
    c, s = np.cos(theta), np.sin(theta)
    return np.array([[-s, c], [c, s]], dtype=float)


def relation_residual_jac(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Residual/Jacobian for (H(a)H(b))^3 = I."""
    a, b = map(float, x)
    A, B = H(a), H(b)
    R = A @ B
    R2 = R @ R
    R3 = R2 @ R
    dRs = (dH(a) @ B, A @ dH(b))
    cols = []
    for dR in dRs:
        dR3 = dR @ R2 + R @ dR @ R + R2 @ dR
        cols.append(dR3.reshape(-1))
    return (R3 - np.eye(2)).reshape(-1), np.column_stack(cols)


@dataclass(frozen=True)
class Anchor:
    op: str
    input_axis: int
    port: int


def anchor_value_jac(x: np.ndarray, anchor: Anchor) -> tuple[float, np.ndarray]:
    idx = 0 if anchor.op == "s" else 1
    q = np.eye(2)[:, anchor.input_axis]
    yv = H(float(x[idx])) @ q
    dy = dH(float(x[idx])) @ q
    g = np.zeros(2)
    g[idx] = dy[anchor.port]
    return float(yv[anchor.port]), g


def choose_anchor(x0: np.ndarray) -> tuple[Anchor, float]:
    """Choose the scalar observation that best completes relation rank."""
    _, Jr = relation_residual_jac(x0)
    best: tuple[float, Anchor] | None = None
    for op in ("s", "t"):
        for q in (0, 1):
            for port in (0, 1):
                a = Anchor(op, q, port)
                _, g = anchor_value_jac(x0, a)
                J = np.vstack([Jr, g])
                s = np.linalg.svd(J, compute_uv=False)
                score = float(s[-1])
                if best is None or score > best[0]:
                    best = (score, a)
    assert best is not None
    return best[1], best[0]


def behavior_residual_jac(
    x: np.ndarray, x0: np.ndarray, anchor: Anchor
) -> tuple[np.ndarray, np.ndarray]:
    rr, Jr = relation_residual_jac(x)
    y, g = anchor_value_jac(x, anchor)
    y0, _ = anchor_value_jac(x0, anchor)
    return np.concatenate([rr, [y - y0]]), np.vstack([Jr, g])


def gauss_newton(
    x: np.ndarray,
    residual_jac,
    iterations: int = 8,
    lam: float = 1e-6,
) -> np.ndarray:
    x = np.asarray(x, dtype=float).copy()
    for _ in range(iterations):
        r, J = residual_jac(x)
        step = np.linalg.solve(J.T @ J + lam * np.eye(x.size), -J.T @ r)
        x += step
        if np.linalg.norm(step) < 1e-12:
            break
    return x


def demo(damage_a: float = 0.07, damage_b: float = -0.02):
    x0 = np.array([0.37, 0.37 - 2 * np.pi / 3])
    anchor, anchor_score = choose_anchor(x0)
    damaged = x0 + np.array([damage_a, damage_b])

    _, Jr0 = relation_residual_jac(x0)
    _, Jb0 = behavior_residual_jac(x0, x0, anchor)
    rd, _ = relation_residual_jac(damaged)
    bd, _ = behavior_residual_jac(damaged, x0, anchor)

    # Relation-only repair. It can remove differential error but has no opinion
    # about common-mode/gauge drift.
    relation_repaired = gauss_newton(damaged, relation_residual_jac)

    # Relation + one observable anchor. This fixes the previously invisible gauge.
    full_repaired = gauss_newton(
        damaged, lambda x: behavior_residual_jac(x, x0, anchor)
    )

    # Pure common-mode damage is a clean identifiability control: relation says
    # "nothing changed" even though the external port moved.
    common = x0 + np.array([0.05, 0.05])
    common_relation, _ = relation_residual_jac(common)
    common_behavior, _ = behavior_residual_jac(common, x0, anchor)

    return {
        "x0": x0,
        "damaged": damaged,
        "anchor": anchor,
        "anchor_score": anchor_score,
        "relation_rank": int(np.linalg.matrix_rank(Jr0, tol=1e-9)),
        "behavior_rank": int(np.linalg.matrix_rank(Jb0, tol=1e-9)),
        "damage_relation_defect": float(np.linalg.norm(rd)),
        "damage_behavior_defect": float(np.linalg.norm(bd)),
        "relation_repaired": relation_repaired,
        "full_repaired": full_repaired,
        "relation_repair_defect": float(np.linalg.norm(relation_residual_jac(relation_repaired)[0])),
        "relation_repair_port_error": float(abs(behavior_residual_jac(relation_repaired, x0, anchor)[0][-1])),
        "full_repair_defect": float(np.linalg.norm(behavior_residual_jac(full_repaired, x0, anchor)[0])),
        "common_relation_defect": float(np.linalg.norm(common_relation)),
        "common_behavior_defect": float(np.linalg.norm(common_behavior)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--damage-a", type=float, default=0.07)
    ap.add_argument("--damage-b", type=float, default=-0.02)
    args = ap.parse_args()
    r = demo(args.damage_a, args.damage_b)
    a = r["anchor"]

    print("=" * 78)
    print("SOL-06  BEHAVIOR / GAUGE DOCTOR — self-calibrate relations plus the port")
    print("=" * 78)
    print(f"nominal angles          : {r['x0']}")
    print(f"hidden damage           : {r['damaged'] - r['x0']}")
    print("target relation          : s^2=t^2=I, (st)^3=I")
    print(f"relation Jacobian rank  : {r['relation_rank']}/2  (one gauge direction invisible)")
    print(f"chosen active anchor    : op={a.op}, input=e{a.input_axis}, port={a.port}")
    print(f"combined Jacobian rank  : {r['behavior_rank']}/2")
    print(f"anchor information score: {r['anchor_score']:.6f}")
    print(f"\ndamaged relation defect : {r['damage_relation_defect']:.6f}")
    print(f"damaged behavior defect : {r['damage_behavior_defect']:.6f}")

    dr = r["relation_repaired"] - r["x0"]
    df = r["full_repaired"] - r["x0"]
    print("\nRELATION-ONLY REPAIR")
    print(f"remaining angle drift   : {dr}")
    print(f"relation defect         : {r['relation_repair_defect']:.3e}")
    print(f"port error              : {r['relation_repair_port_error']:.6f}")
    print("  -> algebra repaired; common coordinate drift remains, correctly unidentifiable")

    print("\nRELATION + PORT REPAIR")
    print(f"remaining angle drift   : {df}")
    print(f"combined defect         : {r['full_repair_defect']:.3e}")

    print("\nPURE COMMON-MODE CONTROL")
    print(f"relation defect         : {r['common_relation_defect']:.3e}")
    print(f"behavior/port defect    : {r['common_behavior_defect']:.6f}")
    print("  -> the relation alone literally cannot see this drift")

    ok = (
        r["relation_rank"] == 1
        and r["behavior_rank"] == 2
        and r["relation_repair_defect"] < 1e-8
        and r["relation_repair_port_error"] > 1e-3
        and r["full_repair_defect"] < 1e-8
        and r["common_relation_defect"] < 1e-8
        and r["common_behavior_defect"] > 1e-3
    )
    print(f"\nRESULT: {'PASS' if ok else 'FAIL'}")
    print("\nThe useful principle: monitor the behavior you actually require, but also")
    print("measure enough of the interface to kill representation/gauge ambiguity.")
    print("Then repair the smallest observable defect instead of blindly restoring a")
    print("memorized internal matrix.")


if __name__ == "__main__":
    main()
