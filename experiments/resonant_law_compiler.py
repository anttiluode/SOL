"""SOL experiment: Resonant Law Compiler.

Take a flexible RCNet-style complex teacher, learn a local recurrence, then
compile the *behavior* rather than preserving the network weights.

Default demo uses the Mandelbrot local law z' = z^2 + c because recursive
verification is brutally legible: tiny one-step errors become wrong escape
geometry.

This is not a novelty claim. Symbolic regression/distillation already exists.
The SOL question is narrower: can we combine
  learn-soft -> identifiability audit -> sparse behavioral lowering ->
  simple legalization -> recursive verification
into a useful compiler workflow?
"""
from __future__ import annotations

import argparse
import math
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

TERMS = (
    "1", "z", "zbar", "z^2", "z*zbar", "zbar^2",
    "c", "cbar", "z*c", "z*cbar", "zbar*c", "zbar*cbar",
)


def design(z: np.ndarray, c: np.ndarray) -> np.ndarray:
    z = np.asarray(z, dtype=np.complex128)
    c = np.asarray(c, dtype=np.complex128)
    return np.column_stack([
        np.ones_like(z), z, np.conj(z), z**2, z*np.conj(z), np.conj(z)**2,
        c, np.conj(c), z*c, z*np.conj(c), np.conj(z)*c, np.conj(z)*np.conj(c),
    ])


def eval_poly(z: np.ndarray, c: np.ndarray, coeff: np.ndarray) -> np.ndarray:
    shape = np.shape(z)
    P = design(np.asarray(z).reshape(-1), np.asarray(c).reshape(-1))
    return (P @ coeff).reshape(shape)


class ComplexLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc_r = nn.Linear(in_features, out_features, bias=False)
        self.fc_i = nn.Linear(in_features, out_features, bias=False)
        self.bias_r = nn.Parameter(torch.zeros(out_features))
        self.bias_i = nn.Parameter(torch.zeros(out_features))
        nn.init.xavier_normal_(self.fc_r.weight, gain=0.5)
        nn.init.xavier_normal_(self.fc_i.weight, gain=0.5)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return torch.complex(
            self.fc_r(z.real) - self.fc_i(z.imag) + self.bias_r,
            self.fc_r(z.imag) + self.fc_i(z.real) + self.bias_i,
        )


class TinyResonantTeacher(nn.Module):
    """Small version of the ResonantCortex Mandelbrot teacher."""
    def __init__(self, width: int = 32):
        super().__init__()
        self.layer1 = ComplexLinear(2, width)
        self.layer2 = ComplexLinear(width, width)
        self.layer3 = ComplexLinear(width, 1)

    @staticmethod
    def activation(z: torch.Tensor) -> torch.Tensor:
        mag = torch.abs(z)
        phase = torch.angle(z)
        return torch.polar(torch.nn.functional.softplus(mag - 0.5), phase)

    def forward(self, z: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        x = torch.stack([z, c], dim=1)
        h = self.activation(self.layer1(x))
        h = self.activation(self.layer2(h))
        return self.layer3(h).squeeze(1)


def random_complex_batch(n: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    zr = torch.empty(n, device=device).uniform_(-2.0, 2.0)
    zi = torch.empty(n, device=device).uniform_(-2.0, 2.0)
    cr = torch.empty(n, device=device).uniform_(-2.0, 1.0)
    ci = torch.empty(n, device=device).uniform_(-1.5, 1.5)
    return torch.complex(zr, zi), torch.complex(cr, ci)


def train_teacher(steps: int, width: int, seed: int, device: str) -> TinyResonantTeacher:
    torch.manual_seed(seed)
    model = TinyResonantTeacher(width=width).to(device)
    opt = optim.AdamW(model.parameters(), lr=3e-3)
    for _ in range(steps):
        z, c = random_complex_batch(2048, device=device)
        target = z*z + c
        opt.zero_grad()
        pred = model(z, c)
        loss = torch.mean(torch.abs(pred - target)**2)
        loss.backward()
        opt.step()
    return model


def fit_sparse(P: np.ndarray, y: np.ndarray, threshold: float = 0.015, rounds: int = 4) -> np.ndarray:
    keep = np.ones(P.shape[1], dtype=bool)
    coeff = np.zeros(P.shape[1], dtype=np.complex128)
    for _ in range(rounds):
        if not np.any(keep):
            break
        c, *_ = np.linalg.lstsq(P[:, keep], y, rcond=None)
        coeff[:] = 0
        coeff[keep] = c
        new_keep = np.abs(coeff) >= threshold
        if np.array_equal(new_keep, keep):
            break
        keep = new_keep
    if np.any(keep):
        c, *_ = np.linalg.lstsq(P[:, keep], y, rcond=None)
        coeff[:] = 0
        coeff[keep] = c
    return coeff


def simple_snap(coeff: np.ndarray, tol: float = 0.06) -> np.ndarray:
    """Conservative toy legalization: snap coefficients close to 0 or real integers.

    In a real compiler the legal set comes from the declared behavioral contract,
    not from wishful rounding. This demo only tests the compile/verify workflow.
    """
    out = coeff.copy()
    for i, a in enumerate(out):
        if abs(a) < tol:
            out[i] = 0.0
            continue
        r = round(float(a.real))
        if -3 <= r <= 3 and abs(a.real - r) < tol and abs(a.imag) < tol:
            out[i] = complex(r, 0.0)
    return out


def fixed_c_orbit_design(c0: complex = -0.12 + 0.74j, n: int = 80) -> np.ndarray:
    z = 0j
    zs, cs = [], []
    for _ in range(n):
        zs.append(z); cs.append(c0)
        z = z*z + c0
        if abs(z) > 2.0:
            z = 0j
    return design(np.asarray(zs), np.asarray(cs))


def escape_map_numpy(fn, resolution: int, iters: int) -> np.ndarray:
    y, x = np.mgrid[-1.2:1.2:complex(resolution), -2.0:0.8:complex(resolution)]
    c = x + 1j*y
    z = np.zeros_like(c)
    esc = np.full(c.shape, iters, dtype=np.int16)
    active = np.ones(c.shape, dtype=bool)
    for i in range(iters):
        z[active] = fn(z[active], c[active])
        hit = active & (np.abs(z) > 2.0)
        esc[hit] = i
        active[hit] = False
    return esc


def escape_map_teacher(model: nn.Module, resolution: int, iters: int, device: str) -> np.ndarray:
    y, x = np.mgrid[-1.2:1.2:complex(resolution), -2.0:0.8:complex(resolution)]
    c = torch.tensor((x + 1j*y).reshape(-1), dtype=torch.complex64, device=device)
    z = torch.zeros_like(c)
    esc = torch.full((c.numel(),), iters, dtype=torch.int16, device=device)
    active = torch.ones(c.numel(), dtype=torch.bool, device=device)
    with torch.no_grad():
        for i in range(iters):
            idx = torch.where(active)[0]
            if idx.numel() == 0:
                break
            zz = model(z[idx], c[idx])
            z[idx] = zz
            hit = torch.abs(zz) > 2.0
            hit_idx = idx[hit]
            esc[hit_idx] = i
            active[hit_idx] = False
    return esc.cpu().numpy().reshape(resolution, resolution)


def fmt_coeff(a: complex) -> str:
    return f"{a.real:+.4f}{a.imag:+.4f}j"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=600)
    ap.add_argument("--width", type=int, default=32)
    ap.add_argument("--samples", type=int, default=8000)
    ap.add_argument("--resolution", type=int, default=120)
    ap.add_argument("--iters", type=int, default=35)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    rng = np.random.default_rng(args.seed + 1000)

    print("=" * 78)
    print("SOL EXPERIMENT — RESONANT LAW COMPILER")
    print("learn soft -> audit identifiability -> lower behavior -> legalize -> recurse")
    print("=" * 78)

    model = train_teacher(args.steps, args.width, args.seed, device)

    # Query the trained teacher rather than its weights.
    zr = rng.uniform(-2.0, 2.0, args.samples)
    zi = rng.uniform(-2.0, 2.0, args.samples)
    cr = rng.uniform(-2.0, 1.0, args.samples)
    ci = rng.uniform(-1.5, 1.5, args.samples)
    z = zr + 1j*zi
    c = cr + 1j*ci
    zt = torch.tensor(z, dtype=torch.complex64, device=device)
    ct = torch.tensor(c, dtype=torch.complex64, device=device)
    with torch.no_grad():
        teacher_y = model(zt, ct).cpu().numpy().astype(np.complex128)

    truth_y = z*z + c
    P = design(z, c)
    coeff = fit_sparse(P, teacher_y)
    snapped = simple_snap(coeff)

    rank = int(np.linalg.matrix_rank(P))
    cond = float(np.linalg.cond(P))
    passive = fixed_c_orbit_design()
    passive_rank = int(np.linalg.matrix_rank(passive, tol=1e-9))
    passive_cond = float(np.linalg.cond(passive))

    pred_sparse = P @ coeff
    pred_snap = P @ snapped
    mse_teacher_truth = float(np.mean(np.abs(teacher_y - truth_y)**2))
    mse_sparse_truth = float(np.mean(np.abs(pred_sparse - truth_y)**2))
    mse_snap_truth = float(np.mean(np.abs(pred_snap - truth_y)**2))
    mse_sparse_teacher = float(np.mean(np.abs(pred_sparse - teacher_y)**2))

    print(f"teacher          : RCNet-style complex net, width={args.width}, train steps={args.steps}")
    print(f"probe design      : rank {rank}/{P.shape[1]}, condition {cond:.2e}")
    print(f"one fixed-c orbit : rank {passive_rank}/{P.shape[1]}, condition {passive_cond:.2e}")
    print("                     -> output from one trajectory is not enough to identify this library")
    print("\ncompiled sparse recurrence (largest terms):")
    for name, a in sorted(zip(TERMS, coeff), key=lambda kv: -abs(kv[1]))[:8]:
        print(f"  {name:10s} {fmt_coeff(a)}  |a|={abs(a):.4f}")
    print("\nlegalized/simple-snap candidate:")
    for name, a in zip(TERMS, snapped):
        if abs(a) > 0:
            print(f"  {name:10s} {fmt_coeff(a)}")

    print("\none-step MSE against the known demo law:")
    print(f"  neural teacher : {mse_teacher_truth:.6f}")
    print(f"  sparse lowering: {mse_sparse_truth:.6f}")
    print(f"  simple snap    : {mse_snap_truth:.6f}")
    print(f"  lowering-vs-teacher residual: {mse_sparse_teacher:.6f}")

    truth_map = escape_map_numpy(lambda zz, cc: zz*zz + cc, args.resolution, args.iters)
    teacher_map = escape_map_teacher(model, args.resolution, args.iters, device)
    sparse_map = escape_map_numpy(lambda zz, cc: eval_poly(zz, cc, coeff), args.resolution, args.iters)
    snap_map = escape_map_numpy(lambda zz, cc: eval_poly(zz, cc, snapped), args.resolution, args.iters)

    def scores(m):
        exact = float(np.mean(m == truth_map))
        inside = float(np.mean((m == args.iters) == (truth_map == args.iters)))
        return exact, inside

    print("\nrecursive Mandelbrot verification:")
    for label, m in (("neural teacher", teacher_map), ("sparse lowering", sparse_map), ("simple snap", snap_map)):
        exact, inside = scores(m)
        print(f"  {label:15s}: escape-step agreement={100*exact:6.2f}%  inside/outside={100*inside:6.2f}%")

    print("\nInterpretation:")
    print("  RCNet is useful here as an optimization scaffold / black-box teacher.")
    print("  The deployable object need not be its weights. A compiler can query behavior,")
    print("  refuse under-identified experiments, lower to a small recurrence, and verify")
    print("  the recurrence under iteration where tiny local errors become obvious.")
    print("\nBoundary:")
    print("  Symbolic regression/distillation is established work. The SOL hypothesis is")
    print("  the full workflow: behavioral contracts + identifiability + active experiments")
    print("  + recursive verification + deployment lowering. This script does not establish novelty.")


if __name__ == "__main__":
    main()
