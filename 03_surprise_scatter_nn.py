"""SOL-03: Surprise Scatter Cell — an event-written local-scatter recurrent cell.

KYY supplies the norm-stable local reciprocal scatter body. GeometricNeuron V20's
"spend on surprise" becomes a write rule: the state keeps circulating locally,
but new input is committed in proportion to prediction error.

The demo trains the cell to recover a slowly changing hidden sensor level from
noisy observations and compares it with a GRU and an always-write ablation.
No claim that it is a better RNN is made; the interesting metric is accuracy per
state write, not merely accuracy.
"""
from __future__ import annotations
import argparse
import math
import random
import numpy as np

try:
    import torch
    from torch import nn
except ImportError as e:
    raise SystemExit("This demo needs PyTorch: pip install torch") from e


class SurpriseScatterCell(nn.Module):
    def __init__(self, input_dim=1, state_dim=16, sweeps=2, surprise_gate=True):
        super().__init__()
        if state_dim < 4 or state_dim % 2:
            raise ValueError("state_dim must be even and >= 4")
        self.input_dim = input_dim; self.state_dim = state_dim; self.sweeps = sweeps
        self.surprise_gate = surprise_gate
        self.edges = [(i, (i + 1) % state_dim) for i in range(state_dim)]
        # Two disjoint checkerboard phases on an even ring.  This is both the
        # natural local-scatter schedule and much faster than a Python loop over edges.
        p0 = list(range(0, state_dim, 2))
        p1 = list(range(1, state_dim, 2))
        self.register_buffer("p0", torch.tensor(p0, dtype=torch.long))
        self.register_buffer("p1", torch.tensor(p1, dtype=torch.long))
        self.register_buffer("src", torch.tensor([i for i, j in self.edges], dtype=torch.long))
        self.register_buffer("dst", torch.tensor([j for i, j in self.edges], dtype=torch.long))
        self.angle_base = nn.Parameter(torch.empty(sweeps, state_dim))
        nn.init.uniform_(self.angle_base, -0.35, 0.35)
        self.angle_mod = nn.Linear(input_dim, sweeps * state_dim, bias=False)
        nn.init.normal_(self.angle_mod.weight, std=0.05)
        self.predict = nn.Linear(state_dim, input_dim)
        self.drive = nn.Linear(input_dim, state_dim)
        self.readout = nn.Linear(state_dim, 1)
        self.threshold_raw = nn.Parameter(torch.tensor(-1.4))
        self.write_raw = nn.Parameter(torch.tensor(-0.2))
        self.sharp_raw = nn.Parameter(torch.tensor(2.0))

    def _phase(self, h, theta, edge_ids):
        src = self.src[edge_ids]; dst = self.dst[edge_ids]
        th = theta[:, edge_ids]
        c, s = torch.cos(th), torch.sin(th)
        a, b = h[:, src], h[:, dst]
        ap, bp = c * a + s * b, s * a - c * b
        hh = h.clone(); hh[:, src] = ap; hh[:, dst] = bp
        return hh

    def scatter(self, h, x):
        mod = self.angle_mod(x).view(x.shape[0], self.sweeps, self.state_dim)
        for sw in range(self.sweeps):
            theta = math.pi * torch.tanh(self.angle_base[sw][None, :] + mod[:, sw, :])
            h = self._phase(h, theta, self.p0)
            h = self._phase(h, theta, self.p1)
        return h

    def step(self, x, h):
        pred = self.predict(h)
        residual = x - pred
        if self.surprise_gate:
            threshold = torch.nn.functional.softplus(self.threshold_raw) + 1e-4
            sharp = torch.nn.functional.softplus(self.sharp_raw) + 1.0
            surprise = residual.abs().mean(dim=-1, keepdim=True)
            gate = torch.sigmoid(sharp * (surprise - threshold))
        else:
            gate = torch.ones((x.shape[0], 1), device=x.device, dtype=x.dtype)
        h = self.scatter(h, x)
        proposal = torch.tanh(self.drive(residual))
        alpha = 0.05 + 0.90 * torch.sigmoid(self.write_raw)
        write = alpha * gate
        h = (1.0 - write) * h + write * proposal
        y = self.readout(h)
        return y, h, gate, pred

    def forward(self, x):
        # x: B,T,D
        h = torch.zeros((x.shape[0], self.state_dim), device=x.device, dtype=x.dtype)
        ys, gates, preds = [], [], []
        for t in range(x.shape[1]):
            y, h, g, p = self.step(x[:, t], h)
            ys.append(y); gates.append(g); preds.append(p)
        return torch.stack(ys, 1), torch.stack(gates, 1), torch.stack(preds, 1)


class GRUTracker(nn.Module):
    def __init__(self, state_dim=16):
        super().__init__(); self.gru = nn.GRU(1, state_dim, batch_first=True); self.out = nn.Linear(state_dim, 1)
    def forward(self, x):
        h, _ = self.gru(x); return self.out(h)


def make_batch(batch, length, device, jump_p=0.035, noise=0.28):
    levels = torch.tensor([-1.0, -0.35, 0.35, 1.0], device=device)
    idx = torch.randint(0, 4, (batch,), device=device)
    hidden = []
    for _ in range(length):
        jump = torch.rand(batch, device=device) < jump_p
        idx = torch.where(jump, torch.randint(0, 4, (batch,), device=device), idx)
        hidden.append(levels[idx])
    target = torch.stack(hidden, 1).unsqueeze(-1)
    obs = target + noise * torch.randn_like(target)
    return obs, target


def train_model(model, steps, device, length=64, batch=64, lr=2e-3):
    model.to(device); opt = torch.optim.Adam(model.parameters(), lr=lr)
    for step in range(steps):
        x, y = make_batch(batch, length, device)
        if isinstance(model, SurpriseScatterCell):
            pred, gates, one_step = model(x)
            # Track hidden level + make internal predictor actually predict observations.
            loss = ((pred - y) ** 2).mean() + 0.10 * ((one_step[:, 1:] - x[:, 1:]) ** 2).mean()
            if model.surprise_gate:
                loss = loss + 0.003 * gates.mean()
        else:
            pred = model(x); loss = ((pred - y) ** 2).mean()
        opt.zero_grad(); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0); opt.step()
    return model


@torch.no_grad()
def evaluate(model, device, batches=8, length=128, batch=96):
    mses, accs, writes = [], [], []
    levels = torch.tensor([-1.0, -0.35, 0.35, 1.0], device=device)
    for _ in range(batches):
        x, y = make_batch(batch, length, device)
        if isinstance(model, SurpriseScatterCell):
            pred, gates, _ = model(x); writes.append(float(gates.mean().cpu()))
        else:
            pred = model(x); writes.append(1.0)
        mses.append(float(((pred-y)**2).mean().cpu()))
        pi = (pred - levels.view(1,1,-1)).abs().argmin(-1)
        yi = (y - levels.view(1,1,-1)).abs().argmin(-1)
        accs.append(float((pi == yi).float().mean().cpu()))
    return float(np.mean(mses)), float(np.mean(accs)), float(np.mean(writes))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=250)
    ap.add_argument("--state", type=int, default=16)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--cpu", action="store_true")
    args = ap.parse_args()
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    models = [
        ("surprise-scatter", SurpriseScatterCell(state_dim=args.state, surprise_gate=True)),
        ("always-write", SurpriseScatterCell(state_dim=args.state, surprise_gate=False)),
        ("GRU", GRUTracker(state_dim=args.state)),
    ]
    rows = []
    print("=" * 76)
    print("SOL-03  SURPRISE SCATTER CELL — sensor tracking with event-written memory")
    print("=" * 76)
    print(f"device={device} training_steps={args.steps} state_dim={args.state}")
    for name, model in models:
        train_model(model, args.steps, device)
        mse, acc, writes = evaluate(model, device)
        params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        rows.append((name, mse, acc, writes, params))
        print(f"{name:17s} MSE={mse:.5f} level-acc={acc*100:5.1f}% mean-write={writes:.3f} params={params}")

    print("\nThe test to care about is not 'did SOL beat GRU once'. It is whether the")
    print("surprise-gated local scatter body can reach useful tracking quality while")
    print("committing substantially fewer state writes, and whether that survives real")
    print("streams, quantization, and stronger baselines. This file is the first apple.")


if __name__ == "__main__":
    main()
