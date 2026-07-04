"""
Generic conditional-flow NPE, parameterized by a forward config module, so one trainer
serves every material family in the pilot (see the _FORWARD registry below).
Usage:  python -m tpdsbi.inference.neural_posterior graphene_oxide nanodiamond
"""
import os, sys, time, importlib
_FORWARD = {
    "hydrogenated_carbon": "tpdsbi.forward.sim_hydrogenated_carbon",
    "graphene_oxide": "tpdsbi.forward.sim_graphene_oxide",
    "nanodiamond": "tpdsbi.forward.sim_nanodiamond",
    "per_family_order": "tpdsbi.forward.sim_per_family_order",
    "lateral_interaction": "tpdsbi.forward.sim_lateral_interaction",
    "twopeak_demo": "tpdsbi.forward.sim_twopeak_demo",
}
from tpdsbi.paths import DATA, MODELS, FIGURES
import numpy as np, torch, torch.nn as nn
from tpdsbi.forward.batch_simulate import sim_chunked

torch.set_num_threads(os.cpu_count() or 8)


class Coupling(nn.Module):
    def __init__(self, dim, ctx, flip):
        super().__init__()
        self.d = dim // 2
        self.flip = flip
        da = (dim - self.d) if flip else self.d      # size of the conditioning half
        db = dim - da                                 # size of the transformed half
        self.net = nn.Sequential(nn.Linear(da + ctx, 128), nn.ReLU(),
                                 nn.Linear(128, 128), nn.ReLU(),
                                 nn.Linear(128, 2 * db))

    def split(self, z):
        return (z[:, self.d:], z[:, :self.d]) if self.flip else (z[:, :self.d], z[:, self.d:])

    def join(self, a, b):
        return torch.cat([b, a], 1) if self.flip else torch.cat([a, b], 1)

    def forward(self, z, c):
        a, b = self.split(z)
        s, t = self.net(torch.cat([a, c], 1)).chunk(2, 1)
        s = 2.5 * torch.tanh(s / 2.5)
        return self.join(a, b * torch.exp(s) + t), s.sum(1)

    def inverse(self, u, c):
        a, b = self.split(u)
        s, t = self.net(torch.cat([a, c], 1)).chunk(2, 1)
        s = 2.5 * torch.tanh(s / 2.5)
        return self.join(a, (b - t) * torch.exp(-s))


class Flow(nn.Module):
    def __init__(self, d_in, n_par, n_layers=8):
        super().__init__()
        self.n_par = n_par
        self.embed = nn.Sequential(nn.Linear(d_in, 256), nn.ReLU(),
                                   nn.Linear(256, 128), nn.ReLU(), nn.Linear(128, 128))
        self.layers = nn.ModuleList([Coupling(n_par, 128, i % 2 == 1) for i in range(n_layers)])

    def log_prob(self, x, z):
        c = self.embed(x)
        ld = 0.0
        for lay in self.layers:
            z, d = lay(z, c)
            ld = ld + d
        return (-0.5 * (z ** 2 + np.log(2 * np.pi)).sum(1)) + ld

    @torch.no_grad()
    def sample_z(self, x, n=3000):
        c = self.embed(x).expand(n, -1)
        u = torch.randn(n, self.n_par)
        for lay in reversed(self.layers):
            u = lay.inverse(u, c)
        return u.numpy()


class NPE:
    """Bundles a forward config + its trained flow. invert(curve, log_beta) -> samples."""

    def __init__(self, cfg_name, suffix="", n_layers=8):
        self.cfg = importlib.import_module(_FORWARD.get(cfg_name, cfg_name))
        self.name = cfg_name + suffix
        self.model = Flow(len(self.cfg.T_GRID) + 1, self.cfg.NP_, n_layers)
        path = os.path.join(MODELS, f"flow_{self.name}.pt")
        if os.path.exists(path):
            try:
                self.model.load_state_dict(torch.load(path)); self.model.eval()
            except RuntimeError:  # architecture mismatch: try the other depth
                alt = 12 if n_layers == 8 else 8
                self.model = Flow(len(self.cfg.T_GRID) + 1, self.cfg.NP_, alt)
                self.model.load_state_dict(torch.load(path)); self.model.eval()

    def to_z(self, p):
        q = np.clip((p - self.cfg.LO) / (self.cfg.HI - self.cfg.LO), 1e-4, 1 - 1e-4)
        return np.log(q / (1 - q))

    def from_z(self, z):
        p = self.cfg.LO + (self.cfg.HI - self.cfg.LO) / (1.0 + np.exp(-np.clip(z, -30, 30)))
        return p

    def invert(self, curve, log_beta, n=4000):
        x = torch.tensor(np.append(curve, log_beta)[None, :], dtype=torch.float32)
        return self.from_z(self.model.sample_z(x, n))

    def train(self, n_sims=160000, epochs=30, seed=0, noise=0.02):
        """noise: scalar sigma, or (lo, hi) tuple giving a per-curve sigma, so the flow learns
        the noise-marginalized posterior (the correct treatment when noise is unknown)."""
        rng = np.random.default_rng(seed); torch.manual_seed(seed)
        print(f"[{self.name}] simulating {n_sims:,} training curves (noise={noise})...")
        t0 = time.time()
        P = self.cfg.sample_prior(n_sims, rng)
        lb = rng.uniform(*self.cfg.LOG_BETA_RANGE, n_sims)
        # chunked: an unchunked 240k batch at N_EA=256 is a ~46 GB (B,E,T) tensor
        clean = sim_chunked(self.cfg.simulate, P, lb, noise=0.0)
        sig = (rng.uniform(noise[0], noise[1], (n_sims, 1)) if isinstance(noise, tuple)
               else np.full((n_sims, 1), noise))
        X = np.column_stack([np.clip(clean + sig * rng.standard_normal(clean.shape)
                                     .astype(clean.dtype), 0, None), lb])
        print(f"[{self.name}] sims done in {time.time()-t0:.0f}s; training flow...")
        Xt = torch.tensor(X, dtype=torch.float32)
        Zt = torch.tensor(self.to_z(P), dtype=torch.float32)
        opt = torch.optim.Adam(self.model.parameters(), 5e-4)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, epochs)
        idx = np.arange(n_sims)
        for ep in range(epochs):
            rng.shuffle(idx); tot = 0.0
            for b in range(0, n_sims, 512):
                j = idx[b:b + 512]
                loss = -self.model.log_prob(Xt[j], Zt[j]).mean()
                opt.zero_grad(); loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.0)
                opt.step(); tot += loss.item() * len(j)
            sched.step()
            if ep % 5 == 0 or ep == epochs - 1:
                print(f"[{self.name}] epoch {ep:2d}  NLL {tot/n_sims:+.3f}")
        torch.save(self.model.state_dict(), os.path.join(MODELS, f"flow_{self.name}.pt"))

        # calibration
        M = 250
        Pt = self.cfg.sample_prior(M, rng)
        lbt = rng.uniform(*self.cfg.LOG_BETA_RANGE, M)
        sv = (rng.uniform(noise[0], noise[1], (M, 1)) if isinstance(noise, tuple)
              else np.full((M, 1), noise))
        Xv = np.clip(self.cfg.simulate(Pt, lbt) +
                     sv * rng.standard_normal((M, len(self.cfg.T_GRID))).astype(np.float32),
                     0, None)
        cov = np.zeros(self.cfg.NP_)
        for i in range(M):
            s = self.invert(Xv[i], lbt[i], 800)
            lo, hi = np.percentile(s, [5, 95], axis=0)
            cov += (Pt[i] >= lo) & (Pt[i] <= hi)
        cov /= M
        print(f"[{self.name}] 90% CI coverage per param: "
              + " ".join(f"{c:.2f}" for c in cov) + f"  | mean {cov.mean():.2f}")


if __name__ == "__main__":
    for cfg in sys.argv[1:]:
        NPE(cfg).train()
