"""
Amortized NPE for TPD inversion. Trains a mixture-density network mapping a TPD curve
(200 pts) to a posterior over the 6 desorption-distribution parameters
[w1, mu1, sig1, mu2, sig2, log_nu], recovering the Ea-nu compensation ridge.
Run: python -m tpdsbi.inference.demo_mdn   (CPU, ~2-3 min)
"""
import os, numpy as np, torch, torch.nn as nn
from tpdsbi.paths import DATA, MODELS, FIGURES
from tpdsbi.forward.sim_twopeak_demo import simulate, sample_prior, PRIOR, PNAMES

DEV = "cpu"
K = 10          # mixture components (diagonal Gaussians in 6D)
NP = len(PNAMES)
LO = np.array([PRIOR[k][0] for k in PNAMES]); HI = np.array([PRIOR[k][1] for k in PNAMES])


def norm_p(p):  return (p - LO) / (HI - LO)          # params -> [0,1]
def denorm(p): return LO + p * (HI - LO)


class MDN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(200, 256), nn.ReLU(), nn.Linear(256, 256),
                                 nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.pi = nn.Linear(128, K)
        self.mu = nn.Linear(128, K * NP)
        self.ls = nn.Linear(128, K * NP)   # log sigma

    def forward(self, x):
        h = self.net(x)
        return (torch.log_softmax(self.pi(h), -1),
                self.mu(h).view(-1, K, NP),
                self.ls(h).view(-1, K, NP).clamp(-5, 1))

    def nll(self, x, y):
        lpi, mu, ls = self(x)
        ll = -0.5 * (((y[:, None, :] - mu) / ls.exp()) ** 2 + 2 * ls + np.log(2 * np.pi)).sum(-1)
        return -torch.logsumexp(lpi + ll, -1).mean()

    @torch.no_grad()
    def sample(self, x, n=2000):
        lpi, mu, ls = self(x)
        comp = torch.multinomial(lpi.exp()[0], n, replacement=True)
        z = mu[0, comp] + ls.exp()[0, comp] * torch.randn(n, NP)
        return denorm(z.numpy())


def main():
    rng = np.random.default_rng(0); torch.manual_seed(0)
    print("Simulating training set...")
    P = sample_prior(60000, rng)
    X = simulate(P, noise=0.02, rng=rng)                 # realistic mass-spec noise
    Xt = torch.tensor(X, dtype=torch.float32)
    Yt = torch.tensor(norm_p(P), dtype=torch.float32)
    model = MDN().to(DEV)
    opt = torch.optim.Adam(model.parameters(), 1e-3)
    n = len(Xt); idx = np.arange(n)
    for ep in range(18):
        rng.shuffle(idx); tot = 0.0
        for b in range(0, n, 512):
            j = idx[b:b + 512]
            loss = model.nll(Xt[j], Yt[j])
            opt.zero_grad(); loss.backward(); opt.step(); tot += float(loss) * len(j)
        if ep % 3 == 0 or ep == 17:
            print(f"  epoch {ep:2d}  NLL {tot/n:+.3f}")
    torch.save(model.state_dict(), os.path.join(MODELS, "npe.pt"))
    print(f"saved {os.path.join(MODELS, 'npe.pt')}")

    # calibration check: fraction of truths inside the 90% credible interval
    Pt = sample_prior(300, rng); Xv = simulate(Pt, noise=0.02, rng=rng)
    cov = np.zeros(NP)
    for i in range(300):
        s = model.sample(torch.tensor(Xv[i:i+1], dtype=torch.float32), 1000)
        lo, hi = np.percentile(s, [5, 95], axis=0)
        cov += (Pt[i] >= lo) & (Pt[i] <= hi)
    print("calibration - 90% CI coverage per parameter (want ~0.90):")
    for k, c in zip(PNAMES, cov / 300):
        print(f"   {k:8s} {c:.2f}")


if __name__ == "__main__":
    main()
