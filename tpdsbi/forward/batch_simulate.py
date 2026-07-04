"""Chunked simulation wrappers. The forward models build a (B, E, T) tensor, so a
200k-parameter batch would allocate ~30 GB; simulate in CHUNK slices instead."""
import numpy as np

CHUNK = 8000


def sim_chunked(simulate, P, log_beta, **kw):
    outs = []
    for i in range(0, len(P), CHUNK):
        lb = log_beta if np.isscalar(log_beta) else log_beta[i:i + CHUNK]
        outs.append(simulate(P[i:i + CHUNK], lb, **kw))
    return np.vstack(outs)


def peakT_chunked(peak_T, P, log_beta):
    return np.concatenate([peak_T(P[i:i + CHUNK], log_beta) for i in range(0, len(P), CHUNK)])
