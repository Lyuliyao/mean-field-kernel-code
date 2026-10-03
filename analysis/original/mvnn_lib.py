"""
Reusable library for post-hoc analysis of trained MVNN checkpoints.

Reimplements the MVNN forward pass (tanh-MLP embedding + interaction network with
mean-pooling over the empirical measure) in pure NumPy so that we can evaluate the
LEARNED drift b_theta(x, mu) on arbitrary measures, and compares it to the
analytically known TRUE drift b*(x, mu).

Architecture (matches CASE*/TRAINING*/model.py):
    feature = feature_net(x)                 # tanh-MLP, last layer linear
    mu_emb  = mean_j feature(x_j)            # < phi_emb , empirical measure >
    drift   = describe_net([x, mu_emb])      # tanh-MLP, last layer linear
"""
import numpy as np
import glob, os


# ----------------------------------------------------------------------------
# checkpoint loading + forward pass
# ----------------------------------------------------------------------------
def load_params(ckpt_path):
    ck = np.load(ckpt_path, allow_pickle=True)
    return ck['params'].item()


def latest_ckpt(model_save_path):
    fs = sorted(glob.glob(os.path.join(model_save_path, "jax_ckpt_*.npz")))
    if not fs:
        raise FileNotFoundError(f"no checkpoints in {model_save_path}")
    return fs[-1]


def _mlp(params_net, x):
    """tanh-MLP: tanh on all Dense layers except the last (linear)."""
    p = params_net['params']
    n_layers = len([k for k in p if k.startswith('Dense_')])
    for i in range(n_layers):
        W = np.asarray(p[f'Dense_{i}']['kernel'])
        b = np.asarray(p[f'Dense_{i}']['bias'])
        x = x @ W + b
        if i < n_layers - 1:
            x = np.tanh(x)
    return x


def embedding_mean(params, X):
    """mu_emb = mean over particles of feature_net(X).  X: (N, d) -> (k,)"""
    feat = _mlp(params['feature_net_param'], X)          # (N, k)
    return feat.mean(axis=0)                              # (k,)


def mvnn_drift(params, Xq, mu_emb):
    """Learned drift at query points Xq given pooled measure-embedding mu_emb.
    Xq: (Q, d), mu_emb: (k,)  ->  (Q, d_out)"""
    Q = Xq.shape[0]
    inp = np.concatenate([Xq, np.tile(mu_emb[None, :], (Q, 1))], axis=1)
    return _mlp(params['describe_net_param'], inp)


def mvnn_drift_full(params, X):
    """Convenience: evaluate learned drift at all particles X (measure = X)."""
    mu_emb = embedding_mean(params, X)
    return mvnn_drift(params, X, mu_emb)


# ----------------------------------------------------------------------------
# analytic TRUE drifts (chunked O(Q*N), memory-safe)
# ----------------------------------------------------------------------------
def true_drift_motsch_tadmor(Xq, Xmeas, ell=0.5, exclude_self=True, chunk=512):
    """Normalized (Motsch-Tadmor) 1D/ d-D drift:
        b*(x) = sum_j w_ij (x_j - x_i) / sum_j w_ij ,   w = exp(-(|x_j-x_i|/ell)^2)
    Xq: (Q, d), Xmeas: (N, d) -> (Q, d)"""
    Xq = np.atleast_2d(Xq); Xmeas = np.atleast_2d(Xmeas)
    Q, d = Xq.shape
    out = np.zeros((Q, d))
    for s in range(0, Q, chunk):
        xi = Xq[s:s+chunk]                                   # (b, d)
        diff = Xmeas[None, :, :] - xi[:, None, :]            # (b, N, d)
        r2 = np.sum(diff*diff, axis=-1)                      # (b, N)
        w = np.exp(-r2/(ell*ell))                            # (b, N)
        if exclude_self:
            same = (r2 == 0.0)
            w = np.where(same, 0.0, w)
        num = np.sum(w[..., None]*diff, axis=1)              # (b, d)
        den = np.sum(w, axis=1)[:, None]                     # (b, 1)
        out[s:s+chunk] = num/np.maximum(den, 1e-30)
    return out


def true_drift_attraction_repulsion(Xq, Xmeas, c_rep=1.0, ell_rep=0.5,
                                    c_att=0.7, ell_att=2.0, avg=True,
                                    exclude_self=True, chunk=256):
    """Two-scale attraction-repulsion drift (2D aggregation case):
        b*(x) = (1/N) sum_j k(|x_j-x_i|) (x_j-x_i),
        k(r) = c_rep exp(-(r/ell_rep)^2) - c_att exp(-(r/ell_att)^2)
    Xq: (Q, d), Xmeas: (N, d) -> (Q, d)"""
    Xq = np.atleast_2d(Xq); Xmeas = np.atleast_2d(Xmeas)
    Q, d = Xq.shape
    N = Xmeas.shape[0]
    out = np.zeros((Q, d))
    for s in range(0, Q, chunk):
        xi = Xq[s:s+chunk]
        diff = Xmeas[None, :, :] - xi[:, None, :]            # (b, N, d)
        r2 = np.sum(diff*diff, axis=-1)
        r = np.sqrt(np.maximum(r2, 1e-24))
        k = c_rep*np.exp(-(r2)/(ell_rep*ell_rep)) - c_att*np.exp(-(r2)/(ell_att*ell_att))
        if exclude_self:
            k = np.where(r2 == 0.0, 0.0, k)
        contrib = np.sum(k[..., None]*diff, axis=1)          # (b, d)
        out[s:s+chunk] = contrib/(N if avg else 1.0)
    return out


# ----------------------------------------------------------------------------
# multi-group (CASE7) forward pass + true bump-kernel drift
# ----------------------------------------------------------------------------
def embedding_mean_mg(params, X, k):
    """mean embedding for group k (k in {1,2,3}) using feature_net_param{k}."""
    feat = _mlp(params[f'feature_net_param{k}'], X)
    return feat.mean(axis=0)


def mvnn_drift_mg(params, Xq, mu1, mu2, mu3, k):
    """learned drift for group-k query points given the three pooled embeddings."""
    Q = Xq.shape[0]
    mus = np.concatenate([mu1, mu2, mu3])[None, :]
    inp = np.concatenate([Xq, np.tile(mus, (Q, 1))], axis=1)
    return _mlp(params[f'describe_net_param{k}'], inp)


def _phi_bump(u):
    a = np.abs(u)
    out = np.zeros_like(a)
    m = a < 1.0
    um = a[m]
    out[m] = np.exp(1.0 - 1.0/(1.0 - um**10))
    return out


def true_drift_bump(Xq, Xsrc, Dij, Rj, exclude_self=True, chunk=512):
    """One source-group contribution to a target-group drift (1D hierarchy case):
        (1/den) sum_j K_ij(x_i - x_j),  K_ij(z) = -Dij * phi_bump(z/Rj) * z
    den = N_src - 1 if exclude_self else N_src.   Xq:(Q,1), Xsrc:(N,1) -> (Q,1)"""
    Xq = np.atleast_2d(Xq); Xsrc = np.atleast_2d(Xsrc)
    Q = Xq.shape[0]; N = Xsrc.shape[0]
    den = (N - 1) if (exclude_self and N > 1) else N
    out = np.zeros((Q, 1))
    for s in range(0, Q, chunk):
        xi = Xq[s:s+chunk]                          # (b,1)
        z = xi - Xsrc[:, 0][None, :]                # (b,N)  z = x_i - x_j
        K = -Dij * _phi_bump(z/Rj) * z              # (b,N)
        out[s:s+chunk, 0] = K.sum(axis=1)/max(den, 1)
    return out


# influence matrix D and radii R used in CASE7 (paper hierarchy)
CASE7_D = dict(D11=5.0, D12=10.0, D22=2.0, D23=5.0, D33=1.0)
CASE7_R = dict(R1=1.0, R2=2.5, R3=5.0)


def true_drift_mg_group(k, x1, x2, x3, Xq=None):
    """Total true drift for group k (1,2,3) evaluated at Xq (default = that group's
    own particles).  Encodes the upper-triangular hierarchy of the paper."""
    D, R = CASE7_D, CASE7_R
    if k == 1:
        q = x1 if Xq is None else Xq
        return (true_drift_bump(q, x1, D['D11'], R['R1'], exclude_self=True)
                + true_drift_bump(q, x2, D['D12'], R['R2'], exclude_self=False))
    if k == 2:
        q = x2 if Xq is None else Xq
        return (true_drift_bump(q, x3, D['D23'], R['R3'], exclude_self=False)
                + true_drift_bump(q, x2, D['D22'], R['R2'], exclude_self=True))
    if k == 3:
        q = x3 if Xq is None else Xq
        return true_drift_bump(q, x3, D['D33'], R['R3'], exclude_self=True)
    raise ValueError(k)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def rel_l2(a, b):
    """relative L2 error ||a-b|| / ||b||  (Frobenius over all entries)."""
    return float(np.linalg.norm(a-b)/max(np.linalg.norm(b), 1e-30))


def r2_score(pred, true):
    ss_res = np.sum((pred-true)**2)
    ss_tot = np.sum((true-true.mean())**2)
    return float(1 - ss_res/max(ss_tot, 1e-30))
