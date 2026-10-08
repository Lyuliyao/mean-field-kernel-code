import sys
import os
import numpy as np
from mpi4py import MPI
from PlotDynamicsComparison import create_interpolated_kernels

outdir = "data_time_sim"

# ---------------- constants ----------------
if len(sys.argv) >= 3:
    N = int(sys.argv[2]) # agents
else:
    N = 16000

STEPS    = 20            # time steps
DT       = 1e-2*5           # step size (Euler)
DIM      = 1              # opinion is 1D; kept for symmetry

# interaction kernel: smooth weight (Gaussian of opinion difference)
ELL      = 0.5            # smooth scale ℓ in exp(-(r/ℓ)^2)

# blocking to reduce memory (tune for cache/fewer temporaries)
BI = 400
BJ = 400

x_grid = np.load("x_grid.npy")
y_pred = np.load("y_pred.npy")

interp_fns = create_interpolated_kernels(x_grid, y_pred)

# ---------------- MPI setup ----------------
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def split_indices(n, size):
    counts = np.array([n // size + (1 if r < n % size else 0) for r in range(size)], dtype=np.int32)
    displs = np.zeros(size, dtype=np.int32)
    displs[1:] = np.cumsum(counts[:-1])
    return counts, displs

COUNTS, DISPLS = split_indices(N, size)
i0 = int(DISPLS[rank])
i1 = int(i0 + COUNTS[rank])

# # ---------------- init ----------------
# def init_opinion(N, seed=0):
#     rng = np.random.default_rng(seed)
#     num_gaussian = rng.integers(low=2, high=9, size=1)[0]
#     value = rng.uniform(low=0, high=3, size=num_gaussian)
#     p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
#     x = 0.5 * rng.standard_normal(N) + rng.choice(value, size=N, p=p)
#     x = x - np.mean(x, axis=0)
#     return x.astype(np.float64), x.copy().astype(np.float64), rng

# ---------------- init ----------------
def init_opinion(N, seed=0):
    rng = np.random.default_rng(seed)
    num_gaussian = rng.integers(low=9, high=10, size=1)[0]
    value = rng.uniform(low=0, high=3, size=num_gaussian)
    p = rng.dirichlet(np.ones(num_gaussian), size=1)[0]
    x = 0.5 * rng.standard_normal(N) + rng.choice(value, size=N, p=p)
    x = x - np.mean(x, axis=0)
    return x.astype(np.float64), x.copy().astype(np.float64), rng

# ---------------- influence (blocked, local slice) ----------------
def blocked_influence_local(x, i_start, i_end, bi=BI, bj=BJ):
    """
    Compute local, state-dependent influence for i in [i_start, i_end):
      g_i = [ sum_j w_ij * (x_j - x_i) ] / [ sum_j w_ij ]
    where w_ij = phi(x_j - x_i). Self-terms are removed.
    Returns g slice of shape (i_end - i_start,).
    """
    N = x.shape[0]
    m = i_end - i_start
    g_loc = np.zeros(m, dtype=x.dtype)

    # numerator and denominator for normalization
    num = np.zeros(m, dtype=x.dtype)
    den = np.zeros(m, dtype=x.dtype)

    # loop only over our i-slice, still sweep all j-blocks
    ib = i_start
    while ib < i_end:
        i1 = min(ib + bi, i_end)
        ii0 = ib - i_start           # local offset
        xi = x[ib:i1][:, None]       # (Bi,1)
        num_b = np.zeros((i1 - ib,), dtype=x.dtype)
        den_b = np.zeros((i1 - ib,), dtype=x.dtype)

        jb = 0
        while jb < N:
            j1 = min(jb + bj, N)
            xj = x[jb:j1][None, :]   # (1,Bj)

            diff = xj - xi           # (Bi,Bj)
            w = interp_fns(diff)            # (Bi,Bj)

            # remove self when blocks overlap (i==j)
            if (ib < j1) and (jb < i1):
                mself = min(i1 - max(ib, jb), j1 - max(ib, jb))
                # indices that line up within the overlap
                # align ranges within [ib,i1) and [jb,j1)
                a0 = max(ib, jb) - ib
                b0 = max(ib, jb) - jb
                w[np.arange(a0, a0 + mself), np.arange(b0, b0 + mself)] = 0.0

            sum_w = w.sum(axis=1)                    # (Bi,)
            num_b += (w @ x[jb:j1]) - (sum_w * x[ib:i1])
            den_b += sum_w

            jb = j1

        num[ii0:ii0 + (i1 - ib)] = num_b
        den[ii0:ii0 + (i1 - ib)] = den_b + 1e-12     # avoid divide-by-zero
        ib = i1

    g_loc = num / den
    return g_loc

# ---------------- one step ----------------
def step_allreduce(x):
    # each rank computes g for its i-slice
    g_loc = blocked_influence_local(x, i0, i1, bi=BI, bj=BJ)

    # assemble g on every rank via Allgatherv
    g = np.empty_like(x)
    comm.Allgatherv(
        [g_loc, MPI.DOUBLE],
        [g, (COUNTS, DISPLS), MPI.DOUBLE]
    )

    # explicit Euler
    return x + DT * g

# ---------------- main ----------------
def main():
    start_time = MPI.Wtime()

    if len(sys.argv) >= 2:
        seed = int(sys.argv[1])
    else:
        seed = 1

    # Only rank 0 initializes; broadcast to the rest
    if rank == 0:
        x, x0, rng = init_opinion(N, seed=seed)
        # buffers for saving on rank 0
        X_save = np.empty((STEPS + 1, N), dtype=np.float64)
        VAR_save = np.empty(STEPS + 1, dtype=np.float64)
        X_save[0] = x
        VAR_save[0] = x.var()
    else:
        x = np.empty(N, dtype=np.float64)
        x0 = np.empty(N, dtype=np.float64)  # kept for API symmetry; not used

    # Broadcast initial state
    comm.Bcast([x, MPI.DOUBLE], root=0)
    comm.Bcast([x0, MPI.DOUBLE], root=0)

    # time stepping
    for t in range(1, STEPS + 1):
        x = step_allreduce(x)

        if rank == 0:
            X_save[t] = x
            VAR_save[t] = x.var()

    comm.Barrier()
    end_time = MPI.Wtime()

    if rank == 0:
        os.makedirs(outdir, exist_ok=True)
        np.save(f"{outdir}/opinion_traj_gp_{N}_{seed}.npy", X_save)
        np.save(f"{outdir}/opinion_var_gp_{N}_{seed}.npy", VAR_save)

        # x_true = np.load(f"/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/test_data3/data_test_tf=1.0/opinion_traj_{N}_{seed}.npy")
        # test_err = np.linalg.norm(x_true-X_save)
        print(f"N = {N}")
        # print(f"Test error: {test_err}")
        print(f"Runtime: {end_time - start_time:.3f} seconds")

if __name__ == "__main__":
    main()

