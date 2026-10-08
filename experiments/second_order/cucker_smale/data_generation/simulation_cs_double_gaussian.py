#!/usr/bin/env python3
import os, sys, time
import numpy as np
from mpi4py import MPI

# -----------------------
# Helpers & kernel
# -----------------------
def ic_gaussian_fixed(N, d, mean, std, seed=0):
    rng=np.random.default_rng(seed)
    x = rng.normal(loc=mean, scale=std, size=(N, d))
    return x.astype(np.float32)

def ic_double_gaussian(N, d, mean1, std1, mean2, std2, seed=0):
    rng = np.random.default_rng(seed)
    
    N1 = N // 2
    N2 = N - N1   # handles odd N cleanly

    # Make stds vary with the seed
    std_scale = rng.uniform(0.1, 10) # scale std based on seed
    std1 = std1 * std_scale
    std2 = std2 * std_scale

    x1 = rng.normal(loc=mean1, scale=std1, size=(N1, d))
    x2 = rng.normal(loc=mean2, scale=std2, size=(N2, d))

    return np.concatenate([x1, x2], axis=0).astype(np.float32)

def kernel_profile_two_scale(r, c_rep, ell_rep, c_att, ell_att):
    # k(r) = c_rep*exp(-(r/ell_rep)^2) - c_att*exp(-(r/ell_att)^2)
    # r shape: (Bi, Bj)
    return (c_rep * np.exp(-(r*r) / (ell_rep*ell_rep))
            - c_att * np.exp(-(r*r) / (ell_att*ell_att)))

def compute_force_blocked_local(
    x, v,                   # (N, d) full positions (replicated on all ranks)
    i_start, i_end,      # local i-slice [i_start, i_end)
    *,
    N, d, bi, bj,
    c_rep, ell_rep, c_att, ell_att,
    cutoff=True, rcut=np.inf,
    avg=True, dtype=np.float64
):
    """Compute force for i in [i_start, i_end) against all j in [0, N)."""

    invN = (1.0 / N) if avg else 1.0
    n_i = i_end - i_start
    f_local = np.zeros((n_i, d), dtype=dtype)

    # loop over i-blocks in my local slice
    ii_off = 0
    for ib in range(i_start, i_end, bi):
        i1 = min(ib + bi, i_end)
        xi = x[ib:i1].astype(dtype, copy=False)             # (Bi,d)
        vi = v[ib:i1].astype(dtype, copy=False)
        Bi = i1 - ib

        fi = np.zeros((Bi, d), dtype=dtype)

        # loop over j-blocks over the whole domain
        for jb in range(0, N, bj):
            j1 = min(jb + bj, N)
            xj = x[jb:j1].astype(dtype, copy=False)         # (Bj,d)
            vj = v[jb:j1].astype(dtype, copy=False)
            Bj = j1 - jb

            # position differences: (Bi,Bj,d) ; NOTE: uses broadcasting
            x_diff = xj[None, :, :] - xi[:, None, :]          # (Bi,Bj,d)
            r2 = np.sum(x_diff*x_diff, axis=-1)                 # (Bi,Bj)
            same_mask = (r2 == 0.0)                         # i==j pairs
            r = np.sqrt(np.maximum(r2, 1e-24)).astype(dtype)

            k = kernel_profile_two_scale(r, c_rep, ell_rep, c_att, ell_att)  # (Bi,Bj)
            if cutoff:
                k[r >= rcut] = 0.0
            # eliminate self-interaction robustly
            k[same_mask] = 0.0

            # velocity differences
            v_diff = vj[None, :, :] - vi[:, None, :]
            # sum_j k * (vj - vi)
            # contrib shape: (Bi,d)
            contrib = np.sum(k[..., None] * v_diff, axis=1, dtype=dtype)
            fi += contrib

            # free memory early (optional)
            del x_diff, v_diff, r2, same_mask, r, k, contrib

        # scale and write into local buffer
        f_local[ii_off:ii_off+Bi, :] = fi * invN
        ii_off += Bi

    return f_local


def partition_counts_offsets(N, size):
    """Return counts (#particles per rank) and offsets (start index) for 1D block partition."""
    base = N // size
    rem  = N % size
    counts = [base + (1 if r < rem else 0) for r in range(size)]
    offs = [0]
    for r in range(1, size):
        offs.append(offs[r-1] + counts[r-1])
    return counts, offs


# -----------------------
# Main simulation (MPI)
# -----------------------
def main():
    # ---------------- MPI setup ----------------
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # -------------- Parameters -----------------
    # CLI: seed (int)
    if len(sys.argv) < 2:
        if rank == 0:
            print("Usage: python3 simulation.py <seed:int>")
        sys.exit(0)
    seed = int(sys.argv[1])

    # Physical / kernel params
    d = 2
    N = 16000            # increase later (e.g., 40000)
    J = 200             # time steps (saves every 'save_every')
    dt = 1e-2
    avg = True

    # two-scale kernel
    c_rep   = 1.0
    ell_rep = 0.5
    c_att   = 0.7
    ell_att = 2.0

    cutoff = True
    rcut_factor = 4.0
    ell_max = max(ell_rep, ell_att)
    rcut = rcut_factor * ell_max if cutoff else np.inf

    # blocking (tune for memory/cache)
    bi = 128
    bj = 128

    dtype = np.float64

    # trajectory saving
    save_every = 1     # save every k steps to reduce memory
    outdir = "data"
    os.makedirs(outdir, exist_ok=True)
    outfile = os.path.join(outdir, f"simulation_{seed}.npz")  # save both x and v

    # ---------------- Initialization ----------------
    if rank == 0:
        x = ic_double_gaussian(N, d, mean1=(0.5,-0.5), std1=1.0, mean2=(-0.5,0.5), std2=1.0, seed=seed).astype(dtype, copy=False)
        v = ic_gaussian_fixed(N, d, mean=(0.0,0.0), std=0.25, seed=seed).astype(dtype, copy=False)
    else:
        x = np.empty((N, d), dtype=dtype)
        v = np.empty((N, d), dtype=dtype)

    comm.Bcast(x, root=0)
    comm.Bcast(v, root=0)

    # partition along i dimension
    counts, offs = partition_counts_offsets(N, size)
    i_start = offs[rank]
    i_end   = i_start + counts[rank]

    # prepare Allgatherv metadata for gathering x each step
    counts_elems = np.array([c * d for c in counts], dtype=np.int32)
    displs_elems = np.array([o * d for o in offs], dtype=np.int32)

    # Trajectory buffer on root (optional)
    if rank == 0:
        n_frames = 1 + (J-1)//save_every
        X_save = np.empty((n_frames, N, d), dtype=dtype)
        V_save = np.empty((n_frames, N, d), dtype=dtype)
        X_save[0] = x
        V_save[0] = v
        frame_idx = 1
    else:
        X_save = None
        V_save = None
        frame_idx = None

    if rank == 0:
        print(f"[MPI {size} ranks] N={N}, d={d}, steps={J}, dt={dt}, bi={bi}, bj={bj}, "
              f"dtype={dtype}, cutoff={cutoff}, rcut={rcut:.3f}")
        sys.stdout.flush()

    # ---------------- Time stepping ----------------
    t0 = time.time()
    for step in range(1, J):
        # compute local force for my i-slice
        f_loc = compute_force_blocked_local(
            x, v, i_start, i_end,
            N=N, d=d, bi=bi, bj=bj,
            c_rep=c_rep, ell_rep=ell_rep, c_att=c_att, ell_att=ell_att,
            cutoff=cutoff, rcut=rcut, avg=avg, dtype=dtype
        )

        # second order update
        # velocity update with damping
        v_local_next = v[i_start:i_end] + dt * f_loc
        # position update (Euler-Cromer)
        x_local_next = x[i_start:i_end] + dt * v[i_start:i_end]

        # gather all local x to every rank (in-place buffer)
        x_next = np.empty_like(x)
        comm.Allgatherv(
            x_local_next.reshape(-1),
            [x_next.reshape(-1), counts_elems, displs_elems, MPI.DOUBLE]
        )

        x = x_next # next step uses the fresh global positions

        # gather velocities if needed (optional)
        v_next = np.empty_like(v)
        comm.Allgatherv(
            v_local_next.reshape(-1),
            [v_next.reshape(-1), counts_elems, displs_elems, MPI.DOUBLE]
        )

        v = v_next

        # save trajectory on root
        if rank == 0 and (step % save_every == 0):
            X_save[frame_idx] = x
            V_save[frame_idx] = v
            frame_idx += 1

        # optional progress
        if rank == 0 and (step % max(1, J//10) == 0):
            elapsed = time.time() - t0
            print(f"  step {step}/{J}  elapsed {elapsed:.1f}s")
            sys.stdout.flush()

    # ensure last frame saved if not on a save boundary
    if rank == 0 and ((J-1) % save_every != 0):
        X_save[frame_idx-1] = x  # overwrite last reserved slot with final
        V_save[frame_idx-1] = v

    # ---------------- Save result ----------------
    if rank == 0:
        np.savez(outfile, x=X_save, v=V_save)
        print(f"Saved trajectory to {outfile} with shape {X_save.shape} and dtype {X_save.dtype}")
        print(f"Total elapsed: {time.time() - t0:.2f}s")

if __name__ == "__main__":
    main()