# Original code from Charles Kulick for the paper "Learning particle swarming models from data with Gaussian processes" by J. Feng, C. Kulick, Y. Ren, S. Tang (2023)

import argparse
import logging
import numpy as np
import torch
import glob

from GetPairwiseDists import get_pairwise_dists
from ConstructIntraSpeciesKernel import construct_intra_species_kernel
from K_rExplicit import k_r_explicit
from GPPredictionUtils import cov_matern
from HyperparameterOptimization import optimize_species_hyperparameters

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
logging.info(f"Using device: {device}")

# Parser for command line/bash calls
parser = argparse.ArgumentParser()
parser.add_argument('--N1',           type=int,   default=16)
parser.add_argument('--L',            type=int,   default=20)
parser.add_argument('--M',            type=int,   default=9)
parser.add_argument('--noise',        type=float, default=0.001)
parser.add_argument('--plotting_edge_left',  type=float, default=0.0)
parser.add_argument('--plotting_edge_right', type=float, default=5.0)
parser.add_argument('--t_end',        type=float, default=4.0)

args = parser.parse_args()

N1              = args.N1
L               = args.L
M               = args.M
noise_sigma     = args.noise
jitter          = noise_sigma
plotting_edge   = args.plotting_edge_left
plotting_edge_right = args.plotting_edge_right
t_end           = args.t_end

# Fixed configuration details for dataset/plotting construction
d                = 1      # spatial dimension
nu               = 1.5    # Matern smoothness
t_begin          = 0.0
init_method      = 'random_uniform'
plotting_tag     = "main"    # id for plot storage
axis_font_size   = 30
axis_font_size_dynamics = 30
L_plot           = 100    # time points for higher resolution trajectory visualization
n_error_points   = 1000   # grid points for L_inf error computation
n_density_bins   = 1000   # bins for rho density estimation
start_bin_err_comp = 0    # first bin index used in L^2 error (0 = use all bins, can cut near 0 if singular issues)
hyp_opt_iters    = 100    # hyperparameter runs, shared for both species
OPTIMIZE_HYPS    = True
results_dir      = "Result_Files/"

DT = 1e-2*5

x_save = None
v_save = None

data_file_list = glob.glob("/Users/tracy/OneDrive/Documents/Research UCLA/mean_field_neural_network/simple/case5/data_gen3/data_train_tf=1.0/opinion_traj_[1-9].npy")
for data_file in data_file_list:
    data = np.load(data_file)
    x = data[:-1]
    v = (data[1:]-data[:-1])/DT
    if x_save is None:
        x_save = x
        v_save = v
    else:
        x_save = np.concatenate([x_save, x], axis=0)
        v_save = np.concatenate([v_save, v], axis=0)

x_save = x_save.reshape(M,L,N1).transpose(1,2,0)
v_save = v_save.reshape(M,L,N1).transpose(1,2,0)


def define_interaction_function():
    ell = 0.5
    def g11(r):
        return np.exp(-(r/ell)**2)
    return g11

def compute_kernel_predictions(x, hyps, species1_data_3d, true_kernels, prod_mat_s1, pinv_s1):
    """
    Evaluate true kernels, GP posterior means, and posterior variances on grid x.
    Returns dicts keyed by 'g11','g12','g21','g22' for y_true, y_pred, variance.
    """
    n = len(x)
    y_true = np.zeros(n)
    y_pred = np.zeros(n)
    var    = np.zeros(n)

    for i, r in enumerate(x):
        if i % 200 == 0:
            logging.info(f"Kernel prediction progress: {i}/{n}")

        y_true[i] = true_kernels(r)

        Kr_e = k_r_explicit(r, species1_data_3d, N1, d, L, hyps, nu, 'intra')
        y_pred[i] = (Kr_e @ prod_mat_s1).item()

        prior_var = cov_matern(r, r, np.exp(hyps[0]), np.exp(hyps[1]), nu)
        var[i] = max(0.0, (prior_var - Kr_e @ pinv_s1 @ Kr_e.T).item())

    return y_true, y_pred, var

# Core of the main experiment
logging.info(f"Config: N1={N1}, L={L}, M={M}")

interaction_function = define_interaction_function()

system_params = {'t_begin': t_begin, 't_end': t_end, 'init_method': init_method}

d11 = get_pairwise_dists(x_save, d, N1, L, M)

# Transpose to (N, L, M) layout expected by kernel builders
species1_data_3d  = x_save.transpose(1, 0, 2)
species1_derivs_3d = v_save.transpose(1, 0, 2)

Y_species1 = species1_derivs_3d.flatten(order='F')

# Initial (fixed) hyperparameters on log scale: [log(delta), log(omega)] x 4
# hyps = np.array([2.5, -0.5])
hyps = np.array([-4.0, 7.0])

if OPTIMIZE_HYPS:
    logging.info(f"Optimizing hyperparameters (initial: {hyps})")

    result1 = optimize_species_hyperparameters(
        species1_data_3d, Y_species1,
        hyps, N1, d, M, L, nu, jitter,
        max_iterations=hyp_opt_iters, device=device
    )
    hyps = result1.x
    logging.info(f"After species 1 opt: {hyps}")

# Build final kernel matrices
logging.info("Building kernel matrices.")
K11 = construct_intra_species_kernel(species1_data_3d, np.exp(hyps[1]), np.exp(hyps[0]), N1, d, M, L, nu, False, device=device)
logging.info(f"Kernel shapes: K11{K11.shape}")

K_species1 = K11 + jitter * np.eye(K11.shape[0])

# Pseudoinverse on device
logging.info("Computing pseudoinverses.")
pinv_s1 = torch.linalg.pinv(torch.from_numpy(K_species1).to(device)).cpu().numpy()

prod_mat_s1 = pinv_s1 @ Y_species1

true_kernels = interaction_function

# Evaluate kernel predictions on plotting grid
x_grid = np.arange(plotting_edge, plotting_edge_right, 0.01)
logging.info("Computing kernel predictions.")
y_true, y_pred, var = compute_kernel_predictions(
    x_grid, hyps, species1_data_3d, 
    true_kernels, prod_mat_s1, pinv_s1
)

np.save("x_grid.npy", x_grid)
np.save("y_pred.npy", y_pred)

logging.info("Done.")