import argparse

def _str2bool(v):
    if isinstance(v, bool):
        return v
    s = str(v).strip().lower()
    if s in ("true", "t", "1", "yes", "y"):
        return True
    if s in ("false", "f", "0", "no", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Boolean value expected, got '{v}'.")

def get_parser():

    parser = argparse.ArgumentParser(description='SL modelling by herculens.')

    parser.add_argument('--data_path', type=str, default='sims/sim_sl.fits', help='Path to data')
    parser.add_argument('--noise_path', type=str, default='sims/sim_sl_noise.fits', help='Path to noise')
    parser.add_argument('--psf_path', type=str, default='sims/sim_sl_psf.fits', help='Path to psf')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--pixel_scale', type=float, default=0.03, help='Pixel scale')

    # general settings
    parser.add_argument('--use_nnls', type=_str2bool, default=True, help='Use nnls')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--crop_size', type=int, default=None, help='Crop size')
    parser.add_argument('--sampler', type=str, default='optax', choices=['optax', 'emcee', 'nautilus'],
        help="Sampling / optimization method: 'optax' (multi-chain chi2), 'emcee' (ensemble MCMC), 'nautilus' (nested sampling).",
    )
    # parser.add_argument('--init_test', action='store_true', help='Test initial parameters')
    parser.add_argument('--test_ps_positions', action='store_true', help='Test point source positions')
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--ps_random_seed', type=int, default=123456)
    parser.add_argument('--init_params_path', type=str, default=None, help='Path to initial parameters')
    parser.add_argument('--mask_ps', action='store_true', help='Mask point sources')
    parser.add_argument('--exclude_ps', action='store_true', help='Exclude point sources')
    parser.add_argument('--linear_amp_jax_iter', type=int, default=200, help='Iterations for JAX projected-gradient NNLS.')
    parser.add_argument('--gpus', type=str, default='7', help='CUDA visible devices (e.g. "0" or "0,1").')
    
    # optax optimization settings
    parser.add_argument('--num_steps_optax', type=int, default=50_000, help='Number of optax steps to run in this invocation')
    parser.add_argument('--step_size_optax', type=float, default=1e-3, help='Step size for optax')
    parser.add_argument('--num_chains_optax', type=int, default=8, help='Number of chains for optax')
    parser.add_argument('--init_jitter_scale_optax', type=float, default=1e-3, help='Initial jitter scale for optax')
    parser.add_argument('--clip_norm_optax', type=float, default=1.0, help='Clip norm for optax')

    # emcee settings
    parser.add_argument('--n_walkers_emcee', type=int, default=128, help='Number of walkers for emcee')
    parser.add_argument('--n_steps_emcee', type=int, default=50_000, help='Number of steps for emcee')
    parser.add_argument('--n_burn_emcee', type=int, default=10_000, help='Number of burn-in steps for emcee')
    parser.add_argument('--jitter_scale_emcee', type=float, default=1e-3, help='Jitter scale for emcee')

    # nautilus settings
    parser.add_argument('--n_live_nautilus', type=int, default=1000, help='Number of live points for nautilus')
    parser.add_argument('--n_eff_nautilus', type=int, default=500, help='Number of effective samples for nautilus')
    parser.add_argument('--n_batch_nautilus', type=int, default=100, help='Number of batches for nautilus')
    parser.add_argument('--exploration_factor_nautilus', type=float, default=0.1, help='Exploration factor for nautilus')

    args = parser.parse_args()

    return args