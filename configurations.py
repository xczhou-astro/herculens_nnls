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

    # data settings
    parser.add_argument('--data_path', type=str, default='F115W/data_with_lens.fits', help='Path to data')
    parser.add_argument('--noise_path', type=str, default='F115W/noise.fits', help='Path to noise')
    parser.add_argument('--psf_path', type=str, default='F115W/psf.fits', help='Path to psf')
    parser.add_argument('--mask_path', type=str, default=None, help='Path to mask')
    parser.add_argument('--save_path', type=str, default=None, help='Path to save results')
    parser.add_argument('--pixel_scale', type=float, default=0.03, help='Pixel scale')

    # general settings
    parser.add_argument('--use_nnls', type=_str2bool, default=True, help='Use nnls')
    parser.add_argument('--random_seed', type=int, default=42, help='Random seed')
    parser.add_argument('--crop_size', type=int, default=61, help='Crop size')
    parser.add_argument(
        '--modeling_config',
        type=str,
        default=None,
        help='YAML file with modeling.lens_mass / modeling.lens_light (default: modeling_config.yaml next to run_herculens.py if present).',
    )
    parser.add_argument('--sampler', type=str, default='optax', choices=['optax', 'emcee', 'nautilus'],
        help="Sampling / optimization method: 'optax' (multi-chain chi2), 'emcee' (ensemble MCMC), 'nautilus' (nested sampling).",
    )
    parser.add_argument('--debug', action='store_true', help='Debug mode')
    parser.add_argument('--init_params_path', type=str, default=None, help='Path to initial parameters')
    parser.add_argument('--linear_amp_jax_iter', type=int, default=100, help='Iterations for JAX projected-gradient NNLS.')
    parser.add_argument('--gpus', type=str, default='7', help='CUDA visible devices (e.g. "0" or "0,1").')

    # for point sources
    parser.add_argument('--image_positions_catalog', type=str, default='photometry/source_catalog_multiband.csv', help='Path to image positions catalog')
    parser.add_argument('--num_sources', type=int, default=1, help='Number of sources in source plane')
    # Per-source catalog row indices (comma-separated), e.g. --images_idx_1=1,2,3 --images_idx_2=4,5
    _max_ps_sources = 20
    for _k in range(1, _max_ps_sources + 1):
        parser.add_argument(
            f'--images_indices_{_k}',
            type=str,
            default=None,
            dest=f'images_indices_{_k}',
            help=f'Comma-separated catalog row indices for point source {_k} (requires num_sources >= {_k}).',
        )
    parser.add_argument('--relieve_mask_indices', type=str, default=None, help='Unmask indices of point sources')
    parser.add_argument('--exclude_ps', type=_str2bool, default=False, help='Exclude point sources')
    
    # optax optimization settings
    parser.add_argument('--num_steps_optax', type=int, default=50_000, help='Number of optax steps to run in this invocation')
    parser.add_argument('--step_size_optax', type=float, default=1e-3, help='Step size for optax')
    parser.add_argument('--min_step_size_optax', type=float, default=1e-6, help='Lower bound for decayed optax step size')
    parser.add_argument('--enable_lr_decay_optax', type=_str2bool, default=True, help='Enable patience-based learning-rate decay in optax')
    parser.add_argument('--lr_decay_factor_optax', type=float, default=0.5, help='Multiplicative decay factor applied to step size when plateau is detected')
    parser.add_argument('--lr_patience_optax', type=int, default=1000, help='Number of non-improving optax steps before LR decay')
    parser.add_argument('--lr_min_delta_optax', type=float, default=0.0, help='Minimum chi2 improvement required to reset LR plateau counter')
    parser.add_argument('--enable_early_stopping_optax', type=_str2bool, default=False, help='Enable early stopping based on plateaued chi2')
    parser.add_argument('--early_stopping_patience_optax', type=int, default=3000, help='Number of non-improving optax steps before early stopping')
    parser.add_argument('--early_stopping_min_delta_optax', type=float, default=0.0, help='Minimum chi2 improvement required to reset early stopping counter')
    parser.add_argument('--num_chains_optax', type=int, default=4, help='Number of chains for optax')
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