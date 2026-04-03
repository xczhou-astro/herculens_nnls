import jax
import optax
import time
import numpy as np
import jax.numpy as jnp
import emcee
import json
from jax.flatten_util import ravel_pytree
import matplotlib.pyplot as plt
from nautilus import Sampler
from scipy.stats import norm

from .models import (
    _nnls_jax_fista,
    apply_nnls_coefficients_to_kwargs_jax,
    build_linear_amp_design_matrix_jax,
    _normalize_link_spec,
    get_init_params,
)

def run_optax(
    prob_model, 
    use_nnls=False,
    num_linear_amps=None,
    param_list=None, 
    type_list=None,
    init_params=None,
    lens_image=None, 
    image_data=None,
    noise_map=None, 
    step_size=1e-3,
    num_steps=5000, 
    clip_norm=1.0,
    num_chains=8, 
    init_jitter_scale=1e-3,
    random_seed=42,
    return_history=True,
    jax_n_iter=200,
):
    print('Running Optax chi2 optimization...')
    print(f'  num_chains={num_chains}, num_steps={num_steps}, step_size={step_size}')
    print(f'  use_nnls={use_nnls}, num_linear_amps={num_linear_amps}')

    rng_key = jax.random.PRNGKey(random_seed)

    image_data_j = jnp.asarray(image_data)
    noise_map_j = jnp.asarray(noise_map)

    if init_params is None:
        init_params = get_init_params(
            prob_model, 
            param_list, 
            type_list, 
            init_params_path=None,
            save_path=None,
            random_seed=random_seed
        )

    init_params_u = prob_model.unconstrain(init_params)

    # 1. Define the Single-Chain Loss Function
    def chi2_loss_single(params_u_single, prev_coefs_single):
        params = prob_model.constrain(params_u_single)
        kwargs = prob_model.params2kwargs(params)
        
        if use_nnls:
            A = build_linear_amp_design_matrix_jax(lens_image, kwargs, param_list, type_list)
            
            data_1d = image_data_j.ravel()
            sig_1d = noise_map_j.ravel()
            # Protect against division by zero (optional but safe)
            sig_1d = jnp.where(sig_1d > 0.0, sig_1d, 1e10)
            
            Aw = A / sig_1d[:, None]
            yw = data_1d / sig_1d
            
            # Solve NNLS
            coefs = _nnls_jax_fista(Aw, yw, prev_coefs_single, n_iter=jax_n_iter)
            
            # CRITICAL: Stop gradients from propagating back through the FISTA loop
            coefs_sg = jax.lax.stop_gradient(coefs)
            
            kwargs = apply_nnls_coefficients_to_kwargs_jax(kwargs, coefs_sg, param_list, type_list)
            new_coefs = coefs_sg
        else:
            new_coefs = prev_coefs_single
            
        model_img = lens_image.model(**kwargs)
        resid = (model_img - image_data_j) / noise_map_j
        chi2 = jnp.sum(resid ** 2)
        chi2 = jnp.where(jnp.isfinite(chi2), chi2, jnp.inf)
        
        return chi2, new_coefs

    # 2. Vectorize the Loss and Gradients across all chains
    valgrad_single = jax.value_and_grad(chi2_loss_single, has_aux=True)
    valgrad_batched = jax.vmap(valgrad_single)

    optimizer = optax.chain(
        optax.clip_by_global_norm(clip_norm),
        optax.adam(step_size),
    )

    def _add_jitter(key, x):
        x = jnp.asarray(x)
        scale = init_jitter_scale * jnp.maximum(jnp.abs(x), 1.0)
        return x + scale * jax.random.normal(key, shape=x.shape, dtype=x.dtype)

    # Initialize batched states
    leaves, treedef = jax.tree_util.tree_flatten(init_params_u)
    n_leaves = len(leaves)
    keys_all = jax.random.split(rng_key, num_chains * max(n_leaves, 1)).reshape((num_chains, max(n_leaves, 1), 2))

    batched_leaves = []
    for j, leaf in enumerate(leaves):
        ks = keys_all[:, j, :]
        batched_leaf = jax.vmap(lambda k: _add_jitter(k, leaf))(ks)
        batched_leaves.append(batched_leaf)

    params_u0 = jax.tree_util.tree_unflatten(treedef, batched_leaves)
    opt_state0 = jax.vmap(optimizer.init)(params_u0)
    best_params_u0 = params_u0
    best_chi2_0 = jnp.full((num_chains,), jnp.inf)
    prev_coefs0 = jnp.zeros((num_chains, num_linear_amps), dtype=image_data_j.dtype)
    # Coefficients that achieved the best chi2 per chain (same NNLS state as the loss at that step).
    best_coefs0 = jnp.zeros((num_chains, num_linear_amps), dtype=image_data_j.dtype)

    def tree_take0(pytree, idx):
        return jax.tree_util.tree_map(lambda x: x[idx], pytree)

    # 3. The Fully Compiled Optimization Step
    def step(carry, _):
        params_u, opt_state, best_params_u, best_chi2, prev_coefs, best_coefs = carry

        (chi2_vals, new_coefs), grads = valgrad_batched(params_u, prev_coefs)

        is_better = chi2_vals < best_chi2
        best_chi2 = jnp.minimum(best_chi2, chi2_vals)
        best_params_u = jax.tree_util.tree_map(
            lambda bp, p: jnp.where(
                is_better.reshape((num_chains,) + (1,) * (p.ndim - 1)), p, bp
            ),
            best_params_u,
            params_u,
        )
        best_coefs = jnp.where(is_better[:, None], new_coefs, best_coefs)

        updates, opt_state = jax.vmap(optimizer.update)(grads, opt_state, params_u)
        params_u = optax.apply_updates(params_u, updates)

        return (params_u, opt_state, best_params_u, best_chi2, new_coefs, best_coefs), chi2_vals

    # 4. Execute the JAX Scan
    start_time = time.time()
    carry0 = (params_u0, opt_state0, best_params_u0, best_chi2_0, prev_coefs0, best_coefs0)
    carry_f, chi2_hist = jax.lax.scan(step, carry0, xs=None, length=int(num_steps))
    end_time = time.time()
    print(f"[optax] Optimization run time: {end_time - start_time:.2f} seconds")
    # 5. Extract Best Global Chain

    _, _, best_params_u_f, best_chi2_f, _, best_coefs_f = carry_f
    best_chain = jnp.argmin(best_chi2_f)
    best_params_u = tree_take0(best_params_u_f, best_chain)
    best_params = prob_model.constrain(best_params_u)
    best_chi2 = float(best_chi2_f[best_chain])
    best_nnls_coefs = (
        np.asarray(best_coefs_f[best_chain]) if use_nnls else None
    )

    if return_history:
        return best_params, np.asarray(chi2_hist), best_chi2, best_nnls_coefs

    return best_params, None, best_chi2, best_nnls_coefs


def run_emcee(
    prob_model,
    use_nnls=False, 
    num_linear_amps=None,
    param_list=None,
    type_list=None,
    init_params=None,
    lens_image=None,
    image_data=None,
    noise_map=None,
    n_walkers=128,
    n_steps=3000,
    n_burn=1000,
    jitter_scale=1e-3,
    unflatten_batch_size=1024,
    progress=True,
    bad_logp_threshold=-1e50,
    bad_init_log_path=None,
    max_logged_bad_init=200,
    jax_n_iter=200,
    random_seed=None,
):
    """
    Run emcee in constrained parameter space.
    - If init_params is provided, initialize walkers around that point.
    - If init_params is None, initialize walkers from fresh prior samples.
    Includes JAX-compiled fast NNLS for linear amplitudes using Profile Likelihood.
    """

    print('Running MCMC by emcee...')
    print(f'  n_walkers={n_walkers}, n_steps={n_steps}, n_burn={n_burn}')
    print(f'  use_nnls={use_nnls}, num_linear_amps={num_linear_amps}')

    rng_np = np.random.default_rng(random_seed)
    base_seed_jax = int(rng_np.integers(0, 2**31 - 1))

    init_mode = init_params is not None
    template_params = None
    if init_mode:
        flat_init, unflatten_fn = ravel_pytree(init_params)
        flat_init = np.asarray(flat_init, dtype=float)
        ndim = flat_init.size
    else:
        template_params = prob_model.get_sample(jax.random.PRNGKey(base_seed_jax))
        flat_template, unflatten_fn = ravel_pytree(template_params)
        ndim = int(np.asarray(flat_template).size)
        print("[emcee] init_params not provided; initializing walkers from prior samples.")

    if n_walkers < 2 * ndim:
        n_walkers = 2 * ndim
        print(f"[emcee] Increased n_walkers to {n_walkers} (>= 2 * ndim).")

    # Small relative perturbation around init point per walker (used only in init_mode).
    if init_mode:
        base_scale = np.maximum(np.abs(flat_init), 1.0)

    def _spec_bounds(param_spec):
        if not isinstance(param_spec, list):
            return None, None
        if len(param_spec) == 4:
            return float(param_spec[2]), float(param_spec[3])
        if len(param_spec) == 2:
            # LogNormal-like prior specs are strictly positive.
            return 1e-12, np.inf
        return None, None

    def _site_bounds_map():
        if param_list is None:
            return {}
        out = {}
        for i, lens_mass_model in enumerate(param_list.get('lens_mass_params_list', [])):
            for key, param in lens_mass_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    out[f'lens_{key}_{i}'] = _spec_bounds(param)
        for i, lens_light_model in enumerate(param_list.get('lens_light_params_list', [])):
            for key, param in lens_light_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    out[f'lens_light_{key}_{i}'] = _spec_bounds(param)
        source_types = [] if type_list is None else type_list.get('source_light_type_list', [])
        pixelated_only = source_types == ['PIXELATED']
        if not pixelated_only:
            for i, source_light_model in enumerate(param_list.get('source_light_params_list', [])):
                for key, param in source_light_model.items():
                    if _normalize_link_spec(param) is None and isinstance(param, list):
                        out[f'source_{key}_{i}'] = _spec_bounds(param)
        ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
        for i, point_source_model in enumerate(param_list.get('point_source_params_list', [])):
            ps_type = ps_types[i] if i < len(ps_types) else None
            for key, param in point_source_model.items():
                if key in ('n_images', 'sigma_image', 'sigma_source'):
                    continue
                if _normalize_link_spec(param) is not None:
                    continue
                if not isinstance(param, list):
                    continue
                if ps_type == 'IMAGE_POSITIONS' and key == 'amp' and len(param) == 2:
                    out[f'ps_{key}_{i}'] = _spec_bounds(param)
                elif ps_type != 'IMAGE_POSITIONS':
                    out[f'ps_{key}_{i}'] = _spec_bounds(param)
        return out

    bounds_ref_params = init_params if init_mode else template_params
    bounds_map = _site_bounds_map()
    low_tree = {}
    high_tree = {}
    for k, v in bounds_ref_params.items():
        arr = np.asarray(v, dtype=float)
        lo, hi = bounds_map.get(k, (None, None))
        lo_arr = np.full(arr.shape, -np.inf, dtype=float) if lo is None else np.full(arr.shape, lo, dtype=float)
        hi_arr = np.full(arr.shape, np.inf, dtype=float) if hi is None else np.full(arr.shape, hi, dtype=float)
        low_tree[k] = jnp.asarray(lo_arr)
        high_tree[k] = jnp.asarray(hi_arr)
    flat_low, _ = ravel_pytree(low_tree)
    flat_high, _ = ravel_pytree(high_tree)
    flat_low = np.asarray(flat_low, dtype=float)
    flat_high = np.asarray(flat_high, dtype=float)

    # =========================================================================
    # JAX LIKELIHOOD DEFINITIONS
    # =========================================================================
    if use_nnls:
        if lens_image is None or image_data is None or noise_map is None:
            raise ValueError(
                "When use_nnls=True, lens_image/image_data/noise_map must be provided."
            )
        if param_list is None or type_list is None:
            raise ValueError(
                "When use_nnls=True, param_list and type_list must be provided."
            )
            
        image_data_j = jnp.asarray(image_data)
        noise_map_j = jnp.asarray(noise_map)
        data_1d = image_data_j.ravel()
        sig_1d = jnp.where(noise_map_j.ravel() > 0.0, noise_map_j.ravel(), 1e10)
        log_2pi = jnp.log(2.0 * jnp.pi)
        
        ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
        sigma_source = 1e-3
        try:
            for ps in param_list.get('point_source_params_list', []):
                if isinstance(ps, dict) and 'sigma_source' in ps:
                    sigma_source = float(ps['sigma_source'])
                    break
        except Exception:
            pass

        @jax.jit
        def _log_prob_one_nnls(theta):
            params_constrained = unflatten_fn(theta)
            
            # 1. Extract pure Prior probability
            lp_full = prob_model.log_prob(params_constrained, constrained=True)
            ll_placeholder = prob_model.log_likelihood(params_constrained)
            log_prior = lp_full - ll_placeholder

            # 2. Firewalled NNLS Solver
            def _valid_prior_fn(_):
                kwargs = prob_model.params2kwargs(params_constrained)
                
                A = build_linear_amp_design_matrix_jax(lens_image, kwargs, param_list, type_list)
                Aw = A / sig_1d[:, None]
                yw = data_1d / sig_1d
                x0 = jnp.zeros(A.shape[1], dtype=Aw.dtype)
                coefs = _nnls_jax_fista(Aw, yw, x0, n_iter=jax_n_iter)
                
                kwargs_solved = apply_nnls_coefficients_to_kwargs_jax(kwargs, coefs, param_list, type_list)
                
                model_img = lens_image.model(**kwargs_solved)
                model_var = lens_image.Noise.C_D_model(model_img)
                resid = image_data_j - model_img
                log_like = -0.5 * jnp.sum((resid * resid) / model_var + jnp.log(model_var) + log_2pi)

                if len(ps_types) > 0 and 'IMAGE_POSITIONS' in ps_types:
                    log_like = log_like + lens_image.PointSourceModel.log_prob_source_plane(
                        kwargs_solved, sigma_source=sigma_source
                    )
                return log_prior + log_like

            out = jax.lax.cond(
                jnp.isfinite(log_prior),
                _valid_prior_fn,
                lambda _: -jnp.inf,
                operand=None
            )
            return jnp.where(jnp.isfinite(out), out, -jnp.inf)

        @jax.jit
        def _log_prob_batch_nnls(theta_batch):
            vals = jax.vmap(_log_prob_one_nnls)(theta_batch)
            return jnp.where(jnp.isfinite(vals), vals, -jnp.inf)
            
    else:
        @jax.jit
        def _log_prob_one(theta):
            params_constrained = unflatten_fn(theta)
            return prob_model.log_prob(params_constrained, constrained=True)

        @jax.jit
        def _log_prob_batch(theta_batch):
            vals = jax.vmap(_log_prob_one)(theta_batch)
            vals = jnp.where(jnp.isfinite(vals), vals, -jnp.inf)
            return vals

    # =========================================================================
    # EMCEE WRAPPERS AND INITIALIZATION
    # =========================================================================
    batch_log_prob_error_count = 0
    single_log_prob_error_count = 0

    def _log_prob_single_safe(theta_1d):
        nonlocal single_log_prob_error_count
        try:
            theta_1d = np.asarray(theta_1d, dtype=np.float64)
            if not np.isfinite(theta_1d).all():
                return -np.inf
            if use_nnls:
                lp = float(_log_prob_one_nnls(jnp.asarray(theta_1d)))
            else:
                lp = float(_log_prob_one(jnp.asarray(theta_1d)))
            if not np.isfinite(lp):
                return -np.inf
            return lp
        except Exception as e:
            single_log_prob_error_count += 1
            if single_log_prob_error_count <= 5:
                print(f"[emcee] Warning: single-walker log_prob exception (#{single_log_prob_error_count}): {e}")
            return -np.inf

    def log_prob_fn(theta):
        # emcee vectorized callback: theta shape (n_walkers, ndim)
        nonlocal batch_log_prob_error_count
        try:
            theta = np.asarray(theta, dtype=np.float64)
            if theta.ndim == 1:
                theta = theta[None, :]
            bad_theta = ~np.isfinite(theta).all(axis=1)
            
            if use_nnls:
                vals = np.array(_log_prob_batch_nnls(jnp.asarray(theta)), dtype=np.float64, copy=True)
            else:
                vals = np.array(_log_prob_batch(jnp.asarray(theta)), dtype=np.float64, copy=True)
                
            vals[~np.isfinite(vals)] = -np.inf
            vals[bad_theta] = -np.inf
            return vals
        except Exception as e:
            batch_log_prob_error_count += 1
            if batch_log_prob_error_count <= 5:
                print(f"[emcee] Warning: batched log_prob exception (#{batch_log_prob_error_count}); "
                      f"falling back to per-walker evaluation. Error: {e}")
            theta = np.asarray(theta, dtype=np.float64)
            if theta.ndim == 1:
                theta = theta[None, :]
            vals = np.array([_log_prob_single_safe(t) for t in theta], dtype=np.float64)
            return vals

    def _propose_walkers(n, scale):
        return flat_init[None, :] + rng_np.standard_normal((n, ndim)) * (scale * base_scale[None, :])

    def _in_bounds(theta_1d):
        return bool(np.all(theta_1d >= flat_low) and np.all(theta_1d <= flat_high))

    def _eval_log_prob(theta_1d):
        """Robust single-point log_prob evaluation (avoids vmap init brittleness)."""
        try:
            theta_1d = np.asarray(theta_1d, dtype=np.float64)
            if not np.isfinite(theta_1d).all():
                return -np.inf
            if use_nnls:
                lp = float(_log_prob_one_nnls(jnp.asarray(theta_1d)))
            else:
                lp = float(prob_model.log_prob(unflatten_fn(jnp.asarray(theta_1d)), constrained=True))
            if not np.isfinite(lp):
                return -np.inf
            return lp
        except Exception:
            return -np.inf

    def _make_valid_initial_walkers():
        """Construct p0 with finite log_prob for all walkers when possible."""
        if not _in_bounds(flat_init):
            raise RuntimeError("[emcee] init_params is out of defined parameter bounds.")
        init_lp = _eval_log_prob(flat_init)
        if not np.isfinite(init_lp):
            raise RuntimeError(
                "[emcee] init_params has non-finite log_prob; cannot initialize valid walkers."
            )

        p0_local = np.empty((n_walkers, ndim), dtype=np.float64)
        accepted = 0
        reject_oob = 0
        reject_bad = 0
        bad_in_bound_records = []
        attempts_per_walker = 40
        trial_scales = [jitter_scale * s for s in (1.0, 0.5, 0.2, 0.1, 0.05, 0.01, 1e-3)]

        for i in range(n_walkers):
            found = False
            for scale in trial_scales:
                for _ in range(attempts_per_walker):
                    cand = _propose_walkers(1, max(scale, 1e-12))[0]
                    if not _in_bounds(cand):
                        reject_oob += 1
                        continue
                    lp = _eval_log_prob(cand)
                    if np.isfinite(lp) and lp >= max(bad_logp_threshold, init_lp - 1e8):
                        p0_local[i] = cand
                        found = True
                        accepted += 1
                        break
                    reject_bad += 1
                    if len(bad_in_bound_records) < max_logged_bad_init:
                        bad_in_bound_records.append(
                            {
                                "walker_index": int(i),
                                "trial_scale": float(scale),
                                "log_prob": None if not np.isfinite(lp) else float(lp),
                                "reason": "non_finite_log_prob" if not np.isfinite(lp) else "below_threshold",
                                "theta": cand.tolist(),
                            }
                        )
                if found:
                    break
            if not found:
                # Guaranteed finite fallback (checked above)
                p0_local[i] = flat_init
                accepted += 1

        print(
            f"[emcee] Initial walkers validated: {accepted}/{n_walkers} "
            f"(out-of-bounds rejects={reject_oob}, in-bounds bad-logp rejects={reject_bad}, fallback-to-init enabled)."
        )
        if bad_init_log_path is not None:
            try:
                with open(bad_init_log_path, "w") as f:
                    json.dump(
                        {
                            "n_logged": len(bad_in_bound_records),
                            "max_logged": int(max_logged_bad_init),
                            "bad_logp_threshold": float(bad_logp_threshold),
                            "records": bad_in_bound_records,
                        },
                        f,
                        indent=2,
                    )
                print(f"[emcee] Logged in-bounds bad-init proposals to '{bad_init_log_path}'.")
            except Exception as e:
                print(f"[emcee] Failed to save bad-init proposal log: {e}")
        return p0_local

    def _make_prior_initial_walkers():
        """Construct p0 from prior draws only (no init-point dependence)."""
        p0_local = np.empty((n_walkers, ndim), dtype=np.float64)
        accepted = 0
        attempts = 0
        max_attempts = max(1000, 20 * n_walkers)
        reject_oob = 0

        while accepted < n_walkers and attempts < max_attempts:
            key = jax.random.PRNGKey(base_seed_jax + 10_000 + attempts)
            cand_params = prob_model.get_sample(key)
            cand_flat, _ = ravel_pytree(cand_params)
            cand = np.asarray(cand_flat, dtype=np.float64)
            if cand.size != ndim:
                attempts += 1
                continue
            if not _in_bounds(cand):
                reject_oob += 1
                attempts += 1
                continue
            lp = _eval_log_prob(cand)
            if np.isfinite(lp):
                p0_local[accepted] = cand
                accepted += 1
            attempts += 1

        if accepted < n_walkers:
            raise RuntimeError(
                f"[emcee] Could not build enough valid prior-initialized walkers "
                f"({accepted}/{n_walkers}) after {attempts} attempts."
            )

        print(
            f"[emcee] Initial walkers from prior: {accepted}/{n_walkers} accepted "
            f"(out-of-bounds rejects={reject_oob})."
        )
        return p0_local

    # =========================================================================
    # RUN MCMC
    # =========================================================================
    start_time = time.time()
    if init_mode:
        p0 = _make_valid_initial_walkers()
    else:
        p0 = _make_prior_initial_walkers()

    sampler = emcee.EnsembleSampler(
        n_walkers, ndim, log_prob_fn, vectorize=True
    )
    sampler.run_mcmc(p0, n_steps, progress=progress)

    end_time = time.time()
    print(f"[emcee] MCMC run time: {end_time - start_time:.2f} seconds")

    flat_samples = np.asarray(sampler.get_chain(discard=n_burn, flat=True))
    flat_log_prob = np.asarray(sampler.get_log_prob(discard=n_burn, flat=True))
    best_idx = int(np.nanargmax(flat_log_prob))
    best_params = unflatten_fn(jnp.asarray(flat_samples[best_idx]))

    # Convert flat samples to dict-of-arrays expected by downstream post-processing.
    # Use chunked batched unflatten to avoid large GPU memory spikes.
    @jax.jit
    def _unflatten_batch(theta_batch):
        return jax.vmap(unflatten_fn)(theta_batch)

    n_total = flat_samples.shape[0]
    chunked = {k: [] for k in best_params.keys()}
    for i0 in range(0, n_total, unflatten_batch_size):
        i1 = min(i0 + unflatten_batch_size, n_total)
        chunk_tree = _unflatten_batch(jnp.asarray(flat_samples[i0:i1]))
        for k in chunked:
            chunked[k].append(np.asarray(chunk_tree[k], dtype=np.float64))
    samples_dict = {k: np.concatenate(v, axis=0) for k, v in chunked.items()}

    return sampler, samples_dict, best_params, flat_samples, flat_log_prob

@jax.jit
def _count_rejections(chain_arr):
    """Count MCMC rejections (repeated positions) - JAX compiled."""
    diff = chain_arr[1:, :, :] - chain_arr[:-1, :, :]  # (n_steps-1, n_walkers, n_dim)
    is_rejection = jnp.all(jnp.abs(diff) < 1e-10, axis=2)
    return jnp.sum(is_rejection)

def assess_mcmc_convergence(flat_samples, n_walkers, n_run, n_burn,
                            param_names=None, save_path=None, plot_trace=True):
    """
    Assess MCMC convergence using integrated autocorrelation time, ESS, and trace plots.
    Uses post-burn-in samples only.
    """

    n_steps_kept = int(n_run) - int(n_burn)
    if n_steps_kept <= 1:
        raise ValueError(
            f"Invalid run/burn combination: n_run={n_run}, n_burn={n_burn}. "
            "Need at least 2 kept steps."
        )

    n_dim = flat_samples.shape[1]
    expected = n_steps_kept * int(n_walkers)
    if flat_samples.shape[0] != expected:
        raise ValueError(
            "flat_samples length mismatch: "
            f"got {flat_samples.shape[0]}, expected {expected} "
            f"(n_steps_kept={n_steps_kept} * n_walkers={n_walkers})."
        )
    chain = flat_samples.reshape(n_steps_kept, n_walkers, n_dim)

    if param_names is None:
        param_names = [f'param_{i}' for i in range(n_dim)]

    # Integrated autocorrelation time (emcee-recommended diagnostic)
    tau_list = []
    for i in range(n_dim):
        try:
            # emcee expects shape (n_steps, n_walkers) for 2D chains.
            tau_i = emcee.autocorr.integrated_time(chain[:, :, i], quiet=True)
            tau_list.append(float(np.squeeze(tau_i)))
        except Exception as e:
            print(f"  Warning: Could not compute τ for param {i} ({param_names[i]}): {e}")
            tau_list.append(np.nan)

    tau_arr = np.asarray(tau_list, dtype=float)
    tau_mean = np.nanmean(tau_arr) if np.any(np.isfinite(tau_arr)) else np.nan
    n_eff = (n_steps_kept * n_walkers) / tau_mean if not np.isnan(tau_mean) else np.nan
    min_chain_length = 50 * tau_mean if not np.isnan(tau_mean) else np.nan

    print("\n--- MCMC Convergence Diagnostics ---")
    print(f"Mean autocorrelation time τ ≈ {tau_mean:.1f}")
    print(f"Effective sample size ≈ {n_eff:.0f}")
    print(f"Convergence check: chain length {n_steps_kept} should be >> 50*τ = {min_chain_length:.0f}")
    if np.isnan(tau_mean):
        print(f"  WARNING: Autocorrelation time is NaN; convergence cannot be assessed from τ.")
    elif n_steps_kept < min_chain_length:
        print(f"  WARNING: Chain may not be converged (n_steps < 50*τ)")
    else:
        print(f"  OK: Chain length sufficient for convergence")

    # Acceptance rate (from rejection count)
    total_rejections = int(_count_rejections(jnp.array(chain)))
    total_steps = (n_steps_kept - 1) * n_walkers
    acceptance_rate = 1.0 - (total_rejections / total_steps) if total_steps > 0 else np.nan
    print(f"Acceptance rate: {acceptance_rate:.2f}")
    if not np.isnan(acceptance_rate) and acceptance_rate < 0.05:
        print("  WARNING: Very low acceptance rate; chain may be stuck (diagnostics unreliable).")

    # Per-parameter τ (for diagnostics)
    print("\nPer-parameter autocorrelation time τ:")
    for i, name in enumerate(param_names):
        tau_val = tau_list[i] if i < len(tau_list) else np.nan
        print(f"  {name}: τ ≈ {tau_val:.1f}")

    # Trace plot
    if save_path and plot_trace:
        samples_flat = flat_samples  # (n_steps_kept * n_walkers, n_dim)
        fig, axes = plt.subplots(n_dim, 1, figsize=(10, 2 * n_dim), sharex=True)
        if n_dim == 1:
            axes = [axes]
        for i, ax in enumerate(axes):
            ax.plot(samples_flat[:, i], alpha=0.5)
            ax.set_ylabel(param_names[i] if i < len(param_names) else f'param_{i}')
        axes[-1].set_xlabel('Sample index')
        plt.tight_layout()
        plt.savefig(f'{save_path}/mcmc_trace_plot.png', dpi=150, bbox_inches='tight')
        plt.close()

    # Save diagnostics summary
    diagnostics = {
        'mean_autocorr_time': float(tau_mean),
        'effective_sample_size': float(n_eff),
        'min_recommended_chain_length': float(min_chain_length),
        'acceptance_rate': float(acceptance_rate),
        'n_walkers': n_walkers,
        'n_run': n_run,
        'n_burn': n_burn,
        'n_steps_kept': n_steps_kept,
        'per_param_tau': {param_names[i]: float(tau_list[i]) for i in range(min(len(param_names), len(tau_list)))},
    }
    if save_path is not None:
        with open(f'{save_path}/mcmc_convergence.json', 'w') as f:
            json.dump(diagnostics, f, indent=2)

    print("-------------------------------------\n")

def run_nautilus(
    prob_model, 
    use_nnls=False, 
    num_linear_amps=None, 
    param_list=None, 
    type_list=None, 
    init_params=None,
    lens_image=None, 
    image_data=None,
    noise_map=None,
    n_live=1000,
    n_eff=500,
    n_batch=100,
    exploration_factor=0.1,
    verbose=True,
    random_seed=42, 
    jax_n_iter=200,
):
    print('Running Nautilus nested sampling...')
    print(f'  n_live={n_live}, n_eff={n_eff}, n_batch={n_batch}')
    print(f'  use_nnls={use_nnls}, num_linear_amps={num_linear_amps}')

    has_user_init = init_params is not None
    if init_params is None:
        init_params = prob_model.get_sample(jax.random.PRNGKey(int(random_seed)))
    flat_init, unflatten_fn = ravel_pytree(init_params)
    flat_init = np.asarray(flat_init, dtype=float)
    ndim = int(flat_init.size)

    def _spec_bounds(param_spec):
        if not isinstance(param_spec, list):
            return None, None
        if len(param_spec) == 4:
            return float(param_spec[2]), float(param_spec[3])
        if len(param_spec) == 2:
            return 1e-12, np.inf
        return None, None

    def _site_bounds_map():
        if param_list is None:
            return {}
        out = {}
        for i, lens_mass_model in enumerate(param_list.get('lens_mass_params_list', [])):
            for key, param in lens_mass_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    out[f'lens_{key}_{i}'] = _spec_bounds(param)
        for i, lens_light_model in enumerate(param_list.get('lens_light_params_list', [])):
            for key, param in lens_light_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    out[f'lens_light_{key}_{i}'] = _spec_bounds(param)
        source_types = [] if type_list is None else type_list.get('source_light_type_list', [])
        pixelated_only = source_types == ['PIXELATED']
        if not pixelated_only:
            for i, source_light_model in enumerate(param_list.get('source_light_params_list', [])):
                for key, param in source_light_model.items():
                    if _normalize_link_spec(param) is None and isinstance(param, list):
                        out[f'source_{key}_{i}'] = _spec_bounds(param)
        ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
        for i, point_source_model in enumerate(param_list.get('point_source_params_list', [])):
            ps_type = ps_types[i] if i < len(ps_types) else None
            for key, param in point_source_model.items():
                if key in ('n_images', 'sigma_image', 'sigma_source'):
                    continue
                if _normalize_link_spec(param) is not None:
                    continue
                if not isinstance(param, list):
                    continue
                if ps_type == 'IMAGE_POSITIONS' and key == 'amp' and len(param) == 2:
                    out[f'ps_{key}_{i}'] = _spec_bounds(param)
                elif ps_type != 'IMAGE_POSITIONS':
                    out[f'ps_{key}_{i}'] = _spec_bounds(param)
        return out

    bounds_map = _site_bounds_map()
    low_tree = {}
    high_tree = {}
    for k, v in init_params.items():
        arr = np.asarray(v, dtype=float)
        lo, hi = bounds_map.get(k, (None, None))
        lo_arr = np.full(arr.shape, -np.inf, dtype=float) if lo is None else np.full(arr.shape, lo, dtype=float)
        hi_arr = np.full(arr.shape, np.inf, dtype=float) if hi is None else np.full(arr.shape, hi, dtype=float)
        low_tree[k] = jnp.asarray(lo_arr)
        high_tree[k] = jnp.asarray(hi_arr)
    flat_low, _ = ravel_pytree(low_tree)
    flat_high, _ = ravel_pytree(high_tree)
    flat_low = np.asarray(flat_low, dtype=float)
    flat_high = np.asarray(flat_high, dtype=float)

    # If user provides init_params, restrict Nautilus search to a local box
    # [v - exploration_factor*v, v + exploration_factor*v], clipped to global parameter bounds.
    local_low = None
    local_high = None
    if has_user_init:
        delta = exploration_factor * flat_init
        lo0 = np.minimum(flat_init - delta, flat_init + delta)
        hi0 = np.maximum(flat_init - delta, flat_init + delta)
        local_low = np.where(np.isfinite(flat_low), np.maximum(lo0, flat_low), lo0)
        local_high = np.where(np.isfinite(flat_high), np.minimum(hi0, flat_high), hi0)
        bad = local_low >= local_high
        if np.any(bad):
            eps = 1e-8
            local_low[bad] = np.where(np.isfinite(flat_low[bad]), flat_low[bad], flat_init[bad] - eps)
            local_high[bad] = np.where(np.isfinite(flat_high[bad]), flat_high[bad], flat_init[bad] + eps)
            still_bad = local_low >= local_high
            if np.any(still_bad):
                local_low[still_bad] = flat_init[still_bad] - eps
                local_high[still_bad] = flat_init[still_bad] + eps
        print(f"[Nautilus] Using local prior box around init_params (±{exploration_factor*100}%), clipped by global bounds.")

    # Unit-hypercube to constrained-parameter transform.
    # Finite bounds -> uniform transform; semi-infinite/unbounded -> Gaussian-tail transform.
    def prior_transform(u):
        u = np.asarray(u, dtype=np.float64)
        single = u.ndim == 1
        if single:
            u = u[None, :]
        uu = np.clip(u, 1e-12, 1.0 - 1e-12)
        x = np.empty_like(uu)
        if has_user_init:
            x = local_low[None, :] + uu * (local_high - local_low)[None, :]
        else:
            z = norm.ppf(uu)
            both = np.isfinite(flat_low) & np.isfinite(flat_high)
            low_only = np.isfinite(flat_low) & ~np.isfinite(flat_high)
            high_only = ~np.isfinite(flat_low) & np.isfinite(flat_high)
            unbounded = ~np.isfinite(flat_low) & ~np.isfinite(flat_high)
            if np.any(both):
                x[:, both] = flat_low[both] + uu[:, both] * (flat_high[both] - flat_low[both])
            if np.any(low_only):
                x[:, low_only] = flat_low[low_only] + np.exp(np.clip(z[:, low_only], -20.0, 20.0))
            if np.any(high_only):
                x[:, high_only] = flat_high[high_only] - np.exp(np.clip(z[:, high_only], -20.0, 20.0))
            if np.any(unbounded):
                x[:, unbounded] = z[:, unbounded]
        return x[0] if single else x

    if use_nnls:
        if lens_image is None or image_data is None or noise_map is None:
            raise ValueError("When use_nnls=True, lens_image/image_data/noise_map must be provided.")
        if param_list is None or type_list is None:
            raise ValueError("When use_nnls=True, param_list and type_list must be provided.")
        image_data_j = jnp.asarray(image_data)
        noise_map_j = jnp.asarray(noise_map)
        data_1d = image_data_j.ravel()
        sig_1d = jnp.where(noise_map_j.ravel() > 0.0, noise_map_j.ravel(), 1e10)
        log_2pi = jnp.log(2.0 * jnp.pi)
        ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
        sigma_source = 1e-3
        try:
            for ps in param_list.get('point_source_params_list', []):
                if isinstance(ps, dict) and 'sigma_source' in ps:
                    sigma_source = float(ps['sigma_source'])
                    break
        except Exception:
            pass

        @jax.jit
        def _log_like_one(theta):
            params_constrained = unflatten_fn(theta)
            kwargs = prob_model.params2kwargs(params_constrained)
            A = build_linear_amp_design_matrix_jax(lens_image, kwargs, param_list, type_list)
            Aw = A / sig_1d[:, None]
            yw = data_1d / sig_1d
            x0 = jnp.zeros(A.shape[1], dtype=Aw.dtype)
            coefs = _nnls_jax_fista(Aw, yw, x0, n_iter=jax_n_iter)
            kwargs_solved = apply_nnls_coefficients_to_kwargs_jax(kwargs, coefs, param_list, type_list)
            model_img = lens_image.model(**kwargs_solved)
            model_var = lens_image.Noise.C_D_model(model_img)
            model_var = jnp.where(model_var > 0.0, model_var, 1e10)
            resid = image_data_j - model_img
            log_like = -0.5 * jnp.sum((resid * resid) / model_var + jnp.log(model_var) + log_2pi)
            if len(ps_types) > 0 and 'IMAGE_POSITIONS' in ps_types:
                log_like = log_like + lens_image.PointSourceModel.log_prob_source_plane(
                    kwargs_solved, sigma_source=sigma_source
                )
            return jnp.where(jnp.isfinite(log_like), log_like, -jnp.inf)
    else:
        @jax.jit
        def _log_like_one(theta):
            params_constrained = unflatten_fn(theta)
            val = prob_model.log_likelihood(params_constrained)
            return jnp.where(jnp.isfinite(val), val, -jnp.inf)

    @jax.jit
    def _log_like_batch(theta_batch):
        vals = jax.vmap(_log_like_one)(theta_batch)
        return jnp.where(jnp.isfinite(vals), vals, -jnp.inf)

    def log_likelihood_fn(theta_batch):
        theta = np.asarray(theta_batch, dtype=np.float64)
        if theta.ndim == 1:
            theta = theta[None, :]
        vals = np.array(_log_like_batch(jnp.asarray(theta)), dtype=np.float64, copy=True)
        vals[~np.isfinite(vals)] = -np.inf
        return vals

    sampler = Sampler(
        prior=prior_transform,
        likelihood=log_likelihood_fn,
        n_dim=ndim,
        n_live=n_live,
        vectorized=True,
        pass_dict=False,
        seed=random_seed,
    )

    start_time = time.time()
    try:
        sampler.run(n_eff=n_eff, n_batch=n_batch, verbose=verbose)
    except TypeError:
        sampler.run(verbose=verbose)
    end_time = time.time()
    print(f"[Nautilus] Time taken: {end_time - start_time:.2f} seconds")

    points, log_w, log_l = sampler.posterior()
    points = np.asarray(points, dtype=np.float64)
    log_l = np.asarray(log_l, dtype=np.float64)
    best_idx = int(np.nanargmax(log_l))
    best_params = unflatten_fn(jnp.asarray(points[best_idx]))
    log_z = sampler.evidence()
    print(f"[Nautilus] Log Bayesian Evidence (ln Z): {float(log_z):.2f}")

    return sampler, None, best_params, points, log_l


def diagnose_nautilus(
    sampler,
    *,
    param_names=None,
    save_path=None,
    points=None,
    log_w=None,
    log_l=None,
):
    """
    Summarize nested-sampling output: posterior size, log-evidence, weighted ESS,
    and log-likelihood range. Optionally write JSON next to the run.
    """
    if points is None or log_w is None or log_l is None:
        points, log_w, log_l = sampler.posterior()

    points = np.asarray(points, dtype=np.float64)
    log_w = np.asarray(log_w, dtype=np.float64)
    log_l = np.asarray(log_l, dtype=np.float64)

    n = points.shape[0]
    ndim = points.shape[1] if points.ndim == 2 else 0

    if param_names is None:
        param_names = [f'param_{i}' for i in range(ndim)]
    elif len(param_names) != ndim:
        param_names = [f'param_{i}' for i in range(ndim)]

    log_w_max = float(np.max(log_w))
    w = np.exp(log_w - log_w_max)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        w_norm = np.full_like(w, 1.0 / max(n, 1))
        ess = float(n)
        print("[Nautilus diagnostics] Warning: invalid weights; ESS fallback to n.")
    else:
        w_norm = w / w_sum
        ess = float(1.0 / np.sum(w_norm ** 2))

    try:
        log_z = float(sampler.evidence())
    except Exception as e:
        log_z = float('nan')
        print(f"[Nautilus diagnostics] Could not read evidence: {e}")

    log_l_max = float(np.nanmax(log_l))
    log_l_med = float(np.quantile(log_l, 0.5))

    print("\n--- Nautilus diagnostics ---")
    print(f"Posterior samples: {n}, ndim: {ndim}")
    print(f"Log evidence ln Z ≈ {log_z:.4f}")
    print(f"Weighted ESS (importance weights) ≈ {ess:.1f}")
    print(f"log L range: max={log_l_max:.4f}, median={log_l_med:.4f}")

    per_param = {}
    for i, name in enumerate(param_names):
        mu = float(np.sum(w_norm * points[:, i]))
        per_param[name] = {'weighted_mean': mu}

    diagnostics = {
        'n_posterior_samples': int(n),
        'ndim': int(ndim),
        'log_evidence_ln_z': log_z,
        'weighted_ess': ess,
        'log_like_max': log_l_max,
        'log_like_median': log_l_med,
        'per_param_weighted_mean': per_param,
    }
    if save_path is not None:
        out = f'{save_path}/nautilus_diagnostics.json'
        with open(out, 'w') as f:
            json.dump(diagnostics, f, indent=2)
        print(f"Saved diagnostics to {out}")
    print("-----------------------------\n")

    return diagnostics