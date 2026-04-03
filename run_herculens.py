import numpy as np
import os
import json
import datetime
import sys

def _configure_cuda_from_args(args):
    """Configure CUDA before importing JAX."""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpus)
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


if __name__ == '__main__':
    from herculens_nnls.configurations import get_parser

    args = get_parser()
    _configure_cuda_from_args(args)

    # JAX (import only after CUDA is configured)
    import jax
    import jax.numpy as jnp
    jax.config.update("jax_enable_x64", True)

    # Herculens
    from herculens.RegulModel.regul_model import RegularizationModel
    from herculens.Analysis.plot import Plotter

    from herculens_nnls.utils import (
        Tee,
        json_serializer,
        center_crop,
        get_fits_data,
        print_emcee_parameter_uncertainties,
        _pytree_flat_param_labels,
    )
    from herculens_nnls.models import (
        create_prob_model,
        solve_linear_amplitudes_jax,
        count_sampling_parameters,
        linear_amp_component_labels,
        apply_nnls_coefficients_to_kwargs_jax,
        create_lens_image,
    )
    from herculens_nnls.samplers import (
        run_optax,
        run_emcee,
        run_nautilus,
        assess_mcmc_convergence,
        diagnose_nautilus,
    )
    from herculens_nnls.visualizations import (
        display,
        plot_loss_curve,
        plot_image_plane,
        plot_source_plane,
        plot_corner_traced_params,
        post_process,
        plot_input_data,
        plot_corner_nautilus,
        display_init,
    )

    if args.save_path is None:

        formatted_datetime = datetime.datetime.now().strftime("%Y%m%d%H%M")
        save_path = f'herculens_run/{formatted_datetime}'
        
        if args.debug:
            save_path = f'herculens_run/{args.sampler}_test'

        if args.test_ps_positions:
            save_path = f'herculens_run_ps/{formatted_datetime}_{args.ps_random_seed}'
    else:
        save_path = args.save_path
    
    os.makedirs(save_path, exist_ok=True)
    print(f"Starting new run in directory: {save_path}")

    log_file = open(f'{save_path}/log.txt', 'w')

    sys.stdout = Tee(sys.stdout, log_file)
    sys.stderr = Tee(sys.stderr, log_file)

    image_data = get_fits_data(args.data_path)
    noise_map = get_fits_data(args.noise_path)
    psf_data = get_fits_data(args.psf_path)

    if args.crop_size is not None:
        image_data = center_crop(image_data, args.crop_size)
        noise_map = center_crop(noise_map, args.crop_size)

    lens_mass_type_list = ['SIE', 'SHEAR']
    lens_mass_params_list = [
        {
            'theta_E': [1.5, 0.1, 1.0, 2.0],
            'e1': [0.0, 0.1, -0.5, 0.5],
            'e2': [0.0, 0.1, -0.5, 0.5],
            'center_x': 0.0,
            'center_y': 0.0,
        },
        {
            'ra_0': 0.0,
            'dec_0': 0.0,
            'gamma1': [0.0, 0.1, -0.2, 0.2],
            'gamma2': [0.0, 0.1, -0.2, 0.2],
        }
    ]

    lens_light_type_list = []
    lens_light_params_list = []

    source_light_type_list = ['SERSIC_ELLIPSE', 'GAUSSIAN']
    source_light_params_list = [
        {
            'amp': [1.0, 0.1],
            'e1': [0.0, 0.1, -0.1, 0.1],
            'e2': [0.0, 0.1, -0.1, 0.1],
            'R_sersic': [0.2, 0.1, 0.01, 0.5],
            'n_sersic': [1.5, 0.5, 0.1, 4.0],
            'center_x': [0.0, 0.1, -0.3, 0.3],
            'center_y': [0.0, 0.1, -0.3, 0.3],
        },
        {
            'amp': [1.0, 0.1],
            'sigma': [0.05, 0.01, 0.01, 0.1],
            'center_x': [0.0, 0.1, -0.5, 0.5],
            'center_y': [0.0, 0.1, -0.5, 0.5],
        }
    ]

    point_source_type_list = ['SOURCE_POSITION']
    point_source_params_list = [
        {
            'amp': [1.0, 0.1],
            'ra': [0.0, 0.1, -0.5, 0.5],
            'dec': [0.0, 0.1, -0.5, 0.5],
        }
    ]

    kwargs_numerics_fit = {
        'supersampling_factor': 2,
    }

    kwargs_lens_equation_solver_model = {
        'nsolutions': 5,
        'niter': 5,
        'scale_factor': 2,
        'nsubdivisions': 3,
    }

    param_list = {
        'lens_mass_params_list': lens_mass_params_list,
        'lens_light_params_list': lens_light_params_list,
        'source_light_params_list': source_light_params_list,
        'point_source_params_list': point_source_params_list,
    }

    type_list = {
        'lens_mass_type_list': lens_mass_type_list,
        'lens_light_type_list': lens_light_type_list,
        'source_light_type_list': source_light_type_list,
        'point_source_type_list': point_source_type_list,
    }

    if args.exclude_ps:
        param_list.pop('point_source_params_list')
        type_list.pop('point_source_type_list')

    lens_image = create_lens_image(
        param_list=param_list,
        type_list=type_list,
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args.pixel_scale,
        kwargs_numerics=kwargs_numerics_fit,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver_model,
    )

    plot_input_data(
        image_data=image_data,
        noise_map=noise_map,
        psf_data=psf_data,
        pixel_scale=args.pixel_scale,
        save_path=save_path,
        point_source_type_list=point_source_type_list if not args.exclude_ps else None,
        point_source_params_list=point_source_params_list if not args.exclude_ps else None,
    )

    if source_light_type_list[0] == 'PIXELATED':
        regularization_terms = [
            ['source', 0, 'SPARSITY_STARLET'],
            ['source', 0, 'SPARSITY_BLWAVELET'],
        ]
        regul_model = RegularizationModel(regularization_terms)
    else:
        regul_model = None

    prob_model = create_prob_model(
        param_list, type_list, lens_image, image_data, noise_map, 
        regul_model=regul_model,
        nnls_linear_amps=args.use_nnls,
    )

    num_params_non_linear, num_linear_amps = count_sampling_parameters(
        param_list=param_list,
        type_list=type_list,
        use_nnls=args.use_nnls,
    )

    with open(f'{save_path}/config.json', 'w') as f:
        json.dump(
            {
                'type_list': type_list,
                'param_list': param_list,
                'num_params_non_linear': num_params_non_linear,
                'num_linear_amps': num_linear_amps,
                'kwargs_numerics_fit': kwargs_numerics_fit,
                'kwargs_lens_equation_solver_model': kwargs_lens_equation_solver_model,
                'ps_random_seed': args.ps_random_seed,
                'sampler': args.sampler,
                'use_nnls': args.use_nnls,
                'linear_amp_jax_iter': args.linear_amp_jax_iter,
            }, f, indent=4, default=json_serializer
        )

    init_params = display_init(
        prob_model=prob_model,
        init_params_path=args.init_params_path,
        use_nnls=args.use_nnls,
        lens_image=lens_image,
        image_data=image_data,
        noise_map=noise_map,
        pixel_scale=args.pixel_scale,
        param_list=param_list,
        type_list=type_list,
        jax_n_iter=args.linear_amp_jax_iter,
        save_path=save_path,
        random_seed=args.random_seed,
    )

    if args.sampler == 'emcee':

        print(f"[emcee] Running with walkers={args.n_walkers_emcee}, steps={args.n_steps_emcee}, burn={args.n_burn_emcee}, jitter={args.jitter_scale_emcee}")

        emcee_kwargs = dict(
            prob_model=prob_model,
            use_nnls=args.use_nnls,
            num_linear_amps=num_linear_amps,
            param_list=param_list,
            type_list=type_list,
            lens_image=lens_image,
            image_data=image_data,
            noise_map=noise_map,
            n_walkers=args.n_walkers_emcee,
            n_steps=args.n_steps_emcee,
            n_burn=args.n_burn_emcee,
            jitter_scale=args.jitter_scale_emcee,
            jax_n_iter=args.linear_amp_jax_iter,
        )
        if init_params is not None:
            emcee_kwargs['init_params'] = init_params

        sampler_emcee, samples_emcee, best_params_emcee, flat_samples, flat_log_prob = run_emcee(**emcee_kwargs)

        print_emcee_parameter_uncertainties(flat_samples, init_params if init_params is not None else best_params_emcee)

        np.save(f"{save_path}/emcee_flat_samples.npy", flat_samples)
        np.save(f"{save_path}/emcee_flat_log_prob.npy", flat_log_prob)

        kwargs_best_emcee = prob_model.params2kwargs(best_params_emcee)
        if args.use_nnls:
            kwargs_best_emcee, linear_amp_coefs_emcee, _ = solve_linear_amplitudes_jax(
                lens_image,
                kwargs_best_emcee,
                image_data,
                noise_map,
                param_list,
                type_list,
                jax_n_iter=args.linear_amp_jax_iter,
            )
            np.save(f'{save_path}/linear_amp_coeffs_emcee.npy', np.asarray(linear_amp_coefs_emcee))
            with open(f'{save_path}/linear_amp_labels_emcee.json', 'w') as f:
                json.dump(linear_amp_component_labels(param_list, type_list), f, indent=2)
        with open(f'{save_path}/kwargs_result.json', 'w') as f:
            json.dump(kwargs_best_emcee, f, indent=4, default=json_serializer)

        if not args.use_nnls:
            # Reuse existing posterior post-processing utilities.
            post_process(
                prob_model, lens_image, samples_emcee,
                image_data, noise_map, args.pixel_scale,
                num_params_non_linear, save_path, ll_batch_size=1024
            )
        else:
            best_fit_model = lens_image.model(**kwargs_best_emcee)
            chi2_best = np.sum(((best_fit_model - image_data) / noise_map) ** 2)
            print(f'[emcee+NNLS] Chi^2 of best fit model: {chi2_best:.2f}')
            display(
                [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
                titles=['Best fit model', 'image data', f'Residuals (chi2 = {chi2_best:.2f})'],
                pixel_scale=args.pixel_scale,
                savefilename=f'{save_path}/best_fit_model.png'
            )
            plot_image_plane(lens_image, kwargs_best_emcee, args.pixel_scale, image_data, noise_map, save_path)
            plot_source_plane(lens_image, kwargs_best_emcee, save_path)

        try:
            plot_corner_traced_params(samples_emcee, save_path)
        except Exception as e:
            print(f"[emcee] Corner plot skipped: {e}")

        try:
            params_for_labels = init_params if init_params is not None else best_params_emcee
            param_names = _pytree_flat_param_labels(params_for_labels)
            assess_mcmc_convergence(
                flat_samples=flat_samples,
                n_walkers=args.n_walkers_emcee,
                n_run=args.n_steps_emcee,
                n_burn=args.n_burn_emcee,
                param_names=param_names,
                save_path=save_path,
            )
        except Exception as e:
            print(f"[emcee] MCMC convergence assessment skipped: {e}")

    elif args.sampler == 'optax':

        best_params, chi2_curve, best_chi2, best_nnls_coefs = run_optax(
            prob_model=prob_model,
            use_nnls=args.use_nnls,
            num_linear_amps=num_linear_amps,
            param_list=param_list,
            type_list=type_list,
            init_params=init_params,
            lens_image=lens_image,
            image_data=image_data,
            noise_map=noise_map,
            step_size=args.step_size_optax, 
            num_steps=args.num_steps_optax,
            clip_norm=args.clip_norm_optax,
            num_chains=args.num_chains_optax,
            init_jitter_scale=args.init_jitter_scale_optax,
            random_seed=args.random_seed,
            return_history=True,
            jax_n_iter=args.linear_amp_jax_iter
        )

        print(f"[Optax] Best chi^2: {best_chi2:.2f}")
        np.save(f"{save_path}/optax_chi2_curve.npy", chi2_curve)
        plot_loss_curve(chi2_curve, save_path)
        
        kwargs_best = prob_model.params2kwargs(best_params)
        linear_amp_coefs_out = None
        if args.use_nnls:
            # Use the same NNLS coefficients as at the best-chi2 step (avoids cold re-solve mismatch).
            if best_nnls_coefs is not None:
                linear_amp_coefs_out = best_nnls_coefs
                kwargs_best = apply_nnls_coefficients_to_kwargs_jax(
                    kwargs_best, linear_amp_coefs_out, param_list, type_list
                )
            else:
                kwargs_best, linear_amp_coefs_out, _ = solve_linear_amplitudes_jax(
                    lens_image,
                    kwargs_best,
                    image_data,
                    noise_map,
                    param_list,
                    type_list,
                    jax_n_iter=args.linear_amp_jax_iter,
                )
        with open(f'{save_path}/kwargs_result.json', 'w') as f:
            json.dump(kwargs_best, f, indent=4, default=json_serializer)

        if args.use_nnls and linear_amp_coefs_out is not None:
            np.save(f'{save_path}/linear_amp_coeffs.npy', np.asarray(linear_amp_coefs_out))
            with open(f'{save_path}/linear_amp_labels.json', 'w') as f:
                json.dump(linear_amp_component_labels(param_list, type_list), f, indent=2)

        if args.use_nnls:
            # NumPyro obs site uses unit amplitudes; match Gaussian log-likelihood to Optax chi^2.
            best_log_likelihood = -0.5 * float(best_chi2)
        else:
            best_log_likelihood = float(prob_model.log_likelihood(best_params))
        total_pixels = image_data.size
        bic = num_params_non_linear * np.log(total_pixels) - 2 * best_log_likelihood
        print(f'BIC: {bic:.2f}')
        print(f'Best log likelihood: {best_log_likelihood:.2f}')

        best_fit_model = lens_image.model(**kwargs_best)
        chi2_best = np.sum(((best_fit_model - image_data) / noise_map) ** 2)
        print(f"[Optax] Chi^2 of best fit model: {chi2_best:.2f}")

        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump(
                {
                    'BIC': float(bic),
                    'CHI2': float(chi2_best),
                }, f, indent=4, default=json_serializer
        )

        savez_kw = dict(
            best_fit_model=best_fit_model,
            image_data=image_data,
            noise_map=noise_map,
        )
        if args.use_nnls and linear_amp_coefs_out is not None:
            savez_kw['linear_amp_coeffs'] = np.asarray(linear_amp_coefs_out)
        np.savez_compressed(f'{save_path}/modeling_result.npz', **savez_kw)

        display(
            [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
            titles=['Best fit model', 'image data', f'Residuals (chi2 = {chi2_best:.2f})'],
            pixel_scale=args.pixel_scale,
            savefilename=f'{save_path}/best_fit_model.png'
        )

        plot_image_plane(lens_image, kwargs_best, args.pixel_scale, image_data, noise_map, save_path)
        plot_source_plane(lens_image, kwargs_best, save_path)
 
        try:
            plotter = Plotter(base_fontsize=18, flux_vmin=1e-3, flux_vmax=1e0, res_vmax=6)
            fig = plotter.model_summary(lens_image, kwargs_best)
            fig.savefig(f'{save_path}/model_summary.png')
        except Exception as e:
            print(f"Error plotting model_summary: {e}")

    elif args.sampler == 'nautilus':

        print(f"[Nautilus] Running with n_live={args.n_live_nautilus}, n_eff={args.n_eff_nautilus}, n_batch={args.n_batch_nautilus}")

        sampler_nautilus, _, best_params_nautilus, flat_samples, flat_log_like = run_nautilus(
            prob_model=prob_model,
            use_nnls=args.use_nnls,
            num_linear_amps=num_linear_amps,
            param_list=param_list,
            type_list=type_list,
            init_params=init_params,
            lens_image=lens_image,
            image_data=image_data,
            noise_map=noise_map,
            n_live=args.n_live_nautilus,
            n_eff=args.n_eff_nautilus,
            n_batch=args.n_batch_nautilus,
            exploration_factor=args.exploration_factor_nautilus,
            verbose=True,
            random_seed=args.random_seed if args.random_seed is not None else 42,
            jax_n_iter=args.linear_amp_jax_iter,
        )

        np.save(f"{save_path}/nautilus_flat_samples.npy", np.asarray(flat_samples))
        np.save(f"{save_path}/nautilus_flat_log_like.npy", np.asarray(flat_log_like))

        pts_naut, logw_naut, logl_naut = sampler_nautilus.posterior()
        np.save(f"{save_path}/nautilus_flat_log_weights.npy", np.asarray(logw_naut))

        param_names_nautilus = _pytree_flat_param_labels(best_params_nautilus)
        try:
            diagnose_nautilus(
                sampler_nautilus,
                param_names=param_names_nautilus,
                save_path=save_path,
                points=pts_naut,
                log_w=logw_naut,
                log_l=logl_naut,
            )
        except Exception as e:
            print(f"[Nautilus] Diagnostics skipped: {e}")
        try:
            plot_corner_nautilus(
                pts_naut,
                logw_naut,
                param_names_nautilus,
                save_path,
                random_seed=args.random_seed,
            )
        except Exception as e:
            print(f"[Nautilus] Corner plot skipped: {e}")

        kwargs_best_nautilus = prob_model.params2kwargs(best_params_nautilus)
        linear_amp_coefs_nautilus = None
        if args.use_nnls:
            kwargs_best_nautilus, linear_amp_coefs_nautilus, _ = solve_linear_amplitudes_jax(
                lens_image,
                kwargs_best_nautilus,
                image_data,
                noise_map,
                param_list,
                type_list,
                jax_n_iter=args.linear_amp_jax_iter,
            )
            np.save(f'{save_path}/linear_amp_coeffs_nautilus.npy', np.asarray(linear_amp_coefs_nautilus))
            with open(f'{save_path}/linear_amp_labels_nautilus.json', 'w') as f:
                json.dump(linear_amp_component_labels(param_list, type_list), f, indent=2)

        with open(f'{save_path}/kwargs_result.json', 'w') as f:
            json.dump(kwargs_best_nautilus, f, indent=4, default=json_serializer)

        best_fit_model = lens_image.model(**kwargs_best_nautilus)
        chi2_best = np.sum(((best_fit_model - image_data) / noise_map) ** 2)
        print(f"[Nautilus] Chi^2 of best fit model: {chi2_best:.2f}")

        with open(f'{save_path}/metrics.json', 'w') as f:
            json.dump(
                {
                    'CHI2': float(chi2_best),
                }, f, indent=4, default=json_serializer
            )

        savez_kw = dict(
            best_fit_model=best_fit_model,
            image_data=image_data,
            noise_map=noise_map,
        )
        if args.use_nnls and linear_amp_coefs_nautilus is not None:
            savez_kw['linear_amp_coeffs'] = np.asarray(linear_amp_coefs_nautilus)
        np.savez_compressed(f'{save_path}/modeling_result.npz', **savez_kw)

        display(
            [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
            titles=['Best fit model', 'image data', f'Residuals (chi2 = {chi2_best:.2f})'],
            pixel_scale=args.pixel_scale,
            savefilename=f'{save_path}/best_fit_model.png'
        )

        plot_image_plane(lens_image, kwargs_best_nautilus, args.pixel_scale, image_data, noise_map, save_path)
        plot_source_plane(lens_image, kwargs_best_nautilus, save_path)

        try:
            plotter = Plotter(base_fontsize=18, flux_vmin=1e-3, flux_vmax=1e0, res_vmax=6)
            fig = plotter.model_summary(lens_image, kwargs_best_nautilus)
            fig.savefig(f'{save_path}/model_summary.png')
        except Exception as e:
            print(f"Error plotting model_summary: {e}")
