import matplotlib.pyplot as plt
import numpy as np
from herculens.Util import model_util
import corner

from .utils import json_serializer
from .models import batch_log_likelihood, get_init_params, solve_linear_amplitudes_jax
import joblib
import jax.numpy as jnp
import json

def display(plot_data, titles, pixel_scale, savefilename=None):
    
    num = len(plot_data)
    fig, axes = plt.subplots(1, num, figsize=(4 * num + 2, 4))
    for i in range(num):
        
        ny, nx = plot_data[i].shape
        x_center = nx // 2
        y_center = ny // 2
        extent = [
            -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
            -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
        ]
        
        im = axes[i].imshow(plot_data[i], origin='lower', cmap='magma', extent=extent)
        axes[i].set_xlabel('arcsec')
        axes[i].set_ylabel('arcsec')
        axes[i].set_title(titles[i])
        plt.colorbar(im, ax=axes[i])
    plt.tight_layout()
    if savefilename is not None:
        plt.savefig(savefilename, dpi=300, bbox_inches='tight')
    plt.close()

def plot_input_data(
    image_data,
    noise_map,
    psf_data,
    pixel_scale,
    save_path=None,
    point_source_type_list=None,
    point_source_params_list=None,
):
    """
    Plot image data, noise map, and PSF kernel with arcsec axes centered at (0,0).
    If point sources use IMAGE_POSITIONS, overlay provided (ra, dec) positions on the image data.
    """
    ny, nx = image_data.shape
    x_center = nx // 2
    y_center = ny // 2
    extent = [
        -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3 + 2, 5))

    im0 = axes[0].imshow(np.log10(np.clip(image_data, 1e-10, None)), origin='lower', cmap='magma', extent=extent)
    axes[0].set_title('Image data')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label='log10')

    # Overlay provided image-plane point-source positions (IMAGE_POSITIONS)
    if (
        point_source_type_list is not None
        and point_source_params_list is not None
        and any(t == 'IMAGE_POSITIONS' for t in point_source_type_list)
    ):
        colors = ['deepskyblue', 'tomato']
        for i, (t, ps) in enumerate(zip(point_source_type_list, point_source_params_list)):
            if t != 'IMAGE_POSITIONS':
                continue
            ras = np.atleast_1d(np.asarray(ps.get('ra', []), dtype=float))
            decs = np.atleast_1d(np.asarray(ps.get('dec', []), dtype=float))
            if ras.size and decs.size:
                axes[0].scatter(
                    ras,
                    decs,
                    s=40,
                    marker='o',
                    facecolors='none',
                    edgecolors=colors[i % len(colors)],
                    linewidths=1.5,
                    label=f'PS {i + 1}',
                )
        axes[0].legend(loc='best', fontsize=8)

    im1 = axes[1].imshow(np.log10(np.clip(noise_map, 1e-12, None)), origin='lower', cmap='viridis', extent=extent)
    axes[1].set_title('Noise map')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=axes[1], label='log10')

    # PSF is typically not on the same pixel scale / centering as the cutout; just show it in pixels
    im2 = axes[2].imshow(np.log10(np.clip(psf_data, 1e-12, None)), origin='lower', cmap='magma')
    axes[2].set_title('PSF kernel')
    axes[2].set_xlabel('pixel')
    axes[2].set_ylabel('pixel')
    plt.colorbar(im2, ax=axes[2], label='log10')

    plt.tight_layout()
    if save_path is not None:
        plt.savefig(f'{save_path}/input_data.png', dpi=200, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_image_plane(
    lens_image,
    kwargs_result, 
    pixel_scale,
    image_data, 
    noise_map,
    save_path, 
):
    
    ny, nx = image_data.shape
    
    # Plot components separately; explicitly turn off others to avoid default additions.
    model_extended = lens_image.model(
        **kwargs_result,
        source_add=True,
        lens_light_add=False,
        point_source_add=False,
    )
    
    model_lens_light = np.zeros((ny, nx))
    if 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result, lens_light_add=True, source_add=False, 
            point_source_add=False
        )
    
    model_point_sources = np.zeros((ny, nx))
    if 'kwargs_point_source' in kwargs_result:
        model_point_sources = lens_image.model(
            **kwargs_result,
            source_add=False,
            lens_light_add=False,
            point_source_add=True,
        )
        # For display: values at ~1e-20 are effectively zero
        # model_point_sources = np.where(model_point_sources < 1e-12, 0.0, model_point_sources)
        
        theta_x, theta_y, amps = lens_image.PointSourceModel.get_multiple_images(
            kwargs_result['kwargs_point_source'],
            kwargs_lens=kwargs_result['kwargs_lens'],
            kwargs_solver=lens_image.kwargs_lens_equation_solver,
            with_amplitude=True
        )
        # Treat numerical floor as zero for reporting
        # amps = [np.where(np.asarray(a) < 1e-12, 0.0, np.asarray(a)) for a in amps]
        
        ra_image_list = []
        dec_image_list = []
        
        for i in range(len(theta_x)):
            ra_image_list.append(np.array(theta_x[i]))    
            dec_image_list.append(np.array(theta_y[i]))
            
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            
            # Print arrays without scientific notation
            np.set_printoptions(suppress=True)
            print(f'RA for lensed point source {i}: {", ".join(str(r) for r in ras)}')
            print(f'Dec for lensed point source {i}: {", ".join(str(d) for d in decs)}')
            print(f'Amplitudes for lensed point source {i}: {", ".join(str(a) for a in amps[i])}')
            print()
            # Reset to default for safety if needed elsewhere (optional)
            # np.set_printoptions(suppress=False)
            
    model_composite = lens_image.model(**kwargs_result, source_add=True, 
                                point_source_add=True)
    
    residuals = (model_composite - image_data) / noise_map
    
    ny, nx = model_composite.shape
    x_center = nx // 2
    y_center = ny // 2
    extent = [
        -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
    ]

    color = ['cyan', 'green']
    
    fig, ax = plt.subplots(2, 3, figsize=(3 * 5 + 2, 5 * 2))
    
    # 1. Image plane by extended source
    im0 = ax[0, 0].imshow(np.log10(np.clip(model_extended, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            ax[0, 0].scatter(ras, decs, s=20, marker='x', color=color[i])
    
    ax[0, 0].set_title('Extended Source (Lensed)')
    ax[0, 0].set_xlabel('arcsec')
    ax[0, 0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=ax[0, 0], label='log10')
    
    # 2. Image plane by lens light
    im1 = ax[0, 1].imshow(np.log10(np.clip(model_lens_light, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    ax[0, 1].set_title('Lens Light')
    ax[0, 1].set_xlabel('arcsec')
    ax[0, 1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=ax[0, 1], label='log10')
    
    # 3. Image plane by point sources
    im2 = ax[0, 2].imshow(np.log10(np.clip(model_point_sources, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    ax[0, 2].set_title('Point Sources')
    ax[0, 2].set_xlabel('arcsec')
    ax[0, 2].set_ylabel('arcsec')
    plt.colorbar(im2, ax=ax[0, 2], label='log10')
    
    # 3. Composite
    im3 = ax[1, 0].imshow(np.log10(np.clip(model_composite, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            ax[1, 0].scatter(ras, decs, s=20, marker='x', color=color[i])
    ax[1, 0].set_title('Composite (Extended + Point Sources)')
    ax[1, 0].set_xlabel('arcsec')
    ax[1, 0].set_ylabel('arcsec')
    plt.colorbar(im3, ax=ax[1, 0], label='log10')
    
    im4 = ax[1, 1].imshow(np.log10(np.clip(image_data, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            ax[1, 1].scatter(ras, decs, s=20, marker='x', color=color[i])
    ax[1, 1].set_title('Image Data')
    ax[1, 1].set_xlabel('arcsec')
    ax[1, 1].set_ylabel('arcsec')
    plt.colorbar(im4, ax=ax[1, 1], label='log10')
    
    im5 = ax[1, 2].imshow(residuals, origin='lower', cmap='RdBu_r', extent=extent)
    ax[1, 2].set_title('Residuals')
    ax[1, 2].set_xlabel('arcsec')
    ax[1, 2].set_ylabel('arcsec')
    plt.colorbar(im5, ax=ax[1, 2])
    
    plt.tight_layout()
    plt.savefig(f'{save_path}/image_plane.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_source_plane(lens_image, kwargs_result, save_path,
                      source_pixel_scale=0.01, num_pixel=200, plot_caustics=True):
    """
    Plot the source plane: extended source surface brightness, point source positions,
    and optionally caustics.
    """
    is_pixelated = (
        'kwargs_source' in kwargs_result
        and len(kwargs_result['kwargs_source']) > 0
        and isinstance(kwargs_result['kwargs_source'][0], dict)
        and 'pixels' in kwargs_result['kwargs_source'][0]
    )

    if is_pixelated:
        # For pixelated source models, plot the reconstructed source on its native grid.
        source_galaxy = np.array(kwargs_result['kwargs_source'][0]['pixels'])
        extent = list(lens_image.SourceModel.pixel_grid.extent)
        source_for_plot = np.log10(np.clip(source_galaxy, 1e-8, None) + 1e-10)
        cbar_label = 'log10'
    else:
        fov = num_pixel * source_pixel_scale
        x = np.linspace(-fov / 2, fov / 2, num_pixel)
        y = np.linspace(-fov / 2, fov / 2, num_pixel)
        xx, yy = np.meshgrid(x, y)
        # Parametric source models: evaluate on a regular source-plane grid.
        source_galaxy = np.array(
            lens_image.SourceModel.surface_brightness(xx, yy, kwargs_result['kwargs_source'])
        )
        extent = [-fov / 2, fov / 2, -fov / 2, fov / 2]
        source_for_plot = np.log10(np.clip(source_galaxy, 1e-8, None) + 1e-10)
        cbar_label = 'log10'

    # Point source positions in source plane
    
    if 'kwargs_point_source' in kwargs_result:
    
        beta_x, beta_y = lens_image.PointSourceModel.get_source_plane_points(
            kwargs_result['kwargs_point_source'],
            kwargs_lens=kwargs_result['kwargs_lens'],
            with_amplitude=False,
        )
        ra_source_list = [np.atleast_1d(np.asarray(b)) for b in beta_x]
        dec_source_list = [np.atleast_1d(np.asarray(d)) for d in beta_y]

    # Caustics (optional)
    caustics = []
    if plot_caustics:
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'],
                supersampling=5,
            )
        except Exception as e:
            print(f"Could not compute caustics: {e}")

    colors = ['cyan', 'green', 'yellow', 'orange']

    fig, axes = plt.subplots(1, 3, figsize=(5 * 3 + 2, 5))

    # 1. Extended source only
    im0 = axes[0].imshow(source_for_plot, origin='lower', extent=extent, cmap='magma')
    axes[0].set_title('Extended Source')
    axes[0].set_xlabel('arcsec')
    axes[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=axes[0], label=cbar_label)

    # 2. Point sources + caustics
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_source_list, dec_source_list)):
            axes[1].scatter(ras, decs, s=30, marker='*', color=colors[i % len(colors)],
                            label=f'Point Source {i}')
    for caust_x, caust_y in caustics:
        axes[1].plot(caust_x, caust_y, color='lime', lw=1.0)
    if caustics:
        axes[1].plot([], [], color='lime', lw=1.0, label='Caustic')
    axes[1].axvline(0, color='red', lw=0.5, linestyle='--')
    axes[1].axhline(0, color='red', lw=0.5, linestyle='--')
    axes[1].set_title('Point Sources + Caustics')
    axes[1].set_xlabel('arcsec')
    axes[1].set_ylabel('arcsec')
    axes[1].legend()
    axes[1].set_xlim(extent[0], extent[1])
    axes[1].set_ylim(extent[2], extent[3])

    # 3. Combined: extended source + point sources + caustics
    im2 = axes[2].imshow(source_for_plot, origin='lower', extent=extent, cmap='magma')
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_source_list, dec_source_list)):
            axes[2].scatter(ras, decs, s=30, marker='*', color=colors[i % len(colors)])
    for caust_x, caust_y in caustics:
        axes[2].plot(caust_x, caust_y, color='lime', lw=1.0)
    axes[2].set_title('Source Plane Reconstruction')
    axes[2].set_xlabel('arcsec')
    axes[2].set_ylabel('arcsec')
    axes[2].set_xlim(extent[0], extent[1])
    axes[2].set_ylim(extent[2], extent[3])
    plt.colorbar(im2, ax=axes[2], label=cbar_label)

    plt.tight_layout()
    plt.savefig(f'{save_path}/source_plane.png', dpi=300, bbox_inches='tight')
    plt.close()

def plot_corner_traced_params(samples, save_path, max_samples=15_000):
    """
    Corner plot for HMC traces. Omits pixelated source (`source_pixels`) and any
    parameter with more than 32 columns when flattened.
    """

    exclude = {'source_pixels'}
    cols = []
    labels = []
    for name in sorted(samples.keys()):
        if name in exclude or name.startswith('ps_'):
            continue
        arr = np.asarray(samples[name])
        if arr.ndim == 1:
            cols.append(arr)
            labels.append(name)
        elif arr.ndim == 2 and arr.shape[1] <= 32:
            for j in range(arr.shape[1]):
                cols.append(arr[:, j])
                labels.append(f'{name}[{j}]')
        else:
            continue

    if len(cols) < 2:
        print(
            f"Corner plot skipped: need at least 2 traced scalars (got {len(cols)})."
        )
        return

    data = np.column_stack(cols)
    n = data.shape[0]
    if n > max_samples:
        rng = np.random.default_rng(42)
        data = data[rng.choice(n, size=max_samples, replace=False)]

    fig = corner.corner(
        data,
        labels=labels,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
    )
    out = f'{save_path}/corner_traced_params.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved corner plot to {out}")

def plot_loss_curve(loss_curve, save_path):
    n_total = len(loss_curve)
    if n_total == 0:
        return

    loss_curve = np.asarray(loss_curve)
    x_all = np.arange(1, n_total + 1)
    begin_idx = int(n_total * 0.8)
    begin_idx = min(begin_idx, n_total - 1)
    x_tail = np.arange(begin_idx + 1, n_total + 1)
    y_tail = loss_curve[begin_idx:]

    fig, (ax_full, ax_tail) = plt.subplots(2, 1, figsize=(10, 8))

    ax_full.plot(x_all, loss_curve, color='tab:blue')
    ax_full.set_xlabel('Iteration')
    ax_full.set_ylabel('Loss')
    ax_full.set_title('Loss Curve (Full)')
    ax_full.grid(True, alpha=0.3)

    ax_tail.plot(x_tail, y_tail, color='tab:red')
    ax_tail.set_xlabel('Iteration')
    ax_tail.set_ylabel('Loss')
    ax_tail.set_title('Loss Curve (Last 20%)')
    ax_tail.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{save_path}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_ps_photometry(ps_sources, first_images_idx, second_images_idx, save_path):

    bands = "F090W,F115W,F150W,F200W,F277W,F356W,F444W"
    bands = [b.strip() for b in bands.split(",") if b.strip()]

    for idx in first_images_idx:
        line = ps_sources.iloc[idx]
        fluxes = []
        for b in bands:
            fluxes.append(line[f"flux_{b}"])
        fluxes = np.array(fluxes, dtype=float)

        plt.plot(bands, fluxes, c='red', marker='o', label='PS 1')

    for idx in second_images_idx:
        line = ps_sources.iloc[idx]
        fluxes = []
        for b in bands:
            fluxes.append(line[f"flux_{b}"])
        fluxes = np.array(fluxes, dtype=float)

        plt.plot(bands, fluxes, c='blue', marker='x', label='PS 2')

    plt.legend()
    plt.ylabel('Flux')
    plt.xlabel('Band')
    plt.savefig(f'{save_path}/ps_photometry.png', dpi=300, bbox_inches='tight')
    plt.close()

def post_process(prob_model, lens_model, samples,
                 image_data, noise_map, pixel_scale,
                 num_params, save_path, ll_batch_size=1024):
    
    with open(f'{save_path}/samples.pkl', 'wb') as f:
        joblib.dump(samples, f)
        
    param_names = list(samples.keys())
    n_total = samples[param_names[0]].shape[0]
    log_l_vals = batch_log_likelihood(prob_model, samples, chunk_size=ll_batch_size)
    n_invalid = int(np.sum(log_l_vals == -np.inf))
    if n_invalid > 0:
        print(f"  Warning: {n_invalid} sample(s) have log_likelihood = -inf (non-finite or invalid).")

    best_idx = int(np.argmax(log_l_vals))
    best_log_likelihood = log_l_vals[best_idx]
    
    best_sample = {name: jnp.asarray(samples[name][best_idx]) for name in param_names}
    kwargs_best = prob_model.params2kwargs(best_sample)
    
    with open(f'{save_path}/kwargs_result.json', 'w') as f:
        json.dump(kwargs_best, f, indent=4, default=json_serializer)
    
    total_pixels = image_data.size
    bic = num_params * np.log(total_pixels) - 2 * best_log_likelihood
    
    print(f'BIC: {bic:.2f}')
    print(f'Best log likelihood: {best_log_likelihood:.2f}')
    
    best_fit_model = lens_model.model(**kwargs_best)
    chi2_best = np.sum(((best_fit_model - image_data) / noise_map) ** 2)
    print(f'Chi^2 of best fit model: {chi2_best:.2f}')
    
    display(
        [best_fit_model, image_data, (best_fit_model - image_data) / noise_map],
        titles=['Best fit model', 'image data', f'Residuals (chi2 = {chi2_best:.2f})'], 
        pixel_scale=pixel_scale,
        savefilename=f'{save_path}/best_fit_model.png'
    )
    
    plot_image_plane(lens_model, kwargs_best, pixel_scale, image_data, noise_map, save_path)
    plot_source_plane(lens_model, kwargs_best, save_path)
    
    return kwargs_best

def plot_corner_nautilus(
    points,
    log_w,
    param_names,
    save_path,
    max_samples=15_000,
    random_seed=42,
    max_dims_corner=32,
):
    """
    Corner plot of the Nautilus posterior from weighted samples.
    Resamples with replacement using normalized importance weights.
    """
    points = np.asarray(points, dtype=np.float64)
    log_w = np.asarray(log_w, dtype=np.float64)
    if points.ndim != 2:
        print("[Nautilus corner] Expected points 2D array (n, ndim).")
        return
    n, ndim = points.shape
    if n < 2:
        print("[Nautilus corner] Need at least 2 posterior samples.")
        return
    if len(param_names) != ndim:
        param_names = [f'param_{i}' for i in range(ndim)]

    log_w_max = float(np.max(log_w))
    w = np.exp(log_w - log_w_max)
    w_sum = float(np.sum(w))
    if w_sum <= 0.0 or not np.isfinite(w_sum):
        p = np.full(n, 1.0 / n)
    else:
        p = w / w_sum

    n_draw = min(int(max_samples), n)
    rng = np.random.default_rng(random_seed)
    idx = rng.choice(n, size=n_draw, p=p, replace=True)
    samples = points[idx]

    if ndim > max_dims_corner:
        print(
            f"[Nautilus corner] ndim={ndim} > {max_dims_corner}; plotting first {max_dims_corner} parameters."
        )
        samples = samples[:, :max_dims_corner]
        labels = list(param_names[:max_dims_corner])
    else:
        labels = list(param_names)

    if samples.shape[1] < 2:
        print("[Nautilus corner] Need at least 2 parameters for a corner plot.")
        return

    fig = corner.corner(
        samples,
        labels=labels,
        show_titles=True,
        title_fmt='.3f',
        quantiles=[0.16, 0.5, 0.84],
    )
    out = f'{save_path}/corner_nautilus.png'
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Saved Nautilus corner plot to {out}")

def display_init(
    prob_model, 
    init_params_path=None,
    use_nnls=False, 
    lens_image=None,
    image_data=None,
    noise_map=None,
    pixel_scale=None,
    param_list=None,
    type_list=None,
    jax_n_iter=200,
    save_path=None,
    random_seed=42,
):

    if init_params_path is not None:

        init_params = get_init_params(
            prob_model, param_list, type_list, init_params_path=init_params_path, 
            save_path=save_path, random_seed=random_seed, use_nnls=use_nnls
        )

    else:

        init_params = get_init_params(
            prob_model, param_list, type_list, init_params_path=None, 
            save_path=None, random_seed=random_seed, use_nnls=use_nnls
        )

    kwargs_init = prob_model.params2kwargs(init_params)
    if use_nnls:
        kwargs_init, _, _ = solve_linear_amplitudes_jax(
            lens_image=lens_image,
            kw_model=kwargs_init,
            image_data=image_data,
            noise_map=noise_map,
            param_list=param_list,
            type_list=type_list,
            jax_n_iter=jax_n_iter,
        )

    initial_model = lens_image.model(**kwargs_init)
    init_chi2 = np.sum(((initial_model - image_data) / noise_map) ** 2)
    print(f'Initial chi^2: {init_chi2:.2f}')
    
    print('Displaying initial guess model...')

    display(
        [initial_model, image_data, (initial_model - image_data) / noise_map],
        titles=['Initial guess model', 'image data', f'Residuals (chi2 = {init_chi2:.2f})'],
        pixel_scale=pixel_scale,
        savefilename=f'{save_path}/initial_guess_model.png'
    )

    if init_params_path is not None:
        return init_params
    else:
        return None