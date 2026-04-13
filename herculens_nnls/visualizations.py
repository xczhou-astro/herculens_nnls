import matplotlib.pyplot as plt
import numpy as np
from herculens.Util import model_util
import corner

from .utils import json_serializer


def _point_source_colors(n: int):
    """Distinct colors for overlays; cycles tab10 for large n."""
    n = int(max(n, 1))
    cmap = plt.get_cmap("tab10")
    return [cmap(i % 10) for i in range(n)]
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
        n_ps = sum(1 for t in point_source_type_list if t == "IMAGE_POSITIONS")
        colors = _point_source_colors(n_ps)
        k = 0
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
                    edgecolors=colors[k],
                    linewidths=1.5,
                    label=f'PS {k + 1}',
                )
                k += 1
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

    n_ps = len(ra_image_list) if "kwargs_point_source" in kwargs_result else 0
    ps_colors = _point_source_colors(n_ps) if n_ps else []

    fig, ax = plt.subplots(2, 3, figsize=(3 * 5 + 2, 5 * 2))
    
    # 1. Image plane by extended source
    im0 = ax[0, 0].imshow(np.log10(np.clip(model_extended, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            ax[0, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    
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
            ax[1, 0].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
    ax[1, 0].set_title('Composite (Extended + Point Sources)')
    ax[1, 0].set_xlabel('arcsec')
    ax[1, 0].set_ylabel('arcsec')
    plt.colorbar(im3, ax=ax[1, 0], label='log10')
    
    im4 = ax[1, 1].imshow(np.log10(np.clip(image_data, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    if 'kwargs_point_source' in kwargs_result:
        for i, (ras, decs) in enumerate(zip(ra_image_list, dec_image_list)):
            ax[1, 1].scatter(ras, decs, s=20, marker='x', color=ps_colors[i])
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
    ra_source_list = []
    dec_source_list = []
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

    n_ps_src = len(ra_source_list)
    colors = _point_source_colors(n_ps_src) if n_ps_src else []

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
            axes[1].scatter(ras, decs, s=30, marker='*', color=colors[i],
                            label=f'Point Source {i + 1}')
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
            axes[2].scatter(ras, decs, s=30, marker='*', color=colors[i])
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


def plot_catalog_source_trace(
    lens_image,
    kwargs_result,
    image_data,
    pixel_scale,
    save_path,
    catalog_path='photometry/source_catalog_multiband.csv',
    start_row=1,
    id_column='id',
    plot_caustics=True,
):
    """
    Plot catalog image-plane positions and their ray-traced source-plane positions
    (style similar to ps_test_optax/catalog_source_trace.png).
    """
    cat = np.genfromtxt(catalog_path, delimiter=',', names=True, dtype=None, encoding='utf-8')
    cat = np.atleast_1d(cat)
    if cat.size == 0:
        print(f"[plot_catalog_source_trace] Empty catalog: {catalog_path}")
        return
    if 'ra_arcsec' not in cat.dtype.names or 'dec_arcsec' not in cat.dtype.names:
        print(
            f"[plot_catalog_source_trace] Missing columns ra_arcsec/dec_arcsec in {catalog_path}; "
            "skip plotting."
        )
        return
    if start_row < 0 or start_row >= len(cat):
        print(
            f"[plot_catalog_source_trace] start_row={start_row} out of range for "
            f"{len(cat)} rows; skip plotting."
        )
        return

    cat = cat[start_row:]
    theta_ra = np.asarray(cat['ra_arcsec'], dtype=float)
    theta_dec = np.asarray(cat['dec_arcsec'], dtype=float)
    beta_x, beta_y = lens_image.MassModel.ray_shooting(theta_ra, theta_dec, kwargs_result['kwargs_lens'])
    beta_x = np.asarray(beta_x, dtype=float).ravel()
    beta_y = np.asarray(beta_y, dtype=float).ravel()

    if id_column in cat.dtype.names:
        ids = [int(v) for v in np.asarray(cat[id_column]).ravel()]
    else:
        ids = [int(start_row + i) for i in range(len(theta_ra))]

    def _circled_label(n: int) -> str:
        if n == 0:
            return "\u24EA"  # ⓪
        if 1 <= n <= 20:
            return chr(0x245F + n)  # ①..⑳
        if 21 <= n <= 35:
            return chr(0x3250 + (n - 20))  # ㉑..㉟
        if 36 <= n <= 50:
            return chr(0x32B1 + (n - 36))  # ㊱..㊿
        return f"({n})"

    labels = [_circled_label(v) for v in ids]

    caustics = []
    if plot_caustics:
        try:
            _, caustics = model_util.critical_lines_caustics(
                lens_image, kwargs_result['kwargs_lens'],
                supersampling=5,
            )
        except Exception as e:
            print(f"[plot_catalog_source_trace] Could not compute caustics: {e}")

    ny, nx = image_data.shape
    x_center = nx // 2
    y_center = ny // 2
    extent = [
        -x_center * pixel_scale, (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale, (ny - y_center - 1) * pixel_scale,
    ]

    fig, axes = plt.subplots(1, 3, figsize=(20, 6.5))

    ax_img = axes[0]
    vmin, vmax = np.percentile(image_data, [5, 99.5])
    ax_img.imshow(
        image_data,
        origin='lower',
        cmap='magma',
        vmin=vmin,
        vmax=vmax,
        interpolation='nearest',
        extent=extent,
    )
    for x, y, lab in zip(theta_ra, theta_dec, labels):
        ax_img.text(
            x, y, lab,
            fontsize=14,
            color='cyan',
            fontweight='bold',
            ha='center',
            va='center',
            zorder=5,
        )
    ax_img.set_xlabel('arcsec')
    ax_img.set_ylabel('arcsec')
    ax_img.set_title('Image plane (catalog positions)')

    ax_src = axes[1]
    for x, y, lab in zip(beta_x, beta_y, labels):
        ax_src.text(
            x, y, lab,
            fontsize=14,
            color='magenta',
            fontweight='bold',
            ha='center',
            va='center',
            zorder=5,
        )
    for i_c, (caust_x, caust_y) in enumerate(caustics):
        lbl = 'Caustic' if i_c == 0 else None
        ax_src.plot(caust_x, caust_y, color='lime', lw=1.2, alpha=0.9, label=lbl)
    ax_src.axvline(0.0, color='red', lw=0.7, linestyle='--', alpha=0.7)
    ax_src.axhline(0.0, color='red', lw=0.7, linestyle='--', alpha=0.7)
    ax_src.set_xlabel('source x (arcsec)')
    ax_src.set_ylabel('source y (arcsec)')
    ax_src.set_title('Source plane (ray-traced)')
    ax_src.set_aspect('equal', adjustable='box')
    ax_src.grid(alpha=0.2)
    if caustics:
        ax_src.legend(loc='best', fontsize=8)

    xs = list(beta_x.ravel())
    ys = list(beta_y.ravel())
    for caust_x, caust_y in caustics:
        xs.extend(np.asarray(caust_x, dtype=float).ravel().tolist())
        ys.extend(np.asarray(caust_y, dtype=float).ravel().tolist())
    if len(xs) > 0:
        x_arr = np.asarray(xs, dtype=float)
        y_arr = np.asarray(ys, dtype=float)
        x_min, x_max = np.nanmin(x_arr), np.nanmax(x_arr)
        y_min, y_max = np.nanmin(y_arr), np.nanmax(y_arr)
        dx = max(x_max - x_min, 0.05)
        dy = max(y_max - y_min, 0.05)
        pad = 0.15 * max(dx, dy)
        ax_src.set_xlim(x_min - pad, x_max + pad)
        ax_src.set_ylim(y_min - pad, y_max + pad)

    # Third panel: same angular frame as the image — θ vs β only (no image underlay).
    ax_mix = axes[2]
    for tx, ty, bx, by in zip(theta_ra, theta_dec, beta_x, beta_y):
        ax_mix.plot(
            [tx, bx], [ty, by],
            color='0.35',
            lw=1.0,
            alpha=0.85,
            zorder=4,
        )
    for x, y, lab in zip(theta_ra, theta_dec, labels):
        ax_mix.text(
            x, y, lab,
            fontsize=14,
            color='cyan',
            fontweight='bold',
            ha='center',
            va='center',
            zorder=6,
        )
    for x, y, lab in zip(beta_x, beta_y, labels):
        ax_mix.text(
            x, y, lab,
            fontsize=14,
            color='magenta',
            fontweight='bold',
            ha='center',
            va='center',
            zorder=6,
        )
    ax_mix.set_xlabel('arcsec')
    ax_mix.set_ylabel('arcsec')
    ax_mix.set_title('θ (cyan) vs β (magenta); tie = deflection')
    ax_mix.set_xlim(extent[0], extent[1])
    ax_mix.set_ylim(extent[2], extent[3])
    ax_mix.set_aspect('equal', adjustable='box')
    ax_mix.grid(alpha=0.25)

    h_img = plt.Line2D([], [], color='cyan', marker='o', linestyle='None', markersize=0)
    h_src = plt.Line2D([], [], color='magenta', marker='o', linestyle='None', markersize=0)
    h_line = plt.Line2D([], [], color='0.35', lw=1.0, alpha=0.85)
    ax_mix.legend(
        [h_img, h_src, h_line],
        ['Catalog θ (image plane)', 'Ray-traced β (source plane)', 'θ–β'],
        loc='best',
        fontsize=8,
    )

    fig.tight_layout()
    fig.savefig(f'{save_path}/catalog_source_trace.png', dpi=200, bbox_inches='tight')
    plt.close(fig)

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

def plot_loss_curve(loss_curve, save_path, lr_curve=None):
    n_total = len(loss_curve)
    if n_total == 0:
        return

    loss_curve = np.asarray(loss_curve)
    if loss_curve.ndim == 1:
        loss_curve_2d = loss_curve[:, None]
    else:
        loss_curve_2d = loss_curve

    x_all = np.arange(1, n_total + 1)
    lr_arr = None
    if lr_curve is not None:
        lr_arr = np.asarray(lr_curve).reshape(-1)
        if lr_arr.shape[0] != n_total:
            lr_arr = None
    # Lower panel: last 20% of iterations with mean over 100-step windows.
    begin_idx = int(n_total * 0.8)
    begin_idx = min(begin_idx, n_total - 1)
    tail = loss_curve_2d[begin_idx:]
    window = 100
    n_tail = tail.shape[0]
    n_bins = n_tail // window

    if n_bins >= 1:
        trimmed_tail = tail[: n_bins * window]
        y_tail = trimmed_tail.reshape(n_bins, window, tail.shape[1]).mean(axis=1)
        x_tail = (begin_idx + 1) + (np.arange(n_bins) * window) + (window / 2.0)
    else:
        # Fallback for very short tails.
        x_tail = np.arange(begin_idx + 1, n_total + 1)
        y_tail = tail

    fig, (ax_full, ax_tail) = plt.subplots(2, 1, figsize=(10, 8))

    loss_lines = ax_full.plot(x_all, loss_curve_2d, color='tab:blue')
    if loss_lines:
        loss_lines[0].set_label('Loss')
        for ln in loss_lines[1:]:
            ln.set_label('_nolegend_')
    ax_full.set_xlabel('Iteration')
    ax_full.set_ylabel('Loss')
    ax_full.set_title('Loss Curve (Full)')
    ax_full.grid(True, alpha=0.3)
    if lr_arr is not None:
        ax_lr = ax_full.twinx()
        (lr_line,) = ax_lr.plot(
            x_all, lr_arr, color='tab:orange', alpha=0.85, linewidth=1.5, label='Step size'
        )
        ax_lr.set_ylabel('Step size (LR)')
        ax_lr.set_yscale('log')
        legend_handles = [loss_lines[0], lr_line] if loss_lines else [lr_line]
        legend_labels = ['Loss', 'Step size'] if loss_lines else ['Step size']
        ax_full.legend(legend_handles, legend_labels, loc='upper right', framealpha=0.9)

    ax_tail.plot(x_tail, y_tail, color='tab:red')
    ax_tail.set_xlabel('Iteration')
    ax_tail.set_ylabel('Loss')
    ax_tail.set_title(f'Loss Curve (Last 20%, mean over {window} steps)')
    ax_tail.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(f'{save_path}/loss_curve.png', dpi=150, bbox_inches='tight')
    plt.close(fig)

def plot_ps_photometry(ps_sources, images_idx_groups, save_path):
    """
    Plot per-band flux for catalog rows grouped by point source.

    Parameters
    ----------
    images_idx_groups
        Sequence of length ``num_sources``; each entry is an iterable of catalog row indices
        (same convention as ``--images_idx_K``).
    """
    plt.figure(figsize=(9, 5.5))
    bands = "F090W,F115W,F150W,F200W,F277W,F356W,F444W"
    bands = [b.strip() for b in bands.split(",") if b.strip()]

    groups = [np.atleast_1d(np.asarray(g, dtype=int)).ravel() for g in images_idx_groups]
    n_groups = len(groups)
    cmap = _point_source_colors(max(n_groups, 1))
    markers = ["o", "s", "^", "D", "v", "<", ">", "p", "*", "h", "P", "X"]

    for k, group in enumerate(groups):
        c = cmap[k]
        m = markers[k % len(markers)]
        for j, idx in enumerate(group):
            line = ps_sources.iloc[int(idx)]
            fluxes = np.array([line[f"flux_{b}"] for b in bands], dtype=float)
            lbl = f"PS {k + 1}" if j == 0 else "_nolegend_"
            plt.plot(bands, fluxes, c=c, marker=m, label=lbl)

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

def plot_lens_light_subtracted_image(
    lens_image,
    kwargs_result,
    pixel_scale,
    image_data,
    noise_map=None,
    save_path=None,
):
    """
    Display/save the data with the best-fit lens light subtracted.
    """
    ny, nx = image_data.shape

    model_lens_light = np.zeros((ny, nx))
    if 'kwargs_lens_light' in kwargs_result:
        model_lens_light = lens_image.model(
            **kwargs_result,
            lens_light_add=True,
            source_add=False,
            point_source_add=False,
        )

    subtracted = image_data - model_lens_light

    ny, nx = image_data.shape
    x_center = nx // 2
    y_center = ny // 2
    extent = [
        -x_center * pixel_scale,
        (nx - x_center - 1) * pixel_scale,
        -y_center * pixel_scale,
        (ny - y_center - 1) * pixel_scale,
    ]

    fig, ax = plt.subplots(1, 3, figsize=(16, 5))

    im0 = ax[0].imshow(np.log10(np.clip(image_data, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    ax[0].set_title('Image data')
    ax[0].set_xlabel('arcsec')
    ax[0].set_ylabel('arcsec')
    plt.colorbar(im0, ax=ax[0], label='log10')

    im1 = ax[1].imshow(np.log10(np.clip(model_lens_light, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
    ax[1].set_title('Lens light model')
    ax[1].set_xlabel('arcsec')
    ax[1].set_ylabel('arcsec')
    plt.colorbar(im1, ax=ax[1], label='log10')

    if noise_map is not None:
        sub_for_plot = subtracted / noise_map
        im2 = ax[2].imshow(sub_for_plot, origin='lower', cmap='RdBu_r', extent=extent)
        ax[2].set_title('Data - Lens light (S/N)')
        plt.colorbar(im2, ax=ax[2])
    else:
        im2 = ax[2].imshow(np.log10(np.clip(subtracted, 1e-8, None)), origin='lower', cmap='magma', extent=extent)
        ax[2].set_title('Data - Lens light')
        plt.colorbar(im2, ax=ax[2], label='log10')
    ax[2].set_xlabel('arcsec')
    ax[2].set_ylabel('arcsec')

    plt.tight_layout()
    fig.savefig(f'{save_path}/lens_light_subtracted_image.png', dpi=300, bbox_inches='tight')
    plt.close(fig)