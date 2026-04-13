import numpyro
from numpyro.distributions import constraints
import numpyro.distributions as dist
import numpy as np
import jax.numpy as jnp
from herculens.Inference.ProbModel.numpyro import NumpyroModel
from copy import deepcopy
import jax
import json
import shutil

from herculens.Coordinates.pixel_grid import PixelGrid
from herculens.LensImage.lens_image import LensImage
from herculens.Instrument.psf import PSF
from herculens.Instrument.noise import Noise
from herculens.MassModel.mass_model import MassModel
from herculens.LightModel.light_model import LightModel
from herculens.PointSourceModel.point_source_model import PointSourceModel
from herculens.RegulModel.regul_model import RegularizationModel
from herculens.Analysis.plot import Plotter


def _is_correlated_param(param):
    """
    Flexible correlation syntax:
      ['correlated', component, index, param_name]
    """
    return isinstance(param, (tuple, list)) and len(param) == 4 and param[0] == 'correlated'

def _normalize_link_spec(param):
    """
    Convert a correlation spec into canonical (component, index, key).
    Component names are normalized to lenstronomy constraint groups.
    Returns None if param is not a correlation spec.
    """
    if _is_correlated_param(param):
        _, comp, idx, key = param
        comp_map = {
            'lens': 'lens',
            'lens_mass': 'lens',
            'mass': 'lens',
            'lens_light': 'lens_light',
            'source': 'source',
            'source_light': 'source',
            'point_source': 'point_source',
            'ps': 'point_source',
        }
        if comp not in comp_map:
            raise ValueError(f"Unknown correlated component '{comp}'. Expected one of {sorted(comp_map.keys())}.")
        return (comp_map[comp], int(idx), str(key))

    return None

def _resolve_link(bank, spec, *, context=""):
    """Resolve (component, index, key) against already-built component dicts."""
    comp, idx, key = spec
    if comp not in bank:
        raise ValueError(f"Cannot resolve linked param {spec} ({context}): component '{comp}' not available yet.")
    arr = bank[comp]
    if idx < 0 or idx >= len(arr):
        raise IndexError(
            f"Cannot resolve linked param {spec} ({context}): index {idx} out of range for component '{comp}' "
            f"(len={len(arr)})."
        )
    if key not in arr[idx]:
        raise KeyError(
            f"Cannot resolve linked param {spec} ({context}): key '{key}' not found in {comp}[{idx}]. "
            f"Available keys: {sorted(arr[idx].keys())}"
        )
    return arr[idx][key]

def create_prob_model(
    param_list,
    type_list,
    lens_image,
    image_data,
    noise_map,
    regul_model=None,
    nnls_linear_amps=False,
):

    noise = Noise(nx=image_data.shape[0], ny=image_data.shape[0], noise_map=noise_map)
    
    class ProbModel(NumpyroModel):
        
        def model(self):
            
            prior_lens_mass = []
            bank = {'lens': prior_lens_mass}
            for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
                model = {}
                for key, param in lens_mass_model.items():
                    link_spec = _normalize_link_spec(param)
                    if link_spec is not None:
                        model[key] = _resolve_link(bank, link_spec, context=f"lens_mass[{i}].{key}")
                    elif isinstance(param, list):
                        if key == 'amp' and nnls_linear_amps:
                            model[key] = 1.0
                        elif key == 'amp':
                            model[key] = numpyro.sample(
                                f'lens_{key}_{i}', 
                                dist.LogNormal(param[0], param[1])
                            )
                        else:
                            model[key] = numpyro.sample(
                                f'lens_{key}_{i}', 
                                dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3])
                            )
                    else:
                        model[key] = param
                
                prior_lens_mass.append(model)
            
            prior_lens_light = []
            if 'lens_light_params_list' in param_list:
                bank['lens_light'] = prior_lens_light
                
                for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
                    model = {}
                    for key, param in lens_light_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            model[key] = _resolve_link(bank, link_spec, context=f"lens_light[{i}].{key}")
                        elif isinstance(param, list):
                            if key == 'amp' and nnls_linear_amps:
                                model[key] = 1.0
                            elif key == 'amp':
                                model[key] = numpyro.sample(
                                    f'lens_light_{key}_{i}', 
                                    dist.LogNormal(param[0], param[1])
                                )
                            else:
                                model[key] = numpyro.sample(
                                    f'lens_light_{key}_{i}', 
                                    dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3])
                                )
                        else:
                            model[key] = param
                            
                    prior_lens_light.append(model)
            
            
            if type_list['source_light_type_list'] == ['PIXELATED']:
                # IMPORTANT: use the actual source pixel grid shape from lens_image.
                # This keeps source_pixels shape consistent with regularization
                # transforms/weights initialized from lens_image.SourceModel.
                ny, nx = lens_image.SourceModel.pixel_grid.num_pixel_axes
                source_pixels = numpyro.param(
                    'source_pixels', 
                    init_value=np.zeros((ny, nx)) + 1e-2,
                    event_dim=(ny, nx),
                    constraint=constraints.greater_than(0.0)
                )
                prior_source_light = [{'pixels': source_pixels}]
            else:
                prior_source_light = []
                bank['source'] = prior_source_light
                for i, source_light_model in enumerate(param_list['source_light_params_list']):
                    model = {}
                    for key, param in source_light_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            model[key] = _resolve_link(bank, link_spec, context=f"source_light[{i}].{key}")
                        elif isinstance(param, list):
                            if key == 'amp' and nnls_linear_amps:
                                model[key] = 1.0
                            elif key == 'amp':
                                model[key] = numpyro.sample(
                                    f'source_{key}_{i}', 
                                    dist.LogNormal(param[0], param[1])
                                )
                            else:
                                model[key] = numpyro.sample(
                                    f'source_{key}_{i}', 
                                    dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3])
                                )
                        else:
                            model[key] = param
                    
                    prior_source_light.append(model)
            
            prior_point_source = []
            if 'point_source_params_list' in param_list:
                bank['point_source'] = prior_point_source
                for i, point_source_model in enumerate(param_list['point_source_params_list']):
                    ps_type = type_list.get('point_source_type_list', [None] * len(param_list['point_source_params_list']))[i]
                    n_img = int(point_source_model.get('n_images', 4)) if isinstance(point_source_model, dict) else 4
                    sigma_image = float(point_source_model.get('sigma_image', 3e-3)) if isinstance(point_source_model, dict) else 3e-3
                    model = {}
                    for key, param in point_source_model.items():
                        # skip helper keys
                        if key in ('n_images', 'sigma_image', 'sigma_source'):
                            continue
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            model[key] = _resolve_link(bank, link_spec, context=f"point_source[{i}].{key}")
                        elif ps_type == 'IMAGE_POSITIONS' and key in ('ra', 'dec'):
                            # IMAGE_POSITIONS: ra/dec are image-plane coordinates (vector length n_img)
                            if isinstance(param, (list, tuple, np.ndarray)) and len(param) == n_img and all(
                                isinstance(v, (int, float, np.floating)) for v in param
                            ):
                                loc = jnp.asarray(param)
                                # Allow local freedom, but keep each image position within +/-0.1 arcsec.
                                pos_bound = float(point_source_model.get('pos_bound', 0.1)) if isinstance(point_source_model, dict) else 0.1
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.TruncatedNormal(
                                        loc=loc,
                                        scale=sigma_image,
                                        low=loc - pos_bound,
                                        high=loc + pos_bound,
                                    ).to_event(1),
                                )
                            else:
                                raise ValueError(
                                    f"For IMAGE_POSITIONS, point_source[{i}].{key} must be a length-{n_img} "
                                    f"list/array of observed image positions."
                                )
                        elif ps_type == 'IMAGE_POSITIONS' and key == 'amp':
                            # IMAGE_POSITIONS: allow one amplitude per image
                            if nnls_linear_amps:
                                model[key] = jnp.ones((n_img,))
                            elif isinstance(param, (list, tuple, np.ndarray)) and len(param) == n_img and all(
                                isinstance(v, (int, float, np.floating)) for v in param
                            ):
                                model[key] = jnp.asarray(param)
                            elif isinstance(param, list) and len(param) == 2:
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.LogNormal(param[0], param[1]).expand((n_img,)).to_event(1),
                                )
                            elif isinstance(param, (int, float, np.floating)):
                                model[key] = jnp.ones((n_img,)) * float(param)
                            else:
                                raise ValueError(
                                    f"For IMAGE_POSITIONS, point_source[{i}].amp must be a length-{n_img} "
                                    f"list/array, a scalar, or a LogNormal prior [mu, sigma]."
                                )
                        elif isinstance(param, list):
                            # SOURCE_POSITION (or other scalar PS params)
                            if key == 'amp' and nnls_linear_amps:
                                model[key] = 1.0
                            elif key == 'amp':
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.LogNormal(param[0], param[1])
                                )
                            else:
                                model[key] = numpyro.sample(
                                    f'ps_{key}_{i}',
                                    dist.TruncatedNormal(param[0], param[1], low=param[2], high=param[3])
                                )
                        else:
                            model[key] = param
                    prior_point_source.append(model)
                
            
            model_params = dict(
                kwargs_lens=prior_lens_mass,
                kwargs_source=prior_source_light,
                # kwargs_point_source=point_source_list,
            )
            
            if len(prior_lens_light) > 0:
                model_params['kwargs_lens_light'] = prior_lens_light
                
            if len(prior_point_source) > 0:
                model_params['kwargs_point_source'] = prior_point_source
            
            # Linear amplitudes (nnls_linear_amps): use unit placeholders here; NNLS is solved in
            # run_optax via solve_linear_amplitudes + chi2_loss_fixed_amps.
            model_image = lens_image.model(**model_params)
            model_var = noise.C_D_model(model_image)
            model_std = jnp.sqrt(model_var)
            # Ensure observed data is a JAX array for consistent device/dtype in NumPyro
            obs = jnp.asarray(image_data)
            numpyro.sample('obs', dist.Independent(dist.Normal(model_image, model_std), 2), obs=obs)
            
            hyperparams = [
                {'lambda_0': 5.0, 'lambda_1': 3.0},  # SPARSITY_STARLET
                {'lambda_0': 5.0},  # SPARSITY_BLWAVELET
                # {'strength': 3.0},  # POSITIVITY — must match one entry per regularization_terms item
            ]

            if regul_model is not None:

                # numpyro.factor(
                #     'source_regul', 
                #     regul_model.log_prob(model_params, hyperparams)
                # )
                # Option A: apply regularization only after regul_model.initialize(...)
                # has set method.transform / method.weights.
                regul_ready = True
                for method in regul_model.method_list:
                    if hasattr(method, 'transform') and method.transform is None:
                        regul_ready = False
                        break
                if regul_ready:
                    numpyro.factor(
                        'source_regul',
                        regul_model.log_prob(model_params, hyperparams)
                    )

            # If using IMAGE_POSITIONS point sources, enforce that images map back to a single source.
            if 'point_source_type_list' in type_list and 'IMAGE_POSITIONS' in type_list['point_source_type_list']:
                # Use a mild default; can be overridden per point source dict via 'sigma_source'
                sigma_source = 1e-3
                try:
                    # If any point source provides sigma_source, take the first one
                    for ps in param_list.get('point_source_params_list', []):
                        if isinstance(ps, dict) and 'sigma_source' in ps:
                            sigma_source = float(ps['sigma_source'])
                            break
                except Exception:
                    pass
                numpyro.factor(
                    'ps_source_plane_penalty',
                    lens_image.PointSourceModel.log_prob_source_plane(
                        model_params,
                        sigma_source=sigma_source,
                    )
                )
            
            
        def params2kwargs(self, params):
            
            kwargs_lens = []
            bank = {'lens': kwargs_lens}
            for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
                kw = {}
                for key, param in lens_mass_model.items():
                    link_spec = _normalize_link_spec(param)
                    if link_spec is not None:
                        kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs lens_mass[{i}].{key}")
                    elif isinstance(param, list):
                        if key == 'amp' and nnls_linear_amps:
                            kw[key] = 1.0
                        else:
                            kw[key] = params[f'lens_{key}_{i}']
                    else:
                        kw[key] = param
                kwargs_lens.append(kw)
                
            kwargs_lens_light = []
            if 'lens_light_params_list' in param_list:
                bank['lens_light'] = kwargs_lens_light
                for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
                    kw = {}
                    for key, param in lens_light_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs lens_light[{i}].{key}")
                        elif isinstance(param, list):
                            if key == 'amp' and nnls_linear_amps:
                                kw[key] = 1.0
                            else:
                                kw[key] = params[f'lens_light_{key}_{i}']
                        else:
                            kw[key] = param
                    kwargs_lens_light.append(kw)
                
            kwargs_source = []
            if type_list['source_light_type_list'] == ['PIXELATED']:
                kwargs_source = [{'pixels': params['source_pixels']}]
            else:
                bank['source'] = kwargs_source
                for i, source_light_model in enumerate(param_list['source_light_params_list']):
                    kw = {}
                    for key, param in source_light_model.items():
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs source_light[{i}].{key}")
                        elif isinstance(param, list):
                            if key == 'amp' and nnls_linear_amps:
                                kw[key] = 1.0
                            else:
                                kw[key] = params[f'source_{key}_{i}']
                        else:
                            kw[key] = param
                    kwargs_source.append(kw)
            
            kwargs_point_source = []
            if 'point_source_params_list' in param_list:
                bank['point_source'] = kwargs_point_source
                for i, point_source_model in enumerate(param_list['point_source_params_list']):
                    ps_type = type_list.get('point_source_type_list', [None] * len(param_list['point_source_params_list']))[i]
                    kw = {}
                    for key, param in point_source_model.items():
                        if key in ('n_images', 'sigma_image', 'sigma_source'):
                            continue
                        link_spec = _normalize_link_spec(param)
                        if link_spec is not None:
                            kw[key] = _resolve_link(bank, link_spec, context=f"params2kwargs point_source[{i}].{key}")
                        elif ps_type == 'IMAGE_POSITIONS' and key in ('ra', 'dec'):
                            kw[key] = params[f'ps_{key}_{i}']
                        elif ps_type == 'IMAGE_POSITIONS' and key == 'amp':
                            if nnls_linear_amps:
                                n_img = int(point_source_model.get('n_images', 4)) if isinstance(point_source_model, dict) else 4
                                kw[key] = jnp.ones((n_img,))
                            else:
                                kw[key] = params[f'ps_{key}_{i}']
                        elif isinstance(param, list):
                            if key == 'amp' and nnls_linear_amps:
                                kw[key] = 1.0
                            else:
                                kw[key] = params[f'ps_{key}_{i}']
                        else:
                            kw[key] = param
                    kwargs_point_source.append(kw)
                
            kw_model = {
                'kwargs_lens': kwargs_lens,
                'kwargs_source': kwargs_source,
            }
            
            if len(kwargs_lens_light) > 0:
                kw_model['kwargs_lens_light'] = kwargs_lens_light
                
            if len(kwargs_point_source) > 0:
                kw_model['kwargs_point_source'] = kwargs_point_source
            
            return kw_model
        
    prob_model = ProbModel()
    
    return prob_model

def _linear_amp_n_images(point_dict, type_list, idx):
    ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
    ps_type = ps_types[idx] if idx < len(ps_types) else None
    if ps_type == 'IMAGE_POSITIONS':
        return int(point_dict.get('n_images', 4))
    return 1

def _point_source_amp_is_free(param_dict, ps_type, n_img):
    if 'amp' not in param_dict:
        return False
    p = param_dict['amp']
    if _normalize_link_spec(p) is not None:
        return False
    if ps_type == 'IMAGE_POSITIONS':
        if isinstance(p, list) and len(p) == 2:
            return True
        if isinstance(p, (list, tuple, np.ndarray)) and len(p) == n_img and all(
            isinstance(v, (int, float, np.floating)) for v in p
        ):
            return True
        if isinstance(p, (int, float, np.floating)):
            return True
        return False
    return isinstance(p, list)

def linear_amp_component_labels(param_list, type_list):
    """
    Human-readable names for each entry of the NNLS amplitude vector, in the same order as
    columns from build_linear_amp_design_matrix / apply_fnnls_coefficients_to_kwargs.
    """
    labels = []
    for i, m in enumerate(param_list.get('lens_mass_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        labels.append(f'kwargs_lens[{i}].amp')

    for i, m in enumerate(param_list.get('lens_light_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        labels.append(f'kwargs_lens_light[{i}].amp')

    for i, m in enumerate(param_list.get('source_light_params_list', [])):
        if not isinstance(m, dict):
            continue
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        labels.append(f'kwargs_source[{i}].amp')

    ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
    for i, psd in enumerate(param_list.get('point_source_params_list', [])):
        ps_type = ps_types[i] if i < len(ps_types) else None
        n_img = _linear_amp_n_images(psd, type_list, i)
        if not _point_source_amp_is_free(psd, ps_type, n_img):
            continue
        if ps_type == 'IMAGE_POSITIONS':
            for j in range(n_img):
                labels.append(f'kwargs_point_source[{i}].amp[{j}]')
        else:
            labels.append(f'kwargs_point_source[{i}].amp')

    return labels

@jax.jit
def _nnls_jax_fista(Aw, yw, x0, n_iter=200):
    """
    Accelerated projected-gradient NNLS (FISTA) with Gram matrix precomputation:
        min_x 0.5 ||Aw x - yw||^2  s.t. x >= 0
    """
    # 1. Precompute Gram matrix and vector (Massive O(N_pix) -> O(N_amp) speedup)
    AtA = Aw.T @ Aw
    Aty = Aw.T @ yw
    
    # 2. Optimal Step Size
    # The exact Lipschitz constant is the largest eigenvalue of the AtA matrix.
    # Using the exact max eigenvalue allows us to take the largest possible safe steps.
    L = jnp.linalg.eigvalsh(AtA)[-1] + 1e-12
    step = 1.0 / L

    def body(_, state):
        x, y_mom, t = state
        
        # 3. Fast Gradient using the precomputed tiny matrices
        grad = AtA @ y_mom - Aty
        
        # 4. Projected Gradient Step (ReLU)
        x_next = jnp.maximum(y_mom - step * grad, 0.0)
        
        # 5. FISTA Momentum Update
        t_next = (1.0 + jnp.sqrt(1.0 + 4.0 * t**2)) / 2.0
        y_next = x_next + ((t - 1.0) / t_next) * (x_next - x)
        
        return (x_next, y_next, t_next)

    # State layout: (current_x, momentum_y, time_step_t)
    initial_state = (x0, x0, 1.0)
    final_state = jax.lax.fori_loop(0, n_iter, body, initial_state)
    
    return final_state[0]

def solve_linear_amplitudes_jax(
    lens_image,
    kw_model,
    image_data,
    noise_map,
    param_list,
    type_list,
    jax_n_iter=200,
    x0_warm=None,
):
    """
    JAX-native pipeline: design matrix stacked with jnp, weighted NNLS via FISTA.
    Use x0_warm (previous iterate) to reduce FISTA iterations needed after the first step.
    Returns coefs as JAX arrays (no numpy conversion).
    """
    A = build_linear_amp_design_matrix_jax(lens_image, kw_model, param_list, type_list)
    if int(A.shape[1]) == 0:
        return deepcopy(kw_model), jnp.zeros((0,), dtype=jnp.float64), A

    y = jnp.asarray(image_data).ravel()
    sig = jnp.asarray(noise_map).ravel()
    sig = jnp.where(sig > 0.0, sig, 1e10)
    Aw = A / sig[:, None]
    yw = y / sig
    n_amp = int(A.shape[1])
    if x0_warm is not None:
        x0 = jnp.asarray(x0_warm, dtype=Aw.dtype).reshape((n_amp,))
        x0 = jnp.maximum(x0, 0.0)
    else:
        x0 = jnp.zeros((n_amp,), dtype=Aw.dtype)

    coefs = _nnls_jax_fista(Aw, yw, x0, n_iter=int(jax_n_iter))
    kw_filled = apply_nnls_coefficients_to_kwargs_jax(kw_model, coefs, param_list, type_list)
    return kw_filled, coefs, A

def count_sampling_parameters(param_list, type_list=None, use_nnls=False):
    """
    Count scalar sampled dimensions implied by `param_list` / `type_list`.

    Rules mirror create_prob_model():
    - linked parameters are never sampled;
    - regular list priors contribute 1 sampled scalar;
    - IMAGE_POSITIONS ra/dec contribute n_images each;
    - IMAGE_POSITIONS amp contributes n_images only for LogNormal prior [mu, sigma]
      when use_nnls is False;
    - when use_nnls is True, all free `amp` terms are treated as linear and not sampled.
    """
    n_lens_mass = 0
    n_lens_light = 0
    n_source_light = 0
    n_point_sources = 0
    n_linear_amps = 0

    # Lens mass
    for lens_mass_model in param_list.get('lens_mass_params_list', []):
        for key, param in lens_mass_model.items():
            if _normalize_link_spec(param) is not None or not isinstance(param, list):
                continue
            if key == 'amp':
                if use_nnls:
                    n_linear_amps += 1
                    continue
            n_lens_mass += 1

    # Lens light
    for lens_light_model in param_list.get('lens_light_params_list', []):
        for key, param in lens_light_model.items():
            if _normalize_link_spec(param) is not None or not isinstance(param, list):
                continue
            if key == 'amp':
                if use_nnls:
                    n_linear_amps += 1
                    continue
            n_lens_light += 1

    # Source light (parametric only; pixelated source uses numpyro.param, not sample)
    src_types = [] if type_list is None else type_list.get('source_light_type_list', [])
    if src_types != ['PIXELATED']:
        for source_light_model in param_list.get('source_light_params_list', []):
            if not isinstance(source_light_model, dict):
                continue
            for key, param in source_light_model.items():
                if _normalize_link_spec(param) is not None or not isinstance(param, list):
                    continue
                if key == 'amp':
                    if use_nnls:
                        n_linear_amps += 1
                        continue
                n_source_light += 1

    # Point sources
    ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
    for i, point_source_model in enumerate(param_list.get('point_source_params_list', [])):
        if not isinstance(point_source_model, dict):
            continue
        ps_type = ps_types[i] if i < len(ps_types) else None
        n_img = int(point_source_model.get('n_images', 4))

        for key, param in point_source_model.items():
            if key in ('n_images', 'sigma_image', 'sigma_source', 'pos_bound'):
                continue
            if _normalize_link_spec(param) is not None:
                continue

            if ps_type == 'IMAGE_POSITIONS':
                if key in ('ra', 'dec'):
                    # In create_prob_model these are vector sample sites (length n_img).
                    n_point_sources += n_img
                    continue
                if key == 'amp':
                    if use_nnls:
                        # IMAGE_POSITIONS amplitudes have one linear coefficient per image.
                        n_linear_amps += n_img
                        continue
                    # For IMAGE_POSITIONS amplitudes are sampled only for [mu, sigma].
                    if isinstance(param, list) and len(param) == 2:
                        n_point_sources += n_img
                    continue
                if isinstance(param, list):
                    n_point_sources += 1
            else:
                if isinstance(param, list):
                    if key == 'amp':
                        if use_nnls:
                            n_linear_amps += 1
                            continue
                    n_point_sources += 1

    n_params = n_lens_mass + n_lens_light + n_source_light + n_point_sources
    print(
        "Parameter count breakdown (sampled/nonlinear dims): "
        f"lens_mass={n_lens_mass}, lens_light={n_lens_light}, "
        f"source_light={n_source_light}, point_source={n_point_sources}, total={n_params}"
    )
    print(
        "Linear amplitude count (NNLS dims): "
        f"num_linear_amps={n_linear_amps} (use_nnls={use_nnls})"
    )
    return int(n_params), int(n_linear_amps)

def kwargs2params(param_list, kwargs, type_list=None):
    """
    Convert kwargs (kwargs_lens, kwargs_source, kwargs_point_source, ...) to the
    constrained params dict expected by NumPyro (site names -> values).
    Inverse of prob_model.params2kwargs. Values are returned as JAX arrays.
    """
    params = {}
    for i, lens_mass_model in enumerate(param_list['lens_mass_params_list']):
        for key, param in lens_mass_model.items():
            if _normalize_link_spec(param) is None and isinstance(param, list):
                params[f'lens_{key}_{i}'] = jnp.asarray(kwargs['kwargs_lens'][i][key])
    if 'lens_light_params_list' in param_list and 'kwargs_lens_light' in kwargs:
        for i, lens_light_model in enumerate(param_list['lens_light_params_list']):
            for key, param in lens_light_model.items():
                if _normalize_link_spec(param) is None and isinstance(param, list):
                    params[f'lens_light_{key}_{i}'] = jnp.asarray(kwargs['kwargs_lens_light'][i][key])
    if 'kwargs_source' in kwargs and len(kwargs['kwargs_source']) > 0:
        k0 = kwargs['kwargs_source'][0]
        if isinstance(k0, dict) and 'pixels' in k0:
            params['source_pixels'] = jnp.asarray(k0['pixels'])
    if 'source_light_params_list' in param_list:
        if 'source_pixels' not in params:
            for i, source_light_model in enumerate(param_list['source_light_params_list']):
                for key, param in source_light_model.items():
                    if _normalize_link_spec(param) is None and isinstance(param, list):
                        params[f'source_{key}_{i}'] = jnp.asarray(kwargs['kwargs_source'][i][key])
    if 'point_source_params_list' in param_list and 'kwargs_point_source' in kwargs:
        ps_type_list = [None] * len(param_list['point_source_params_list'])
        for i, point_source_model in enumerate(param_list['point_source_params_list']):
            ps_type = ps_type_list[i]
            if isinstance(type_list, dict):
                ps_type = type_list.get('point_source_type_list', ps_type_list)[i]
            for key, param in point_source_model.items():
                if key in ('n_images', 'sigma_image', 'sigma_source'):
                    continue
                if _normalize_link_spec(param) is not None:
                    continue
                if ps_type == 'IMAGE_POSITIONS' and key in ('ra', 'dec'):
                    # In the model these are sampled vector sites around provided image positions.
                    params[f'ps_{key}_{i}'] = jnp.asarray(kwargs['kwargs_point_source'][i][key])
                    continue
                if ps_type == 'IMAGE_POSITIONS' and key == 'amp':
                    # IMAGE_POSITIONS amp is sampled only for LogNormal prior [mu, sigma].
                    if isinstance(param, list) and len(param) == 2:
                        params[f'ps_{key}_{i}'] = jnp.asarray(kwargs['kwargs_point_source'][i][key])
                    continue
                if isinstance(param, list):
                    params[f'ps_{key}_{i}'] = jnp.asarray(kwargs['kwargs_point_source'][i][key])
    return params

def batch_log_likelihood(prob_model, samples, chunk_size=1024):
    """
    Evaluate log-likelihood for many parameter samples in one vectorized (GPU-friendly) call.

    Parameters
    ----------
    prob_model : NumpyroModel
        Probabilistic model with log_likelihood(params) -> scalar.
    samples : dict[str, array]
        Keys are parameter names; each value has shape (n_samples,) or (n_samples, ...).
        Can be numpy or JAX arrays.

    Returns
    -------
    np.ndarray
        Shape (n_samples,). Log-likelihood per sample; non-finite values are replaced with -inf.
    """
    param_names = list(samples.keys())
    n_total = samples[param_names[0]].shape[0]
    # Ensure JAX arrays so vmap runs on device
    samples_jax = {k: jnp.asarray(samples[k]) for k in param_names}
    # vmap over the batch dimension (axis 0) of the params dict
    batched_ll = jax.jit(jax.vmap(prob_model.log_likelihood, in_axes=0))

    out = np.empty((n_total,), dtype=np.float64)
    for i0 in range(0, n_total, chunk_size):
        i1 = min(i0 + chunk_size, n_total)
        chunk = {k: v[i0:i1] for k, v in samples_jax.items()}
        ll_chunk = np.asarray(batched_ll(chunk), dtype=np.float64)
        ll_chunk = np.where(np.isfinite(ll_chunk), ll_chunk, -np.inf)
        out[i0:i1] = ll_chunk
    return out

def _functional_set_amp(kw, comp_key, idx, val):
    """Safely updates an amplitude inside a nested dictionary without breaking JAX tracers."""
    new_kw = {k: v for k, v in kw.items()}
    if comp_key in new_kw:
        comp_list = list(new_kw[comp_key])
        if idx < len(comp_list):
            new_dict = dict(comp_list[idx])
            new_dict['amp'] = val
            comp_list[idx] = new_dict
        new_kw[comp_key] = comp_list
    return new_kw

def apply_nnls_coefficients_to_kwargs_jax(kw_model, coefs, param_list, type_list):
    """JIT-traceable version of applying NNLS amplitudes to the model dictionary."""
    kw = {k: v for k, v in kw_model.items()}
    for k in ['kwargs_lens', 'kwargs_lens_light', 'kwargs_source', 'kwargs_point_source']:
        if k not in kw: kw[k] = []

    idx = 0
    for i, m in enumerate(param_list.get('lens_mass_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None): continue
        kw = _functional_set_amp(kw, 'kwargs_lens', i, coefs[idx])
        idx += 1

    for i, m in enumerate(param_list.get('lens_light_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None): continue
        kw = _functional_set_amp(kw, 'kwargs_lens_light', i, coefs[idx])
        idx += 1

    for i, m in enumerate(param_list.get('source_light_params_list', [])):
        if not isinstance(m, dict): continue
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None): continue
        kw = _functional_set_amp(kw, 'kwargs_source', i, coefs[idx])
        idx += 1

    ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
    for i, psd in enumerate(param_list.get('point_source_params_list', [])):
        ps_type = ps_types[i] if i < len(ps_types) else None
        n_img = _linear_amp_n_images(psd, type_list, i)
        if not _point_source_amp_is_free(psd, ps_type, n_img): continue
        
        if ps_type == 'IMAGE_POSITIONS':
            kw = _functional_set_amp(kw, 'kwargs_point_source', i, coefs[idx:idx + n_img])
            idx += n_img
        else:
            kw = _functional_set_amp(kw, 'kwargs_point_source', i, coefs[idx])
            idx += 1

    return kw

def build_linear_amp_design_matrix_jax(lens_image, kw_model, param_list, type_list):
    """
    JIT-traceable function to build a design matrix of basis images.
    """
    cols = []

    # 1. Initialize base kwargs functionally (no deepcopy)
    kw_base = {k: v for k, v in kw_model.items()}
    for k in ['kwargs_lens', 'kwargs_lens_light', 'kwargs_source', 'kwargs_point_source']:
        if k not in kw_base:
            kw_base[k] = []

    # 2. ZERO-OUT ALL FREE AMPLITUDES
    # To get isolated basis images, all other components must be dark (amp=0.0).
    for comp_name, kw_key in [('lens_mass', 'kwargs_lens'), 
                              ('lens_light', 'kwargs_lens_light'), 
                              ('source_light', 'kwargs_source')]:
        for i, m in enumerate(param_list.get(f'{comp_name}_params_list', [])):
            if isinstance(m, dict) and isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None:
                kw_base = _functional_set_amp(kw_base, kw_key, i, 0.0)

    ps_types = [] if type_list is None else type_list.get('point_source_type_list', [])
    for i, psd in enumerate(param_list.get('point_source_params_list', [])):
        ps_type = ps_types[i] if i < len(ps_types) else None
        n_img = _linear_amp_n_images(psd, type_list, i)
        if _point_source_amp_is_free(psd, ps_type, n_img):
            zero_val = jnp.zeros(n_img, dtype=jnp.float64) if ps_type == 'IMAGE_POSITIONS' else 0.0
            kw_base = _functional_set_amp(kw_base, 'kwargs_point_source', i, zero_val)

    # 3. BUILD THE BASIS IMAGES (One-hot encoding the amplitudes)
    # Lens Mass
    for i, m in enumerate(param_list.get('lens_mass_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        kw = _functional_set_amp(kw_base, 'kwargs_lens', i, 1.0)
        col_img = jnp.asarray(lens_image.model(**kw))
        cols.append(col_img.reshape(-1))

    # Lens Light
    for i, m in enumerate(param_list.get('lens_light_params_list', [])):
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        kw = _functional_set_amp(kw_base, 'kwargs_lens_light', i, 1.0)
        col_img = jnp.asarray(lens_image.model(**kw))
        cols.append(col_img.reshape(-1))

    # Source Light
    for i, m in enumerate(param_list.get('source_light_params_list', [])):
        if not isinstance(m, dict):
            continue
        if not (isinstance(m.get('amp'), list) and _normalize_link_spec(m['amp']) is None):
            continue
        kw = _functional_set_amp(kw_base, 'kwargs_source', i, 1.0)
        col_img = jnp.asarray(lens_image.model(**kw))
        cols.append(col_img.reshape(-1))

    # Point Sources
    for i, psd in enumerate(param_list.get('point_source_params_list', [])):
        ps_type = ps_types[i] if i < len(ps_types) else None
        n_img = _linear_amp_n_images(psd, type_list, i)
        if not _point_source_amp_is_free(psd, ps_type, n_img):
            continue

        if ps_type == 'IMAGE_POSITIONS':
            for j in range(n_img):
                # JAX requires functional array updates using .at[].set()
                a = jnp.zeros(n_img, dtype=jnp.float64)
                a = a.at[j].set(1.0)
                kw = _functional_set_amp(kw_base, 'kwargs_point_source', i, a)
                col_img = jnp.asarray(lens_image.model(**kw))
                cols.append(col_img.reshape(-1))
        else:
            kw = _functional_set_amp(kw_base, 'kwargs_point_source', i, 1.0)
            col_img = jnp.asarray(lens_image.model(**kw))
            cols.append(col_img.reshape(-1))

    if len(cols) == 0:
        return jnp.zeros((0, 0), dtype=jnp.float64)
        
    return jnp.column_stack(cols)

def get_init_params(prob_model, param_list, type_list, init_params_path=None,
                    save_path=None, random_seed=42, use_nnls=False):

    master_key = jax.random.PRNGKey(random_seed)
    key, key_init = jax.random.split(master_key)

    init_params = prob_model.get_sample(key_init)

    if init_params_path is not None:

        shutil.copy(init_params_path, f'{save_path}/kwargs_init.json')
        with open(f'{save_path}/kwargs_init.json', 'r') as f:
            init_info = json.load(f)

        if isinstance(init_info, dict) and 'kwargs_lens' in init_info:
            init_params = kwargs2params(param_list, init_info, type_list=type_list)
            if use_nnls:
                # In NNLS mode, amplitudes are linear coefficients (not sampled non-linear sites).
                # Drop any amplitude entries loaded from kwargs init files to match prob_model sites.
                init_params = {k: v for k, v in init_params.items() if '_amp_' not in k}
            print(f"[Init] Loaded kwargs init from '{init_params_path}'")
        else:
            init_params = {k: jnp.asarray(v) for k, v in init_info.items()}
            print(f"[Init] Loaded site init from '{init_params_path}'")

    # Avoid exact-zero amplitudes at initialization (can violate positive-amplitude priors).
    for k, v in list(init_params.items()):
        if '_amp_' not in k:
            continue
        arr = jnp.asarray(v)
        init_params[k] = jnp.where(arr == 0.0, 1e-8, arr)

    return init_params

def create_pixel_grids(npix, pix_scl):

    half_size = npix * pix_scl / 2
    ra_at_xy_0 = dec_at_xy_0 = -half_size + pix_scl / 2  # position of the (0, 0) with respect to bottom left pixel
    transform_pix2angle = pix_scl * np.eye(2)  # transformation matrix pixel <-> angle
    kwargs_pixel = {'nx': npix, 'ny': npix,
                    'ra_at_xy_0': ra_at_xy_0, 'dec_at_xy_0': dec_at_xy_0,
                    'transform_pix2angle': transform_pix2angle}
    
    pixel_grid = PixelGrid(**kwargs_pixel)
    
    ps_grid_npix = 2 * npix + 1
    ps_grid_pix_scl = (pix_scl * npix) / ps_grid_npix
    ps_grid_half_size = ps_grid_npix * ps_grid_pix_scl / 2.
    ps_grid_ra_at_xy_0 = ps_grid_dec_at_xy_0 = -ps_grid_half_size + ps_grid_pix_scl / 2.
    ps_grid_transform_pix2angle = ps_grid_pix_scl * np.eye(2)
    kwargs_ps_grid = {'nx': ps_grid_npix, 'ny': ps_grid_npix,
                    'ra_at_xy_0': ps_grid_ra_at_xy_0, 'dec_at_xy_0': ps_grid_dec_at_xy_0,
                    'transform_pix2angle': ps_grid_transform_pix2angle}
    ps_grid = PixelGrid(**kwargs_ps_grid)
    
    return pixel_grid, ps_grid

def create_lens_image(
    param_list, 
    type_list, 
    image_data, 
    noise_map, 
    psf_data,
    pixel_scale,
    kwargs_numerics=None,
    kwargs_lens_equation_solver=None,
):

    num_pixels = image_data.shape[0]
    psf = PSF(psf_type='PIXEL', kernel_point_source=psf_data, pixel_size=pixel_scale)
    noise = Noise(nx=num_pixels, ny=num_pixels, noise_map=noise_map)
    pixel_grid, ps_grid = create_pixel_grids(num_pixels, pixel_scale)

    lens_mass_model = None
    lens_light_model = None
    source_light_model = None
    point_source_model = None

    if 'lens_mass_type_list' in type_list:
        if 'lens_mass_params_list' not in param_list:
            raise ValueError('lens_mass_params_list not found in param_list')

        lens_mass_model = MassModel(type_list['lens_mass_type_list'])
        
    if 'lens_light_type_list' in type_list:
        if 'lens_light_params_list' not in param_list:
            raise ValueError('lens_light_params_list not found in param_list')

        lens_light_model = LightModel(type_list['lens_light_type_list'])
        
    if 'source_light_type_list' in type_list:
        if 'source_light_params_list' not in param_list:
            raise ValueError('source_light_params_list not found in param_list')

        source_light_model = LightModel(type_list['source_light_type_list'])
    
    if 'point_source_type_list' in type_list:
        if 'point_source_params_list' not in param_list:
            raise ValueError('point_source_params_list not found in param_list')

        point_source_model = PointSourceModel(type_list['point_source_type_list'], lens_mass_model, ps_grid)
    
    if kwargs_numerics is None:
        kwargs_numerics = {'supersampling_factor': 1}

    lens_image = LensImage(
        grid_class=pixel_grid,
        psf_class=psf,
        noise_class=noise,
        lens_mass_model_class=lens_mass_model,
        lens_light_model_class=lens_light_model,
        source_model_class=source_light_model,
        point_source_model_class=point_source_model,
        kwargs_numerics=kwargs_numerics,
        kwargs_lens_equation_solver=kwargs_lens_equation_solver,
    )

    return lens_image

def validate_param_list(type_list, param_list):
    """
    Ensure type_list / param_list use matching keys and equal list lengths per component,
    and that every ``['correlated', component, index, param_name]`` entry points to a
    base parameter that exists and is visible when the probabilistic model is built
    (same rules as ``create_prob_model`` / ``_resolve_link``).
    """
    if not isinstance(type_list, dict) or not isinstance(param_list, dict):
        raise TypeError("type_list and param_list must be dicts.")

    _PAIRS = (
        ("lens_mass_type_list", "lens_mass_params_list"),
        ("lens_light_type_list", "lens_light_params_list"),
        ("source_light_type_list", "source_light_params_list"),
        ("point_source_type_list", "point_source_params_list"),
    )

    for type_key, param_key in _PAIRS:
        has_t = type_key in type_list
        has_p = param_key in param_list
        if has_t != has_p:
            raise ValueError(
                f"type_list and param_list must both contain '{type_key}' and '{param_key}', "
                f"or omit both; found type={has_t}, params={has_p}."
            )
        if has_t:
            tl, pl = type_list[type_key], param_list[param_key]
            if not isinstance(tl, (list, tuple)):
                raise TypeError(f"{type_key} must be a list, got {type(tl).__name__}.")
            if not isinstance(pl, (list, tuple)):
                raise TypeError(f"{param_key} must be a list, got {type(pl).__name__}.")
            if len(tl) != len(pl):
                raise ValueError(
                    f"Length mismatch: {type_key} has {len(tl)} entries but {param_key} has {len(pl)}."
                )

    # Bank construction order in create_prob_model: lens → lens_light → source → point_source
    _SECTION_RANK = {
        "lens_mass_params_list": 0,
        "lens_light_params_list": 1,
        "source_light_params_list": 2,
        "point_source_params_list": 3,
    }
    _TARGET_RANK = {"lens": 0, "lens_light": 1, "source": 2, "point_source": 3}
    _PARAM_LIST_BY_TARGET = {
        "lens": "lens_mass_params_list",
        "lens_light": "lens_light_params_list",
        "source": "source_light_params_list",
        "point_source": "point_source_params_list",
    }

    def _lengths():
        return {
            "lens": len(param_list.get("lens_mass_params_list", [])),
            "lens_light": len(param_list.get("lens_light_params_list", [])),
            "source": len(param_list.get("source_light_params_list", [])),
            "point_source": len(param_list.get("point_source_params_list", [])),
        }

    def _check_link(section_key, model_idx, field_name, param_val):
        if not _is_correlated_param(param_val):
            return
        ctx = f"{section_key}[{model_idx}].{field_name}"
        try:
            spec = _normalize_link_spec(param_val)
        except ValueError as e:
            raise ValueError(f"Invalid correlated spec at {ctx}: {e}") from e
        if spec is None:
            return
        tgt_comp, tgt_idx, tgt_key = spec
        lens = _lengths()
        if tgt_comp not in lens:
            raise ValueError(f"Correlated spec at {ctx} has unknown target component {tgt_comp!r}.")
        n = lens[tgt_comp]
        if tgt_idx < 0 or tgt_idx >= n:
            raise IndexError(
                f"Correlated spec at {ctx} targets {tgt_comp}[{tgt_idx}], but there are only {n} such component(s)."
            )
        plist_key = _PARAM_LIST_BY_TARGET[tgt_comp]
        base_model = param_list[plist_key][tgt_idx]
        if not isinstance(base_model, dict):
            raise TypeError(f"Expected dict at {plist_key}[{tgt_idx}], got {type(base_model).__name__}.")
        if tgt_key not in base_model:
            raise KeyError(
                f"Correlated field at {ctx} points to {tgt_comp}[{tgt_idx}].{tgt_key!r}, "
                f"but that base entry is missing. Available keys: {sorted(base_model.keys())}."
            )

        from_rank = _SECTION_RANK[section_key]
        tgt_r = _TARGET_RANK[tgt_comp]
        if tgt_r > from_rank:
            raise ValueError(
                f"Correlated field at {ctx} references {tgt_comp} (rank {tgt_r}), which is not built yet "
                f"when sampling {section_key} (rank {from_rank}). Use an earlier component as the base."
            )
        if tgt_r == from_rank and tgt_idx >= model_idx:
            raise ValueError(
                f"Correlated field at {ctx} must reference a strictly earlier model in the same section "
                f"(need index < {model_idx}, got {tgt_idx})."
            )

    for i, model in enumerate(param_list.get("lens_mass_params_list", [])):
        if not isinstance(model, dict):
            raise TypeError(f"lens_mass_params_list[{i}] must be a dict, got {type(model).__name__}.")
        for k, v in model.items():
            _check_link("lens_mass_params_list", i, k, v)

    for i, model in enumerate(param_list.get("lens_light_params_list", [])):
        if not isinstance(model, dict):
            raise TypeError(f"lens_light_params_list[{i}] must be a dict, got {type(model).__name__}.")
        for k, v in model.items():
            _check_link("lens_light_params_list", i, k, v)

    for i, model in enumerate(param_list.get("source_light_params_list", [])):
        if not isinstance(model, dict):
            raise TypeError(f"source_light_params_list[{i}] must be a dict, got {type(model).__name__}.")
        for k, v in model.items():
            _check_link("source_light_params_list", i, k, v)

    for i, model in enumerate(param_list.get("point_source_params_list", [])):
        if not isinstance(model, dict):
            raise TypeError(f"point_source_params_list[{i}] must be a dict, got {type(model).__name__}.")
        for k, v in model.items():
            if k in ("n_images", "sigma_image", "sigma_source"):
                continue
            _check_link("point_source_params_list", i, k, v)