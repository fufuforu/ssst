"""Additional gradient contracts required only by the new coupling variants."""
import math


def validate_coupling_smoke(receipt, preflight):
    recipe = preflight['extra_checkpoint_metadata']['instance_coupling_recipe']
    for row in receipt['ranks']:
        checks = row['coupling_checks']
        if recipe['lift']:
            if not checks['lift_geometry_gradient_absent'] or not row['lift_head_changed']:
                raise ValueError('Lift geometry detach or head update contract failed')
            if not all(math.isfinite(checks[k]) and checks[k] > 0 for k in
                       ('lift_feature_gradient_norm', 'lift_residual_norm', 'lift_visible_gaussians')):
                raise ValueError('Inactive or nonfinite lifting path')
        if recipe['boundary']:
            if not checks['boundary_instance_gradient_absent'] or not checks['boundary_weights_detached']:
                raise ValueError('Boundary source must be detached')
            if not math.isfinite(checks['boundary_rgb_gradient_norm']) or checks['boundary_rgb_gradient_norm'] <= 0:
                raise ValueError('Missing boundary RGB gradient')
    return True
