"""Standard-library identity for serializer-only recovery of reading controls."""
import ast
import hashlib
from pathlib import Path

FILES = (
    'tokengs/models/object_locus_reading_controls_v1.py',
    'tokengs/models/object_locus_panoptic_v1.py',
    'tokengs/models/object_locus_panoptic_v1_controller.py',
    'tokengs/models/object_locus_frozen_vggt_posefree.py',
    'tokengs/models/object_locus_mask_support_readback_v1.py',
    'tokengs/models/object_locus_feedback_factorial_v1.py',
    'tokengs/models/object_locus_posefree_geometry.py',
    'scripts/object_locus_frozen_vggt_posefree_runtime.py',
    'scripts/train_vggt_classification_readback_v1.py',
    'scripts/probe_vggt_raw_query_readout_v1.py',
    'scripts/runtime_bootstrap.py',
)


def contract(root):
    root = Path(root)
    result = {file: hashlib.sha256((root / file).read_bytes()).hexdigest() for file in FILES}
    tree = ast.parse((root / 'scripts/train_vggt_reading_controls_v1.py').read_text())
    functions = {node.name: node for node in tree.body if isinstance(node, ast.FunctionDef)}
    for name in ('config', 'build', 'slots', 'worker', 'zero_reproduction', 'apply_state'):
        result['AST:' + name] = hashlib.sha256(ast.dump(functions[name], include_attributes=False).encode()).hexdigest()
    loop = next(node for node in ast.walk(functions['run']) if isinstance(node, ast.For) and isinstance(node.target, ast.Name) and node.target.id == 'update')
    result['AST:joint_training_update_loop'] = hashlib.sha256(ast.dump(loop, include_attributes=False).encode()).hexdigest()
    return result
