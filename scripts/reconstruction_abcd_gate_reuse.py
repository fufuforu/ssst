"""Validate explicit reuse of completed A/B/D GPU gates; never accepts C reuse."""
import ast,hashlib,json
from pathlib import Path

def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()

class StripC(ast.NodeTransformer):
    def visit_If(self,node):
        if "arm == 'C'" in ast.unparse(node.test):return [self.visit(x) for x in node.orelse]
        return self.generic_visit(node)

def update_signature(path):
    tree=ast.parse(Path(path).read_text())
    loops=[n for n in ast.walk(tree) if isinstance(n,ast.For) and ast.unparse(n.target)=='step' and ast.unparse(n.iter)=='range(limit)']
    assert len(loops)==1
    body=StripC().visit(loops[0]);return hashlib.sha256(ast.dump(body,include_attributes=False).encode()).hexdigest()

def task_signature(path):
    tree=ast.parse(Path(path).read_text());node=next(n for n in tree.body if isinstance(n,ast.ClassDef) and n.name=='Task')
    return hashlib.sha256(ast.dump(node,include_attributes=False).encode()).hexdigest()

def validate(run,cfg,arm):
    gate_path=run/'smoke/SMOKE_PASSED.json';gate=json.loads(gate_path.read_text())
    if gate['status']!='PASSED' or gate['world_size']!=8 or gate['added_updates']!=2 or gate.get('arm')!=arm:raise ValueError('Strict GPU gate missing')
    if gate['code_sha']==cfg['code_sha']:return gate
    if arm not in ('A','B','D'):raise ValueError('C must pass a fresh complete smoke')
    receipt=json.loads((run/'REUSED_SMOKE.json').read_text())
    if receipt['source_code_sha']!=gate['code_sha'] or receipt['target_code_sha']!=cfg['code_sha'] or receipt['source_checkpoint_sha256']!=cfg['source_sha256']:raise ValueError('Reused gate identity differs')
    root=Path(cfg['code_snapshot'])
    if update_signature(root/'scripts/train_reconstruction_abcd.py')!=receipt['non_C_update_ast_sha256']:raise ValueError('A/B/D update loop changed')
    if task_signature(root/'scripts/train_reconstruction_abcd.py')!=receipt['task_ast_sha256']:raise ValueError('Training Task changed')
    for relative,expected in receipt['unchanged_science_files'].items():
        if sha(root/relative)!=expected:raise ValueError('Science code changed '+relative)
    for relative,expected in receipt['evidence_files'].items():
        if sha(run/'smoke'/relative)!=expected:raise ValueError('Reused GPU receipt changed '+relative)
    return gate
